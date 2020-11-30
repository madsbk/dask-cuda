from collections import defaultdict

from numba.core.types.containers import BaseTuple
from dask_cuda.proxy_object import dev_id, unproxy
import functools
import threading
import time
import weakref
import os
from typing import Any, Dict, MutableMapping
from dask.sizeof import sizeof

from zict import Buffer, File, Func
from zict.common import ZictBase

import dask
from distributed.protocol import (
    dask_deserialize,
    dask_serialize,
    deserialize,
    deserialize_bytes,
    serialize,
    serialize_bytelist,
)
from distributed.utils import nbytes
from distributed.worker import weight

from . import proxy_object, device_host_file
from .proxify_device_object import proxify_device_object
from .is_device_object import is_device_object
from .utils import nvtx_annotate


class DynamicHostFile(MutableMapping):
    """Manages serialization/deserialization of objects.

    TODO: Three LRU cache levels are controlled, for device, host and disk.
    Each level takes care of serializing objects once its limit has been
    reached and pass it to the subsequent level. Similarly, each cache
    may deserialize the object, but storing it back in the appropriate
    cache, depending on the type of object being deserialized.

    Parameters
    ----------
    device_memory_limit: int
        Number of bytes of CUDA device memory for device LRU cache,
        spills to host cache once filled.
    TODO: memory_limit: int
        Number of bytes of host memory for host LRU cache, spills to
        disk once filled. Setting this to 0 means unlimited host memory,
        implies no spilling to disk.
    local_directory: path
        Path where to store serialized objects on disk
    """

    def __init__(
        self,
        device_memory_limit: int,
        **kwargs,
    ):
        self.device_memory_limit = device_memory_limit
        self.store = {}
        self.lock = threading.RLock()
        import cupy
        import cudf
        import gc
        import time
        dummpy1 = cupy.random.random(2**30//8)
        dummpy2 = cudf.datasets.timeseries(
            start="2000-01-01",
            end="2001-01-02",
            freq="1s",
        )

        from .proxy_object import dev_id, dev_used_mem
        did = dev_id()
        if did == 1 and False:
            gc.collect()
            time.sleep(1)
            gc.collect()
            m1 = dev_used_mem()
            d = cudf.datasets.timeseries(
                start="2000-01-01",
                end="2001-01-02",
                freq="1s",
            )
            #d = cupy.random.random(2**30)
            d_size = sizeof(d)/1024/1024/1024
            gc.collect()
            m2 = dev_used_mem()
            del d
            gc.collect()
            m3 = dev_used_mem()

            print("DynamicHostFile() - dev_id: %d, sizeof(d): %.4f, %.4f => %.4f => %.4f GB" % (did, d_size, m1, m2, m3))

        # self.proxied_id_to_proxy = Dict[int, proxy_object.ProxyObject] = {}
        # self.proxy_id_to_proxy: Dict[int, proxy_object.ProxyObject] = {}
        # self.proxies: Dict[int, proxy_object.ProxyObject] = {}

    def __contains__(self, key):
        return key in self.store

    def __len__(self):
        return len(self.store)

    def __iter__(self):
        return iter(self.store)

    def unspilled_proxies(self):
        found_proxies = []
        proxied_id_to_proxy = {}
        bases = set()
        proxify_device_object(self.store, proxied_id_to_proxy, found_proxies, bases)
        ret = list(proxied_id_to_proxy.values())
        assert len(ret) == len(set(id(p) for p in ret))  # No duplicates
        return ret

    def dev_mem_usage(self):
        from .get_device_memory_objects import get_device_memory_objects
        with self.lock:
            ret = 0
            dev_ids = set()
            for p in self.unspilled_proxies():
                for m in get_device_memory_objects(p._obj_pxy["obj"]):
                    if id(m) not in dev_ids:
                        ret += sizeof(m)
                        dev_ids.add(id(m))
            return ret

    def proxied_id_to_proxy(self):
        ret = {}
        for p in self.unspilled_proxies():
            _id = id(p._obj_pxy["obj"])
            assert _id not in ret
            ret[_id] = p
        return ret

    def __setitem__(self, key, value):
        from .get_device_memory_objects import get_device_memory_objects
        self.check_alias()
        with self.lock:
            found_proxies = []
            bases = set()
            self.store[key] = proxify_device_object(
                value, self.proxied_id_to_proxy(), found_proxies, bases
            )
            last_access = time.time()
            self_weakref = weakref.ref(self)
            for p in found_proxies:
                p._obj_pxy["hostfile"] = self_weakref
                p._obj_pxy["last_access"] = last_access

            dev_objs = []
            for p in self.unspilled_proxies():
                dev_objs.extend(get_device_memory_objects(p._obj_pxy["obj"]))

            if len(dev_objs) != len(set(dev_objs)):
                msg = f"\nkey: {key}\n"

                from pprint import pprint
                print(dev_objs)
                pprint(value)



            assert len(dev_objs) == len(set(dev_objs)), key
            self.maybe_evict()

    def __getitem__(self, key):
        self.check_alias()
        return self.store[key]

    def __delitem__(self, key):
        self.check_alias()
        del self.store[key]

    def evict(self, proxy):
        self.check_alias()
        proxy._obj_pxy_serialize(serializers=["dask", "pickle"], check_leak=True)

    def evict_all(self, ignores=()):
        ignores = set(id(p) for p in ignores)
        for p in self.unspilled_proxies():
            if id(p) not in ignores:
                self.evict(p)

    def check_alias(self):
        return
        import cudf
        bases = set()
        for p in self.unspilled_proxies():
            o = p._obj_pxy['obj']
            if isinstance(o, cudf.DataFrame):
                for col_name in o.columns:
                    col = o[col_name]
                    i = id(col.data)
                    if i in bases:
                        print("check_alias()")
                    assert i not in bases
                    bases.add(i)

        bases = set()
        bases_obj = set()
        from .proxy_object import get_owners
        from rmm._lib.device_buffer import DeviceBuffer
        for p in self.unspilled_proxies():
            for o in get_owners(p._obj_pxy['obj']):
                i = (o)
                if i in bases:
                    print(f"check_alias() - o: {repr(o)}, bases: {bases_obj}")
                    assert False
                #assert i not in bases, repr(o)
                bases.add(i)
                bases_obj.add(o)





    def evict_oldest(self):
        with self.lock:
            in_dev_mem = []
            total_dev_mem = 0
            for p in self.unspilled_proxies():
                last_access = p._obj_pxy.get("last_access", 0)
                size = sizeof(p._obj_pxy["obj"])
                in_dev_mem.append((last_access, size, p))
                total_dev_mem += size
            sorted(in_dev_mem, key=lambda x: (x[0], -x[1]))
            last_access, size, p = in_dev_mem[0]
            self.evict(p)
            return p

    def maybe_evict(self, extra_dev_mem=0, ignores=()):
        self.check_alias()
        with self.lock:
            in_dev_mem = []
            total_dev_mem = 0
            ignores = set(id(p) for p in ignores)
            for p in list(self.unspilled_proxies()):
                if id(p) not in ignores:
                    last_access = p._obj_pxy.get("last_access", 0)
                    size = sizeof(p._obj_pxy["obj"])
                    in_dev_mem.append((last_access, size, p))
                    total_dev_mem += size

            total_dev_mem += extra_dev_mem
            if total_dev_mem > self.device_memory_limit:
                sorted(in_dev_mem, key=lambda x: (x[0], -x[1]))
                for last_access, size, p in in_dev_mem:
                    self.evict(p)
                    total_dev_mem -= size
                    if total_dev_mem <= self.device_memory_limit:
                        break
            if total_dev_mem > self.device_memory_limit:
                print("Warning maybe_evict() - total_dev_mem: %.4f GB, device_memory_limit: %.4f GB" % (total_dev_mem/1024/1024/1024, self.device_memory_limit/1024/1024/1024))


    #def maybe_evict_correct(self, extra_dev_mem=0, ignores=()):


