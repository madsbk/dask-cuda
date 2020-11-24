from collections import defaultdict
import functools
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

from . import proxy_object
from .proxify_device_object import proxify_device_object
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
    ):
        self.device_memory_limit = device_memory_limit
        self.store = {}

        # self.proxied_id_to_proxy = Dict[int, proxy_object.ProxyObject] = {}
        # self.proxy_id_to_proxy: Dict[int, proxy_object.ProxyObject] = {}
        # self.proxies: Dict[int, proxy_object.ProxyObject] = {}

    def __contains__(self, key):
        return key in self.store

    def __len__(self):
        return len(self.store)

    def __iter__(self):
        return iter(self.store)

    # @property
    # def proxies(self):
    #     found_proxies = []
    #     proxied_id_to_proxy = {}
    #     proxify_device_object(self.store, proxied_id_to_proxy, found_proxies)
    #     assert len(found_proxies) == len(set(id(p) for p in found_proxies))  # No duplicates
    #     return found_proxies

    def unspilled_proxies(self):
        found_proxies = []
        proxied_id_to_proxy = {}
        proxify_device_object(self.store, proxied_id_to_proxy, found_proxies)
        ret = proxied_id_to_proxy.values()
        assert len(ret) == len(set(id(p) for p in ret))  # No duplicates
        return ret

    @property
    def proxied_id_to_proxy(self):
        ret = {}
        for p in self.unspilled_proxies():
            _id = id(p._obj_pxy["obj"])
            assert _id not in ret
            ret[_id] = p
        return ret

    def __setitem__(self, key, value):
        found_proxies = []
        self.store[key] = proxify_device_object(
            value, self.proxied_id_to_proxy, found_proxies
        )
        last_access = time.time()
        self_weakref = weakref.ref(self)
        for p in found_proxies:
            p._obj_pxy["hostfile"] = self_weakref
            p._obj_pxy["last_access"] = last_access
        self.maybe_evict()

    def __getitem__(self, key):
        return self.store[key]

    def __delitem__(self, key):
        del self.store[key]

    def evict(self, proxy):
        proxy._obj_pxy_serialize(serializers=["dask", "pickle"])

    def maybe_evict(self, extra_dev_mem=0):
        in_dev_mem = []
        for p in self.unspilled_proxies():
            last_access = p._obj_pxy.get("last_access", 0)
            size = sizeof(p._obj_pxy["obj"])
            in_dev_mem.append((last_access, size, p))
        if len(in_dev_mem) > 1:
            total_dev_mem = functools.reduce(lambda x, y: x[1] + y[1], in_dev_mem)
        elif len(in_dev_mem) == 1:
            total_dev_mem = in_dev_mem[0][1]
        else:
            total_dev_mem = 0
        total_dev_mem += extra_dev_mem
        if total_dev_mem > self.device_memory_limit:
            sorted(in_dev_mem, key=lambda x: (x[0], -x[1]))
            for last_access, size, p in in_dev_mem:
                self.evict(p)
                total_dev_mem -= size
                if total_dev_mem <= self.device_memory_limit:
                    break

    # def validate(self):
    #     proxies_in_store = []
    #     for value in self.store.values():
    #         proxify_device_object(value, self.proxies, proxies_in_store)

    #     assert [id(p) for p in proxies_in_store]
