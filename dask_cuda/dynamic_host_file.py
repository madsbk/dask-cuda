from collections import defaultdict
import functools
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
        self.proxies: Dict[int, proxy_object.ProxyObject] = {}

    def __contains__(self, key):
        return key in self.store

    def __len__(self):
        return len(self.store)

    def __iter__(self):
        return iter(self.store)

    def __setitem__(self, key, value):
        value = proxify_device_object(value, self)
        self.store[key] = value

    def __getitem__(self, key):
        return self.store[key]

    def __delitem__(self, key):
        del self.store[key]

    def evict(self, proxy):
        proxy._obj_pxy_serialize(serializers=["dask", "pickle"])

    def maybe_evict(self, ignore):
        in_dev_mem = []
        for p in self.proxies.values():
            if p._obj_pxy["serializers"] is None and p not in ignore:
                last_time = p._obj_pxy.get("last_time", 0)
                size = sizeof(p._obj_pxy["obj"])
                in_dev_mem.append((last_time, size, p))

        total_dev_mem = functools.reduce(lambda x: x[1], in_dev_mem)
        if total_dev_mem > self.device_memory_limit:
            sorted(in_dev_mem, key=lambda x: (x[0], -x[1]))
            print(f"maybe_evict() - total_dev_mem: {total_dev_mem}")
            print("in_dev_mem: ", in_dev_mem)
            for last_time, size, p in in_dev_mem:
                print(f"evicting : ", (last_time, size, p))
                self.evict(p)
                total_dev_mem -= size
                if total_dev_mem <= self.device_memory_limit:
                    break










    # def validate(self):
    #     unique_proxies = set()
    #     for p in self.proxies.values():
    #         assert p.__
