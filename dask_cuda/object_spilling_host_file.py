import abc
import threading
import time
import weakref
from typing import DefaultDict, Dict, Hashable, List, MutableMapping, Set
from dask.sizeof import sizeof

from .proxify_device_object import proxify_device_object
from .get_device_memory_objects import get_device_memory_objects
from .proxy_object import ProxyObject


from collections import MutableSet


class IdentitySet(MutableSet):
    key = id  # should return a hashable object

    def __init__(self, iterable=()):
        self.map = {}  # id -> object
        self |= iterable  # add elements from iterable to the set (union)

    def __len__(self):  # Sized
        return len(self.map)

    def __iter__(self):  # Iterable
        return iter(self.map.values())

    def __contains__(self, x):  # Container
        return self.key(x) in self.map

    def add(self, value):  # MutableSet
        """Add an element."""
        self.map[self.key(value)] = value

    def discard(self, value):  # MutableSet
        """Remove an element.  Do not raise an exception if absent."""
        self.map.pop(self.key(value), None)

    def __repr__(self):
        if not self:
            return "%s()" % (self.__class__.__name__,)
        return "%s(%r)" % (self.__class__.__name__, list(self))


class ObjectSpillingHostFile(MutableMapping):
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

        self.key_to_proxies: Dict[Hashable, IdentitySet] = DefaultDict(IdentitySet)
        self.proxied_id_to_proxy: Dict[int, ProxyObject] = {}

    def __contains__(self, key):
        return key in self.store

    def __len__(self):
        return len(self.store)

    def __iter__(self):
        return iter(self.store)

    def unspilled_proxies(self) -> List[ProxyObject]:
        found_proxies = []
        proxied_id_to_proxy = {}
        proxify_device_object(self.store, proxied_id_to_proxy, found_proxies)
        ret = list(proxied_id_to_proxy.values())
        assert len(ret) == len(set(id(p) for p in ret))  # No duplicates
        return ret

    def obj_mappings(self):
        # TODO: simplify and optimize
        proxied_id_to_proxy = {}
        buffer_to_proxies = {}

        for p in self.unspilled_proxies():
            proxied = p._obj_pxy["obj"]
            proxied_id = id(proxied)
            assert proxied_id not in proxied_id_to_proxy
            proxied_id_to_proxy[proxied_id] = p
            for buf in p._obj_pxy_get_device_memory_objects():
                l = buffer_to_proxies.get(buf, [])
                if id(p) not in set(id(i) for i in l):
                    l.append(p)
                buffer_to_proxies[buf] = l
        return proxied_id_to_proxy, buffer_to_proxies

    def get_proxied_id_to_proxy(self) -> Dict[int, ProxyObject]:
        with self.lock:
            ret = {}
            for proxies in self.key_to_proxies.values():
                for proxy in proxies:
                    if proxy._obj_pxy_serialized():
                        proxied = proxy._obj_pxy["obj"]
                        proxied_id = id(proxied)
                        p = ret.get(proxied_id, None)
                        if p is None:
                            ret[proxied_id] = proxied
                        else:
                            assert id(p) == id(proxy)  # No duplicates
            return ret

    def __setitem__(self, key, value):
        with self.lock:
            found_proxies = []
            proxied_id_to_proxy = self.get_proxied_id_to_proxy()
            self.store[key] = proxify_device_object(
                value, proxied_id_to_proxy, found_proxies
            )
            last_access = time.time()
            self_weakref = weakref.ref(self)
            for p in found_proxies:
                p._obj_pxy["hostfile"] = self_weakref
                p._obj_pxy["last_access"] = last_access
                self.key_to_proxies[key].add(p)

            self.maybe_evict()

    def __getitem__(self, key):
        return self.store[key]

    def __delitem__(self, key):
        del self.store[key]
        self.key_to_proxies.pop(key, None)

    def evict(self, proxy):
        proxy._obj_pxy_serialize(serializers=("dask", "pickle"))

    def buffer_info(self, ignores=()):
        buffers = []
        dev_mem_usage = 0
        for buf, proxies in self.obj_mappings()[1].items():
            last_access = max(p._obj_pxy.get("last_access", 0) for p in proxies)
            size = sizeof(buf)
            buffers.append((last_access, size, proxies))
            dev_mem_usage += size
        return buffers, dev_mem_usage

    def maybe_evict(self, extra_dev_mem=0, ignores=()):
        buffers, dev_mem_usage = self.buffer_info()
        dev_mem_usage += extra_dev_mem
        if dev_mem_usage > self.device_memory_limit:
            buffers.sort(key=lambda x: (x[0], -x[1]))
            for _, size, proxies in buffers:
                for p in proxies:
                    self.evict(p)
                dev_mem_usage -= size
                if dev_mem_usage <= self.device_memory_limit:
                    break
