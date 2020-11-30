import time

from dask.utils import Dispatch
from .proxy_object import ProxyObject, asproxy
from dask.sizeof import sizeof

proxify_device_object = Dispatch(name="proxify_device_object")


def proxify(obj, proxied_id_to_proxy, found_proxies, bases):
    ptr = id(obj)
    if ptr in proxied_id_to_proxy:
        print("resusing!\n\n\n")
        ret = proxied_id_to_proxy[ptr]
    else:
        proxied_id_to_proxy[ptr] = ret = asproxy(obj)

    # print(f"proxify() - ptr: {hex(ptr)}, obj: {repr(obj)}, ret: {repr(ret)}")
    found_proxies.append(ret)


    # from .proxy_object import get_owners
    # for o in get_owners(obj):
    #     assert o not in bases

    # for o in get_owners(obj):
    #     bases.add(o)

    return ret


@proxify_device_object.register(object)
def proxify_device_object_default(obj, proxied_id_to_proxy, found_proxies, bases):
    if hasattr(obj, "__cuda_array_interface__"):
        return proxify(obj, proxied_id_to_proxy, found_proxies, bases)
    else:
        if sizeof(obj)> 2*20 or True:
            print("proxify_device_object_default:", type(obj))
    return obj


@proxify_device_object.register(ProxyObject)
def proxify_device_object_proxy_object(obj, proxied_id_to_proxy, found_proxies, bases):
    if obj._obj_pxy["serializers"] is None:
        ptr = id(obj)
        if ptr in proxied_id_to_proxy:
            obj = proxied_id_to_proxy[ptr]
        else:
            proxied_id_to_proxy[ptr] = obj

        # from .proxy_object import get_owners
        # for o in get_owners(obj):
        #     assert o not in bases

        # for o in get_owners(obj):
        #     bases.add(o)

    found_proxies.append(obj)
    return obj


@proxify_device_object.register(list)
@proxify_device_object.register(tuple)
@proxify_device_object.register(set)
@proxify_device_object.register(frozenset)
def proxify_device_object_python_collection(seq, proxied_id_to_proxy, found_proxies, bases):
    return type(seq)(
        proxify_device_object(o, proxied_id_to_proxy, found_proxies, bases) for o in seq
    )


@proxify_device_object.register(dict)
def proxify_device_object_python_dict(seq, proxied_id_to_proxy, found_proxies, bases):
    for k,v in seq.items():
        assert v is not seq
    return {
        k: proxify_device_object(v, proxied_id_to_proxy, found_proxies, bases)
        for k, v in seq.items()
    }


@proxify_device_object.register_lazy("cudf")
def register_cudf():
    import cudf

    @proxify_device_object.register(cudf.DataFrame)
    @proxify_device_object.register(cudf.Series)
    @proxify_device_object.register(cudf.Index)
    def proxify_device_object_cudf_dataframe(obj, proxied_id_to_proxy, found_proxies, bases):
        return proxify(obj, proxied_id_to_proxy, found_proxies, bases)
