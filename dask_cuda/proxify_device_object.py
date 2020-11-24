import weakref

from dask.utils import Dispatch
from .proxy_object import ProxyObject, asproxy

proxify_device_object = Dispatch(name="proxify_device_object")


def proxify(obj, existing_proxies, host_file):
    ptr = id(obj)
    if ptr in existing_proxies:
        return existing_proxies[ptr]

    obj = asproxy(obj)
    existing_proxies[ptr] = obj
    obj._obj_pxy["host_file"] = weakref.ref(host_file)
    return obj


@proxify_device_object.register(object)
def proxify_device_object_default(obj, existing_proxies, host_file):
    if hasattr(obj, "__cuda_array_interface__"):
        return proxify(obj, existing_proxies, host_file)
    return obj


@proxify_device_object.register(ProxyObject)
def proxify_device_object_proxy_object(obj, existing_proxies, host_file):
    obj._obj_pxy["host_file"] = weakref.ref(host_file)
    return obj


@proxify_device_object.register(list)
@proxify_device_object.register(tuple)
@proxify_device_object.register(set)
@proxify_device_object.register(frozenset)
def proxify_device_object_python_collection(seq, existing_proxies, host_file):
    return type(seq)(proxify_device_object(o, existing_proxies, host_file) for o in seq)


@proxify_device_object.register(dict)
def proxify_device_object_python_dict(seq, existing_proxies, host_file):
    return {
        k: proxify_device_object(v, existing_proxies, host_file) for k, v in seq.items()
    }


@proxify_device_object.register_lazy("cudf")
def register_cudf():
    import cudf

    @proxify_device_object.register(cudf.DataFrame)
    @proxify_device_object.register(cudf.Series)
    @proxify_device_object.register(cudf.Index)
    def proxify_device_object_cudf_dataframe(obj, existing_proxies, host_file):
        return proxify(obj, existing_proxies, host_file)
