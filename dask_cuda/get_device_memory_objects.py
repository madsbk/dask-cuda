from dask.utils import Dispatch


get_device_memory_objects = Dispatch(name="get_device_memory_objects")


@get_device_memory_objects.register(object)
def get_device_memory_objects_default(obj):
    if hasattr(obj, "_obj_pxy"):
        if obj._obj_pxy["serializers"] is None:
            return get_device_memory_objects(obj._obj_pxy["obj"])
        else:
            return []
    if hasattr(obj, "columns"):
        ret = []
        for col_name in obj.columns:
            ret += get_device_memory_objects(obj[col_name])
        return ret
    if hasattr(obj, "data"):
        return get_device_memory_objects(obj.data)
    if hasattr(obj, "_owner") and obj._owner is not None:
        return get_device_memory_objects(obj._owner)
    if hasattr(obj, "__cuda_array_interface__"):
        return [obj]
    return []


@get_device_memory_objects.register(list)
@get_device_memory_objects.register(tuple)
@get_device_memory_objects.register(set)
@get_device_memory_objects.register(frozenset)
def get_device_memory_objects_python_sequence(seq):
    ret = []
    for s in seq:
        ret.extend(get_device_memory_objects(s))
    return ret


@get_device_memory_objects.register(dict)
def get_device_memory_objects_python_dict(seq):
    ret = []
    for s in seq.values():
        ret.extend(get_device_memory_objects(s))
    return ret
