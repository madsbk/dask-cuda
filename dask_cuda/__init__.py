import functools
from typing import Any, List, Tuple
import dask
import dask.dataframe.core
import dask.dataframe.shuffle

from ._version import get_versions
from .cuda_worker import CUDAWorker
from .explicit_comms.dataframe.shuffle import get_rearrange_by_column_tasks_wrapper
from .local_cuda_cluster import LocalCUDACluster
from .proxify_device_objects import proxify_decorator, unproxify_decorator, ProxyObject

__version__ = get_versions()["version"]
del get_versions


# Monkey patching Dask to make use of explicit-comms when `DASK_EXPLICIT_COMMS=True`
dask.dataframe.shuffle.rearrange_by_column_tasks = get_rearrange_by_column_tasks_wrapper(
    dask.dataframe.shuffle.rearrange_by_column_tasks
)


# Monkey patching Dask to make use of proxify and unproxify in compatibility mode
dask.dataframe.shuffle.shuffle_group = proxify_decorator(
    dask.dataframe.shuffle.shuffle_group
)

total_time = [0]
import time


def is_pandas_spilled(proxy):
    try:
        return proxy._obj_pxy["serializer"] == "to-pandas"
    except:
        return False

def extract_to_pandas_serialized(args) -> Tuple[List[ProxyObject], List[Any]]:
    to_pandas = []
    others = []
    for a in args:
        if is_pandas_spilled(a):
            to_pandas.append(a)
        else:
            others.append(a)
    return to_pandas, others


def concat_to_pandas(func, args, ignore_index=False):
    try:
        if len(args) == 1:
            return func(args, ignore_index)
        pds, others = extract_to_pandas_serialized(args)
        if pds:
            res1 = pds[0]._obj_pxy_copy()

            pd = func([p._obj_pxy["obj"][1][0] for p in pds], ignore_index)
            # print("pd: ", type(pd))
            # print("pds[0]._obj_pxy: ", repr(pds[0]._obj_pxy["obj"]))
            res1._obj_pxy["obj"] = (pds[0]._obj_pxy["obj"][0], [pd])
        if others:
            res2 = func(others, ignore_index)
        if pds and others:
            return func([res1, res2], ignore_index)
        elif pds:
            return res1
        else:
            return res2
    except BaseException as e:
        print("concat_to_pandas() ", repr(e))
        import traceback
        traceback.print_exc()
        traceback.print_stack()
        raise



def concat_decorator(func):
    @functools.wraps(func)
    def wrapper(args, ignore_index=False):
        proxies = [is_pandas_spilled(a) for a in args]
        trues = sum(proxies)
        t1 = time.monotonic()
        ret = concat_to_pandas(func, args, ignore_index)
        if trues:
            total_time[0] += time.monotonic() - t1
            #print("concat: ", proxies, trues, total_time)
        return ret

    return wrapper


#dask.dataframe.core._concat = concat_decorator(dask.dataframe.core._concat)
