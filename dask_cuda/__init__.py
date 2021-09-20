from dask_cuda.proxy_operation import ProxyOperation
import functools
from dask_cuda.proxy_object import ProxyObject
import dask
import dask.dataframe.core
import dask.dataframe.shuffle

from ._version import get_versions
from .cuda_worker import CUDAWorker
from .explicit_comms.dataframe.shuffle import get_rearrange_by_column_tasks_wrapper
from .local_cuda_cluster import LocalCUDACluster
from .proxify_device_objects import proxify_decorator, unproxify_decorator

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


def concat_decorator(func):
    @functools.wraps(func)
    def wrapper(args, ignore_index=False):
        if len(args) > 1:
            if all(isinstance(a, ProxyObject) for a in args):
                if any(a._obj_pxy_is_serialized() for a in args):
                    return ProxyOperation(
                        func, args, {"ignore_index": ignore_index}, args[0].__class__
                    )
        return func(args, ignore_index)

    return wrapper


dask.dataframe.core._concat = concat_decorator(dask.dataframe.core._concat)
# dask.dataframe.core._concat = unproxify_decorator(dask.dataframe.core._concat)
