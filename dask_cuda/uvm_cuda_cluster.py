import threading
from concurrent.futures import ThreadPoolExecutor

import cupy
import cupy.cuda.runtime

from distributed import LocalCluster

import rmm.mr
from rmm.allocators.cupy import rmm_cupy_allocator

from dask_cuda.utils import cuda_visible_devices, parse_cuda_visible_device


class ThreadPoolExecutorCUDA(ThreadPoolExecutor):
    def submit(self, fn, /, *args, **kwargs):
        print(f"submit() - id: {threading.get_native_id()}")
        return super().submit(fn, *args, **kwargs)


def enable_uvm() -> None:
    for i in range(cupy.cuda.runtime.getDeviceCount()):
        for j in range(cupy.cuda.runtime.getDeviceCount()):
            if i != j:
                with cupy.cuda.Device(i):
                    cupy.cuda.runtime.deviceEnablePeerAccess(j)


class UvmCUDACluster(LocalCluster):
    def __init__(
        self, CUDA_VISIBLE_DEVICES=None, n_workers=None, rmm_reinitialize=True
    ):
        if CUDA_VISIBLE_DEVICES is None:
            CUDA_VISIBLE_DEVICES = cuda_visible_devices(0)
        if isinstance(CUDA_VISIBLE_DEVICES, str):
            CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES.split(",")
        CUDA_VISIBLE_DEVICES = list(
            map(parse_cuda_visible_device, CUDA_VISIBLE_DEVICES)
        )
        if n_workers is None:
            n_workers = len(CUDA_VISIBLE_DEVICES)
        if n_workers < 1:
            raise ValueError("Number of workers cannot be less than 1.")

        if rmm_reinitialize:
            rmm.reinitialize(
                managed_memory=False, devices=CUDA_VISIBLE_DEVICES, pool_allocator=True
            )
            cupy.cuda.set_allocator(rmm_cupy_allocator)

        print(
            f"UvmCUDACluster() - CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES}, "
            f"n_workers: {n_workers}, rmm_reinitialize: {rmm_reinitialize}"
        )
        enable_uvm()
        super().__init__(
            n_workers=2,
            threads_per_worker=1,
            processes=False,
            protocol="inproc",
            data={},
            memory_limit=None,
        )
        assert len(self.workers) == n_workers

        for i, worker in enumerate(self.workers.values()):
            worker.executors = {
                n: ThreadPoolExecutor(
                    max_workers=1,
                    initializer=lambda dev: cupy.cuda.Device(dev).use(),
                    initargs=(i,),
                )
                for n, e in worker.executors.items()
            }
