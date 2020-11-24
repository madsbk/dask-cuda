import os
from random import randint

import numpy as np
import pytest

import dask.array
from distributed.protocol import (
    deserialize,
    deserialize_bytes,
    serialize,
    serialize_bytelist,
)
from distributed.protocol.pickle import HIGHEST_PROTOCOL

from dask_cuda.dynamic_host_file import DynamicHostFile

cupy = pytest.importorskip("cupy")
itemsize = cupy.arange(1).nbytes


def test_one_item_limit():
    dhf = DynamicHostFile(device_memory_limit=itemsize)
    dhf["k1"] = cupy.arange(1) + 1
    dhf["k2"] = cupy.arange(1) + 2

    # Check k1 is spilled because of the newer k2
    k1 = dhf["k1"]
    assert k1._obj_pxy_serialized()
    assert not dhf["k2"]._obj_pxy_serialized()

    # Accessing k1 spills k2 and unspill k1
    k1_val = k1[0]
    assert k1_val == 1
    k2 = dhf["k2"]
    assert k2._obj_pxy_serialized()

    # Duplicate arrays changes nothing
    dhf["k3"] = [k1, k2]
    assert not k1._obj_pxy_serialized()
    assert k2._obj_pxy_serialized()

    # Adding a new array spills k1 and k2
    dhf["k4"] = cupy.arange(1) + 4
    assert k1._obj_pxy_serialized()
    assert k2._obj_pxy_serialized()
    assert not dhf["k4"]._obj_pxy_serialized()

    # Deleting k2 does change anything since k3 still holds a
    # reference to the underlying proxy object
    dhf["k2"][0]
    assert dhf["k1"]._obj_pxy_serialized()
    assert not dhf["k2"]._obj_pxy_serialized()
    assert dhf["k4"]._obj_pxy_serialized()
    del dhf["k2"]
    assert not dhf["k3"][1]._obj_pxy_serialized()

