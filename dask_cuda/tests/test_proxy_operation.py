import pytest
from pandas.testing import assert_frame_equal

from dask.dataframe.core import _concat as df_concat
from distributed.protocol.serialize import deserialize, serialize

from dask_cuda import proxy_object
from dask_cuda.proxy_operation import ProxyOperation


def create_proxy_cat_operation(ser1=("dask",), ser2=("dask",)):
    """Help function to create a concat proxy operation"""
    cudf = pytest.importorskip("cudf")
    df1 = cudf.DataFrame({"a": [1, 2, 3]})
    df2 = cudf.DataFrame({"a": [4, 5]})
    org = df_concat([df1, df2])
    p1 = proxy_object.asproxy(df1, serializers=ser1)
    p2 = proxy_object.asproxy(df2, serializers=ser2)
    return org, ProxyOperation(df_concat, (p1, p2), {}, type(df1))


@pytest.mark.parametrize("serializers1", [None, ("dask",)])
@pytest.mark.parametrize("serializers2", [None, ("dask",)])
def test_cudf_concat(serializers1, serializers2):
    expected, pxy_op = create_proxy_cat_operation(serializers1, serializers2)
    assert_frame_equal(expected.to_pandas(), pxy_op.to_pandas())


def test_cudf_serialize():
    org, pxy = create_proxy_cat_operation()
    header, frames = serialize(pxy)
    res = deserialize(header, frames)
    assert_frame_equal(org.to_pandas(), res.to_pandas())
