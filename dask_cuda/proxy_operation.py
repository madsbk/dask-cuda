import functools
import operator
from typing import Callable, Iterable, Mapping, Type

import dask
import dask.array.core
import dask.dataframe.methods
import dask.dataframe.utils
import dask.utils
import distributed.protocol
import distributed.utils
from dask.sizeof import sizeof
from distributed.protocol import pickle

try:
    from dask.dataframe.dispatch import make_meta_dispatch as make_meta_dispatch
except ImportError:
    from dask.dataframe.utils import make_meta as make_meta_dispatch

import dask.dataframe.core

from .is_device_object import is_device_object
from .proxy_object import ProxyObject


class ProxyOperation:
    def __init__(
        self,
        func: Callable,
        func_args: Iterable[ProxyObject],
        func_kwargs: Mapping,
        output_type: Type,
    ):
        self._pxy_func = func
        self._pxy_func_args = func_args
        self._pxy_func_kwargs = func_kwargs
        self._pxy_output_type = output_type
        self._pxy_output_value = None
        self._pxy_done = False
        self.__class__ = output_type

    def _pxy_apply(self):
        if self._pxy_done:
            assert self._pxy_func_args is None
            assert self._pxy_func_kwargs is None
        else:
            self._pxy_output_value = self._pxy_func(
                self._pxy_func_args, **self._pxy_func_kwargs
            )
            self._pxy_done = True
            self._pxy_func_args = None
            self._pxy_func_kwargs = None
        assert isinstance(self._pxy_output_value, self._pxy_output_type)
        return self._pxy_output_value

    def _pxy_map_reduce(self, map_func, reduce_func):
        if self._pxy_done:
            return map_func(self._pxy_output_value)
        else:
            return reduce_func(map_func(p) for p in self._pxy_func_args)

    def __getattr__(self, name):
        return getattr(self._pxy_apply(), name)

    def __setattr__(self, name, val):
        if name.startswith("_pxy_"):
            return object.__setattr__(self, name, val)
        object.__setattr__(self._pxy_apply(), name, val)

    def __str__(self):
        return str(self._pxy_apply())

    def __repr__(self):
        if self._pxy_done:
            of = repr(self._pxy_output_type)
        else:
            of = repr(self._pxy_func_args)
        return f"<{dask.utils.typename(type(self))} at {hex(id(self))} of {of}>"

    def __sizeof__(self):
        return self._pxy_map_reduce(sizeof, sum)

    def __len__(self):
        return self._pxy_map_reduce(len, sum)

    def __contains__(self, value):
        return value in self._pxy_apply()

    def __getitem__(self, key):
        return self._pxy_apply()[key]

    def __setitem__(self, key, value):
        self._pxy_apply()[key] = value

    def __delitem__(self, key):
        del self._pxy_apply()[key]

    def __getslice__(self, i, j):
        return self._pxy_apply()[i:j]

    def __setslice__(self, i, j, value):
        self._pxy_apply()[i:j] = value

    def __delslice__(self, i, j):
        del self._pxy_apply()[i:j]

    def __iter__(self):
        return iter(self._pxy_apply())

    def __array__(self, *args, **kwargs):
        return getattr(self._pxy_apply(), "__array__")(*args, **kwargs)

    def __lt__(self, other):
        return self._pxy_apply() < other

    def __le__(self, other):
        return self._pxy_apply() <= other

    def __eq__(self, other):
        return self._pxy_apply() == other

    def __ne__(self, other):
        return self._pxy_apply() != other

    def __gt__(self, other):
        return self._pxy_apply() > other

    def __ge__(self, other):
        return self._pxy_apply() >= other

    def __add__(self, other):
        return self._pxy_apply() + other

    def __sub__(self, other):
        return self._pxy_apply() - other

    def __mul__(self, other):
        return self._pxy_apply() * other

    def __truediv__(self, other):
        return operator.truediv(self._pxy_apply(), other)

    def __floordiv__(self, other):
        return self._pxy_apply() // other

    def __mod__(self, other):
        return self._pxy_apply() % other

    def __divmod__(self, other):
        return divmod(self._pxy_apply(), other)

    def __pow__(self, other):
        return pow(self._pxy_apply(), other)

    def __lshift__(self, other):
        return self._pxy_apply() << other

    def __rshift__(self, other):
        return self._pxy_apply() >> other

    def __and__(self, other):
        return self._pxy_apply() & other

    def __xor__(self, other):
        return self._pxy_apply() ^ other

    def __or__(self, other):
        return self._pxy_apply() | other

    def __radd__(self, other):
        return other + self._pxy_apply()

    def __rsub__(self, other):
        return other - self._pxy_apply()

    def __rmul__(self, other):
        return other * self._pxy_apply()

    def __rtruediv__(self, other):
        return operator.truediv(other, self._pxy_apply())

    def __rfloordiv__(self, other):
        return other // self._pxy_apply()

    def __rmod__(self, other):
        return other % self._pxy_apply()

    def __rdivmod__(self, other):
        return divmod(other, self._pxy_apply())

    def __rpow__(self, other, *args):
        return pow(other, self._pxy_apply(), *args)

    def __rlshift__(self, other):
        return other << self._pxy_apply()

    def __rrshift__(self, other):
        return other >> self._pxy_apply()

    def __rand__(self, other):
        return other & self._pxy_apply()

    def __rxor__(self, other):
        return other ^ self._pxy_apply()

    def __ror__(self, other):
        return other | self._pxy_apply()

    def __iadd__(self, other):
        proxied = self._pxy_apply()
        proxied += other
        return self

    def __isub__(self, other):
        proxied = self._pxy_apply()
        proxied -= other
        return self

    def __imul__(self, other):
        proxied = self._pxy_apply()
        proxied *= other
        return self

    def __itruediv__(self, other):
        proxied = self._pxy_apply()
        self._pxy_output_value["obj"] = operator.itruediv(proxied, other)
        return self

    def __ifloordiv__(self, other):
        proxied = self._pxy_apply()
        proxied //= other
        return self

    def __imod__(self, other):
        proxied = self._pxy_apply()
        proxied %= other
        return self

    def __ipow__(self, other):
        proxied = self._pxy_apply()
        proxied **= other
        return self

    def __ilshift__(self, other):
        proxied = self._pxy_apply()
        proxied <<= other
        return self

    def __irshift__(self, other):
        proxied = self._pxy_apply()
        proxied >>= other
        return self

    def __iand__(self, other):
        proxied = self._pxy_apply()
        proxied &= other
        return self

    def __ixor__(self, other):
        proxied = self._pxy_apply()
        proxied ^= other
        return self

    def __ior__(self, other):
        proxied = self._pxy_apply()
        proxied |= other
        return self

    def __neg__(self):
        return -self._pxy_apply()

    def __pos__(self):
        return +self._pxy_apply()

    def __abs__(self):
        return abs(self._pxy_apply())

    def __invert__(self):
        return ~self._pxy_apply()

    def __int__(self):
        return int(self._pxy_apply())

    def __float__(self):
        return float(self._pxy_apply())

    def __complex__(self):
        return complex(self._pxy_apply())

    def __index__(self):
        return operator.index(self._pxy_apply())


@is_device_object.register(ProxyOperation)
def pxy_op_is_device(obj: ProxyOperation):
    return obj._pxy_map_reduce(is_device_object, sum)


@distributed.protocol.dask_serialize.register(ProxyOperation)
def pxy_op_dask_serialize(obj: ProxyOperation):
    """
    The generic serialization of ProxyOperation used by Dask when
    communicating ProxyOperation.
    """
    if obj._pxy_done:
        sub_header, sub_frames = distributed.protocol.serialize(obj._pxy_output_value)
        return {"done": True, "sub-header": sub_header}, sub_frames

    sub_headers = []
    frames = []
    for pxy in obj._pxy_func_args:
        sub_header, sub_frames = distributed.protocol.serialize(pxy)
        sub_headers.append((len(frames), len(frames) + len(sub_frames), sub_header))
        frames.extend(sub_frames)

    return (
        {
            "done": False,
            "sub-headers": sub_headers,
            "func": pickle.dumps(obj._pxy_func),
            "func_kwargs": pickle.dumps(obj._pxy_func_kwargs),
            "output_type": pickle.dumps(obj._pxy_output_type),
        },
        frames,
    )


@distributed.protocol.dask_deserialize.register(ProxyOperation)
def pxy_op_dask_deserialize(header, frames):
    frames = list(frames)
    if header["done"]:
        return distributed.protocol.deserialize(header["sub-header"], frames)

    proxies = []
    for start, stop, sub_header in header["sub-headers"]:
        proxies.append(distributed.protocol.deserialize(sub_header, frames[start:stop]))
    return ProxyOperation(
        func=pickle.loads(header["func"]),
        func_args=proxies,
        func_kwargs=pickle.loads(header["func_kwargs"]),
        output_type=pickle.loads(header["output_type"]),
    )


@dask.dataframe.core.get_parallel_type.register(ProxyOperation)
def get_parallel_type_proxy_operation(obj: ProxyOperation):
    # Notice, `get_parallel_type()` needs a instance not a type object
    return dask.dataframe.core.get_parallel_type(obj.__class__.__new__(obj.__class__))


def unproxy(obj):
    if isinstance(obj, ProxyOperation):
        return obj._pxy_apply()
    else:
        return obj


def unproxify_input_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = [unproxy(d) for d in args]
        kwargs = {k: unproxy(v) for k, v in kwargs.items()}
        return func(*args, **kwargs)

    return wrapper


# Register dispatch of ProxyOperation on all known dispatch objects
for dispatch in (
    dask.dataframe.core.hash_object_dispatch,
    make_meta_dispatch,
    dask.dataframe.utils.make_scalar,
    dask.dataframe.core.group_split_dispatch,
    dask.array.core.tensordot_lookup,
    dask.array.core.einsum_lookup,
    dask.array.core.concatenate_lookup,
):
    dispatch.register(ProxyOperation, unproxify_input_wrapper(dispatch))

dask.dataframe.methods.concat_dispatch.register(
    ProxyOperation, unproxify_input_wrapper(dask.dataframe.methods.concat)
)
