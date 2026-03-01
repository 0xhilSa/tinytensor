from functools import reduce
from typing import List, Optional, Union
from tinytensor import dtypes
from tinytensor.engine.cpu.cpu import dtype
from tinytensor.tensor import Tensor
from tinytensor.engine import cpu, cuda
from tinytensor.engine.constants import pi, e, euler_gamma, inf, nan
from tinytensor.helpers import reshape
from tinytensor.device import Device, DeviceLike
from tinytensor import version
from tinytensor.dtypes import (bool, int8, uint8, int16, uint16,
                               int32, uint32, int64, uint64,
                               float16, float32, float64,
                               complex64, complex128)


__version__ = version.__version__

def cuda_available(): return hasattr(cuda, "device_count") and cuda.device_count() > 0

def tensor(
  buf:Union[List[dtypes.ConstType],dtypes.ConstType],
  dtype:Optional[Union[dtypes.DType,dtypes.ConstType]]=None,
  device:Union[str,Device,None]=None,
  requires_grad:bool=False, # type: ignore[valid-type]
  const:bool=False # type: ignore[valid-type]
): return Tensor(buf=buf, dtype=dtype, device=device, requires_grad=requires_grad, const=const)

def arange(
  start:int=0, stop:int=0, step:int=1,
  dtype:Union[dtypes.ConstType,dtypes.DType]=int64,
  device:Union[str,Device,None]=None,
  requires_grad:bool=False, # type: ignore[valid-type]
  const:bool=False # type: ignore[valid-type]
):
  if step == 0: raise ValueError("step parameter cannot be 0")
  return Tensor(list(range(start, stop, step)), dtype=dtype, device=device, requires_grad=requires_grad, const=const)

def linspace(
  start:int, end:int, steps:int,
  *,
  dtype:Optional[dtypes.DType]=None,
  device:Union[str,Device]="cpu",
  requires_grad:bool=False, # type: ignore[valid-type]
  const:bool=False # type: ignore[valid-type]
) -> Tensor:
  if steps <= 0: raise ValueError("linspace steps must be a positive integer")
  if dtype is None:
    if isinstance(start, complex) or isinstance(end, complex): dtype = dtypes.complex64
    elif isinstance(start, float) or isinstance(end, float): dtype = dtypes.float32
    else: dtype = dtypes.float32
  if steps == 1: data = [start]
  else:
    step = (end - start) / (steps - 1)
    data = [start + i * step for i in range(steps)]
  return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad, const=const)

def ones(
  *shape,
  dtype:Union[dtypes.DType,dtypes.ConstType]=dtypes.float32,
  device:Union[str,Device]="cpu",
  requires_grad:bool=False, # type: ignore[valid-type]
  const:bool=False # type: ignore[valid-type]
) -> Tensor:
  if len(shape) == 1 and isinstance(shape, (list, tuple)): shape = shape[0]
  length = 1
  for x in shape: length *= x
  return Tensor(reshape([1] * length, shape), dtype=dtype, device=device, requires_grad=requires_grad, const=const)

def zeros(
  *shape,
  dtype:Union[dtypes.DType,dtypes.ConstType]=dtypes.float32,
  device:Union[str,Device]="cpu",
  requires_grad:bool=False, # type: ignore[valid-type]
  const:bool=False # type: ignore[valid-type]
) -> Tensor:
  if len(shape) == 1 and isinstance(shape, (list, tuple)): shape = shape[0]
  length = 1
  for x in shape: length *= x
  return Tensor(reshape([0] * length, shape), dtype=dtype, device=device, requires_grad=requires_grad, const=const)

def fill(
  value,
  *shape,
  dtype:Union[dtypes.DType,dtypes.ConstType]=dtypes.float32,
  device:Union[str,Device]="cpu",
  requires_grad:bool=False, # type: ignore[valid-type]
  const:bool=False # type: ignore[valid-type]
) -> Tensor:
  if len(shape) == 1 and isinstance(shape, (list, tuple)): shape = shape[0]
  length = 1
  for x in shape: length *= x
  return Tensor(reshape([value] * length, shape), dtype=dtype, device=device, requires_grad=requires_grad, const=const)

def empty(
  *shape,
  dtype:dtypes.DType=dtypes.float32,
  device:Union[str,Device]="cpu",
  requires_grad:bool=False, # type: ignore[valid-type]
  const:bool=False # type: ignore[valid-type]
) -> Tensor:
  if len(shape) == 1 and isinstance(shape, (list, tuple)): shape = shape[0]
  if not isinstance(shape, tuple): raise TypeError("shape must be a tuple")
  if len(shape) == 0: raise ValueError("shape cannot be empty")
  for dim in shape:
    if not isinstance(dim, int): raise TypeError("shape elements must be int")
    if dim < 0: raise ValueError("negative dimensions not allowed")
  if not isinstance(device, Device) or device is None: device = Device(device)
  caps = cpu.empty(shape, dtype.fmt)
  return Tensor._from_backend(caps, dtype=dtype, device=device, requires_grad=requires_grad, const=const)

@staticmethod
def eye(m:int, n:int|None=None, dtype:dtypes.DType=dtypes.float32, device:Union[str,Device]="cpu", requires_grad:bool=False, const:bool=False): # type: ignore
  if n is None: n = m
  if not isinstance(device, Device): device = Device(device)
  buf = cpu.eye(m, n, dtype.fmt) if device.type == "CPU" else cuda.eye(m, n, dtype.fmt)
  return Tensor._from_backend(buf, dtype=dtype, device=device, requires_grad=requires_grad, const=const)

def __ensure_tensor(x):
  if isinstance(x,Tensor): return x
  if isinstance(x,list): return Tensor(x)
  return Tensor(x)

def __binop(op, *args:Union[Tensor,List,dtypes.ConstType], res_dtype:dtypes.DType|None=None) -> Tensor:
  if len(args) < 2: raise ValueError("At least two arguments requires")
  tensors = [__ensure_tensor(a) for a in args]
  return reduce(op, tensors)

def __unary_backend(cpu_op, cuda_op, x, requires_grad=False, const=False):
  x = __ensure_tensor(x)
  if x.device.type == "CPU": out_buf = cpu_op(x.buf)
  else: out_buf = cuda_op(x.buf)
  requires_grad = x.requires_grad or requires_grad
  return Tensor._from_backend(out_buf, dtype=dtypes.DType.from_ctype(dtype(out_buf)), device=x.device, requires_grad=requires_grad, const=const)

def add(*args:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __binop(lambda x,y: x + y, *args)
def sub(*args:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __binop(lambda x,y: x - y, *args)
def mul(*args:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __binop(lambda x,y: x * y, *args)
def tdiv(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __binop(lambda x,y: x / y, x,y)
def fdiv(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __binop(lambda x,y: x // y, x,y)
def mod(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __binop(lambda x,y: x % y, x,y)
def pow(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __binop(lambda x,y: x ** y, x,y)

def eq(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __binop(lambda x,y: x == y, x,y)
def ne(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __binop(lambda x,y: x != y, x,y)
def gt(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __binop(lambda x,y: x > y, x,y)
def ge(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __binop(lambda x,y: x >= y, x,y)
def lt(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __binop(lambda x,y: x < y, x,y)
def le(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __binop(lambda x,y: x <= y, x,y)

def lshift(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __binop(lambda x,y: x << y, x,y)
def rshift(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __binop(lambda x,y: x >> y, x,y)

def bitwise_and(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __binop(lambda x,y: x & y, x,y)
def bitwise_or(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __binop(lambda x,y: x | y, x,y)
def bitwise_xor(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __binop(lambda x,y: x ^ y, x,y)
def bitwise_not(x:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __unary_backend(cpu.bitwise_not, cuda.bitwise_not, x)
def bitwise_nand(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __binop(lambda x,y: ~(x & y), x,y)
def bitwise_nor(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __binop(lambda x,y: ~(x | y), x,y)
def bitwise_xnor(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.ConstType]) -> Tensor: return __binop(lambda x,y: ~(x ^ y), x,y)

def logical_and(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.DType]) -> Tensor:
  x, y = __ensure_tensor(x), __ensure_tensor(y)
  out = cpu.logical_and(x.buf, y.buf) if x.device.type == "CPU" else cuda.logical_and(x.buf, y.buf)
  return Tensor._from_backend(out, dtype=dtypes.bool, device=x.device)

def logical_or(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.DType]) -> Tensor:
  x, y = __ensure_tensor(x), __ensure_tensor(y)
  out = cpu.logical_or(x.buf, y.buf) if x.device.type == "CPU" else cuda.logical_or(x.buf, y.buf)
  return Tensor._from_backend(out, dtype=dtypes.bool, device=x.device)

def logical_xor(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.DType]) -> Tensor:
  x, y = __ensure_tensor(x), __ensure_tensor(y)
  out = cpu.logical_xor(x.buf, y.buf) if x.device.type == "CPU" else cuda.logical_xor(x.buf, y.buf)
  return Tensor._from_backend(out, dtype=dtypes.bool, device=x.device)

def logical_not(x:Union[Tensor,List,dtypes.ConstType]) -> Tensor:
  x = __ensure_tensor(x)
  out = cpu.logical_not(x.buf) if x.device.type == "CPU" else cuda.logical_not(x.buf)
  return Tensor._from_backend(out, dtype=dtypes.bool, device=x.device)

def logical_nand(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.DType]) -> Tensor:
  x, y = __ensure_tensor(x), __ensure_tensor(y)
  out = cpu.logical_nand(x.buf, y.buf) if x.device.type == "CPU" else cuda.logical_nand(x.buf, y.buf)
  return Tensor._from_backend(out, dtype=dtypes.bool, device=x.device)

def logical_nor(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.DType]) -> Tensor:
  x, y = __ensure_tensor(x), __ensure_tensor(y)
  out = cpu.logical_nor(x.buf, y.buf) if x.device.type == "CPU" else cuda.logical_nor(x.buf, y.buf)
  return Tensor._from_backend(out, dtype=dtypes.bool, device=x.device)

def logical_xnor(x:Union[Tensor,List,dtypes.ConstType], y:Union[Tensor,List,dtypes.DType]) -> Tensor:
  x, y = __ensure_tensor(x), __ensure_tensor(y)
  out = cpu.logical_xnor(x.buf, y.buf) if x.device.type == "CPU" else cuda.logical_xnor(x.buf, y.buf)
  return Tensor._from_backend(out, dtype=dtypes.bool, device=x.device)

def exp(x:Union[Tensor,List,dtypes.ConstType], requires_grad:bool=False, const:bool=False) -> Tensor: return __unary_backend(cpu.exp, cuda.exp, x, requires_grad, const) # type: ignore[type-valid]
def log(x:Union[Tensor,List,dtypes.ConstType], requires_grad:bool=False, const:bool=False) -> Tensor: return __unary_backend(cpu.log, cuda.log, x, requires_grad, const) # type: ignore[type-valid]
def log2(x:Union[Tensor,List,dtypes.ConstType], requires_grad:bool=False, const:bool=False) -> Tensor: return __unary_backend(cpu.log2, cuda.log2, x, requires_grad, const) # type: ignore[type-valid]
def log10(x:Union[Tensor,List,dtypes.ConstType], requires_grad:bool=False, const:bool=False) -> Tensor: return __unary_backend(cpu.log10, cuda.log10, x, requires_grad, const) # type: ignore[type-valid]
def sin(x:Union[Tensor,List,dtypes.ConstType], requires_grad:bool=False, const:bool=False) -> Tensor: return __unary_backend(cpu.sin, cuda.sin, x, requires_grad, const) # type: ignore[type-valid]
def cos(x:Union[Tensor,List,dtypes.ConstType], requires_grad:bool=False, const:bool=False) -> Tensor: return __unary_backend(cpu.cos, cuda.cos, x, requires_grad, const) # type: ignore[type-valid]
def tan(x:Union[Tensor,List,dtypes.ConstType], requires_grad:bool=False, const:bool=False) -> Tensor: return __unary_backend(cpu.tan, cuda.tan, x, requires_grad, const) # type: ignore[type-valid]
def asin(x:Union[Tensor,List,dtypes.ConstType], requires_grad:bool=False, const:bool=False) -> Tensor: return __unary_backend(cpu.asin, cuda.asin, x, requires_grad, const) # type: ignore[type-valid]
def acos(x:Union[Tensor,List,dtypes.ConstType], requires_grad:bool=False, const:bool=False) -> Tensor: return __unary_backend(cpu.acos, cuda.acos, x, requires_grad, const) # type: ignore[type-valid]
def atan(x:Union[Tensor,List,dtypes.ConstType], requires_grad:bool=False, const:bool=False) -> Tensor: return __unary_backend(cpu.atan, cuda.atan, x, requires_grad, const) # type: ignore[type-valid]
def sinh(x:Union[Tensor,List,dtypes.ConstType], requires_grad:bool=False, const:bool=False) -> Tensor: return __unary_backend(cpu.sinh, cuda.sinh, x, requires_grad, const) # type: ignore[type-valid]
def cosh(x:Union[Tensor,List,dtypes.ConstType], requires_grad:bool=False, const:bool=False) -> Tensor: return __unary_backend(cpu.cosh, cuda.cosh, x, requires_grad, const) # type: ignore[type-valid]
def tanh(x:Union[Tensor,List,dtypes.ConstType], requires_grad:bool=False, const:bool=False) -> Tensor: return __unary_backend(cpu.tanh, cuda.tanh, x, requires_grad, const) # type: ignore[type-valid]
def asinh(x:Union[Tensor,List,dtypes.ConstType], requires_grad:bool=False, const:bool=False) -> Tensor: return __unary_backend(cpu.asinh, cuda.asinh, x, requires_grad, const) # type: ignore[type-valid]
def acosh(x:Union[Tensor,List,dtypes.ConstType], requires_grad:bool=False, const:bool=False) -> Tensor: return __unary_backend(cpu.acosh, cuda.acosh, x, requires_grad, const) # type: ignore[type-valid]
def atanh(x:Union[Tensor,List,dtypes.ConstType], requires_grad:bool=False, const:bool=False) -> Tensor: return __unary_backend(cpu.atanh, cuda.atanh, x, requires_grad, const) # type: ignore[type-valid]
def sgn(x:Union[Tensor,List,dtypes.ConstType], requires_grad:bool=False, const:bool=False) -> Tensor: return __unary_backend(cpu.sgn, cuda.sgn, x, requires_grad, const) # type: ignore[type-valid]

