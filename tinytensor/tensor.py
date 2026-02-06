from __future__ import annotations
from typing import List, Optional, Tuple, Union
from tinytensor import dtypes
from tinytensor.shape import Shape
from tinytensor.engine import cpu, cuda
from tinytensor.helpers import dtype_of, shape_of, flatten, reshape
from tinytensor.device import Device


class Tensor:
  def __init__(
    self,
    buf:Union[List,dtypes.ConstType],
    dtype:Optional[Union[dtypes.DType,dtypes.ConstType]]=None,
    device:Union[str,Device]="cpu",
    requires_grad:bool=False,
    const:bool=False
  ):
    self.__shape = Shape(shape_of(buf))
    self.__ndim = self.__shape.ndim
    buf = flatten(buf) if isinstance(buf, list) else buf
    buf, self.__dtype = dtype_of(buf, dtype)
    self.__device = Device(device) if not isinstance(device, Device) else device
    self.__const = const
    if self.__device.type == "CPU": self.__buf = cpu.tocpu(buf, self.__shape.shape, self.__dtype.fmt)
    elif self.__device.type == "CUDA": self.__buf = cuda.tocuda(buf, self.__shape.shape, self.__dtype.fmt, self.__device.index)
    self.__requires_grad = requires_grad # TODO: yet to be immplemented
  def __repr__(self): return f"Tensor(shape={self.__shape}, dtype='{self.__dtype.ctype}', device={self.__device}, requires_grad={self.__requires_grad}, const={self.__const})"
  @property
  def ndim(self): return self.__ndim
  @property
  def shape(self): return self.__shape
  @property
  def device(self): return self.__device
  @property
  def nbyte(self): return self.__shape.size * self.__dtype.nbyte
  @property
  def dtype(self): return self.__dtype
  @property
  def requires_grad(self): return self.__requires_grad
  @property
  def buf(self): return self.__buf
  def is_const(self): return self.__const
  def cuda(self, device_index:int=0): return Tensor(reshape(cpu.topyobj(self.__buf), self.__shape.shape), dtype=self.__dtype, device=f"cuda:{device_index}") if self.__device.type == "CPU" else self # type: ignore
  def cpu(self): return Tensor(reshape(cuda.topyobj(self.__buf), self.__shape.shape), dtype=self.__dtype, device=f"cpu") if self.__device.type == "CUDA" else self # type: ignore
  def data(self): x = cuda.topyobj(self.__buf) if self.__device.type == "CUDA" else cpu.topyobj(self.__buf); return reshape(x, self.__shape.shape) # type: ignore
  def __len__(self):
    if self.__ndim == 0: raise ValueError("len() of a 0-d tensor")
    return len(cpu.topyobj(self.__buf)) if self.__device.type == "CPU" else len(cuda.topyobj(self.__buf)) # type: ignore
  @staticmethod
  def ones(shape:Tuple[int,...], dtype:Union[dtypes.DType,dtypes.ConstType]=dtypes.int64, device:Union[str,Device]="cpu"):
    length = 1
    for x in shape: length *= x
    return Tensor(reshape([1 for _ in range(length)], shape), dtype=dtype, device=device)
  @staticmethod
  def zeros(shape:Tuple[int,...], dtype:Union[dtypes.DType,dtypes.ConstType]=dtypes.int64, device:Union[str,Device]="cpu"):
    length = 1
    for x in shape: length *= x
    return Tensor(reshape([0 for _ in range(length)], shape), dtype=dtype, device=device)
  @staticmethod
  def fill(value, shape:Tuple[int,...], dtype:Union[dtypes.DType,dtypes.ConstType]=dtypes.int64, device:Union[str,Device]="cpu"):
    length = 1
    for x in shape: length *= x
    return Tensor(reshape([value for _ in range(length)], shape), dtype=dtype, device=device)
  def astype(self, dtype:Union[dtypes.DType,dtypes.ConstType]):
    x = reshape(cuda.topyobj(self.__buf), self.__shape.shape) if self.__device.type == "CUDA" else reshape(cpu.topyobj(self.__buf), self.__shape.shape) # type: ignore
    return Tensor(x, dtype=dtype, device=f"{self.__device.type}:{self.__device.index}")
  @staticmethod
  def _pad_shape(shape: Tuple[int, ...], ndim: int) -> Tuple[int, ...]: return (1,) * (ndim - len(shape)) + shape
  @staticmethod
  def _pad_data(x, ndim: int):
    while len(shape_of(x)) < ndim:x = [x]
    return x
  @staticmethod
  def _get_item(x, idx, shape):
    for i, dim in zip(idx, shape):
      if dim == 1: x = x[0]
      else: x = x[i]
    return x
  @staticmethod
  def _build_tensor(x, shape_x, result_shape, idx=()):
    if len(idx) == len(result_shape): return Tensor._get_item(x, idx, shape_x)
    dim = result_shape[len(idx)]
    return [Tensor._build_tensor(x, shape_x, result_shape, idx + (i,)) for i in range(dim)]
  @staticmethod
  def broadcast(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    if a.device.type != b.device.type: raise ValueError("Cannot broadcast tensors on different devices")
    ax = a.data()
    bx = b.data()
    shape_a = a.shape.shape
    shape_b = b.shape.shape
    ndim = max(len(shape_a), len(shape_b))
    shape_a = Tensor._pad_shape(shape_a, ndim)
    shape_b = Tensor._pad_shape(shape_b, ndim)
    for da, db in zip(shape_a, shape_b):
      if da != db and da != 1 and db != 1: raise ValueError(f"Shapes {shape_a} and {shape_b} are not broadcastable")
    result_shape = tuple(max(da, db) for da, db in zip(shape_a, shape_b))
    ax = Tensor._pad_data(ax, ndim)
    bx = Tensor._pad_data(bx, ndim)
    A = Tensor._build_tensor(ax, shape_a, result_shape)
    B = Tensor._build_tensor(bx, shape_b, result_shape)
    return (Tensor(A, dtype=a.dtype, device=a.device), Tensor(B, dtype=b.dtype, device=b.device),)
  def __add__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.__shape != other.__shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.add(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.add(x.buf, y.buf))
    else: out = cpu.topyobj(cpu.add(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.add(self.buf, other.buf))
    return Tensor(reshape(out, self.__shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore
  def __radd__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.__shape != other.__shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.add(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.add(x.buf, y.buf))
    else: out = cpu.topyobj(cpu.add(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.add(self.buf, other.buf))
    return Tensor(reshape(out, self.__shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore
  def __sub__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.__shape != other.__shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.sub(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.sub(x.buf, y.buf))
    else: out = cpu.topyobj(cpu.sub(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.sub(self.buf, other.buf))
    return Tensor(reshape(out, self.__shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore
  def __rsub__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.__shape != other.__shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.sub(y.buf, x.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.sub(y.buf, x.buf))
    else: out = cpu.topyobj(cpu.sub(other.buf, self.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.sub(other.buf, self.buf))
    return Tensor(reshape(out, self.__shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore
  def __mul__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.__shape != other.__shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.mul(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.mul(x.buf, y.buf))
    else: out = cpu.topyobj(cpu.mul(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.mul(self.buf, other.buf))
    return Tensor(reshape(out, self.__shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore
  def __rmul__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.__shape != other.__shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.mul(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.mul(x.buf, y.buf))
    else: out = cpu.topyobj(cpu.mul(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.mul(self.buf, other.buf))
    return Tensor(reshape(out, self.__shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore
  def __truediv__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.__shape != other.__shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.tdiv(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.tdiv(x.buf, y.buf))
    else: out = cpu.topyobj(cpu.tdiv(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.tdiv(self.buf, other.buf))
    return Tensor(reshape(out, self.__shape.shape), dtype = self.__dtype, device = f"{self.__device.type}:{self.__device.index}") # type: ignore
  def __floordiv__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.__shape != other.__shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.fdiv(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.fdiv(x.buf, y.buf))
    else: out = cpu.topyobj(cpu.fdiv(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.fdiv(self.buf, other.buf))
    return Tensor(reshape(out, self.__shape.shape), dtype = dtypes.int64, device = f"{self.__device.type}:{self.__device.index}") # type: ignore
  def __pow__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.__shape != other.__shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.pow(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.pow(x.buf, y.buf))
    else: out = cpu.topyobj(cpu.pow(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.pow(self.buf, other.buf))
    return Tensor(reshape(out, self.__shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore
  def __mod__(self, other:Union[dtypes.ConstType,Tensor,List]): raise NotImplementedError
  def __eq__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: # type: ignore
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.__shape != other.__shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.eq(x.buf,y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.eq(x.buf, y.buf))
    else: out = cpu.topyobj(cpu.eq(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.eq(self.buf, other.buf))
    return Tensor(reshape(out, self.__shape.shape), dtype=dtypes.bool, device=f"{self.__device.type}:{self.__device.index}") # type: ignore
  def __ne__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: # type: ignore
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.__shape != other.__shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.ne(x.buf,y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.ne(x.buf, y.buf))
    else: out = cpu.topyobj(cpu.ne(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.ne(self.buf, other.buf))
    return Tensor(reshape(out, self.__shape.shape), dtype=dtypes.bool, device=f"{self.__device.type}:{self.__device.index}") # type: ignore
  def __gt__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: # type: ignore
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.__shape != other.__shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.gt(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.gt(x.buf, y.buf))
    else: out = cpu.topyobj(cpu.gt(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.gt(self.buf, other.buf))
    return Tensor(reshape(out, self.__shape.shape), dtype=dtypes.bool, device=f"{self.__device.type}:{self.__device.index}") # type: ignore
  def __ge__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: # type: ignore
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.__shape != other.__shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.ge(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.ge(x.buf, y.buf))
    else: out = cpu.topyobj(cpu.ge(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.ge(self.buf, other.buf))
    return Tensor(reshape(out, self.__shape.shape), dtype=dtypes.bool, device=f"{self.__device.type}:{self.__device.index}") # type: ignore
  def __lt__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: # type: ignore
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.__shape != other.__shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.lt(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.lt(x.buf, y.buf))
    else: out = cpu.topyobj(cpu.lt(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.lt(self.buf, other.buf))
    return Tensor(reshape(out, self.__shape.shape), dtype=dtypes.bool, device=f"{self.__device.type}:{self.__device.index}") # type: ignore
  def __le__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: # type: ignore
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.__shape != other.__shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.le(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.le(x.buf, y.buf))
    else: out = cpu.topyobj(cpu.le(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.le(self.buf, other.buf))
    return Tensor(reshape(out, self.__shape.shape), dtype=dtypes.bool, device=f"{self.__device.type}:{self.__device.index}") # type: ignore
  def __neg__(self): return Tensor(reshape(cpu.topyobj(cpu.neg(self.__buf)), self.__shape.shape), dtype=self.__dtype) if self.__device.type == "CPU" else Tensor(reshape(cuda.topyobj(cuda.neg(self.__buf)) ,self.__shape.shape), dtype=self.__dtype, device=f"cuda:{self.__device.index}") # type: ignore
  def __pos__(self): return Tensor(reshape(cpu.topyobj(cpu.pos(self.__buf)), self.__shape.shape), dtype=self.__dtype) if self.__device.type == "CPU" else Tensor(reshape(cuda.topyobj(cuda.pos(self.__buf)), self.__shape.shape), dtype=self.__dtype, device=f"cuda:{self.__device.index}") # type: ignore
  def __abs__(self):
    if self.__dtype == dtypes.complex64: res_dtype = dtypes.float32
    elif self.__dtype == dtypes.complex128: res_dtype = dtypes.float64
    else: res_dtype = self.__dtype
    return Tensor(reshape(cpu.topyobj(cpu.abs(self.__buf)), self.__shape.shape), dtype=res_dtype) if self.__device.type == "CPU" else Tensor(reshape(cuda.topyobj(cuda.abs(self.__buf)), self.__shape.shape), dtype=res_dtype, device=f"cuda:{self.__device.index}") # type: ignore

