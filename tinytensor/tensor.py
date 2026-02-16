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
    self.__stride = self.__shape.stride
    self.__ndim = self.__shape.ndim
    self.__size = self.__shape.size
    buf = flatten(buf) if isinstance(buf, list) else buf
    buf, self.__dtype = dtype_of(buf, dtype)
    self.__device = Device(device) if not isinstance(device, Device) else device
    self.__const = const
    if self.__device.type == "CPU": self.__buf = cpu.tocpu(buf, self.__shape.shape, self.__dtype.fmt)
    elif self.__device.type == "CUDA": self.__buf = cuda.tocuda(buf, self.__shape.shape, self.__dtype.fmt, self.__device.index)
    self.__requires_grad = requires_grad # TODO: yet to be implemented
  def __repr__(self): return f"Tensor(shape={self.shape}, dtype='{self.__dtype.ctype}', device={self.__device}, requires_grad={self.__requires_grad}, const={self.__const})"
  @property
  def ndim(self): return self.__ndim
  @property
  def shape(self): return self.__shape
  @property
  def stride(self): return self.__stride
  @property
  def size(self): return self.__size
  @property
  def device(self): return self.__device
  @property
  def nbyte(self): return self.shape.size * self.__dtype.nbyte
  @property
  def dtype(self): return self.__dtype
  @property
  def requires_grad(self): return self.__requires_grad
  @property
  def buf(self): return self.__buf
  def is_const(self): return self.__const

  def cuda(self, device_index:int=0): return Tensor(reshape(cpu.topyobj(self.__buf), self.shape.shape), dtype=self.__dtype, device=f"cuda:{device_index}") if self.__device.type == "CPU" else self # type: ignore
  def cpu(self): return Tensor(reshape(cuda.topyobj(self.__buf), self.shape.shape), dtype=self.__dtype, device=f"cpu") if self.__device.type == "CUDA" else self # type: ignore

  @staticmethod
  def _format_number(x):
    if isinstance(x, float):
      ax = abs(x)
      if x != 0 and ax < 1e-4: return float(f"{x:0e}")
      if ax > 1e6: return f"{x:.4e}"
      return float(f"{x:.4f}")
    if isinstance(x, int):
      if abs(x) >= 10**9: return f"{x:.2e}"
      return x
    if isinstance(x, complex):
      r, im = x.real, x.imag
      if r != 0 and abs(r) < 1e-4: r_fmt = f"{r:.0e}"
      else: r_fmt = f"{r:.4f}"
      if im != 0 and abs(im) < 1e-4: im_fmt = f"{im:.0e}"
      else: im_fmt = f"{im:.4f}"
      sign = "+" if im >= 0 else "-"
      return f"{r_fmt}{sign}{abs(float(im_fmt))}j"
    return x
  @staticmethod
  def _format_nested(arr):
    if isinstance(arr, list): return [Tensor._format_nested(v) for v in arr]
    return Tensor._format_number(arr)
  def data(self):
    if self.__device.type == "CUDA": x = cuda.topyobj(self.__buf)
    else: x = cpu.topyobj(self.__buf)
    return Tensor._format_nested(reshape(x, self.shape.shape)) # type: ignore

  @property
  def real(self):
    if self.dtype == dtypes.complex64 or self.dtype == dtypes.complex128:
      out = cpu.real(self.buf) if self.device.type == "CPU" else cuda.real(self.buf)
      out_dtype = dtypes.float32 if self.dtype == dtypes.complex64 else dtypes.float64
      return Tensor._from_view(buf=out, dtype=out_dtype, device=self.device)
    return Tensor._from_view(buf=self.buf, dtype=self.dtype, device=self.device)

  @property
  def imag(self):
    if self.dtype == dtypes.complex64 or self.dtype == dtypes.complex128:
      out = cpu.imag(self.buf) if self.device.type == "CPU" else cuda.imag(self.buf)
      out_dtype = dtypes.float32 if self.dtype == dtypes.complex64 else dtypes.float64
      return Tensor._from_view(buf=out, dtype=out_dtype, device=self.device)
    return Tensor.zeros(self.shape.shape)

  def __len__(self):
    if self.ndim == 0: raise ValueError("len() of a 0-d tensor")
    return len(cpu.topyobj(self.__buf)) if self.__device.type == "CPU" else len(cuda.topyobj(self.__buf)) # type: ignore

  def item(self):
    if self.ndim != 0: raise ValueError("only 0-d tensors can be converted to Python scalars")
    return self.data()

  def __bool__(self):
    if self.ndim != 0: raise ValueError("The truth value of tensor with more than one value is ambiguous. Use tensor.any() or tensor.all()")
    return bool(self.item())

  def any(self):
    if self.size == 0: return Tensor(False, dtype=dtypes.bool)
    if self.ndim == 0: return Tensor(bool(self.item() != 0), dtype=dtypes.bool)
    return Tensor(bool((self != 0).sum().item()), dtype=dtypes.bool)

  def all(self):
    if self.size == 0: return Tensor(True, dtype=dtypes.bool)
    if self.ndim == 0: return Tensor(bool(self.item() != 0), dtype=dtypes.bool)
    return Tensor(bool((self != 0).sum().item() == self.size), dtype=dtypes.bool)

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
    x = reshape(cuda.topyobj(self.__buf), self.shape.shape) if self.__device.type == "CUDA" else reshape(cpu.topyobj(self.__buf), self.shape.shape) # type: ignore
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

  @classmethod
  def _from_view(cls, buf, dtype, device, requires_grad=False, const=False):
    obj = cls.__new__(cls)
    obj.__buf = buf
    obj.__dtype = dtype
    obj.__requires_grad = requires_grad
    obj.__const = const
    obj.__device = device
    if device.type == "CPU":
      shp = cpu.shape(buf)
      obj.__shape = Shape(shp)
      obj.__stride = cpu.stride(buf)
      obj.__ndim = cpu.ndim(buf)
    elif device.type == "CUDA":
      shp = cuda.shape(buf)
      obj.__shape = Shape(shp)
      obj.__stride = cuda.stride(buf)
      obj.__ndim = cuda.ndim(buf)
    return obj

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
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.add(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.add(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.add(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.add(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __radd__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.add(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.add(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.add(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.add(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __sub__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.sub(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.sub(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.sub(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.sub(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __rsub__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.sub(y.buf, x.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.sub(y.buf, x.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.sub(other.buf, self.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.sub(other.buf, self.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __mul__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.mul(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.mul(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.mul(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.mul(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __rmul__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.mul(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.mul(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.mul(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.mul(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __truediv__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.tdiv(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.tdiv(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.tdiv(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.tdiv(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype = self.__dtype, device = f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __floordiv__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.fdiv(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.fdiv(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.fdiv(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.fdiv(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype = dtypes.int64, device = f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __pow__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.pow(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.pow(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.pow(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.pow(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __mod__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.mod(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.mod(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.mod(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.mod(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __eq__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: # type: ignore
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.eq(x.buf,y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.eq(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.eq(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.eq(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=dtypes.bool, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __ne__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: # type: ignore
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.ne(x.buf,y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.ne(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.ne(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.ne(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=dtypes.bool, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __gt__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: # type: ignore
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.gt(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.gt(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.gt(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.gt(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=dtypes.bool, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __ge__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: # type: ignore
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.ge(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.ge(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.ge(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.ge(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=dtypes.bool, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __lt__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: # type: ignore
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.lt(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.lt(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.lt(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.lt(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=dtypes.bool, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __le__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: # type: ignore
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.le(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.le(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.le(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.le(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=dtypes.bool, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __neg__(self): return Tensor(reshape(cpu.topyobj(cpu.neg(self.__buf)), self.shape.shape), dtype=self.__dtype) if self.__device.type == "CPU" else Tensor(reshape(cuda.topyobj(cuda.neg(self.__buf)) ,self.shape.shape), dtype=self.__dtype, device=f"cuda:{self.__device.index}") # type: ignore
  def __pos__(self): return Tensor(reshape(cpu.topyobj(cpu.pos(self.__buf)), self.shape.shape), dtype=self.__dtype) if self.__device.type == "CPU" else Tensor(reshape(cuda.topyobj(cuda.pos(self.__buf)), self.shape.shape), dtype=self.__dtype, device=f"cuda:{self.__device.index}") # type: ignore
  def __abs__(self):
    if self.__dtype == dtypes.complex64: res_dtype = dtypes.float32
    elif self.__dtype == dtypes.complex128: res_dtype = dtypes.float64
    else: res_dtype = self.__dtype
    return Tensor(reshape(cpu.topyobj(cpu.abs(self.__buf)), self.shape.shape), dtype=res_dtype) if self.__device.type == "CPU" else Tensor(reshape(cuda.topyobj(cuda.abs(self.__buf)), self.shape.shape), dtype=res_dtype, device=f"cuda:{self.__device.index}") # type: ignore

  def __and__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.and_(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.and_(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.and_(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.and_(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __rand__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.and_(y.buf, x.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.and_(y.buf, x.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.and_(other.buf, self.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.and_(other.buf, self.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __or__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.or_(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.or_(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.or_(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.or_(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __ror__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.or_(y.buf, x.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.or_(y.buf, x.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.or_(other.buf, self.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.or_(other.buf, self.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __invert__(self):
    out = cpu.topyobj(cpu.not_(self.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.not_(self.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __xor__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.xor_(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.xor_(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.xor_(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.xor_(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __rxor__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.xnor_(y.buf, x.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.xor_(y.buf, x.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.xnor_(other.buf, self.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.xor_(other.buf, self.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def nand(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.nand_(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.nand_(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.nand_(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.nand_(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def nor(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.nor_(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.nor_(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.nor_(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.nor_(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def xnor(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.xnor_(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.xnor_(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.xnor_(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.xnor_(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __rshift__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.rshift(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.rshift(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.rshift(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.rshift(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __rrshift__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.rshift(y.buf, x.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.rshift(y.buf, x.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.rshift(other.buf, self.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.rshift(other.buf, self.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __lshift__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.lshift(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.lshift(x.buf, y.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.lshift(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.lshift(self.buf, other.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def __rlshift__(self, other:Union[dtypes.ConstType,Tensor,List]):
    if not isinstance(other, Tensor): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.shape != other.shape:
      x, y = Tensor.broadcast(self, other)
      out = cpu.topyobj(cpu.lshift(y.buf, x.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.lshift(y.buf, x.buf))
      return Tensor(reshape(out, x.shape.shape), dtype=x.dtype, device=f"{x.device.type}:{x.device.index}") # type: ignore
    out = cpu.topyobj(cpu.lshift(other.buf, self.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.lshift(other.buf, self.buf))
    return Tensor(reshape(out, self.shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def reshape(self, shape:Tuple[int,...]):
    out = cpu.topyobj(self.buf) if self.__device.type == "CPU" else cuda.topyobj(self.buf)
    return Tensor(reshape(out, shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore

  def permute(self, axes:Tuple[int,...]):
    if self.__device.type == "CPU": out = cpu.permute(self.__buf, axes)
    else: out = cuda.permute(self.__buf, axes)
    return Tensor._from_view(buf=out, dtype=self.__dtype, device=self.__device)

  def sum(self, axis:int|None=None, keepdim:bool=False):
    if axis is None:
      result_buf = cpu.sum(self.__buf, None) if self.__device.type == "CPU" else cuda.sum(self.__buf, None)
      result = Tensor._from_view(result_buf, self.__dtype, self.__device, self.__requires_grad, const=self.__const)
      return result
    if axis < 0: axis += self.__ndim
    if axis < 0 or axis >= self.__ndim: raise ValueError(f"axis {axis} is out of range for tensor with {self.__ndim} dimensions")
    result_buf = cpu.sum(self.__buf, axis) if self.__device.type == "CPU" else cuda.sum(self.__buf, axis)
    result = Tensor._from_view(result_buf, self.__dtype, self.__device, self.__requires_grad, const=self.__const)
    if keepdim:
      new_shape = list(self.shape.shape)
      new_shape[axis] = 1
      result = result.reshape(tuple(new_shape))
    return result

  def bmm(self, other:Tensor):
    if not isinstance(other, Tensor): raise ValueError(f"given value must be the Tensor object")
    out = cpu.topyobj(cpu.bmm(self.buf, other.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.bmm(self.buf, other.buf))
    batch_dims = self.shape.shape[:-2]
    m = self.shape.shape[-2]
    n = other.shape.shape[-1]
    out_shape = batch_dims + (m ,n)
    return Tensor(reshape(out, out_shape), dtype=self.dtype, device=f"{self.device.type}:{self.device.index}") # type: ignore

  def numpy(self):
    try:
      import numpy as np
      return np.array(self.data())
    except ImportError:
      import warnings
      warnings.warn("NumPy is required for Tensor.numpy(). Install it with: pip install numpy", RuntimeWarning)
      return None

  def exp(self):
    out = cpu.exp(self.buf) if self.__device.type == "CPU" else NotImplemented
    out_dtype = cpu.dtype(out)
    return Tensor._from_view(out, dtype=dtypes.DType.from_ctype(out_dtype), device=self.device)

  def log(self):
    out = cpu.log(self.buf) if self.__device.type == "CPU" else NotImplemented
    out_dtype = cpu.dtype(out)
    return Tensor._from_view(out, dtype=dtypes.DType.from_ctype(out_dtype), device=self.device)

  def log2(self):
    out = cpu.log2(self.buf) if self.__device.type == "CPU" else NotImplemented
    out_dtype = cpu.dtype(out)
    return Tensor._from_view(out, dtype=dtypes.DType.from_ctype(out_dtype), device=self.device)

  def log10(self):
    out = cpu.log10(self.buf) if self.__device.type == "CPU" else NotImplemented
    out_dtype = cpu.dtype(out)
    return Tensor._from_view(out, dtype=dtypes.DType.from_ctype(out_dtype), device=self.device)

  def sin(self):
    out = cpu.sin(self.buf) if self.__device.type == "CPU" else NotImplemented
    out_dtype = cpu.dtype(out)
    return Tensor._from_view(out, dtype=dtypes.DType.from_ctype(out_dtype), device=self.device)

  def cos(self):
    out = cpu.cos(self.buf) if self.__device.type == "CPU" else NotImplemented
    out_dtype = cpu.dtype(out)
    return Tensor._from_view(out, dtype=dtypes.DType.from_ctype(out_dtype), device=self.device)

  def tan(self):
    out = cpu.tan(self.buf) if self.__device.type == "CPU" else NotImplemented
    out_dtype = cpu.dtype(out)
    return Tensor._from_view(out, dtype=dtypes.DType.from_ctype(out_dtype), device=self.device)

  def asin(self):
    out = cpu.asin(self.buf) if self.__device.type == "CPU" else NotImplemented
    out_dtype = cpu.dtype(out)
    return Tensor._from_view(out, dtype=dtypes.DType.from_ctype(out_dtype), device=self.device)

  def acos(self):
    out = cpu.acos(self.buf) if self.__device.type == "CPU" else NotImplemented
    out_dtype = cpu.dtype(out)
    return Tensor._from_view(out, dtype=dtypes.DType.from_ctype(out_dtype), device=self.device)

  def atan(self):
    out = cpu.atan(self.buf) if self.__device.type == "CPU" else NotImplemented
    out_dtype = cpu.dtype(out)
    return Tensor._from_view(out, dtype=dtypes.DType.from_ctype(out_dtype), device=self.device)

  def sinh(self):
    out = cpu.sinh(self.buf) if self.__device.type == "CPU" else NotImplemented
    out_dtype = cpu.dtype(out)
    return Tensor._from_view(out, dtype=dtypes.DType.from_ctype(out_dtype), device=self.device)

  def cosh(self):
    out = cpu.cosh(self.buf) if self.__device.type == "CPU" else NotImplemented
    out_dtype = cpu.dtype(out)
    return Tensor._from_view(out, dtype=dtypes.DType.from_ctype(out_dtype), device=self.device)

  def tanh(self):
    out = cpu.tanh(self.buf) if self.__device.type == "CPU" else NotImplemented
    out_dtype = cpu.dtype(out)
    return Tensor._from_view(out, dtype=dtypes.DType.from_ctype(out_dtype), device=self.device)

  def asinh(self):
    out = cpu.asinh(self.buf) if self.__device.type == "CPU" else NotImplemented
    out_dtype = cpu.dtype(out)
    return Tensor._from_view(out, dtype=dtypes.DType.from_ctype(out_dtype), device=self.device)

  def acosh(self):
    out = cpu.acosh(self.buf) if self.__device.type == "CPU" else NotImplemented
    out_dtype = cpu.dtype(out)
    return Tensor._from_view(out, dtype=dtypes.DType.from_ctype(out_dtype), device=self.device)

  def atanh(self):
    out = cpu.atanh(self.buf) if self.__device.type == "CPU" else NotImplemented
    out_dtype = cpu.dtype(out)
    return Tensor._from_view(out, dtype=dtypes.DType.from_ctype(out_dtype), device=self.device)

