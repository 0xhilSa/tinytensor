from __future__ import annotations
import ctypes
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
  ) -> None:
    self.__shape = Shape(shape_of(buf))
    self.__stride = self.__shape.stride
    self.__ndim = self.__shape.ndim
    self.__size = self.__shape.size
    buf = flatten(buf) if isinstance(buf, list) else buf
    buf, self.__dtype = dtype_of(buf, dtype)
    self.__device = Device(device) if not isinstance(device, Device) else device
    self.__const = const
    self.__offset = 0
    if self.__device.type == "CPU": self.__buf = cpu.tocpu(buf, self.__shape.shape, self.__dtype.fmt)
    elif self.__device.type == "CUDA": self.__buf = cuda.tocuda(buf, self.__shape.shape, self.__dtype.fmt, self.__device.index)
    self.__requires_grad = requires_grad # TODO: yet to be implemented
  def __repr__(self) -> str: return f"Tensor(shape={self.shape}, dtype='{self.__dtype.ctype}', device={self.__device}, requires_grad={self.__requires_grad}, const={self.__const})"
  @property
  def ndim(self) -> int: return self.__ndim
  @property
  def shape(self) -> Shape: return self.__shape
  @property
  def stride(self) -> Tuple[int,...]: return self.__stride
  @property
  def size(self) -> int: return self.__size
  @property
  def device(self) -> Device: return self.__device
  @property
  def nbyte(self) -> int: return self.shape.size * self.__dtype.nbyte
  @property
  def dtype(self) -> dtypes.DType: return self.__dtype
  @property
  def requires_grad(self) -> bool: return self.__requires_grad
  @property
  def buf(self) -> ctypes.c_void_p: return self.__buf
  def is_const(self): return self.__const

  def cuda(self, device_index:int=0) -> Tensor: return Tensor(reshape(cpu.topyobj(self.__buf), self.shape.shape), dtype=self.__dtype, device=f"cuda:{device_index}") if self.__device.type == "CPU" else self # type: ignore
  def cpu(self) -> Tensor: return Tensor(reshape(cuda.topyobj(self.__buf), self.shape.shape), dtype=self.__dtype, device=f"cpu") if self.__device.type == "CUDA" else self # type: ignore

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
    flat = cpu.topyobj(self.__buf) if self.__device.type == "CPU" else cuda.topyobj(self.__buf)
    if self.__ndim == 0:
      if isinstance(flat, (list, tuple)): return Tensor._format_number(flat[self.__offset])
      return Tensor._format_number(flat)
    def build(indices, dim):
      if dim == self.__ndim:
        real_offset = self.__offset
        for i, s in zip(indices, self.__stride): real_offset += i * s
        return Tensor._format_number(flat[real_offset]) # type: ignore
      return [
        build(indices + [i], dim + 1)
        for i in range(self.__shape.shape[dim])
      ]
    return build([], 0)

  def is_contiguous(self):
    expected = []
    acc = 1
    for size in reversed(self.shape.shape):
      expected.insert(0, acc)
      acc *= size
    return tuple(expected) == self.__stride

  def conj(self) -> Tensor:
    if self.dtype == dtypes.complex64 or self.dtype == dtypes.complex128:
      out = cpu.conj(self.buf) if self.device.type == "CPU" else cuda.conj(self.buf)
      return Tensor._from_backend(buf=out, dtype=self.dtype, device=self.device)
    return self

  @property
  def real(self) -> Tensor:
    if self.dtype == dtypes.complex64 or self.dtype == dtypes.complex128:
      out = cpu.real(self.buf) if self.device.type == "CPU" else cuda.real(self.buf)
      out_dtype = dtypes.float32 if self.dtype == dtypes.complex64 else dtypes.float64
      return Tensor._from_backend(buf=out, dtype=out_dtype, device=self.device)
    return Tensor._from_backend(buf=self.buf, dtype=self.dtype, device=self.device)

  @property
  def imag(self) -> Tensor:
    if self.dtype == dtypes.complex64 or self.dtype == dtypes.complex128:
      out = cpu.imag(self.buf) if self.device.type == "CPU" else cuda.imag(self.buf)
      out_dtype = dtypes.float32 if self.dtype == dtypes.complex64 else dtypes.float64
      return Tensor._from_backend(buf=out, dtype=out_dtype, device=self.device)
    return Tensor.zeros(self.shape.shape)

  def _normalize_index(self, index):
    if not isinstance(index, tuple): index = (index,)
    if Ellipsis in index:
      ell_pos = index.index(Ellipsis)
      missing = self.__ndim - (len(index) - 1)
      index = (index[:ell_pos] + (slice(None),) * missing + index[ell_pos+1:])
    if len(index) < self.__ndim: index = index + (slice(None),) * (self.__ndim - len(index))
    if len(index) > self.__ndim: raise IndexError("too many indices")
    return index

  def _scalar_get(self, index_tuple):
    offset = 0
    for i, stride, size in zip(index_tuple, self.__stride, self.__shape.shape):
      if i < 0: i += size
      if i >= size or i < 0: raise IndexError("index out of range")
      offset += i * stride
    caps = cpu.getitem(self.__buf, offset) if self.__device.type == "CPU" else cuda.getitem(self.__buf, offset)
    return Tensor._from_backend(caps, dtype=self.dtype, device=self.device)

  def _scalar_from_offset(self, offset):
    caps = cpu.getitem(self.__buf, offset) if self.__device.type == "CPU" else cuda.getitem(self.__buf, offset)
    return Tensor._from_backend(caps, self.dtype, self.device)

  def _slice_get(self, index):
    new_shape = []
    new_stride = []
    offset = self.__offset
    for idx, size, stride in zip(index, self.__shape.shape, self.__stride):
      if isinstance(idx, int):
        if idx < 0: idx += size
        if idx >= size or idx < 0: raise IndexError("index out of range")
        offset += idx * stride
      elif isinstance(idx, slice):
        start, stop, step = idx.indices(size)
        if step > 0: length = max(0, (stop - start + (step - 1)) // step)
        else: length = max(0, (start - stop - step - 1) // (-step))
        new_shape.append(length)
        new_stride.append(stride * step)
        offset += start * stride
      else: raise TypeError("invalid index")
    if len(new_shape) == 0: return self._scalar_from_offset(offset)
    return Tensor._from_view(buf=self.__buf, shape=tuple(new_shape), stride=tuple(new_stride), offset=offset, dtype=self.__dtype, device=self.__device)

  def __getitem__(self, index):
    index = self._normalize_index(index)
    if all(isinstance(i, int) for i in index): return self._scalar_get(index)
    return self._slice_get(index)

  def __setitem__(self, index):
    pass

  def __len__(self) -> int:
    if self.ndim == 0: raise ValueError("len() of a 0-d tensor")
    return len(cpu.topyobj(self.__buf)) if self.__device.type == "CPU" else len(cuda.topyobj(self.__buf)) # type: ignore

  def item(self):
    if self.ndim != 0: raise ValueError("only 0-d tensors can be converted to Python scalars")
    return self.data()

  def __bool__(self) -> bool:
    if self.ndim != 0: raise ValueError("The truth value of tensor with more than one value is ambiguous. Use tensor.any() or tensor.all()")
    return bool(self.item())

  def any(self) -> Tensor:
    if self.size == 0: return Tensor(False, dtype=dtypes.bool)
    if self.ndim == 0: return Tensor(bool(self.item() != 0), dtype=dtypes.bool)
    return Tensor(bool((self != 0).sum().item()), dtype=dtypes.bool)

  def all(self) -> Tensor:
    if self.size == 0: return Tensor(True, dtype=dtypes.bool)
    if self.ndim == 0: return Tensor(bool(self.item() != 0), dtype=dtypes.bool)
    return Tensor(bool((self != 0).sum().item() == self.size), dtype=dtypes.bool)

  @staticmethod
  def empty(*shape, dtype:dtypes.DType=dtypes.float32, device:Union[str,Device]="cpu", requires_grad:bool=False, const:bool=False) -> Tensor:
    if len(shape) == 1 and isinstance(shape, (list, tuple)): shape = shape[0]
    if not isinstance(shape, tuple): raise TypeError("shape must be a tuple")
    if len(shape) == 0: raise ValueError("shape cannot be empty")
    for dim in shape:
      if not isinstance(dim, int): raise TypeError("shape elements must be int")
      if dim < 0: raise ValueError("negative dimensions not allowed")
    device = Device(device) if not isinstance(device, Device) else device
    caps = cpu.empty(shape, dtype.fmt) if device.type == "CPU" else cuda.empty(shape, dtype.fmt, device_index=device.index)
    return Tensor._from_backend(caps, dtype=dtype, device=device, requires_grad=requires_grad, const=const)

  @staticmethod
  def ones(*shape, dtype:Union[dtypes.DType,dtypes.ConstType]=dtypes.int64, device:Union[str,Device]="cpu") -> Tensor:
    if len(shape) == 1 and isinstance(shape, (list, tuple)): shape = shape[0]
    length = 1
    for x in shape: length *= x
    return Tensor(reshape([1 for _ in range(length)], shape), dtype=dtype, device=device)

  @staticmethod
  def zeros(*shape, dtype:Union[dtypes.DType,dtypes.ConstType]=dtypes.int64, device:Union[str,Device]="cpu") -> Tensor:
    if len(shape) == 1 and isinstance(shape, (list, tuple)): shape = shape[0]
    length = 1
    for x in shape: length *= x
    return Tensor(reshape([0 for _ in range(length)], shape), dtype=dtype, device=device)

  @staticmethod
  def fill(value, *shape, dtype:Union[dtypes.DType,dtypes.ConstType]=dtypes.int64, device:Union[str,Device]="cpu") -> Tensor:
    if len(shape) == 1 and isinstance(shape, (list, tuple)): shape = shape[0]
    length = 1
    for x in shape: length *= x
    return Tensor(reshape([value for _ in range(length)], shape), dtype=dtype, device=device)

  def astype(self, dtype:Union[dtypes.DType,dtypes.ConstType]) -> Tensor:
    x = reshape(cuda.topyobj(self.__buf), self.shape.shape) if self.__device.type == "CUDA" else reshape(cpu.topyobj(self.__buf), self.shape.shape) # type: ignore
    return Tensor(x, dtype=dtype, device=f"{self.__device.type}:{self.__device.index}")

  @staticmethod
  def _pad_shape(shape:Tuple[int,...], ndim:int) -> Tuple[int,...]: return (1,) * (ndim - len(shape)) + shape

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
  def _from_view(cls, buf, dtype, shape, stride, offset, device, requires_grad=False, const=False):
    obj = cls.__new__(cls)
    obj.__buf = buf
    obj.__shape = Shape(shape)
    obj.__stride = stride
    obj.__offset = offset
    obj.__dtype = dtype
    obj.__device = device
    obj.__requires_grad = requires_grad
    obj.__const = const
    obj.__ndim = len(shape)
    return obj

  @classmethod
  def _from_backend(cls, buf, dtype, device, requires_grad=False, const=False):
    obj = cls.__new__(cls)
    obj.__buf = buf
    obj.__dtype = dtype
    obj.__device = device
    obj.__requires_grad = requires_grad
    obj.__const = const
    obj.__offset = 0
    if device.type == "CPU":
      shp = cpu.shape(buf)
      obj.__shape = Shape(shp)
      obj.__stride = cpu.stride(buf)
      obj.__ndim = cpu.ndim(buf)
    else:
      shp = cuda.shape(buf)
      obj.__shape = Shape(shp)
      obj.__stride = cuda.stride(buf)
      obj.__ndim = cuda.ndim(buf)
    return obj

  @staticmethod
  def _promote_binary(x:Tensor, y:Tensor) -> Tuple[Tensor, Tensor]:
    if x.device != y.device: raise RuntimeError("Tensors must be on same device")
    dx, dy = x.dtype, y.dtype
    complexes = [dtypes.complex64, dtypes.complex128]
    floats = [dtypes.float16, dtypes.float32, dtypes.float64]
    ints = {
      dtypes.int8, dtypes.uint8,
      dtypes.int16, dtypes.uint16,
      dtypes.int32, dtypes.uint32,
      dtypes.int64, dtypes.uint64,
    }
    if dx in complexes or dy in complexes:
      order = {
        dtypes.complex64: 0,
        dtypes.complex128: 1
      }
      if dx not in complexes: dx = dtypes.complex64
      if dy not in complexes: dy = dtypes.complex64
      target = dx if order[dx] > order[dy] else dy
      return x.astype(target), y.astype(target)
    if dx in floats or dy in floats:
      order = {
        dtypes.float16: 0,
        dtypes.float32: 1,
        dtypes.float64: 2
      }
      target = dx if order.get(dx, -1) > order.get(dy, -1) else dy
      return x.astype(target), y.astype(target)
    if dx in ints and dy in ints:
      order = {
        dtypes.int8: 0, dtypes.uint8: 0,
        dtypes.int16: 1, dtypes.uint16: 1,
        dtypes.int32: 2, dtypes.uint32: 2,
        dtypes.int64: 3, dtypes.uint64: 3,
      }
      target = dx if order[dx] > order[dy] else dy
      return x.astype(target), y.astype(target)
    raise TypeError("Unsupported dtype for binary operation")

  @staticmethod
  def _promote_tdiv(x:Tensor, y:Tensor) -> Tuple[Tensor, Tensor]:
    if x.device != y.device: raise RuntimeError("Tensors must be on same device")
    dx, dy = x.dtype, y.dtype
    floats = [dtypes.float16, dtypes.float32, dtypes.float64]
    complexes = [dtypes.complex64, dtypes.complex128]
    ints8  = {dtypes.int8, dtypes.uint8}
    ints16 = {dtypes.int16, dtypes.uint16}
    ints32 = {dtypes.int32, dtypes.uint32}
    ints64 = {dtypes.int64, dtypes.uint64}
    if dx in complexes or dy in complexes:
      order = {
        dtypes.complex64: 0,
        dtypes.complex128: 1
      }
      if dx not in complexes: dx = dtypes.complex64 if dx in floats or dx in ints32 else dtypes.complex128
      if dy not in complexes: dy = dtypes.complex64 if dy in floats or dy in ints32 else dtypes.complex128
      target = dx if order[dx] > order[dy] else dy
      return x.astype(target), y.astype(target)
    if dx in floats or dy in floats:
      order = {
        dtypes.float16: 0,
        dtypes.float32: 1,
        dtypes.float64: 2
      }
      target = dx if order.get(dx, -1) > order.get(dy, -1) else dy
      return x.astype(target), y.astype(target)
    if dx in ints64 or dy in ints64: target = dtypes.float32
    elif dx in ints32 or dy in ints32: target = dtypes.float32
    elif dx in ints16 or dy in ints16: target = dtypes.float32
    elif dx in ints8 or dy in ints8: target = dtypes.float16
    else: target = dtypes.float16
    return x.astype(target), y.astype(target)

  @staticmethod
  def _promote_fdiv(x:Tensor, y:Tensor) -> Tuple[Tensor, Tensor]:
    if x.device != y.device: raise RuntimeError("Tensors must be on same device")
    complexes = {dtypes.complex64, dtypes.complex128}
    if x.dtype in complexes or y.dtype in complexes: raise TypeError("floor div not supported for complex dtype tensor")
    dx, dy = x.dtype, y.dtype
    floats = {dtypes.float16, dtypes.float32, dtypes.float64}
    ints = {
      dtypes.int8, dtypes.uint8,
      dtypes.int16, dtypes.uint16,
      dtypes.int32, dtypes.uint32,
      dtypes.int64, dtypes.uint64,
    }
    if dx in floats or dy in floats:
      order = {
        dtypes.float16: 0,
        dtypes.float32: 1,
        dtypes.float64: 2
      }
      target = dx if order.get(dx, -1) > order.get(dy, -1) else dy
      return x.astype(target), y.astype(target)
    if dx in ints and dy in ints:
      order = {
        dtypes.int8: 0, dtypes.uint8: 0,
        dtypes.int16: 1, dtypes.uint16: 1,
        dtypes.int32: 2, dtypes.uint32: 2,
        dtypes.int64: 3, dtypes.uint64: 3,
      }
      target = dx if order[dx] > order[dy] else dy
      return x.astype(target), y.astype(target)
    raise TypeError("Unsupported dtype for floor division")

  @staticmethod
  def _promote_unary_math(x:Tensor) -> Tensor:
    dt = x.dtype
    if dt in (dtypes.complex64, dtypes.complex128): return x
    if dt in (dtypes.float16, dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128): return x
    if dt in (dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16): return x.astype(dtypes.float16)
    if dt in (dtypes.int32, dtypes.uint32): return x.astype(dtypes.float32)
    if dt in (dtypes.int64, dtypes.uint64): return x.astype(dtypes.float64)
    raise TypeError("Unsupported dtype for math function")

  @staticmethod
  def _promote_bitwise(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    if x.device != y.device: raise RuntimeError("Tensors must be on same device")
    dx, dy = x.dtype, y.dtype
    ints = {
      dtypes.bool: -1,
      dtypes.int8: 0,  dtypes.uint8: 0,
      dtypes.int16: 1, dtypes.uint16: 1,
      dtypes.int32: 2, dtypes.uint32: 2,
      dtypes.int64: 3, dtypes.uint64: 3,
    }
    if dx not in ints or dy not in ints: raise TypeError("Bitwise ops require integer or bool dtype")
    target = dx if ints[dx] > ints[dy] else dy
    return x.astype(target), y.astype(target)

  @staticmethod
  def _promote_logical(x:Tensor, y:Tensor) -> Tuple[Tensor, Tensor]:
    if x.device != y.device: raise RuntimeError("Tensors must be on same device")
    allowed = (
      dtypes.BOOLEAN +
      dtypes.INT +
      dtypes.UINT +
      dtypes.FLOAT +
      dtypes.COMPLEX
    )
    if x.dtype not in allowed or y.dtype not in allowed: raise TypeError("Unsupported dtype for logical operation")
    return Tensor._promote_binary(x, y)

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

  def binop(self, other, op, promote:str|None="binary", reverse: bool = False, res_dtype=None):
    if not isinstance(other, Tensor): other = Tensor(other, device=f"{self.device.type}:{self.device.index}")
    if op in {
      "bitwise_and", "bitwise_or", "bitwise_xor",
      "bitwise_nand", "bitwise_nor", "bitwise_xnor",
      "lshift", "rshift"
    }:
      x, y = Tensor._promote_bitwise(self, other)
    elif op in {
      "logical_and", "logical_or", "logical_xor",
      "logical_nand", "logical_nor", "logical_xnor"
    }:
      x, y = Tensor._promote_logical(self, other)
      res_dtype = dtypes.bool
    elif promote == "tdiv": x, y = Tensor._promote_tdiv(self, other)
    elif promote == "fdiv": x, y = Tensor._promote_fdiv(self, other)
    else: x, y = Tensor._promote_binary(self, other)
    if x.shape != y.shape: x, y = Tensor.broadcast(x, y)
    if reverse: x, y = y, x
    if x.device.type == "CPU":
      kernel = getattr(cpu, op)
      out = cpu.topyobj(kernel(x.buf, y.buf))
    else:
      kernel = getattr(cuda, op)
      out = cuda.topyobj(kernel(x.buf, y.buf))
    dtype = res_dtype if res_dtype is not None else x.dtype
    return Tensor(reshape(out, x.shape.shape), dtype=dtype, device=x.device) # type: ignore

  def uop(self, op:str, promote:bool=False, res_dtype:dtypes.DType|None=None):
    x = Tensor._promote_unary_math(self) if promote else self
    if x.__device.type == "CPU":
      kernel = getattr(cpu, op)
      out = cpu.topyobj(kernel(x.buf))
    else:
      kernel = getattr(cuda, op)
      out = cuda.topyobj(kernel(x.buf))
    dtype = res_dtype if res_dtype is not None else x.dtype
    return Tensor(reshape(out, x.shape.shape), dtype=dtype, device=x.device) # type: ignore

  def __add__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "add")
  def __radd__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "add", reverse=True)
  def __sub__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "sub")
  def __rsub__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "sub", reverse=True)
  def __mul__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "mul")
  def __rmul__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "mul", reverse=True)
  def __truediv__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "tdiv", promote="tdiv")
  def __rtruediv__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "tdiv", promote="tdiv", reverse=True)
  def __floordiv__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "fdiv", promote="fdiv")
  def __rfloordiv__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "fdiv", promote="fdiv", reverse=True)
  def __pow__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "pow")
  def __rpow__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "pow", reverse=True)
  def __mod__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "mod")
  def __rmod__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "mod", reverse=True)

  def __eq__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "eq", res_dtype=dtypes.bool) # type: ignore
  def __ne__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "ne", res_dtype=dtypes.bool) # type: ignore
  def __gt__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "gt", res_dtype=dtypes.bool) # type: ignore
  def __ge__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "ge", res_dtype=dtypes.bool) # type: ignore
  def __lt__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "lt", res_dtype=dtypes.bool) # type: ignore
  def __le__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "le", res_dtype=dtypes.bool) # type: ignore

  def __neg__(self) -> Tensor: return self.uop("neg")
  def __pos__(self) -> Tensor: return self.uop("pos")

  def __abs__(self) -> Tensor:
    if self.__dtype == dtypes.complex64: return self.uop("abs", res_dtype=dtypes.float32)
    if self.__dtype == dtypes.complex128: return self.uop("abs", res_dtype=dtypes.float64)
    return self.uop("abs")

  def __and__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "bitwise_and")
  def __rand__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "bitwise_and", reverse=True)
  def __or__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "bitwise_or")
  def __ror__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "bitwise_or", reverse=True)
  def __xor__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "bitwise_xor")
  def __rxor__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "bitwise_xor", reverse=True)
  def __invert__(self) -> Tensor: return self.uop("bitwise_not")

  def bitwise_and(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "bitwise_and")
  def bitwise_nand(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "bitwise_nand")
  def bitwise_or(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "bitwise_or")
  def bitwise_nor(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "bitwise_nor")
  def bitwise_xor(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "bitwise_xor")
  def bitwise_xnor(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "bitwise_xnor")
  def bitwise_not(self) -> Tensor: return self.uop("bitwise_not")

  def logical_and(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "logical_and", res_dtype=dtypes.bool)
  def logical_nand(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "logical_nand", res_dtype=dtypes.bool)
  def logical_or(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "logical_or", res_dtype=dtypes.bool)
  def logical_nor(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "logical_nor", res_dtype=dtypes.bool)
  def logical_xor(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "logical_xor", res_dtype=dtypes.bool)
  def logical_xnor(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "logical_xnor", res_dtype=dtypes.bool)
  def logical_not(self) -> Tensor: return self.uop("logical_not", res_dtype=dtypes.bool)

  def __rshift__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "rshift")
  def __rrshift__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "rshift", reverse=True)
  def __lshift__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "lshift")
  def __rlshift__(self, other:Union[dtypes.ConstType,Tensor,List]) -> Tensor: return self.binop(other, "lshift", reverse=True)

  def reshape(self, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = shape[0]
    numel = 1
    for d in self.__shape: numel *= d
    new_shape = list(shape)
    unknown_index = -1
    known_product = 1
    for i, dim in enumerate(new_shape):
      if dim == -1:
        if unknown_index != -1: raise ValueError("Only one -1 allowed in reshape")
        unknown_index = i
      else: known_product *= dim
    if unknown_index != -1:
      if numel % known_product != 0: raise ValueError("Shape is not compatible with tensor size")
      new_shape[unknown_index] = numel // known_product
    final_product = 1
    for d in new_shape: final_product *= d
    if final_product != numel: raise ValueError("Shape does not match tensor size")
    new_stride = []
    running = 1
    for dim in reversed(new_shape):
      new_stride.insert(0, running)
      running *= dim
    return Tensor._from_view(
      buf=self.__buf,
      dtype=self.__dtype,
      shape=tuple(new_shape),
      stride=tuple(new_stride),
      offset=self.__offset,
      device=self.__device,
      requires_grad=self.__requires_grad,
      const=self.__const,
    )

  def permute(self, *axes):
    if len(axes) == 1 and isinstance(axes[0], (tuple, list)): axes = axes[0]
    new_shape = tuple(self.__shape.shape[i] for i in axes)
    new_stride = tuple(self.__stride[i] for i in axes)
    return Tensor._from_view(
      buf=self.__buf,
      shape=new_shape,
      stride=new_stride,
      offset=self.__offset,
      dtype=self.__dtype,
      device=self.__device,
      requires_grad=self.__requires_grad,
      const=self.__const
    )

  def sum(self, axis:int|None=None, keepdim:bool=False):
    if axis is None:
      result_buf = cpu.sum(self.__buf, None) if self.__device.type == "CPU" else cuda.sum(self.__buf, None)
      result = Tensor._from_backend(result_buf, self.__dtype, self.__device, self.__requires_grad, const=self.__const)
      return result
    if axis < 0: axis += self.__ndim
    if axis < 0 or axis >= self.__ndim: raise ValueError(f"axis {axis} is out of range for tensor with {self.__ndim} dimensions")
    result_buf = cpu.sum(self.__buf, axis) if self.__device.type == "CPU" else cuda.sum(self.__buf, axis)
    result = Tensor._from_backend(result_buf, self.__dtype, self.__device, self.__requires_grad, const=self.__const)
    if keepdim:
      new_shape = list(self.shape.shape)
      new_shape[axis] = 1
      result = result.reshape(tuple(new_shape))
    return result

  def flatten(self):
    out = cpu.topyobj(self.buf) if self.__device.type == "CPU" else cuda.topyobj(self.buf)
    return Tensor(out, dtype=self.__dtype, device=self.__device) # type: ignore

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
      if self.device.type == "CPU": return np.array(reshape(cpu.topyobj(self.buf), self.shape.shape)) # type: ignore
      return np.array(reshape(cuda.topyobj(self.buf), self.shape.shape)) # type: ignore
    except ImportError:
      import warnings
      warnings.warn("NumPy is required for Tensor.numpy(). Install it with: pip install numpy", RuntimeWarning)
      return None

  def exp(self) -> Tensor: return self.uop("exp", promote=True)
  def log(self) -> Tensor: return self.uop("log", promote=True)
  def log2(self) -> Tensor: return self.uop("log2", promote=True)
  def log10(self) -> Tensor: return self.uop("log10", promote=True)
  def sin(self) -> Tensor: return self.uop("sin", promote=True)
  def cos(self) -> Tensor: return self.uop("cos", promote=True)
  def tan(self) -> Tensor: return self.uop("tan", promote=True)
  def asin(self) -> Tensor: return self.uop("asin", promote=True)
  def acos(self) -> Tensor: return self.uop("acos", promote=True)
  def atan(self) -> Tensor: return self.uop("atan", promote=True)
  def sinh(self) -> Tensor: return self.uop("sinh", promote=True)
  def cosh(self) -> Tensor: return self.uop("cosh", promote=True)
  def tanh(self) -> Tensor: return self.uop("tanh", promote=True)
  def asinh(self) -> Tensor: return self.uop("asinh", promote=True)
  def acosh(self) -> Tensor: return self.uop("acosh", promote=True)
  def atanh(self) -> Tensor: return self.uop("atanh", promote=True)

  def sgn(self) -> Tensor: return self.uop("sgn")

