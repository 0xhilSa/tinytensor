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
    const:bool=False
  ):
    self.__shape = Shape(shape_of(buf))
    self.__ndim = self.__shape.ndim
    buf = flatten(buf)
    buf, self.__dtype = dtype_of(buf, dtype)
    self.__device = Device(device) if not isinstance(device, Device) else device
    self.__const = const
    if self.__device.type == "CPU": self.__buf = cpu.tocpu(buf, self.__shape.shape, self.__dtype.fmt)
    elif self.__device.type == "CUDA": self.__buf = cuda.tocuda(buf, self.__shape.shape, self.__dtype.fmt, self.__device.index)
  def __repr__(self): return f"Tensor(shape={self.__shape}, dtype='{self.__dtype.ctype}', device={self.__device}), const={self.__const})"
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
  def buf(self): return self.__buf
  def is_const(self): return self.__const
  def cuda(self, device_index:int=0): return Tensor(cpu.topyobj(self.__buf), dtype=self.__dtype, device=f"cuda:{device_index}") if self.__device.type == "CPU" else self # type: ignore
  def cpu(self): return Tensor(cuda.topyobj(self.__buf), dtype=self.__dtype, device=f"cpu") if self.__device.type == "CUDA" else self # type: ignore
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
  def typecast(self, dtype:Union[dtypes.DType,dtypes.ConstType]):
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
  def __add__(self, other:Union[dtypes.ConstType,Tensor]):
    if isinstance(other, dtypes.ConstType): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    x, y = Tensor.broadcast(self, other)
    out = cpu.topyobj(cpu.add(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.add(x.buf, y.buf))
    return Tensor(reshape(out, self.__shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore
  def __radd__(self, other:Union[dtypes.ConstType,Tensor]):
    if isinstance(other, dtypes.ConstType): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    x, y = Tensor.broadcast(self, other)
    out = cpu.topyobj(cpu.add(x.buf, y.buf)) if self.__device.type == "CPU" else cuda.topyobj(cuda.add(x.buf, y.buf))
    return Tensor(reshape(out, self.__shape.shape), dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}") # type: ignore
