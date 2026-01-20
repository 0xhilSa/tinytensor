from __future__ import annotations
from typing import List, Optional, Union
from tinytensor import dtypes
from tinytensor.shape import Shape
from tinytensor.engine import cpu, cuda
from tinytensor.helpers import infer_dtype, infer_shape, flatten, reshape
from tinytensor.device import Device


class Tensor:
  def __init__(
    self,
    buf:Union[List,dtypes.ConstType],
    dtype:Optional[Union[dtypes.DType,dtypes.ConstType]]=None,
    device:Union[str,Device]="cpu",
    const:bool=False
  ):
    if isinstance(buf, dtypes.ConstType): buf = [buf]
    self.__shape = Shape(infer_shape(buf))
    buf = flatten(buf)
    buf, self.__dtype = infer_dtype(buf, dtype)
    self.__device = Device(device) if not isinstance(device, Device) else device
    self.__const = const
    if self.__device.type == "CPU": self.__buf = cpu.tocpu(buf, self.__shape.shape, self.__dtype.fmt)
    elif self.__device.type == "CUDA": self.__buf = cuda.tocuda(buf, self.__shape.shape, self.__dtype.fmt, self.__device.index)
  def __repr__(self): return f"Tensor(shape={self.__shape.shape}, dtype='{self.__dtype.ctype}', device='{self.__device.type}:{self.__device.index}', const={self.__const})"
  def cpu(self): return Tensor(reshape(cpu.tolist(cuda.tocpu(self.__buf)), self.__shape.shape), dtype=self.__dtype) if self.__device.type == "CUDA" else self
  def cuda(self, index:int=0): return Tensor(reshape(cpu.tolist(self.__buf), self.__shape.shape), dtype=self.__dtype, device=f"cuda:{index}") if self.__device.type == "CPU" else self
  def tolist(self): return reshape(cpu.tolist(self.__buf), self.__shape.shape) if self.__device.is_cpu() else reshape(cpu.tolist(cuda.tocpu(self.__buf)), self.__shape.shape)
  def dev(self, device:Union[str,Device]): return Tensor(self.tolist(), device="cpu:0") if self.__device.type == "cuda" else Tensor(self.tolist(), device=device)
  def typecast(self, dtype:Union[dtypes.DType,dtypes.ConstType]): return Tensor(cpu.tolist(self.__buf), dtype=dtype) if dtype != self.__dtype else self
  def __add__(self, other:Union[Tensor,dtypes.ConstType]): # prototype
    if isinstance(other, dtypes.ConstType): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    elif isinstance(other, Tensor):
      assert other.device == self.__device, "addition cannot be perform on tensors with different devices"
      assert other.dtype == self.__dtype, "addition cannot be perform on tensors with different dtypes"
    if isinstance(other, dtypes.ConstType): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.__device.is_cpu(): return Tensor(reshape(cpu.tolist(cpu.add(self.__buf, other.buf)), self.__shape.shape), dtype=self.__dtype)
    raise NotImplementedError("__add__ not implemented for tensors on CUDA device")
  def __radd__(self, other:Union[Tensor,dtypes.ConstType]): # prototype
    if isinstance(other, dtypes.ConstType): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    elif isinstance(other, Tensor):
      assert other.device == self.__device, "addition cannot be perform on tensors with different devices"
      assert other.dtype == self.__dtype, "addition cannot be perform on tensors with different dtypes"
    if self.__device.is_cpu(): return Tensor(reshape(cpu.tolist(cpu.add(self.__buf, other.buf)), self.__shape.shape), dtype=self.__dtype)
    raise NotImplementedError("__add__ not implemented for tensors on CUDA device")
  def __sub__(self, other:Union[Tensor,dtypes.ConstType]): # prototype
    if isinstance(other, dtypes.ConstType): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    elif isinstance(other, Tensor):
      assert other.device == self.__device, "addition cannot be perform on tensors with different devices"
      assert other.dtype == self.__dtype, "addition cannot be perform on tensors with different dtypes"
    if isinstance(other, dtypes.ConstType): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    if self.__device.is_cpu(): return Tensor(reshape(cpu.tolist(cpu.sub(self.__buf, other.buf)), self.__shape.shape), dtype=self.__dtype)
    raise NotImplementedError("__add__ not implemented for tensors on CUDA device")
  def __rsub__(self, other:Union[Tensor,dtypes.ConstType]): # prototype
    if isinstance(other, dtypes.ConstType): other = Tensor(other, dtype=self.__dtype, device=f"{self.__device.type}:{self.__device.index}")
    elif isinstance(other, Tensor):
      assert other.device == self.__device, "addition cannot be perform on tensors with different devices"
      assert other.dtype == self.__dtype, "addition cannot be perform on tensors with different dtypes"
    if self.__device.is_cpu(): return Tensor(reshape(cpu.tolist(cpu.sub(self.__buf, other.buf)), self.__shape.shape), dtype=self.__dtype)
    raise NotImplementedError("__add__ not implemented for tensors on CUDA device")
  @property
  def buf(self): return self.__buf
  @property
  def device(self): return self.__device
  @property
  def dtype(self): return self.__dtype


def tensor(
  buf:List,
  dtype:Optional[Union[dtypes.DType,dtypes.ConstType]]=None,
  device:str="cpu",
  const:bool=False
): return Tensor(buf=buf, dtype=dtype, device=device, const=const)
