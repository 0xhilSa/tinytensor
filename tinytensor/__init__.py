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
    device:str="cpu",
    const:bool=False
  ):
    if isinstance(buf, dtypes.ConstType): buf = [buf]
    self.shape = Shape(infer_shape(buf))
    buf = flatten(buf)
    buf, self.dtype = infer_dtype(buf, dtype)
    self.device = Device(device)
    self.const = const
    if self.device.type == "CPU": self.buf = cpu.tocpu(buf, self.shape.shape, self.dtype.fmt)
    elif self.device.type == "CUDA": self.buf =cuda.tocuda(buf, self.shape.shape, self.dtype.fmt, self.device.index)
  def __repr__(self): return f"Tensor(shape={self.shape.shape}, dtype='{self.dtype.ctype}', device='{self.device.type}:{self.device.index}', const={self.const})"
  def cpu(self): return Tensor(reshape(cpu.tolist(cuda.tocpu(self.buf)), self.shape.shape), dtype=self.dtype) if self.device.type == "CUDA" else self
  def cuda(self): return Tensor(reshape(cpu.tolist(self.buf), self.shape.shape), dtype=self.dtype, device="cuda:0") if self.device.type == "CPU" else self
  def tolist(self): return reshape(cpu.tolist(self.buf), self.shape.shape) if self.device.is_cpu() else reshape(cpu.tolist(cuda.tocpu(self.buf)), self.shape.shape)
  def __add__(self, other:Union[Tensor,dtypes.ConstType]): # prototype
    if isinstance(other, dtypes.ConstType): other = Tensor(other, dtype=self.dtype, device=f"{self.device.type}:{self.device.index}")
    elif isinstance(other, Tensor):
      assert other.device == self.device, "addition cannot be perform on tensors with different devices"
      assert other.dtype == self.dtype, "addition cannot be perform on tensors with different dtypes"
    if isinstance(other, dtypes.ConstType): other = Tensor(other, dtype=self.dtype, device=f"{self.device.type}:{self.device.index}")
    if self.device.is_cpu(): return Tensor(reshape(cpu.tolist(cpu.add(self.buf, other.buf)), self.shape.shape), dtype=self.dtype)
    raise NotImplementedError("__add__ not implemented for tensors on CUDA device")
  def __radd__(self, other:Union[Tensor,dtypes.ConstType]): # prototype
    if isinstance(other, dtypes.ConstType): other = Tensor(other, dtype=self.dtype, device=f"{self.device.type}:{self.device.index}")
    elif isinstance(other, Tensor):
      assert other.device == self.device, "addition cannot be perform on tensors with different devices"
      assert other.dtype == self.dtype, "addition cannot be perform on tensors with different dtypes"
    if self.device.is_cpu(): return Tensor(reshape(cpu.tolist(cpu.add(self.buf, other.buf)), self.shape.shape), dtype=self.dtype)
    raise NotImplementedError("__add__ not implemented for tensors on CUDA device")


def tensor(
  buf:List,
  dtype:Optional[Union[dtypes.DType,dtypes.ConstType]]=None,
  device:str="cpu",
  const:bool=False
): return Tensor(buf=buf, dtype=dtype, device=device, const=const)
