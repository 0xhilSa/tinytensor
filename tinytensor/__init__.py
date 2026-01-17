import ctypes
from typing import List, Optional, Union
from tinytensor import dtypes
from tinytensor.shape import Shape
from tinytensor.engine import cpu, gpu_cuda
from tinytensor.helpers import infer_dtype, infer_shape, flatten, reshape
from tinytensor.device import Device


class Tensor:
  def __init__(
    self,
    buf:Union[List,ctypes.c_void_p],
    dtype:Optional[Union[dtypes.DType,dtypes.ConstType]]=None,
    device:str="cpu",
    const:bool=False
  ):
    if isinstance(buf,List):
      self.shape = Shape(infer_shape(buf))
      buf = flatten(buf)
      buf, self.dtype = infer_dtype(buf, dtype)
      self.device = Device(device)
      self.const = const
      if self.device.type == "CPU": self.buf = cpu.tocpu(buf, self.shape.shape, self.dtype.fmt)
      elif self.device.type == "CUDA": self.buf = gpu_cuda.tocuda(buf, self.shape.shape, self.dtype.fmt)
    else: raise NotImplementedError("create tensor from the pointer has not been implemented yet!") # not planned yet
  def __repr__(self): return f"Tensor(shape={self.shape.shape}, dtype='{self.dtype.ctype}', device='{self.device.type}:{self.device.index}', const={self.const})"
  def cpu(self): return Tensor(reshape(cpu.tolist(gpu_cuda.tocpu(self.buf)), self.shape.shape), dtype=self.dtype) if self.device.type == "CUDA" else self
  def cuda(self): return Tensor(reshape(cpu.tolist(self.buf), self.shape.shape), dtype=self.dtype, device="cuda:0") if self.device.type == "CPU" else self


def tensor(
  buf:List,
  dtype:Optional[Union[dtypes.DType,dtypes.ConstType]]=None,
  device:str="cpu",
  const:bool=False
): return Tensor(buf=buf, dtype=dtype, device=device, const=const)
