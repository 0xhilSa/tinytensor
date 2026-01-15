from typing import List, Optional, Union
from tinytensor import dtypes
from tinytensor.engine import cpu, gpu_cuda
from tinytensor.helpers import infer_dtype, infer_shape, flatten
from tinytensor.device import Device
from tinytensor.shape import Shape


class Tensor:
  def __init__(
    self,
    buf:List,
    dtype:Optional[Union[dtypes.DType,dtypes.ConstType]]=None,
    device:str="cpu",
    const:bool=False
  ):
    self.shape = Shape(infer_shape(buf))
    buf = flatten(buf)
    buf, self.dtype = infer_dtype(buf, dtype)
    self.device = Device(device)
    self.const = const
    if self.device.type == "CPU": self.buf = cpu.tocpu(buf, self.shape.shape, self.dtype.fmt)
    elif self.device.type == "CUDA": self.buf = gpu_cuda.tocuda(buf, self.shape.shape, self.dtype.fmt)
  def __repr__(self): return f"Tensor(shape={self.shape.shape}, dtype='{self.dtype.ctype}', device='{self.device.type}:{self.device.index}', const={self.const})"
