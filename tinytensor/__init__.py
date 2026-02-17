from typing import List, Optional, Union
from tinytensor import dtypes
from tinytensor.tensor import Tensor
from tinytensor.engine import cpu, cuda
from tinytensor.engine.constants import pi, e, euler_gamma, inf, nan
from tinytensor.device import Device


def cuda_available(): return hasattr(cuda, "device_count") and cuda.device_count() > 0

def tensor(
  buf:Union[List[dtypes.ConstType],dtypes.ConstType],
  dtype:Optional[Union[dtypes.DType,dtypes.ConstType]]=None,
  device:str="cpu",
  requires_grad:bool=False, # type: ignore (type interference)
  const:bool=False # type: ignore (type interference)
): return Tensor(buf=buf, dtype=dtype, device=device, requires_grad=requires_grad, const=const)

from tinytensor.dtypes import (bool, int8, uint8, int16, uint16,
                               int32, uint32, int64, uint64,
                               float16, float32, float64,
                               complex64, complex128)

def arange(
  start:int=0, stop:int=0, step:int=1,
  dtype:Union[dtypes.ConstType,dtypes.DType]=int64,
  device:Union[str,Device]="cpu",
  requires_grad:bool=False, # type: ignore
  const:bool=False # type: ignore
):
  if step == 0: raise ValueError("step parameter cannot be 0")
  return Tensor(list(range(start, stop, step)), dtype=dtype, device=device, requires_grad=requires_grad, const=const)

__all__ = [
  "cuda_available",
  "arange",
  "tensor",
  "Tensor",
  "Device",
  "dtypes",
  "cpu",
  "cuda",
  "bool",
  "int8",
  "uint8",
  "int16",
  "uint16",
  "int32",
  "uint32",
  "int64",
  "uint64",
  "float16",
  "float32",
  "float64",
  "complex64",
  "complex128",
  "pi",
  "e",
  "euler_gamma",
  "inf",
  "nan",
]
