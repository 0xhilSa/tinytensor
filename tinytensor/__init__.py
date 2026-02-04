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
  const:bool=False # type: ignore (type interference)
): return Tensor(buf=buf, dtype=dtype, device=device, const=const)

from tinytensor.dtypes import (bool, int8, uint8, int16, uint16,
                               int32, uint32, int64, uint64,
                               float32, float64, longdouble,
                               complex64, complex128)

__all__ = [
  "cuda_available",
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
  "float32",
  "float64",
  "longdouble",
  "complex64",
  "complex128",
  "pi",
  "e",
  "euler_gamma",
  "inf",
  "nan",
]
