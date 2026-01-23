from __future__ import annotations
from typing import List, Optional, Union
from tinytensor.tensor import Tensor
from tinytensor.engine import cpu, cuda
from tinytensor import dtypes
from tinytensor.device import Device


def cuda_available(): return hasattr(cuda, "device_count") and cuda.device_count() > 0

def tensor(
  buf:List,
  dtype:Optional[Union[dtypes.DType,dtypes.ConstType]]=None,
  device:str="cpu",
  const:bool=False
): return Tensor(buf=buf, dtype=dtype, device=device, const=const)

__all__ = [
  "cuda_available",
  "tensor",
  "Tensor",
  "Device",
  "dtypes",
  "cpu",
  "cuda"
]
