from tinytensor import dtypes
from tinytensor.engine.cpu import functional_cpu as F_c
from tinytensor.engine.cuda import functional_cuda as F_cu
from tinytensor.tensor import Tensor

def relu(x:Tensor, requires_grad:bool=False, const:bool=False) -> Tensor:
  out = F_c.relu(x.buf) if x.device.type == "CPU" else F_cu.relu(x.buf)
  return Tensor._from_view(out, dtype=x.dtype, device=x.device, requires_grad=requires_grad, const=const)

def gelu(x:Tensor, requires_grad:bool=False, const:bool=False) -> Tensor:
  out = F_c.gelu(x.buf) if x.device.type == "CPU" else F_cu.gelu(x.buf)
  dtype = dtypes.float32 if x.dtype == dtypes.float32 else dtypes.float64
  return Tensor._from_view(out, dtype=dtype, device=x.device, requires_grad=requires_grad, const=const)

def leaky_relu(x:Tensor, negative_slope:float=0.01, requires_grad:bool=False, const:bool=False) -> Tensor:
  out = F_c.leaky_relu(x.buf, negative_slope) if x.device.type == "CPU" else F_cu.leaky_relu(x.buf, negative_slope)
  dtype = dtypes.float32 if x.dtype == dtypes.float32 else dtypes.float64
  return Tensor._from_view(out, dtype=dtype, device=x.device, requires_grad=requires_grad, const=const)

__all__ = [
  "relu",
  "gelu",
  "leaky_relu"
]
