from tinytensor.engine.cpu import functional_cpu as F_c
from tinytensor.tensor import Tensor

def relu(x:Tensor, requires_grad:bool=False, const:bool=False) -> Tensor:
  out = F_c.relu(x.buf)
  return Tensor._from_view(out, dtype=x.dtype, device=x.device, requires_grad=requires_grad, const=const)

def gelu(x:Tensor, requires_grad:bool=False, const:bool=False) -> Tensor:
  out = F_c.gelu(x.buf)
  return Tensor._from_view(out, dtype=x.dtype, device=x.device, requires_grad=requires_grad, const=const)

def leaky_relu(x:Tensor, alpha:float=0.01, requires_grad:bool=False, const:bool=False) -> Tensor:
  out = F_c.leaky_relu(x.buf, alpha)
  return Tensor._from_view(out, dtype=x.dtype, device=x.device, requires_grad=requires_grad, const=const)

__all__ = [
  "relu",
  "gelu",
  "leaky_relu"
]
