from typing import Tuple

class Function:
  def __init__(self, *parents):
    self.parents:Tuple["Tensor",...] = parents # type: ignore
    self.saved_tensors:Tuple["Tensor",...] = () # type: ignore
  def save_for_backward(self, *tensors:Tuple["Tensor",...]): self.saved_tensors = tensors # type: ignore
  def backward(self, grad_out): raise NotImplementedError

class AddBackward(Function):
  def backward(self, grad_out):
    return grad_out, grad_out

class SubBackward(Function):
  def backward(self, grad_out):
    return grad_out, -grad_out

class MulBackward(Function):
  def __init__(self, x, y):
    super().__init__(x,y)
    self.save_for_backward(x,y)
  def backward(self, grad_out):
    x, y = self.saved_tensors
    return grad_out * y, grad_out * x

class TDivBackward(Function):
  def __init__(self, x, y):
    super().__init__(x,y)
    self.save_for_backward(x,y)
  def backward(self, grad_out):
    x, y = self.saved_tensors
    return grad_out / y, -grad_out * x / (y * y)

class PowBackward(Function):
  def __init__(self, x, y):
    super().__init__(x,y)
    self.save_for_backward(x,y)
  def backward(self, grad_out):
    x, y = self.saved_tensors
    pow_xy = x ** y
    dx = grad_out * y * (x ** (y - 1))
    dy = grad_out * pow_xy * x.log()
    return dx, dy

class ExpBackward(Function):
  def __init__(self, x, out):
    super().__init__(x)
    self.save_for_backward(out)
  def backward(self, grad_out):
    (out,) = self.saved_tensors
    return grad_out * out

class LogBackward(Function):
  def __init__(self, x):
    super().__init__(x)
    self.save_for_backward(x)
  def backward(self, grad_out):
    (x,) = self.saved_tensors
    return grad_out / x

class Log2Backward(Function):
  def __init__(self, x):
    super().__init__(x)
    self.save_for_backward(x)
  def backward(self, grad_out):
    (x,) = self.saved_tensors
    log2 = x.__class__(2).log()
    return grad_out / (x * log2)

class Log10Backward(Function):
  def __init__(self, x):
    super().__init__(x)
    self.save_for_backward(x)
  def backward(self, grad_out):
    (x,) = self.saved_tensors
    log10 = x.__class__(10).log()
    return grad_out / (x * log10)

class SinBackward(Function):
  def __init__(self, x):
    super().__init__(x)
    self.save_for_backward(x)
  def backward(self, grad_out):
    (x,) = self.saved_tensors
    return grad_out * x.cos()

class CosBackward(Function):
  def __init__(self, x):
    super().__init__(x)
    self.save_for_backward(x)
  def backward(self, grad_out):
    (x,) = self.saved_tensors
    return -grad_out * x.sin()

class TanBackward(Function):
  def __init__(self, x):
    super().__init__(x)
    self.save_for_backward(x)
  def backward(self, grad_out):
    (x,) = self.saved_tensors
    cos = x.cos()
    return grad_out / (cos * cos)

class ASinBackward(Function):
  def __init__(self, x):
    super().__init__(x)
    self.save_for_backward(x)
  def backward(self, grad_out):
    (x,) = self.saved_tensors
    return grad_out / (1 - x * x).sqrt()

class ACosBackward(Function):
  def __init__(self, x):
    super().__init__(x)
    self.save_for_backward(x)
  def backward(self, grad_out):
    (x,) = self.saved_tensors
    return -grad_out / (1 - x * x).sqrt()

class ATanBackward(Function):
  def __init__(self, x):
    super().__init__(x)
    self.save_for_backward(x)
  def backward(self, grad_out):
    (x,) = self.saved_tensors
    return grad_out / (1 + x * x)

class SinhBackward(Function):
  def __init__(self, x):
    super().__init__(x)
    self.save_for_backward(x)
  def backward(self, grad_out):
    (x,) = self.saved_tensors
    return grad_out * x.cosh()

class CoshBackward(Function):
  def __init__(self, x):
    super().__init__(x)
    self.save_for_backward(x)
  def backward(self, grad_out):
    (x,) = self.saved_tensors
    return grad_out * x.sinh()

class TanhBackward(Function):
  def __init__(self, x):
    super().__init__(x)
    self.save_for_backward(x)
  def backward(self, grad_out):
    (x,) = self.saved_tensors
    t = x.tanh()
    return grad_out * (1 - t * t)

class ASinhBackward(Function):
  def __init__(self, x):
    super().__init__(x)
    self.save_for_backward(x)
  def backward(self, grad_out):
    (x,) = self.saved_tensors
    return grad_out / (1 + x * x).sqrt()

class ACoshBackward(Function):
  def __init__(self, x):
    super().__init__(x)
    self.save_for_backward(x)
  def backward(self, grad_out):
    (x,) = self.saved_tensors
    return grad_out / ((x - 1).sqrt() * (x + 1).sqrt())

class ATanhBackward(Function):
  def __init__(self, x):
    super().__init__(x)
    self.save_for_backward(x)
  def backward(self, grad_out):
    (x,) = self.saved_tensors
    return grad_out / (1 - x * x)

class SqrtBackward(Function):
  def __init__(self, x):
    super().__init__(x)
    self.save_for_backward(x)
  def backward(self, grad_out):
    (x,) = self.saved_tensors
    return grad_out / (2 * x.sqrt())

class CbrtBackward(Function):
  def __init__(self, x):
    super().__init__(x)
    self.save_for_backward(x)
  def backward(self, grad_out):
    (x,) = self.saved_tensors
    c = x.cbrt()
    return grad_out / (3 * c * c)

