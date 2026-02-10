from typing import Tuple, Union

class Shape:
  def __init__(self, shape:Tuple[int,...]):
    self.shape = shape
    self.ndim = len(shape)
    self.size = 1
    for x in shape: self.size *= x
    self.stride = self._compute_stride()
  def _compute_stride(self) -> Tuple[int, ...]:
    if self.ndim == 0: return ()
    stride = [1] * self.ndim
    for i in range(self.ndim - 2, -1, -1): stride[i] = self.shape[i + 1] * stride[i + 1]
    return tuple(stride)
  def __repr__(self):
    return (
      f"Shape(shape={self.shape}, ndim={self.ndim}, "
      f"size={self.size}, stride={self.stride})"
    )
  def __iter__(self): return iter(self.shape)
  def __getitem__(self, index: Union[int, slice]):
    if isinstance(index, int):
      if not 0 <= index < self.ndim: raise IndexError("Index out of range")
      return self.shape[index]
    if isinstance(index, slice): return self.shape[index.start:index.stop:index.step]
  def __eq__(self, other):
    if not isinstance(other, Shape): raise TypeError("Shape type is required")
    return self.shape == other.shape
  def __ne__(self, other): return not self.__eq__(other)
