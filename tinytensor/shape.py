from typing import Tuple, Union

class Shape:
  def __init__(self, shape:Tuple[int,...]):
    if not shape: raise RuntimeError(f"shape can't be empty")
    self.shape = shape
    self.ndim = len(shape)
    self.size = 1
    for x in shape: self.size *= x
  def __repr__(self): return f"Shape(shape={self.shape}, ndim={self.ndim}, size={self.size})"
  def __iter__(self): return iter(self.shape)
  def __getitem__(self, index:Union[int,slice]):
    if isinstance(index, int):
      if not 0 <= index < self.ndim: raise IndexError(f"Index out of range")
      return self.shape[index]
    if isinstance(index, slice):
      return self.shape[index.start:index.stop:index.step]
  def __eq__(self, other):
    if not isinstance(other, Shape): raise TypeError(f"Shape dtype is required")
    return self.shape == other.shape
  def __ne__(self, other): return not self.__eq__(other)
