from typing import Union
from tinytensor.ops import UOp, BinOp

def topo_sort(node):
  visited = set()
  order = []
  def dfs(n):
    if n in visited: return
    visited.add(n)
    for s in n.srcs: dfs(s)
    order.append(n)
  dfs(node)
  return order

class Node:
  def __init__(self, *srcs, op:Union[UOp,BinOp,None]=None, value=None, device=None):
    self.srcs = srcs
    self.op = op
    self.value = value
    if value is not None:
      self.shape = value.shape.shape
      self.dtype = value.dtype
      self.device = value.device
    else:
      self.shape = self.srcs[0].shape
      self.dtype = self.srcs[0].dtype
      self.device = self.srcs[0].device

  def __repr__(self):
    if self.value is not None: return f"Node(value={self.value})"
    return f"Node(op={self.op},srcs={self.srcs})"

  def __add__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.ADD)

  def __radd__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(other, self, op=BinOp.ADD)

  def __sub__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.SUB)

  def __rsub__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(other, self, op=BinOp.SUB)

  def __mul__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.MUL)

  def __rmul__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(other, self, op=BinOp.MUL)

  def __truediv__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.TDIV)

  def __rtruediv__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(other, self, op=BinOp.TDIV)

  def __floordiv__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.FDIV)

  def __rfloordiv__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(other, self, op=BinOp.FDIV)

  def __mod__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.MOD)

  def __rmod__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(other, self, op=BinOp.MOD)

  def __pow__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.POW)

  def __rpow__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(other, self, op=BinOp.POW)

  def __and__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.AND)

  def __rand__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(other, self, op=BinOp.AND)

  def __or__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.OR)

  def __ror__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(other, self, op=BinOp.OR)

  def __invert__(self): return Node(self, op=UOp.NOT)

  def __xor__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.XOR)

  def __rxor__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(other, self, op=BinOp.XOR)

  def __lshift__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.LSHIFT)

  def __rlshift__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(other, self, op=BinOp.LSHIFT)

  def __rshift__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.RSHIFT)

  def __rrshift__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(other, self, op=BinOp.RSHIFT)

  def __eq__(self, other): # type: ignore
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.EQ)

  def __ne__(self, other): # type: ignore
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.NE)

  def __gt__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.GT)

  def __ge__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.GE)

  def __lt__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.LT)

  def __le__(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.LE)

  def __hash__(self): return id(self)

  def __abs__(self): return Node(self, op=UOp.ABS)
  def __neg__(self): return Node(self, op=UOp.NEG)
  def __pos__(self): return Node(self, op=UOp.POS)

  def abs(self): return Node(self, op=UOp.ABS)
  def neg(self): return Node(self, op=UOp.NEG)
  def pos(self): return Node(self, op=UOp.POS)
  def sin(self): return Node(self, op=UOp.SIN)
  def cos(self): return Node(self, op=UOp.COS)
  def tan(self): return Node(self, op=UOp.TAN)
  def asin(self): return Node(self, op=UOp.ASIN)
  def acos(self): return Node(self, op=UOp.ACOS)
  def atan(self): return Node(self, op=UOp.ATAN)
  def sinh(self): return Node(self, op=UOp.SINH)
  def cosh(self): return Node(self, op=UOp.COSH)
  def tanh(self): return Node(self, op=UOp.TANH)
  def asinh(self): return Node(self, op=UOp.ASINH)
  def acosh(self): return Node(self, op=UOp.ACOSH)
  def atanh(self): return Node(self, op=UOp.ATANH)
  def log(self): return Node(self, op=UOp.LOG)
  def log2(self): return Node(self, op=UOp.LOG2)
  def log10(self): return Node(self, op=UOp.LOG10)
  def exp(self): return Node(self, op=UOp.EXP)
  def exp2(self): return Node(self, op=UOp.EXP2)
  def floor(self): return Node(self, op=UOp.FLOOR)
  def ceil(self): return Node(self, op=UOp.CEIL)
  def sqrt(self): return Node(self, op=UOp.SQRT)
  def cbrt(self): return Node(self, op=UOp.CBRT)

  def logical_and(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.AND)

  def logical_or(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.OR)

  def logical_xor(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.XOR)

  def logical_not(self): return Node(self, op=UOp.NOT)

  def bitwise_and(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.AND)

  def bitwise_or(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.OR)

  def bitwise_xor(self, other):
    if not isinstance(other, Node): other = Node(value=other)
    return Node(self, other, op=BinOp.XOR)

  def bitwise_not(self): return Node(self, op=UOp.NOT)
