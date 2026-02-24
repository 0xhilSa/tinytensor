from dataclasses import dataclass
from typing import Literal, Final

Fmts = Literal[
    "?", "b", "B", "h", "H", "i", "I", "l", "L",
    "e", "f", "d", "F", "D"
]
ConstType = int|float|complex|bool

@dataclass(frozen=True, eq=True)
class DType:
  fmt:Fmts
  ctype:str
  nbyte:int
  signed:bool|None
  kind_:str
  @classmethod
  def new(cls, fmt:Fmts, ctype:str, nbyte:int, signed:bool|None, kind:str): return DType(fmt, ctype, nbyte, signed, kind)
  def __repr__(self): return f"<DType(ctype='{self.ctype}', fmt='{self.fmt}', nbyte={self.nbyte}, signed={self.signed}, kind={self.kind_})>"
  @property
  def nbit(self): return self.nbyte * 8
  @property
  def kind(self):
    if self in BOOLEAN: return "bool"
    if self in INT: return "int"
    if self in UINT: return "uint"
    if self in FLOAT: return "float"
    if self in COMPLEX: return "complex"
    raise RuntimeError("Unknown dtype")
  @staticmethod
  def from_ctype(ctype:str):
    if ctype == "bool": return bool
    elif ctype == "char": return int8
    elif ctype == "unsigned char": return uint8
    elif ctype == "short": return int16
    elif ctype == "unsigned short": return uint16
    elif ctype == "int": return int32
    elif ctype == "unsigned int": return uint32
    elif ctype == "long": return int64
    elif ctype == "unsigned long": return uint64
    elif ctype == "half": return float16
    elif ctype == "float": return float32
    elif ctype == "double": return float64
    elif ctype == "float _Complex": return complex64
    elif ctype == "double _Complex": return complex128
    else: raise RuntimeError("unexpected result occured")
  def can_cast(self, dst: "DType") -> bool:
    if self is dst: return True
    sk = self.kind
    dk = dst.kind
    if sk == "bool": return True
    if sk in ("int", "uint") and dk in ("int", "uint"): return True
    if sk in ("int", "uint") and dk == "float": return True
    if sk == "float" and dk in ("int", "uint"): return True
    if sk == "float" and dk == "float": return True
    if sk == "complex" and dk == "complex": return True
    if sk == "float" and dk == "complex": return True
    return False
  def is_safe_cast(self, dst: "DType") -> bool:
    if not self.can_cast(dst): return False
    if self.kind == dst.kind: return self.nbit <= dst.nbit
    if self.kind in ("int", "uint") and dst.kind == "float": return True
    if self.kind == "float" and dst.kind == "complex": return True
    return False

bool:Final = DType("?", "bool", 1, False, "bool")
int8:Final = DType.new("b", "char", 1, True, "int")
uint8:Final = DType.new("B", "unsigned char", 1, False, "uint")
int16:Final = DType.new("h", "short", 2, True, "int")
uint16:Final = DType.new("H", "unsigned short", 2, False, "uint")
int32:Final = DType.new("i", "int", 4, True, "int")
uint32:Final = DType.new("I", "unsigned int", 4, False, "uint")
int64:Final = DType.new("l", "long", 8, True, "int")
uint64:Final = DType.new("L", "unsigned long", 8, False, "uint")
float16:Final = DType.new("e", "half", 2, None, "float")
float32:Final = DType.new("f", "float", 4, None, "float")
float64:Final = DType.new("d", "double", 8, None, "float")
complex64:Final = DType.new("F", "float _Complex", 8, None, "complex")
complex128:Final = DType.new("D", "double _Complex", 16, None, "complex")

BOOLEAN = [bool]
INT = [int8, int16, int32, int64]
UINT = [uint8, uint16, uint32, uint64]
FLOAT = [float16, float32, float64]
COMPLEX = [complex64, complex128]
ALL = BOOLEAN + INT + UINT + FLOAT + COMPLEX
