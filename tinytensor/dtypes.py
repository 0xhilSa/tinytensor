from dataclasses import dataclass
from typing import Literal, Final

Fmts = Literal[
    "?", "b", "B", "h", "H", "i", "I", "l", "L",
    "f", "d", "F", "D"
]
ConstType = int|float|complex|bool

@dataclass(frozen=True, eq=True)
class DType:
  fmt:Fmts
  ctype:str
  nbyte:int
  signed:bool|None
  @classmethod
  def new(cls, fmt:Fmts, ctype:str, nbyte:int, signed:bool|None): return DType(fmt, ctype, nbyte, signed)
  def __repr__(self): return f"<DType(ctype='{self.ctype}', fmt='{self.fmt}', nbyte={self.nbyte}, signed={self.signed})>"
  @property
  def nbit(self): return self.nbyte * 8
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
    elif ctype == "float": return float32
    elif ctype == "double": return float64
    elif ctype == "float _Complex": return complex64
    elif ctype == "double _Complex": return complex128
    else: raise RuntimeError("unexpected result occured")

bool:Final = DType("?", "bool", 1, False)
int8:Final = DType.new("b", "char", 1, True)
uint8:Final = DType.new("B", "unsigned char", 1, False)
int16:Final = DType.new("h", "short", 2, True)
uint16:Final = DType.new("H", "unsigned short", 2, False)
int32:Final = DType.new("i", "int", 4, True)
uint32:Final = DType.new("I", "unsigned int", 4, False)
int64:Final = DType.new("l", "long", 8, True)
uint64:Final = DType.new("L", "unsigned long", 8, False)
float32:Final = DType.new("f", "float", 4, None)
float64:Final = DType.new("d", "double", 8, None)
complex64:Final = DType.new("F", "float _Complex", 8, None)
complex128:Final = DType.new("D", "double _Complex", 16, None)

BOOLEAN = [bool]
INT = [int8, int16, int32, int64]
UINT = [uint8, uint16, uint32, uint64]
FLOAT = [float32, float64]
COMPLEX = [complex64, complex128]
ALL = BOOLEAN + INT + UINT + FLOAT + COMPLEX
