from dataclasses import dataclass
from typing import Literal, Final

Fmts = Literal[
    "?", "b", "B", "h", "H", "i", "I", "l", "L",
    "f", "d", "g", "F", "D", "G"
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
float128:Final = DType.new("g", "long double", 16, None)
complex64:Final = DType.new("F", "float _Complex", 8, None)
complex128:Final = DType.new("D", "double _Complex", 16, None)
complex256:Final = DType.new("G", "long double _Complex", 32, None)

BOOLEAN = [bool]
INT = [int8, int16, int32, int64]
UINT = [uint8, uint16, uint32, uint64]
FLOAT = [float32, float64, float128]
COMPLEX = [complex64, complex128, complex256]
ALL = BOOLEAN + INT + UINT + FLOAT + COMPLEX
