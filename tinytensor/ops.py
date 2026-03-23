from enum import Enum, auto

class BinOp(Enum):
  ADD=auto(); SUB=auto(); MUL=auto(); TDIV=auto(); FDIV=auto(); MOD=auto(); POW=auto()
  MATMUL=auto()
  EQ=auto(); NE=auto(); GT=auto(); GE=auto(); LT=auto(); LE=auto()
  AND=auto(); OR=auto(); XOR=auto()
  LSHIFT=auto(); RSHIFT=auto()

class UOp(Enum):
  NOT=auto()
  ABS=auto(); NEG=auto(); POS=auto()
  SIN=auto(); COS=auto(); TAN=auto()
  ASIN=auto(); ACOS=auto(); ATAN=auto()
  SINH=auto(); COSH=auto(); TANH=auto()
  ASINH=auto(); ACOSH=auto(); ATANH=auto()
  LOG=auto(); LOG2=auto(); LOG10=auto()
  EXP=auto(); EXP2=auto()
  INV=auto()
  FLOOR=auto(); CEIL=auto()
  SQRT=auto(); CBRT=auto()

