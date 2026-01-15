from tinytensor import dtypes
from typing import List, Union, Optional

_CAST = {
  dtypes.bool_: bool,
  dtypes.int8: int,
  dtypes.int16: int,
  dtypes.int32: int,
  dtypes.int64: int,
  dtypes.uint8: int,
  dtypes.uint16: int,
  dtypes.uint32: int,
  dtypes.uint64: int,
  dtypes.float32: float,
  dtypes.float64: float,
  dtypes.float128: float,
  dtypes.complex64: complex,
  dtypes.complex128: complex,
  dtypes.complex256: complex,
}

def infer_dtype(
  buf: List,
  dtype: Optional[Union[dtypes.ConstType,dtypes.DType]] = None
):
  if dtype is None:
    if any(isinstance(x, complex) for x in buf): dtype = dtypes.complex128
    elif any(isinstance(x, float) for x in buf): dtype = dtypes.float64
    elif all(isinstance(x, bool) for x in buf): dtype = dtypes.bool_
    elif all(isinstance(x, int) for x in buf): dtype = dtypes.int64
    else: raise TypeError("Unsupported buf types")
  if dtype is int: dtype = dtypes.int64
  elif dtype is float: dtype = dtypes.float64
  elif dtype is complex: dtype = dtypes.complex128
  elif dtype is bool: dtype = dtypes.bool_
  if not isinstance(dtype, dtypes.DType): raise TypeError("dtype must be a DType or a Python scalar type")
  caster = _CAST.get(dtype)
  if caster is None: raise TypeError(f"No caster defined for dtype {dtype}")
  return [caster(x) for x in buf], dtype

def flatten(buf:list|dtypes.ConstType):
  if not isinstance(buf, list): return [buf]
  flat = []
  for x in buf: flat.extend(flatten(x))
  return flat

def infer_shape(lst) -> tuple:
  if not isinstance(lst, list):return ()
  if len(lst) == 0:return (0,)
  return (len(lst),) + infer_shape(lst[0])

def has_uniform_shape(lst):
  if not isinstance(lst, list): return True
  lengths = [len(x) if isinstance(x, list) else -1 for x in lst]
  if len(set(lengths)) != 1: return False
  return all(has_uniform_shape(x) for x in lst)

