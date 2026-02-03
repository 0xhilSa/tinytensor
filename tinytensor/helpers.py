from typing import List, Tuple, Union, Optional, Any
from tinytensor import dtypes

_CAST = {
  dtypes.bool: bool,
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
  dtypes.longdouble: float,
  dtypes.complex64: complex,
  dtypes.complex128: complex,
}

from typing import Union, Optional, List

def dtype_of(
    buf: Union[List, dtypes.ConstType],
    dtype: Optional[Union[dtypes.ConstType, dtypes.DType]] = None
):
  is_scalar = isinstance(buf, (int, float, complex, bool))
  if is_scalar: buf = [buf]
  elif not isinstance(buf, list): raise TypeError("buf must be a scalar or a list of scalars")

  if dtype is None:
    if not buf: raise TypeError("Cannot infer dtype from empty buffer")
    if any(isinstance(x, complex) for x in buf): dtype = dtypes.complex128
    elif any(isinstance(x, float) for x in buf): dtype = dtypes.float64
    elif all(isinstance(x, bool) for x in buf): dtype = dtypes.bool
    elif all(isinstance(x, int) for x in buf): dtype = dtypes.int64
    else: raise TypeError("Unsupported buffer element types")
  if dtype is int: dtype = dtypes.int64
  elif dtype is float: dtype = dtypes.float64
  elif dtype is complex: dtype = dtypes.complex128
  elif dtype is bool: dtype = dtypes.bool
  if not isinstance(dtype, dtypes.DType): raise TypeError("dtype must be a DType or Python scalar type")
  caster = _CAST.get(dtype)
  if caster is None: raise TypeError(f"No caster defined for dtype {dtype}")
  out = [caster(x) for x in buf]
  if is_scalar: return out[0], dtype
  return out, dtype

def flatten(buf:list|dtypes.ConstType):
  if not isinstance(buf, list): return [buf]
  flat = []
  for x in buf: flat.extend(flatten(x))
  return flat

def shape_of(x):
  if not isinstance(x, list): return ()
  return (len(x),) + shape_of(x[0])

def has_uniform_shape(lst):
  if not isinstance(lst, list): return True
  lengths = [len(x) if isinstance(x, list) else -1 for x in lst]
  if len(set(lengths)) != 1: return False
  return all(has_uniform_shape(x) for x in lst)

def reshape(data: List[Any], shape: Tuple[int, ...]) -> List[Any]:
  if not shape: return data
  total = 1
  for d in shape: total *= d
  if total != len(data): raise ValueError("shape does not match data length")
  def _reshape(flat, shp):
    if len(shp) == 1: return flat[:shp[0]]
    step = 1
    for d in shp[1:]: step *= d
    return [
      _reshape(flat[i * step:(i + 1) * step], shp[1:])
      for i in range(shp[0])
    ]
  return _reshape(data, shape)
