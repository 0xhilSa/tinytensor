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
  dtypes.float16: float,
  dtypes.float32: float,
  dtypes.float64: float,
  dtypes.complex64: complex,
  dtypes.complex128: complex,
}

def dtype_of(buf, dtype=None):
  is_scalar = not isinstance(buf, list)
  if is_scalar: buf = [buf]
  elif not isinstance(buf, list): raise TypeError("buf must be a scalar or a list")
  if dtype is None:
    if not buf: raise TypeError("Cannot infer dtype from empty buffer")
    inferred = None
    for x in buf:
      if isinstance(x, complex):
        inferred = dtypes.complex64
        break
      elif isinstance(x, float): inferred = dtypes.float32
      elif isinstance(x, bool):
        if inferred is None: inferred = dtypes.bool
      elif isinstance(x, int):
        if inferred not in (dtypes.float32, dtypes.complex64): inferred = dtypes.int32
      else: raise TypeError("Unsupported buffer element types")
    dtype = inferred
  if dtype in (int, float, complex, bool):
    dtype = {
      int: dtypes.int32,
      float: dtypes.float32,
      complex: dtypes.complex64,
      bool: dtypes.bool
    }[dtype]
  if not isinstance(dtype, dtypes.DType): raise TypeError("dtype must be DType")
  caster = _CAST[dtype]
  out = [caster(x) for x in buf]
  return (out[0], dtype) if is_scalar else (out, dtype)

def flatten(buf):
  if not isinstance(buf, list): return [buf]
  flat = []
  stack = [buf]
  while stack:
    curr = stack.pop()
    if isinstance(curr, list): stack.extend(reversed(curr))
    else: flat.append(curr)
  return flat

def shape_of(x):
  shape = []
  while isinstance(x, list):
    length = len(x)
    shape.append(length)
    if length == 0: break
    first = x[0]
    for item in x:
      if isinstance(item, list):
        if len(item) != len(first): raise ValueError("Non-uniform shape")
      elif isinstance(first, list): raise ValueError("Non-uniform shape")
    x = first
  return tuple(shape)

def has_uniform_shape(lst):
  if not isinstance(lst, list): return True
  if not lst: return True
  first = lst[0]
  expected = len(first) if isinstance(first, list) else None
  for item in lst:
    if isinstance(item, list):
      if expected is None or len(item) != expected: return False
      if not has_uniform_shape(item): return False
    elif expected is not None: return False
  return True

def reshape(data, shape):
  if not shape: return data
  total = 1
  for d in shape: total *= d
  if not isinstance(data, (list, tuple)):
    if total == 1: data = [data]
    else: raise TypeError("non-iterable data")
  if len(data) != total: raise ValueError("shape mismatch")
  idx = 0
  def build(shp):
    nonlocal idx
    if len(shp) == 1:
      res = data[idx:idx + shp[0]]
      idx += shp[0]
      return res
    return [build(shp[1:]) for _ in range(shp[0])]
  return build(shape)