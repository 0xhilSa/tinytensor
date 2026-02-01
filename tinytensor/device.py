from tinytensor.engine.cuda.cuda import device_count
import functools
import warnings
import re

DEVICES = {"CPU", "CUDA"}

class Device:
  __slots__ = ("type", "index")
  @staticmethod
  @functools.cache
  def _canonicalize(spec:str):
    spec = spec.upper()
    if ":" in spec:
      base, rest = spec.split(":", 1)
      spec = base + ":" + rest
    return re.sub(r":0$", "", spec)
  def __init__(self, spec: str):
    spec = Device._canonicalize(spec)
    if ":" in spec:
      dev_type, idx = spec.split(":", 1)
      try: index = int(idx)
      except ValueError: raise ValueError(f"Invalid device index: {idx}")
    else:
      dev_type = spec
      index = 0
    if dev_type not in DEVICES: raise ValueError(f"Unknown device type: {dev_type}")
    if index < 0: raise ValueError("device index must be non-negative")
    if dev_type == "CPU":
      if index != 0: warnings.warn(f"CPU device index must be 0 (got {index}); defaulting to 0")
      index = 0
    else:
      count = device_count()
      if index >= count: raise ValueError(f"invalid CUDA device index. Expected range [0, {count}), got {index}")
    self.type = dev_type
    self.index = index
  def __repr__(self): return f"Device(type={self.type}, index={self.index})" if self.type == "CUDA" else f"Device(type={self.type})"
  def eq(self, other:"Device", soft:bool=False):
    if not isinstance(other, Device): other = Device(other)
    return self.type == other.type and self.index == other.index if not soft else self.type == other.type
  def ne(self, other:"Device", soft:bool=False): return not self.eq(other, soft=soft)
  def __eq__(self, other:"Device"): return self.eq(other) # type: ignore
  def __ne__(self, other:"Device"): return self.ne(other) # type: ignore
