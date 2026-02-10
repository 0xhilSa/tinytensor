from __future__ import annotations
from tinytensor.engine.cuda.cuda import device_count
from typing import Union, Tuple
import functools
import warnings
import re

DEVICES = {"CPU", "CUDA"}

DeviceSpec = Union[
    str,
    "Device",
    Tuple[str, int],   # ("CUDA", 0)
]

class Device:
  __slots__ = ("type", "index")

  @staticmethod
  @functools.cache
  def _canonicalize(spec:str) -> str:
    spec = spec.upper()
    if ":" in spec:
      base, rest = spec.split(":", 1)
      spec = base + ":" + rest
    return re.sub(r":0$", "", spec)
  def __init__(self, spec:DeviceSpec="cpu"):
    if isinstance(spec, Device):
      self.type = spec.type
      self.index = spec.index
      return
    if isinstance(spec, tuple):
      if len(spec) != 2: raise ValueError("Device tuple must be (type, index)")
      dev_type, index = spec
      if not isinstance(dev_type, str): raise TypeError("Device type must be a string")
      if not isinstance(index, int): raise TypeError("Device index must be an int")
      dev_type = dev_type.upper()
    elif isinstance(spec, str):
      spec = Device._canonicalize(spec)
      if ":" in spec:
        dev_type, idx = spec.split(":", 1)
        try: index = int(idx)
        except ValueError: raise ValueError(f"Invalid device index: {idx}")
      else:
        dev_type = spec
        index = 0
    else: raise TypeError("Device spec must be str, tuple[str,int], or Device")
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
  def __repr__(self):
      if self.type == "CUDA": return f"Device(type={self.type}, index={self.index})"
      return f"Device(type={self.type})"
  def __iter__(self):return iter((self.type, self.index))
  def eq(self, other: DeviceSpec, soft: bool = False):
      other = Device(other)
      return (self.type == other.type and self.index == other.index if not soft else self.type == other.type)
  def ne(self, other:DeviceSpec, soft:bool=False): return not self.eq(other, soft=soft)
  def __eq__(self, other: object): return self.eq(other)  # type: ignore
  def __ne__(self, other: object): return self.ne(other)  # type: ignore
