import warnings

DEVICES = {"CPU", "CUDA"}

class Device:
  __slots__ = ("type", "index")
  def __init__(self, spec:str):
    spec = spec.upper()
    if ":" in spec:
      dev_type, idx =  spec.split(":", 1)
      try: index = int(idx)
      except: raise ValueError(f"Invalid device index: {idx}")
    else:
      dev_type = spec
      index = 0
    if dev_type not in DEVICES: raise ValueError(f"Unknown device type: {dev_type}")
    if index < 0: raise ValueError(f"device index must be non-negative")
    self.type = dev_type
    if self.type == "CPU" and index != 0: warnings.warn(f"CPU device index must be 0 (got {index}) defaulting to 0"); index = 0
    self.index = index
  def __repr__(self): return f"Device('{self.type.lower()}:{self.index}')"
  def __eq__(self, other):
    return (
      isinstance(other, Device)
      and self.type == other.type
      and self.index == other.index
    )
  def __ne__(self, other): return not self.__eq__(other)
  def is_cpu(self): return self.type == "CPU"
  def is_cuda(self): return self.type == "CUDA"
