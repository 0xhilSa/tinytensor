import ctypes

# defined at ./tinytensor/engine/cuda/cuda_ops.c
def add(x:ctypes.c_void_p, y:ctypes.c_void_p) -> ctypes.c_void_p: ...
