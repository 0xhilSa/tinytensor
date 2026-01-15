from typing import List, Tuple, Any
import ctypes

def tocpu(pylist:List[Any], shape:Tuple[int,...], fmt:str) -> ctypes.c_void_p: ...
