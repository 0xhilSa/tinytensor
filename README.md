# TinyTensor
a tiny Python tensor computation library

## Install from the source
```bash
git clone https://github.com/0xhilSa/tinytensor
cd tinytensor
bash compile.sh
```

## Testing
```python3
import tinytensor as tt

x = tt.Tensor([[1,2,3,4],[5,6,7,8]], dtype=tt.dtypes.int8, device="cpu", const=True)
y = tt.Tensor([[1,2,3,4],[5,6,7,8]], dtype=tt.dtypes.int8, device="cuda", const=True)
print("=====on CPU=====")
print(x)
print(x.buf)
print(x.device)
print("=====on CUDA=====")
print(y)
print(y.buf)
print(y.device)
```

## TODOs
- [X] Implement conversion from raw pointers / PyCapsules to Python lists (CPU & CUDA)
- [ ] Extend `.cuda()` and `.cpu()` to support `copy`, `dtype`, `const`, and related parameters
- [ ] Add CUDA device selection and device indexing support
- [ ] Implement basic arithmetic operations (`__add__`, `__sub__`, etc.)
- [ ] Implement relational operations (`__eq__`, `__ne__`, etc.)
- [ ] Vectorize tensor operations for performance
- [ ] Profiler for cpu/cuda based tensors
- [ ] Supports cross dtypes tensor operation
- [ ] Implement `Tensor` dtype casting
- [X] Enable cuda device selection
