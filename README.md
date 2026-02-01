<p align="center">
  <img src="./docs/tt-logo.png" alt="tt-logo.png">
  <p align="center">a tiny Python tensor computation library</p>
</p>


## What is TinyTensor?
a tinytensor is a tiny implementation of tensor computation library, focused on how tensor ops works bts.
kernel launch, fusion ops, memory allocation, free(ing).
The goal is to implement tensor library from scratch using Python, C, & CUDA, avoiding using external libraries like
numpy, scipy, matplotlib, etc.

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
- [X] Implement `copy` method
- [X] Implement tensor broadcasting
- [X] Implement `Tensor` dtype casting
- [X] Enable cuda device selection

## OS(es)
tinytensor runs on
- [X] Linux/MacOS
- [ ] Windows

## LICENSE
[MIT](./LICENSE)
