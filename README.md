# TinyTensor
tending towards the tiny python tensor computation library


## Install from the source
```bash
git clone https://github.com/0xhilSa/tinytensor
cd tinytensor
bash compile
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
- [ ] Implement function to convert pointers/pycapsule into the list (cpu & gpu)
- [ ] CUDA device selection
- [ ] Basic arithmetic operation (`__add__`, `__sub__`, and so on)
- [ ] Relational operation (`__eq__`, `__ne__`, and so on)
- [ ] Vectorize the whole tensor operation
