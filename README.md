<p align="center">
  <img src="./docs/tt-logo.png" alt="tt-logo.png">
  <p align="center">a tiny Python tensor computation library</p>
</p>


## What is TinyTensor?
TinyTensor is a lightweight tensor library written from scratch in pure C, focused on building the core foundations of deep learning frameworks.
It provides basic tensor structures, memory handling, and essential operations with a minimal and educational design.

## Install from the source
```bash
git clone https://github.com/0xhilSa/tinytensor
cd tinytensor
bash compile.sh -run
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
or simply run `pytest -v`

## OS(es)
tinytensor runs on
- [X] Linux
- [ ] Windows

## LICENSE
[MIT](./LICENSE)
