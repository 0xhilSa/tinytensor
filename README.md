<p align="center">
  <img src="./docs/tt-logo.png" alt="tt-logo.png">
  <p align="center">a tiny tensor computation library</p>
</p>

## What is TinyTensor?
TinyTensor is a lightweight tensor library written from scratch in pure C, focused on building the core foundations of deep learning frameworks.
It provides basic tensor structures, memory handling, and essential operations with a minimal and educational design.


## DType supported
| DType      |      Description      | CPU | CUDA |
|------------|-----------------------|-----|------|
|`bool`      | bool dtype            | ✅  |  ✅  |
|`int8`      | 8bit signed integer   | ✅  |  ✅  |
|`uint8`     | 8bit unsigned integer | ✅  |  ✅  |
|`int16`     | 16bit signed integer  | ✅  |  ✅  |
|`uint16`    | 16bit unsigned integer| ✅  |  ✅  |
|`int32`     | 32bit signed integer  | ✅  |  ✅  |
|`uint32`    | 32bit unsigned integer| ✅  |  ✅  |
|`int64`     | 64bit signed integer  | ✅  |  ✅  |
|`uint64`    | 64bit unsigned integer| ✅  |  ✅  |
|`float16`   | 16bit floating point  | ✅  |  ✅  |
|`float32`   | 32bit floating point  | ✅  |  ✅  |
|`float64`   | 64bit floating point  | ✅  |  ✅  |
|`float128`  | 128bit floating point | ❌  |  ❌  |
|`complex32` | 32bit complex dtype   | ❓  |  ❓  |
|`complex64` | 64bit complex dtype   | ✅  |  ✅  |
|`complex128`| 128bit complex dtype  | ✅  |  ✅  |
|`complex256`| 256bit complex dtype  | ❌  |  ❌  |

❓ = not-supported (in future maybe)

## Install from the source
```bash
git clone https://github.com/0xhilSa/tinytensor
cd tinytensor
bash build.sh
```

## Testing
```python3
import tinytensor as tt

with tt.Device("cuda"):
  x = tt.tensor([3.14, 2.71, 3.91, -8.01, 4.63], dtype=tt.float32, requires_grad=True)
  y = tt.tensor([4.41, 1.71, -7.12, 0.81, 6.21], dtype=tt.float32, requires_grad=True)
  z = tt.tensor(2.2, dtype=tt.float32, requires_grad=True)
  a = x * y + z
  a.backward()
  print(a)
  print(a.numpy())
  print(x.grad.numpy())
  print(y.grad.numpy())
  print(z.grad.numpy())
```
or simply run `pytest -v`

## Requirements
- GCC / Clang
- CUDA Toolkit (for GPU support)
- Python ≥ 3.9
- Linux environment

## Contributions
Pull requests are welcome.
If you find a bug or want to suggest an operation, feel free to open an issue.

## LICENSE
[MIT](./LICENSE)
