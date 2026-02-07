
import tinytensor as tt

def test_tensor_add():
  x = tt.Tensor([1, 2, 3, 4, 5], dtype=tt.int8)
  y = tt.Tensor([3], dtype=tt.int8)
  z = x + y
  assert z.data() == [4, 5, 6, 7, 8]

def test_tensor_sub():
  x = tt.Tensor([1, 2, 3, 4, 5], dtype=tt.int8)
  y = tt.Tensor([3], dtype=tt.int8)
  z = x - y
  assert z.data() == [-2, -1, 0, 1, 2]

def test_tensor_mul():
  x = tt.Tensor([1, 2, 3, 4, 5], dtype=tt.int8)
  y = tt.Tensor([3], dtype=tt.int8)
  z = x * y
  assert z.data() == [3, 6, 9, 12, 15]

def test_tensor_truediv():
  x = tt.Tensor([3, 6, 9], dtype=tt.int8)
  y = tt.Tensor([3], dtype=tt.int8)
  z = x / y
  assert z.data() == [1, 2, 3]

def test_tensor_floordiv():
  x = tt.Tensor([1, 2, 3, 4, 5], dtype=tt.int8)
  y = tt.Tensor([3], dtype=tt.int8)
  z = x // y
  assert z.data() == [0, 0, 1, 1, 1]

def test_tensor_pow():
  x = tt.Tensor([1, 2, 3], dtype=tt.int8)
  y = tt.Tensor([2], dtype=tt.int8)
  z = x ** y
  assert z.data() == [1, 4, 9]

def test_tensor_mod():
  x = tt.Tensor([1, 2, 3], dtype=tt.int8)
  y = tt.Tensor([2], dtype=tt.int8)
  z = x % y
  assert z.data() == [1, 0, 1]

def test_cuda_tensor_add():
  x = tt.Tensor([1, 2, 3, 4, 5], dtype=tt.int8, device="cuda")
  y = tt.Tensor([3], dtype=tt.int8, device="cuda")
  z = x + y
  assert z.data() == [4, 5, 6, 7, 8]

def test_cuda_tensor_sub():
  x = tt.Tensor([1, 2, 3, 4, 5], dtype=tt.int8, device="cuda")
  y = tt.Tensor([3], dtype=tt.int8, device="cuda")
  z = x - y
  assert z.data() == [-2, -1, 0, 1, 2]

def test_cuda_tensor_mul():
  x = tt.Tensor([1, 2, 3, 4, 5], dtype=tt.int8, device="cuda")
  y = tt.Tensor([3], dtype=tt.int8, device="cuda")
  z = x * y
  assert z.data() == [3, 6, 9, 12, 15]

def test_cuda_tensor_truediv():
  x = tt.Tensor([3, 6, 9], dtype=tt.int8, device="cuda")
  y = tt.Tensor([3], dtype=tt.int8, device="cuda")
  z = x / y
  assert z.data() == [1, 2, 3]

def test_cuda_tensor_floordiv():
  x = tt.Tensor([1, 2, 3, 4, 5], dtype=tt.int8, device="cuda")
  y = tt.Tensor([3], dtype=tt.int8, device="cuda")
  z = x // y
  assert z.data() == [0, 0, 1, 1, 1]

def test_cuda_tensor_pow():
  x = tt.Tensor([1, 2, 3], dtype=tt.int8, device="cuda")
  y = tt.Tensor([2], dtype=tt.int8, device="cuda")
  z = x ** y
  assert z.data() == [1, 4, 9]

def test_cuda_tensor_mod():
  x = tt.Tensor([1, 2, 3], dtype=tt.int8, device="cuda")
  y = tt.Tensor([2], dtype=tt.int8, device="cuda")
  z = x % y
  assert z.data() == [1, 0, 1]
