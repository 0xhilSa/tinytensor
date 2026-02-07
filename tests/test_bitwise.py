import tinytensor as tt

def test_tensor_and():
  x = tt.Tensor([1, 2, 3], dtype=tt.int8)
  y = tt.Tensor([2], dtype=tt.int8)
  assert (x & y).data() == [0, 2, 2]

def test_tensor_nand():
  x = tt.Tensor([1, 2, 3], dtype=tt.int8)
  y = tt.Tensor([2], dtype=tt.int8)
  assert x.nand(y).data() == [-1, -3, -3]

def test_tensor_or():
  x = tt.Tensor([1, 2, 3], dtype=tt.int8)
  y = tt.Tensor([2], dtype=tt.int8)
  assert (x | y).data() == [3, 2, 3]

def test_tensor_nor():
  x = tt.Tensor([1, 2, 3], dtype=tt.int8)
  y = tt.Tensor([2], dtype=tt.int8)
  assert x.nor(y).data() == [-4, -3, -4]

def test_tensor_not():
  x = tt.Tensor([1, 2, 3], dtype=tt.int8)
  assert (~x).data() == [-2, -3, -4]

def test_tensor_xor():
  x = tt.Tensor([1, 2, 3], dtype=tt.int8)
  y = tt.Tensor([2], dtype=tt.int8)
  assert (x ^ y).data() == [3, 0, 1]

def test_tensor_xnor():
  x = tt.Tensor([1, 2, 3], dtype=tt.int8)
  y = tt.Tensor([2], dtype=tt.int8)
  assert x.xnor(y).data() == [-4, -1, -2]

def test_cuda_tensor_and():
  x = tt.Tensor([1, 2, 3], dtype=tt.int8, device="cuda")
  y = tt.Tensor([2], dtype=tt.int8, device="cuda")
  assert (x & y).data() == [0, 2, 2]

def test_cuda_tensor_nand():
  x = tt.Tensor([1, 2, 3], dtype=tt.int8, device="cuda")
  y = tt.Tensor([2], dtype=tt.int8, device="cuda")
  assert x.nand(y).data() == [-1, -3, -3]

def test_cuda_tensor_or():
  x = tt.Tensor([1, 2, 3], dtype=tt.int8, device="cuda")
  y = tt.Tensor([2], dtype=tt.int8, device="cuda")
  assert (x | y).data() == [3, 2, 3]

def test_cuda_tensor_nor():
  x = tt.Tensor([1, 2, 3], dtype=tt.int8, device="cuda")
  y = tt.Tensor([2], dtype=tt.int8, device="cuda")
  assert x.nor(y).data() == [-4, -3, -4]

def test_cuda_tensor_not():
  x = tt.Tensor([1, 2, 3], dtype=tt.int8, device="cuda")
  assert (~x).data() == [-2, -3, -4]

def test_cuda_tensor_xor():
  x = tt.Tensor([1, 2, 3], dtype=tt.int8, device="cuda")
  y = tt.Tensor([2], dtype=tt.int8, device="cuda")
  assert (x ^ y).data() == [3, 0, 1]

def test_cuda_tensor_xnor():
  x = tt.Tensor([1, 2, 3], dtype=tt.int8, device="cuda")
  y = tt.Tensor([2], dtype=tt.int8, device="cuda")
  assert x.xnor(y).data() == [-4, -1, -2]

def test_tensor_lshift():
  x = tt.Tensor([1, 2, 3], dtype=tt.uint8)
  y = tt.Tensor([3], dtype=tt.uint8)
  assert (x << y).data() == [8, 16, 24]

def test_tensor_rshift():
  x = tt.Tensor([8, 16, 24], dtype=tt.uint8)
  y = tt.Tensor([3], dtype=tt.uint8)
  assert (x >> y).data() == [1, 2, 3]

def test_cuda_tensor_lshift():
  x = tt.Tensor([1, 2, 3], dtype=tt.int16, device="cuda")
  y = tt.Tensor([3], dtype=tt.int16, device="cuda")
  assert (y << x).data() == [6, 12, 24]

def test_cuda_tensor_rshift():
  x = tt.Tensor([6, 12, 24], dtype=tt.int16, device="cuda")
  y = tt.Tensor([3], dtype=tt.int16, device="cuda")
  assert (y >> x).data() == [0, 0, 0]
