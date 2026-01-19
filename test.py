import tinytensor as tt

v = tt.Tensor([[1+3j, 1-2j, -1+7j, -5-9j],[1+3j, 1-2j, -1+7j, -5-9j]], dtype=tt.dtypes.complex128, device="cuda:0")
w = tt.Tensor([[1,2,3],[4,5,6]], dtype=tt.dtypes.float64, device="cpu:0")

x = tt.Tensor([[1,2,3,4],[5,6,7,8]], dtype=tt.dtypes.float64, device="cpu:0")
y = tt.Tensor([[1,2,3,4],[5,6,7,8]], dtype=tt.dtypes.float64, device="cpu:0")
z = tt.Tensor(2, dtype=tt.dtypes.float64)

a = tt.Tensor([[1,2,3,4],[5,6,7,8]], dtype=tt.dtypes.float64, device="cuda:0")

print(x)
print(y)
print(z)
print(a, a.tolist())

print(x+v)
print(x+y, (x+y).tolist())
print(x+z, (x+z).tolist())

#print(x)
#print(x.device)
#print(x.dtype)
#
#print(y)
#print(y.device)
#print(y.dtype)

#print(tt.cuda.device_count())
#print(tt.cuda.device_name(0))
#print(tt.cuda.device_prop(0))
#print(tt.cuda.is_available())
