import tinytensor as tt

x = tt.Tensor([[1,2,3,4],[5,6,7,8]], dtype=tt.dtypes.float64, device="cpu:0")
y = tt.Tensor([[1,2,3,4],[5,6,7,8]], dtype=tt.dtypes.float64, device="cuda:0")

print(x)
print(x.device)
print(x.dtype)

print(y)
print(y.device)
print(y.dtype)

print(tt.cuda.device_count())
print(tt.cuda.device_name(0))
print(tt.cuda.device_prop(0))
print(tt.cuda.is_available())
