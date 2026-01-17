import tinytensor as tt

x = tt.Tensor([[1,2,3,4],[5,6,7,8]], dtype=tt.dtypes.float64, device="cpu:0")
y = tt.Tensor([[1,2,3,4],[5,6,7,8]], dtype=tt.dtypes.float64, device="cuda:0")

print("=====(tensor at CPU initially)=====")
print(x.buf)
print(x)
print(x.cuda())
print("=====(tensor at CUDA initially)=====")
print(y.buf)
print(y)
print(y.cpu())

print(tt.gpu_cuda.device_count())
print(tt.gpu_cuda.device_name(0))
print(tt.gpu_cuda.device_prop(0))
print(tt.gpu_cuda.is_available())
