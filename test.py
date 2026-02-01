import tinytensor as tt

x = tt.Tensor([[1,2,3,4],[5,6,7,8]], dtype=tt.dtypes.uint8, device="cuda:0")
y = tt.Tensor([[1,2,3,4],[5,6,7,8]], dtype=tt.dtypes.uint8, device="cuda:0")
z = tt.Tensor([[1,2,3,4]], dtype=tt.dtypes.uint8, device="cuda:0")

print(x)
print(y)
print(z)

a = x + z
b = x + 2
c = 2 + x

print(f"{x.data()} + {z.data()} = {a.data()}")
print(f"{x.data()} + {2} = {b.data()}")
print(f"{2} + {x.data()} = {c.data()}")
