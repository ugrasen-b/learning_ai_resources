import torch

N, D = 3, 4
x = torch.rand((N, D), requires_grad = True)
y = torch.rand((N, D), requires_grad = True)
z = torch.rand((N, D), requires_grad = True)

a = x * y
b = a + z
c = torch.sum(b)

c.backward()
print(f"Hello PyTorch")