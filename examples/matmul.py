import torch
from torchperf import trace


x = torch.randn(1024, 1024)

result = trace(lambda x: x @ x, x)

print(result)