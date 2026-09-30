import torch
from torchperf import trace


a = torch.randn(128, 256)
b = torch.randn(256, 512)


def fn(a, b):
    x = a @ b
    x = torch.relu(x)
    return x @ x.T

result = trace(fn, a, b)

print(result)

print("\nFLOPs by operator:")
for op, flops in result.flops_by_op.items():
    print(op, flops)

print("\nExecuted operators:")
for op in result.ops:
    print(op)