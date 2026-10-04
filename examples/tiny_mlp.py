import torch
import torch.nn as nn
from torchperf.collectors.profiler import collect_profile


class TinyMLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 64)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

model = TinyMLP()
x = torch.randn(32, 128)

#profile_data = collect_profile(model, x)
#print(profile_data["module_ops"])
#print(profile_data["modules"])

from torchperf import profile

result = profile(model, x)

print(result.module("fc1"))
print(result.module("fc2"))

#for name in profile_data["ops"]:
#    if name.startswith("TORCHPERF_MODULE::"):
#        print(name)