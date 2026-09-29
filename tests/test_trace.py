import torch
from torchperf import trace

def test_plain_callable():
    x = torch.randn(1024, 1024)
    result = trace(lambda x: x@x, x)
    assert result.runtime_ms > 0
    assert result.device == "cpu"

def test_nn_module():
    x = torch.randn(32, 1024)
    model = torch.nn.Linear(1024, 1024)
    result = trace(model, x)

    assert result.runtime_ms > 0
    assert result.device == "cpu"

def test_kwargs():
    x = torch.randn(1024, 1024)

    def fn(x, scale=1.0):
        return (x @ x) * scale
    result = trace(fn, x, scale=2.0)
    assert result.runtime_ms > 0
    assert result.device == "cpu"