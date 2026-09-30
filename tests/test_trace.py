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

def test_matmul_flops():
    a = torch.randn(128, 256)
    b = torch.randn(256, 512)

    result = trace(lambda a, b: a @ b, a, b)

    expected = 2 * 128 * 256 * 512

    assert result.total_flops == expected


def test_by_op():
    a = torch.randn(64, 64)
    b = torch.randn(64, 64)
    result = trace(lambda a, b: a @ b, a, b)
    op = result.by_op("aten::mm")
    assert op is not None
    assert op.name == "aten::mm"

def test_slowest_ops():
    a = torch.randn(64, 64)
    b = torch.randn(64, 64)

    def fn(a, b):
        x = a @ b
        return torch.relu(x)

    result = trace(fn, a, b)

    slowest = result.slowest_ops(2)

    assert len(slowest) == 2
    assert slowest[0].cpu_time_us >= slowest[1].cpu_time_us


def test_by_category():
    a = torch.randn(64, 64)
    b = torch.randn(64, 64)

    result = trace(lambda a, b: a @ b, a, b)

    matmul_ops = result.by_category("MATMUL")

    assert len(matmul_ops) >= 1
    assert all(op.category == "MATMUL" for op in matmul_ops)

def test_layout_ops():
    a = torch.randn(64, 64)

    result = trace(lambda x: x.T, a)

    layout = result.layout_ops()

    assert len(layout) >= 1
    assert all(op.category == "LAYOUT" for op in layout)

def test_copy_ops():
    a = torch.randn(64, 64)

    result = trace(lambda x: x.clone(), a)

    copies = result.copy_ops()

    assert len(copies) >= 1
    assert all(op.category == "COPY" for op in copies)

def test_copy_diagnostic():
    x = torch.randn(64, 64)

    def fn(x):
        x = x.clone()
        x = x.clone()
        x = x.clone()
        return x

    result = trace(fn, x)

    assert len(result.diagnostics) >= 1

    diagnostic = result.diagnostics[0]

    assert diagnostic.code == "TD001"
    assert diagnostic.severity == "medium"

def test_layout_diagnostic():
    x = torch.randn(64, 64)

    def fn(x):
        x = x.T
        x = x.T
        x = x.T
        return x

    result = trace(fn, x)

    codes = [d.code for d in result.diagnostics]

    assert "TD002" in codes