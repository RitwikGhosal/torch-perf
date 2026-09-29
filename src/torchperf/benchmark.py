import torch
import torch.utils.benchmark as benchmark
from .trace import ExecutionTrace

def _infer_device(args, kwargs):
    for obj in list(args) + list(kwargs.values()):
        if isinstance(obj, torch.Tensor):
            return str(obj.device)
    return "cpu"  # Default to CPU if no tensor is found

def trace(fn, *args, **kwargs):
    def runner():
        with torch.inference_mode():
            return fn(*args, **kwargs)

    timer = benchmark.Timer(stmt = "runner()", globals = {"runner": runner})
    measurement = timer.timeit(1)
    return ExecutionTrace(
        runtime_ms=measurement.median * 1000,
        device=_infer_device(args, kwargs),
    )