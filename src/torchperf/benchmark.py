import torch
import torch.utils.benchmark as benchmark

from .trace import ExecutionTrace
from .collectors.flops import collect_flops
from .collectors.profiler import collect_profile
from .operation import OperationTrace
from .categories import categorize_op
from .collectors.memory import collect_peak_memory
from .diagnostics import detect_copy_pressure, detect_layout_pressure
from .collectors.profiler import collect_profile
from .kernel import KernelTrace


def _infer_device(args, kwargs):
    for obj in list(args) + list(kwargs.values()):
        if isinstance(obj, torch.Tensor):
            return str(obj.device)

    return "cpu"


def trace(fn, *args, collect_flops_enabled=True, collect_profile_enabled=True, collect_memory_enabled=True, **kwargs):
    def runner():
        with torch.inference_mode():
            return fn(*args, **kwargs)

    timer = benchmark.Timer(
        stmt="runner()",
        globals={"runner": runner},
    )

    measurement = timer.blocked_autorange()

    if collect_flops_enabled:
        flop_data = collect_flops(fn, *args, **kwargs)
    else:
        flop_data = {
            "total": None,
            "by_op": {},
        }
    if collect_profile_enabled:
        profile_data = collect_profile(fn, *args, **kwargs)
    else:
        profile_data = {
            "ops": {},
            "kernels": {},
        }
    ops = profile_data["ops"]
    kernels = profile_data["kernels"]
    cuda_launch_count = 0

    if "cudaLaunchKernel" in ops:
        cuda_launch_count = ops["cudaLaunchKernel"]["calls"]

    if collect_memory_enabled:
        peak_memory_bytes = collect_peak_memory(fn, *args, **kwargs)
    else:
        peak_memory_bytes = None

    operation_traces = []

    for op_name, op_data in ops.items():
        flop_name = op_name.replace("::", ".")
        flops = flop_data["by_op"].get(flop_name)

        operation_traces.append(
            OperationTrace(
                name=op_name,
                calls=op_data["calls"],
                flops=flops,
                cpu_time_us=op_data["cpu_time_us"],
                cuda_time_us=op_data["cuda_time_us"],
                category=categorize_op(op_name),
            )
        )

    kernel_traces = []

    for kernel_name, kernel_data in kernels.items():
        kernel_traces.append(
            KernelTrace(
                name = kernel_name,
                calls = kernel_data["calls"],
                cuda_time_us = kernel_data["cuda_time_us"],
            )
        )
        
    trace_result =  ExecutionTrace(
        runtime_ms=measurement.median * 1000,
        device=_infer_device(args, kwargs),
        total_flops=flop_data["total"],
        flops_by_op=flop_data["by_op"],
        ops=operation_traces,
        kernels = kernel_traces,
        peak_memory_bytes=peak_memory_bytes,
        cuda_launch_count=cuda_launch_count,
    )

    diagnostic_rules = [
        detect_copy_pressure,
        detect_layout_pressure,
    ]

    for rule in diagnostic_rules:
        diagnostic = rule(trace_result)
        if diagnostic:
            trace_result.diagnostics.append(diagnostic)

    return trace_result