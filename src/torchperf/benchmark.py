import torch
import torch.utils.benchmark as benchmark

from .trace import ExecutionTrace
from .collectors.flops import collect_flops
from .collectors.profiler import collect_ops
from .operation import OperationTrace
from .categories import categorize_op
from .collectors.memory import collect_peak_memory
from .diagnostics import detect_copy_pressure, detect_layout_pressure


def _infer_device(args, kwargs):
    for obj in list(args) + list(kwargs.values()):
        if isinstance(obj, torch.Tensor):
            return str(obj.device)

    return "cpu"


def trace(fn, *args, **kwargs):
    def runner():
        with torch.inference_mode():
            return fn(*args, **kwargs)

    timer = benchmark.Timer(
        stmt="runner()",
        globals={"runner": runner},
    )

    measurement = timer.blocked_autorange()

    flop_data = collect_flops(fn, *args, **kwargs)
    ops = collect_ops(fn, *args, **kwargs)
    peak_memory_bytes = collect_peak_memory(fn, *args, **kwargs)

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
        
    trace_result =  ExecutionTrace(
        runtime_ms=measurement.median * 1000,
        device=_infer_device(args, kwargs),
        total_flops=flop_data["total"],
        flops_by_op=flop_data["by_op"],
        ops=operation_traces,
        peak_memory_bytes=peak_memory_bytes,
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