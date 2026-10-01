from dataclasses import dataclass, field

from .diagnostic import Diagnostic
from .operation import OperationTrace
from .kernel import KernelTrace


@dataclass
class ExecutionTrace:
    runtime_ms: float
    device: str
    total_flops: int | None = None
    flops_by_op: dict = field(default_factory=dict)
    ops: list[OperationTrace] = field(default_factory=list)
    kernels: list[KernelTrace] = field(default_factory=list)
    peak_memory_bytes: int | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def by_op(self, name):
        for op in self.ops:
            if op.name == name:
                return op
        return None

    def by_category(self, category):
        return [op for op in self.ops if op.category == category]

    def slowest_ops(self, n=5):
        return sorted(
            self.ops,
            key=lambda op: op.cpu_time_us if op.cpu_time_us is not None else 0,
            reverse=True
        )[:n]

    def slowest_kernels(self, n=5):
        return sorted(
            self.kernels,
            key = lambda k : k.cuda_time_us,
            reverse = True,
        )[:n]

    def copy_ops(self):
        return self.by_category("COPY")

    def layout_ops(self):
        return self.by_category("LAYOUT")
