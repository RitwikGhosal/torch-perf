from dataclasses import dataclass, field
from unicodedata import name

from .diagnostic import Diagnostic
from .operation import OperationTrace
from .kernel import KernelTrace
from .comparison import TraceComparison
from .module import ModuleTrace


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
    cuda_launch_count: int = 0    
    modules: list[ModuleTrace] = field(default_factory=list)

    def by_op(self, name):
        for op in self.ops:
            if op.name == name:
                return op
        return None

    def by_category(self, category):
        return [op for op in self.ops if op.category == category]

    def slowest_ops(self, n=5):
        aten_ops = [
            op for op in self.ops
            if op.name.startswith("aten::")
        ]

        return sorted(
            aten_ops,
            key=lambda op: op.cpu_time_us if op.cpu_time_us is not None else 0,
            reverse=True,
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

    def module(self, name):
        for module in self.modules:
            if module.name == name:
             return module
        return None

    def report(self):
        lines = []

        lines.append("TorchPerf Report")
        lines.append("=" * 16)
        lines.append("")

        lines.append(f"Runtime: {self.runtime_ms:.3f} ms")
        lines.append(f"Device: {self.device}")

        if self.total_flops is not None:
            lines.append(f"FLOPs: {self.total_flops:,}")

        if self.peak_memory_bytes is not None:
            peak_mib = self.peak_memory_bytes / (1024 ** 2)
            lines.append(f"Peak memory: {peak_mib:.2f} MiB")

        if self.cuda_launch_count:
            lines.append(f"CUDA launches: {self.cuda_launch_count}")

        lines.append("")

        if self.device.startswith("cuda"):
            lines.append("Top operations (CPU-side profiler time)")
            lines.append("-" * 39)
        else:
            lines.append("Top operations")
            lines.append("-" * 14)

        for op in self.slowest_ops(5):
            time_ms = (
                op.cpu_time_us / 1000
                if op.cpu_time_us is not None
                else 0
            )

            lines.append(
                f"{op.name:<24} "
                f"{op.calls:>3} calls   "
                f"{time_ms:>8.3f} ms   "
                f"{op.category or 'OTHER'}"
            )

        if self.kernels:
            lines.append("")
            lines.append("Top CUDA kernels")
            lines.append("-" * 16)

            for kernel in self.slowest_kernels(5):
                time_ms = kernel.cuda_time_us / 1000

                lines.append(
                    f"{kernel.name:<40} "
                    f"{kernel.calls:>3} calls   "
                    f"{time_ms:>8.3f} ms"
                )

        if self.modules:
            lines.append("")
            lines.append("Modules")
            lines.append("-" * 7)

            for module in sorted(
                self.modules,
                key = lambda m: (
                    m.cuda_time_us
                    if self.device.startswith("cuda")
                    else m.cpu_time_us
                ) or 0,
                reverse=True,
            ):
                if self.device.startswith("cuda"):
                    time_us = module.cuda_time_us
                else:
                    time_us = module.cpu_time_us

                time_ms = time_us / 1000 if time_us is not None else 0

                lines.append(
                    f"{module.name:<24} "
                    f"{time_ms:>8.3f} ms"
                )

        if self.diagnostics:
            lines.append("")
            lines.append("Diagnostics")
            lines.append("-" * 11)

            for diagnostic in self.diagnostics:
                lines.append(
                    f"[{diagnostic.severity}] "
                    f"{diagnostic.title}"
                )

                ops = diagnostic.evidence.get("ops", [])

                for op in ops:
                    lines.append(
                        f"  - {op['name']}: {op['calls']} calls"
                    )

        return "\n".join(lines)
    
    def compare(self, other):
        runtime_change_pct = (
            (other.runtime_ms - self.runtime_ms)
            / self.runtime_ms * 100
        )
        
        return TraceComparison(
            runtime_a_ms=self.runtime_ms,
            runtime_b_ms=other.runtime_ms,
            runtime_change_pct=runtime_change_pct,
            kernel_count_a=len(self.kernels),
            kernel_count_b=len(other.kernels),
            peak_memory_a=self.peak_memory_bytes,
            peak_memory_b=other.peak_memory_bytes,
            cuda_launch_count_a=self.cuda_launch_count,
            cuda_launch_count_b=other.cuda_launch_count,
        )