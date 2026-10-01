from dataclasses import dataclass


@dataclass
class TraceComparison:
    runtime_a_ms: float
    runtime_b_ms: float

    runtime_change_pct: float

    kernel_count_a: int
    kernel_count_b: int

    peak_memory_a: int | None
    peak_memory_b: int | None