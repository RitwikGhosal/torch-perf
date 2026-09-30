from dataclasses import dataclass


@dataclass
class OperationTrace:
    name: str
    calls: int
    flops: int | None = None
    cpu_time_us: float | None = None
    category: str | None = None