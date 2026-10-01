from dataclasses import dataclass

@dataclass
class KernelTrace:
    name: str
    calls: int
    cuda_time_us: float