from dataclasses import dataclass, field

from .operation import OperationTrace

@dataclass
class ModuleTrace:
    name : str
    ops : list[OperationTrace] = field(default_factory=list)
    cpu_time_us : float | None = None
    cuda_time_us : float | None = None