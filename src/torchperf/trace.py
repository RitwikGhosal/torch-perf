from dataclasses import dataclass

@dataclass
class ExecutionTrace:
    runtime_ms: float
    device: str