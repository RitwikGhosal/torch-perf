from dataclasses import dataclass

@dataclass
class Diagnostic:
    code: str
    title: str
    severity: str
    evidence: dict