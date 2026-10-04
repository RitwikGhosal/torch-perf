from .diagnostic import Diagnostic


def detect_copy_pressure(trace):
    copies = trace.copy_ops()
    total_copy_calls = sum(op.calls for op in copies)

    if total_copy_calls >= 3:
        return Diagnostic(
            code="TD001",
            title="Repeated copy/materialization operations",
            severity="medium",
            evidence={
                "copy_op_types": len(copies),
                "copy_calls": total_copy_calls,
                "ops": [
                    {
                        "name": op.name,
                        "calls": op.calls,
                    }
                    for op in copies
                ],
            },
        )

    return None

def detect_layout_pressure(trace):
    layout_ops = trace.layout_ops()
    total_layout_calls = sum(op.calls for op in layout_ops)

    if total_layout_calls >= 5:
        return Diagnostic(
            code="TD002",
            title="Repeated layout operations",
            severity="low",
            evidence={
                "layout_op_types": len(layout_ops),
                "layout_calls": total_layout_calls,
                "ops": [
                    {
                        "name": op.name,
                        "calls": op.calls,
                    }
                    for op in layout_ops
                ],
            },
        )

    return None