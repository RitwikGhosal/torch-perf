import torch
from torch.utils.flop_counter import FlopCounterMode

def _normalize_flop_counts(raw_counts):
    global_counts = raw_counts.get("Global", {})
    normalized = {}
    for op, flops in global_counts.items():
        normalized[str(op)] = flops
    return normalized

def collect_flops(fn, *args, **kwargs):
    with FlopCounterMode(display=False) as counter:
        fn(*args, **kwargs)

    raw_counts = counter.get_flop_counts()

    return {
        "total": counter.get_total_flops(),
        "by_op": _normalize_flop_counts(raw_counts),
    }