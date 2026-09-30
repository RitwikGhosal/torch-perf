import torch


def collect_ops(fn, *args, **kwargs):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
    ) as prof:
        fn(*args, **kwargs)

    ops = {}

    for event in prof.key_averages():
        ops[event.key] = {
            "calls": event.count,
            "cpu_time_us": event.cpu_time_total,
        }

    return ops