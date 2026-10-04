import torch

from .modules import add_module_ranges


def collect_profile(fn, *args, **kwargs):
    activities = [torch.profiler.ProfilerActivity.CPU]

    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    handles = []

    if isinstance(fn, torch.nn.Module):
        handles = add_module_ranges(fn)

    try:
        with torch.profiler.profile(
            activities=activities,
        ) as prof:
            fn(*args, **kwargs)

    finally:
        for handle in handles:
            handle.remove()

    ops = {}
    kernels = {}
    modules = {}

    for event in prof.key_averages():
        if event.key.startswith("TORCHPERF_MODULE::"):
            module_name = event.key.removeprefix("TORCHPERF_MODULE::")

            modules[module_name] = {
                "calls": event.count,
                "cpu_time_us": event.cpu_time_total,
                "cuda_time_us": (
                    event.device_time_total
                    if torch.cuda.is_available()
                    else None
                ),
            }

        elif event.device_type == torch.autograd.DeviceType.CUDA:
            kernels[event.key] = {
                "calls": event.count,
                "cuda_time_us": event.device_time_total,
            }

        else:
            ops[event.key] = {
                "calls": event.count,
                "cpu_time_us": event.cpu_time_total,
                "cuda_time_us": (
                    event.device_time_total
                    if torch.cuda.is_available()
                    else None
                ),
            }

    return {
        "ops": ops,
        "kernels": kernels,
        "modules": modules,
    }