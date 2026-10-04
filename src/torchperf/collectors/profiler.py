import torch

from .modules import add_module_ranges

def _find_module_ancestor(event):
    parent = event.cpu_parent
    while parent is not None:
        if parent.name.startswith("TORCHPERF_MODULE::"):
            return parent.name.removeprefix("TORCHPERF_MODULE::")
        parent = parent.cpu_parent
    return None


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
    module_ops = {}

    for event in prof.events():
        if not event.name.startswith("aten::"):
            continue

        module_name = _find_module_ancestor(event)

        if module_name is None:
            continue
        if module_name not in module_ops:
            module_ops[module_name] = {}

        if event.name not in module_ops[module_name]:
            module_ops[module_name][event.name] = {
                "calls": 0,
                "cpu_time_us": 0.0,
            }

        module_ops[module_name][event.name]["calls"] += 1
        module_ops[module_name][event.name]["cpu_time_us"] += event.cpu_time_total

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
        "module_ops": module_ops,
    }