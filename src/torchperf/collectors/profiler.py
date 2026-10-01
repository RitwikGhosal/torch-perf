import torch


def collect_profile(fn, *args, **kwargs):
    activities = [torch.profiler.ProfilerActivity.CPU]

    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
    ) as prof:
        fn(*args, **kwargs)

    ops = {}
    kernels = {}

    for event in prof.key_averages():
        if event.device_type == torch.autograd.DeviceType.CUDA:
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
    }