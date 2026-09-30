import torch


def collect_peak_memory(fn, *args, **kwargs):
    tensors = [
        obj
        for obj in list(args) + list(kwargs.values())
        if isinstance(obj, torch.Tensor)
    ]

    if not tensors:
        return None

    device = tensors[0].device

    if device.type != "cuda":
        return None

    torch.cuda.reset_peak_memory_stats(device)
    fn(*args, **kwargs)

    return torch.cuda.max_memory_allocated(device)

