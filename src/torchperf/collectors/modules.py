import torch


def add_module_ranges(model):
    handles = []
    active_ranges = {}

    for name, module in model.named_modules():
        if name == "":
            continue

        def make_pre_hook(module_name):
            def pre_hook(module, args):
                context = torch.profiler.record_function(
                    f"TORCHPERF_MODULE::{module_name}"
                )
                context.__enter__()
                active_ranges[id(module)] = context

            return pre_hook

        def make_post_hook(module_name):
            def post_hook(module, args, output):
                context = active_ranges.pop(id(module), None)

                if context is not None:
                    context.__exit__(None, None, None)

            return post_hook

        handles.append(
            module.register_forward_pre_hook(
                make_pre_hook(name)
            )
        )

        handles.append(
            module.register_forward_hook(
                make_post_hook(name)
            )
        )

    return handles