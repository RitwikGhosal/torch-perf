def categorize_op(name: str) -> str:
    if name.endswith(("::mm", "::bmm", "::matmul", "::addmm")):
        return "MATMUL"

    elif name.endswith(("::relu", "::sigmoid", "::clamp_min", "::add", "::mul")):
        return "ELEMENTWISE"

    elif name.endswith(("::conv2d", "::conv3d")):
        return "CONVOLUTION"

    elif name.endswith(("::batch_norm", "::layer_norm")):
        return "NORMALIZATION"

    elif name.endswith(("::max_pool2d", "::avg_pool2d")):
        return "POOLING"

    elif name.endswith(("::permute", "::transpose", "::view", "::as_strided", "::reshape")):
        return "LAYOUT"

    elif name.endswith(("::copy_", "::clone", "::contiguous", "::_to_copy")):
        return "COPY"

    elif name.endswith(("::cat", "::stack")):
        return "TENSOR_COMBINE"

    elif name.endswith(("::slice", "::narrow", "::split")):
        return "SLICE"

    elif name.endswith(("::sum", "::mean", "::var", "::std")):
        return "REDUCTION"

    elif name.endswith("::einsum"):
        return "EINSUM"

    return "OTHER"