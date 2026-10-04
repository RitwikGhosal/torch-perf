# TorchPerf

TorchPerf is a lightweight PyTorch execution-analysis library for inspecting **how a model actually runs**.

It combines runtime benchmarking, FLOP counting, PyTorch operator tracing, CUDA kernel inspection, memory tracking, module-level attribution, diagnostics, and trace comparison behind a small API.

The goal is not to replace `torch.profiler`, Nsight, or existing benchmarking tools. Instead, TorchPerf provides a higher-level execution trace that is easier to inspect, query, compare, and reason about.

```python
from torchperf import benchmark, profile

timing = benchmark(model, x)
trace = profile(model, x)

print(trace.report())
```

---

## Why TorchPerf?

PyTorch already exposes powerful profiling and benchmarking primitives, but answering simple questions often requires combining several tools manually.

Questions such as:

- How long does this model actually take to run?
- How many FLOPs does it perform?
- Which PyTorch operations dominate execution?
- Which CUDA kernels were launched?
- How much GPU memory was used?
- Which module caused a particular operation?
- Are repeated copies or layout transformations occurring?
- Did `torch.compile()` actually improve this workload?

TorchPerf collects these signals into a single queryable `ExecutionTrace`.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/RitwikGhosal/torch-perf.git
cd torch-perf
```

Install in editable mode:

```bash
pip install -e .
```

---

# Quickstart

## 1. Clean benchmarking

Use `benchmark()` when you only want runtime measurement without profiler or FLOP-collection overhead.

```python
import torch
from torchperf import benchmark

x = torch.randn(512, 512)

result = benchmark(lambda x: x @ x, x)

print(result.runtime_ms)
```

`benchmark()` intentionally disables the heavier collectors so timing measurements remain as clean as possible.

---

## 2. Full execution profiling

Use `profile()` when you want detailed execution information.

```python
import torch
from torchperf import profile

x = torch.randn(512, 512)

result = profile(lambda x: x @ x, x)

print(result.report())
```

A trace may contain:

- runtime
- device
- FLOPs
- operator traces
- CUDA kernel traces
- peak GPU memory
- CUDA launch count
- module attribution
- diagnostics

---

## 3. Profiling an `nn.Module`

```python
import torch
import torch.nn as nn
from torchperf import profile


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 64)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        return self.fc2(x)


model = TinyMLP()
x = torch.randn(32, 128)

result = profile(model, x)

print(result.report())
```

Example CPU report:

```text
TorchPerf Report
================

Runtime: 0.083 ms
Device: cpu
FLOPs: 3,145,728

Top operations
--------------
aten::linear               2 calls      0.760 ms   OTHER
aten::addmm                2 calls      0.563 ms   MATMUL
aten::t                    2 calls      0.152 ms   OTHER
aten::copy_                2 calls      0.101 ms   COPY
aten::relu                 1 calls      0.100 ms   ELEMENTWISE

Modules
-------
fc1                         0.816 ms
fc2                         0.302 ms
```

---

# CUDA profiling

TorchPerf can also inspect CUDA execution.

```python
model = TinyMLP().cuda()
x = torch.randn(32, 128, device="cuda")

result = profile(model, x)

print(result.report())
```

Example:

```text
TorchPerf Report
================

Runtime: 0.071 ms
Device: cuda:0
FLOPs: 3,145,728
Peak memory: 9.39 MiB
CUDA launches: 3

Top operations (CPU-side profiler time)
---------------------------------------
aten::linear               2 calls      3.217 ms   OTHER
aten::addmm                2 calls      3.121 ms   MATMUL
aten::relu                 1 calls      0.103 ms   ELEMENTWISE

Top CUDA kernels
----------------
volta_sgemm_32x32_sliced1x4_tn                                 2 calls    0.021 ms
void cublasLt::splitKreduce_kernel<32, 16, int, float, fl...   1 calls    0.004 ms

Modules
-------
fc2                         0.022 ms
fc1                         0.012 ms
```

On CUDA, operator timings and kernel timings represent different layers of execution.

`aten::...` timings are CPU-side profiler measurements around PyTorch operations.

CUDA kernel timings represent actual GPU kernel execution.

These should not be summed or interpreted as equivalent quantities.

---

# Module attribution

TorchPerf tracks which PyTorch operations were executed inside individual `nn.Module`s.

```python
result = profile(model, x)

fc1 = result.module("fc1")

print(fc1)
```

You can inspect the operations attributed to that module:

```python
for op in fc1.ops:
    print(op.name, op.calls, op.category)
```

Example:

```text
aten::linear
aten::t
aten::transpose
aten::as_strided
aten::addmm
aten::expand
aten::copy_
aten::resolve_conj
```

TorchPerf uses temporary module hooks and profiler ranges to trace operations back through their execution ancestry.

The hooks are removed immediately after profiling.

---

# Querying a trace

`ExecutionTrace` provides simple query helpers.

## Find an operation

```python
result.by_op("aten::addmm")
```

## Slowest operations

```python
result.slowest_ops(5)
```

## Operations by category

```python
result.by_category("MATMUL")
```

## Copy operations

```python
result.copy_ops()
```

## Layout operations

```python
result.layout_ops()
```

## Slowest CUDA kernels

```python
result.slowest_kernels(5)
```

## Inspect a module

```python
result.module("fc1")
```

---

# Comparing traces

TorchPerf can compare two execution traces.

```python
from torchperf import benchmark

eager = benchmark(model, x)

compiled_model = torch.compile(model)

# Warm the compiled model before benchmarking.
compiled_model(x)

if x.is_cuda:
    torch.cuda.synchronize()

compiled = benchmark(compiled_model, x)

comparison = eager.compare(compiled)

print(comparison)
```

Example:

```text
TraceComparison(
    runtime_a_ms=0.202,
    runtime_b_ms=0.229,
    runtime_change_pct=12.9,
    kernel_count_a=0,
    kernel_count_b=0,
    peak_memory_a=None,
    peak_memory_b=None,
    cuda_launch_count_a=0,
    cuda_launch_count_b=0
)
```

A negative runtime change means trace B was faster.

A positive runtime change means trace B was slower.

For small workloads, `torch.compile()` may be slower because compiler and dispatch overhead can outweigh optimization benefits.

---

# Diagnostics

TorchPerf includes lightweight rule-based diagnostics.

Current examples include:

- repeated copy/materialization operations
- repeated layout operations

Example:

```text
Diagnostics
-----------
[low] Repeated layout operations
  - aten::transpose: 2 calls
  - aten::as_strided: 4 calls
```

Diagnostics are intentionally conservative.

They represent evidence of potentially interesting execution behavior rather than guarantees that an optimization exists.

---

# `benchmark()`, `profile()`, and `trace()`

TorchPerf exposes three entry points.

## `benchmark()`

Use when you want clean timing.

```python
benchmark(fn, *args, **kwargs)
```

Heavy collectors are disabled.

---

## `profile()`

Use when you want full execution analysis.

```python
profile(fn, *args, **kwargs)
```

Enables FLOP collection, profiler collection, memory tracking, diagnostics, module attribution, and CUDA information when available.

---

## `trace()`

`trace()` is the lower-level configurable engine used internally by the two convenience APIs.

```python
trace(
    fn,
    *args,
    collect_flops_enabled=True,
    collect_profile_enabled=True,
    collect_memory_enabled=True,
    **kwargs,
)
```

It can be used when more control over collectors is needed.

---

# Current scope

TorchPerf v0.1 focuses on:

- PyTorch callables
- `nn.Module`
- CPU execution
- single-GPU CUDA execution
- forward or arbitrary callable regions
- runtime benchmarking
- FLOPs
- operator tracing
- CUDA kernels
- peak CUDA memory
- CUDA launch counts
- module attribution
- diagnostics
- trace comparison

---

# Current limitations

TorchPerf v0.1 does not yet attempt to provide:

- distributed profiling
- DDP/FSDP attribution
- multi-GPU execution analysis
- exact DRAM traffic
- roofline analysis
- detailed Triton kernel internals
- automatic code rewriting
- full training-loop monitoring
- guaranteed CUDA kernel-name availability across every profiler backend
- complete per-module FLOP attribution

Profiler measurements also introduce overhead.

For this reason, TorchPerf separates clean benchmarking from detailed profiling rather than treating profiler timings as benchmark timings.

---

# Design philosophy

TorchPerf treats execution analysis as structured data rather than just profiler output.

The core abstraction is:

```python
ExecutionTrace
```

An execution trace can be queried, reported, diagnosed, and compared.

The long-term goal is to make PyTorch computational behavior easier to inspect and reason about without requiring users to manually combine multiple low-level profiling tools.

---

# Development

Run tests with:

```bash
pytest
```

Current development is validated on CPU and single-GPU CUDA workloads.

---

# Status

TorchPerf is currently an early-stage experimental project.

The API may change as the execution model, diagnostics, and attribution system evolve.

Contributions, bug reports, profiling examples, and feedback are welcome.
