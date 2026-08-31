# Forge Execution Model

## High-level execution

```text
Batch
 ↓
Move/prepare on selected Device
 ↓
Model.forward
 ↓
Prediction Tensor
 ↓
Loss
 ↓
Backward
 ↓
Parameter gradients
 ↓
Optimizer.step
 ↓
Gradient reset
```

## Device model
A device identifies where tensor data and operations execute. High-level model/training code should request a device rather than branch on CPU/CUDA internals.

## Autograd model
The initial implementation should favor correctness and inspectability over maximum graph-engine performance. Each differentiable operation must have a well-defined backward rule and tests for its gradients.

## Backend model
Backends provide the numerical implementation behind tensor operations. CPU is the reference behavior. CUDA implementations must match the same semantic contract within normal floating-point tolerance. As of Milestone 8, a real CUDA backend exists for a small forward-only operation set -- see `docs/architecture/cuda-backend.md`.

## Memory
Do not introduce an elaborate memory allocator initially. Use the simplest correct storage model that can support the tensor/device abstraction. Optimize only after benchmarks identify a real bottleneck.
