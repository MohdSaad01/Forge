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
The initial implementation should favor correctness and inspectability over maximum graph-engine performance. Each differentiable operation must have a well-defined backward rule and tests for its gradients. As of Milestone 10, a backward rule is backend-dispatched (one implementation per device that supports it) rather than a single NumPy-only closure -- see `docs/architecture/autograd.md`.

## Backend model
Backends provide the numerical implementation behind tensor operations. CPU is the reference behavior. CUDA implementations must match the same semantic contract within normal floating-point tolerance. As of Milestone 8, a real CUDA backend exists for a small operation set -- see `docs/architecture/cuda-backend.md`. As of Milestone 9, that operation set is reachable through `nn.Module` (`Module.to(device)`, then an ordinary `Linear`/`ReLU` forward pass), not just raw `Tensor` code -- see `docs/architecture/modules.md`. As of Milestone 10, that operation set's *backward* computation also runs on CUDA -- reverse-mode autograd, gradient storage, and `SGD.step()` all dispatch through the same CPU/CUDA `Backend` boundary as forward execution, so a CUDA computation graph never needs to fall back to CPU to be differentiated -- see `docs/architecture/autograd.md`. As of Milestone 12, the full diagram at the top of this file runs on CUDA through `forge.training.Trainer(device="cuda")`: "Move/prepare on selected Device" is `Trainer`'s explicit per-batch `x.to(device)`/`y.to(device)` (never `DataLoader`'s job -- see `docs/architecture/data-system.md`), and `Loss` is `MSELoss` (composed entirely from operations this operation set already covered) or, as of Milestone 14, `CrossEntropyLoss` (which needed four new CUDA primitives -- `exp`/`log`, an axis=1 `sum`, and a column-broadcast `sub` -- added specifically because its own math required them) -- see `docs/architecture/training-engine.md` and `docs/architecture/cuda-backend.md`'s **CUDA losses**/**CUDA CrossEntropyLoss** sections. Every stage of this diagram now runs on CUDA for Forge's basic supervised (regression or classification) workflow.

## Memory
Do not introduce an elaborate memory allocator initially. Use the simplest correct storage model that can support the tensor/device abstraction. Optimize only after benchmarks identify a real bottleneck.
