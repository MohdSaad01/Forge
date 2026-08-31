# Forge System Overview

## Major components

| Component | Responsibility |
|---|---|
| Tensor | Shape, dtype, storage, device, numerical operations |
| Autograd | Computation graph and gradient propagation |
| Parameter | Trainable tensor state |
| Module/Model | Composable neural-network behavior and parameters |
| Loss | Objective calculation |
| Optimizer | Parameter update from gradients |
| Dataset | Sample access and dataset semantics |
| Transform | Data preprocessing |
| DataLoader | Batching/shuffling/iteration |
| Trainer | Training loop and lifecycle |
| Metrics | Task evaluation |
| Serialization | Save/load model state and configuration |
| Device/Backend | CPU/CUDA execution boundary |
| CLI | User-facing commands |
| Benchmark | Reproducible performance measurement |

## Dependency direction
Lower-level numerical components must not depend on the CLI or application examples. Training may depend on model/data/loss/optimizer abstractions. Public APIs should compose lower layers rather than duplicate them.
