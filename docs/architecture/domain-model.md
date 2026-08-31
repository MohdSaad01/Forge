# Forge Domain Model

## Core entities

### Tensor
A typed multidimensional numerical value associated with a device and, when enabled, gradient-tracking state.

### Operation
A numerical transformation on tensors. Differentiable operations expose enough information for autograd to compute input gradients.

### Parameter
A trainable tensor owned by a module/model.

### Module
A composable unit that can own parameters and child modules and implement a forward computation.

### Model
A module intended to represent a complete trainable/inferential network.

### Dataset
A source of samples. The framework should not assume every dataset is file-backed.

### DataLoader
An iterator producing batches from a dataset according to batching and shuffling configuration.

### Optimizer
Owns optimization state and updates parameters from gradients.

### Trainer
Coordinates repeated forward/loss/backward/update steps and records training state/metrics.

### Device
A logical execution target such as CPU or CUDA.

## Relationship

```text
Model
 ├── Modules
 │    └── Parameters
 └── Forward → Tensor operations → Autograd

Dataset → DataLoader → Trainer
                         ├── Model
                         ├── Loss
                         ├── Optimizer
                         └── Metrics
```
