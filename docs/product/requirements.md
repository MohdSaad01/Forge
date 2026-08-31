# Forge Requirements

## Functional requirements

### Tensor computation
- Provide a tensor abstraction suitable for neural-network computation.
- Support the data types and operations required by implemented layers and training.
- Provide shape and dtype validation.

### Automatic differentiation
- Track the computation needed for differentiable operations.
- Compute gradients through supported operations.
- Expose gradients to the optimizer/training system.

### Neural networks
- Provide composable model/layer abstractions.
- Support trainable parameters.
- Begin with a small general architecture and add representative architecture families incrementally.

### Losses and optimization
- Provide reusable loss functions.
- Provide optimizer abstractions and at least one practical optimizer in the early working system.

### Data
- Provide dataset and dataloader abstractions.
- Support arrays directly.
- Support extensible/custom datasets.
- Provide relevant transforms and batching.
- Add image and tabular conveniences without coupling the framework to one data format.

### Training/evaluation
- Provide a training engine or equivalent model training API.
- Support epochs, batches, loss, metrics, device selection, and useful progress information.
- Provide evaluation and inference paths.

### Persistence
- Save enough model state to reliably reconstruct a trained model.
- Load saved models without requiring the original training process.

### Backends
- Provide a backend/device abstraction.
- CPU is the baseline backend.
- CUDA is a first-class target and must use actual CUDA execution for supported operations.

### CLI
- Provide CLI entry points that call the same underlying framework services as the Python API.
- Planned commands include train, evaluate, predict, and benchmark.

### Benchmarking
- Provide reproducible timing/metadata for supported operations.
- Support CPU/CUDA comparisons where both implementations exist.

### Errors and observability
- Errors should identify invalid shapes, dtypes, configuration, data, devices, and resource problems clearly.
- Training should expose useful metrics such as loss, relevant task metrics, elapsed time, samples/sec, and device where applicable.

## Non-functional requirements
- Python-first developer experience.
- Modular, testable architecture.
- No paid/cloud requirement.
- Small-hardware development must remain practical.
- Avoid hidden magic and misleading capability claims.
- Public APIs should be documented when stabilized.
