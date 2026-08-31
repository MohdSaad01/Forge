# Forge Scope

## In scope
- Tensor abstraction and numerical operations.
- Automatic differentiation.
- Neural-network modules/models.
- Losses and optimizers.
- Dataset/data-loader abstractions and relevant preprocessing.
- Training/evaluation/inference.
- Model persistence.
- CPU backend.
- CUDA backend for supported operations.
- CLI.
- Benchmarks and performance reporting.
- Documentation and tests.

## Out of scope
- Chatbots, agents, RAG applications, or API wrappers.
- Wrapping PyTorch/TensorFlow as the core implementation.
- Cloud ML infrastructure.
- Distributed training.
- Large-model training.
- Enterprise-scale orchestration.
- Pretending unsupported hardware/features work.

## Scope rule
Future capabilities may influence interfaces, but implementation should prioritize representative, working functionality over breadth.
