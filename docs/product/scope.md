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

## Feature freeze (Forge 1.0.0)
The implementation is feature-frozen; only maintenance changes follow (`docs/development/maintenance.md`). The following are explicitly not part of Forge and are not planned: Transformer/attention layers, 3D convolution, multi-GPU or distributed training, mixed precision, ONNX or other framework interop, a serving platform, a model registry, hyperparameter search, cloud training, a categorical-data framework, automatic ID-column detection, a pandas requirement, and automatic missing-value imputation.

## Scope rule
Future capabilities may influence interfaces, but implementation should prioritize representative, working functionality over breadth.
