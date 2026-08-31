# Forge Use Cases

## UC1 — Train a classifier
A developer loads labeled data, defines a model, trains it, evaluates accuracy/loss, and performs predictions.

## UC2 — Train a regression model
A developer supplies numeric features and targets, defines a network, trains it, evaluates a regression metric, and predicts numeric outputs.

## UC3 — Image workload
A developer loads a small labeled image dataset, applies transforms, trains an appropriate model, evaluates it, and predicts classes.

## UC4 — Custom dataset
A developer implements a dataset abstraction for data not covered by built-in loaders and feeds it into the normal batching/training path.

## UC5 — CPU/CUDA execution
A developer runs the same high-level model workflow on CPU or CUDA where supported and can compare results/performance.

## UC6 — Persistence
A developer saves a trained model, exits the process, reloads it later, and performs inference.

## UC7 — Benchmark
A developer benchmarks a supported operation or model workload and receives timing plus hardware/backend metadata.
