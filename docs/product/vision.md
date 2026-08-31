# Forge Product Vision

## Purpose
Forge is a general deep-learning framework for constructing, training, evaluating, saving, loading, and executing neural-network models.

## Vision
Build a usable framework that combines approachable model development, transparent training behavior, and progressively optimized CPU/CUDA execution.

## Primary user
A Python developer who wants to define neural networks and training workflows through Forge's APIs without depending on PyTorch or TensorFlow for the core learning machinery.

## Core workflow
Dataset → preprocessing → batches → model → forward pass → loss → autograd/backpropagation → optimizer → training → evaluation → persistence → inference.

## Supported workload direction
Forge is general-purpose. Classification, regression, image workloads, and additional neural-network tasks should emerge from reusable primitives rather than separate application-specific systems.

## Success
A developer can use Forge to build and train real small models, evaluate them, persist them, reload them, perform inference, and execute supported workloads on CPU and CUDA where available.
