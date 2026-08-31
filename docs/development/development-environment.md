# Forge Development Environment

## Hardware

### CPU
Intel Core i5-7200U

### RAM
8 GB

### GPU
NVIDIA GeForce 940MX
- VRAM: 2 GB
- Compute Capability: 5.0

### Storage
240 GB SSD
~95 GB available at baseline

## Software

### OS
Windows 10

### Python
Python 3.13.5

### CUDA
CUDA Toolkit 12.6
nvcc 12.6.85

### NVIDIA Driver
582.53

### Compiler
Visual Studio 2022 / MSVC 19.44

## CUDA Verification
The development machine has successfully compiled and executed an
`sm_50` CUDA kernel on the 940MX.

Therefore, CUDA development is considered technically feasible on
the primary development machine.

## Development Constraint
Development workloads should remain appropriate for 8 GB RAM and
2 GB VRAM. Forge is not expected to train large modern models on
this machine.