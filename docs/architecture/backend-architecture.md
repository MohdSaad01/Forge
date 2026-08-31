# Forge Backend Architecture

## Goal
Allow CPU and CUDA execution behind a stable tensor/framework API.

## Boundary

```text
Tensor / Operation API
        ↓
Backend dispatch
   ┌────┴────┐
 CPU       CUDA
```

The backend boundary should answer:
- Which device owns this tensor?
- Which implementation performs this operation?
- What dtype/shape combinations are supported?
- How are device transfers performed?
- What error is raised for unsupported combinations?

## CPU
CPU is the reference backend for correctness and testability.

## CUDA
CUDA is a real execution backend, not a label. Initial CUDA support should target a small, measured operation set and the verified 940MX environment. Kernel/toolchain choices should be compatible with Compute Capability 5.0 unless a deliberate compatibility decision changes this.

## Consistency
For operations implemented on both backends, tests should compare CPU and CUDA results using appropriate tolerances.

## Performance
Do not optimize every abstraction prematurely. Establish correctness first, benchmark, then optimize measured hot paths.
