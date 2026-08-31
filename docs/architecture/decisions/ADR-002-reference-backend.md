# ADR-002: CPU as Reference Backend

## Status
Accepted

## Decision
Treat CPU execution as the reference semantic implementation and use it for the majority of deterministic correctness tests. CUDA implementations must satisfy the same operation contracts within documented floating-point tolerances.

## Rationale
The development GPU is constrained and CUDA debugging is more expensive. A trustworthy CPU reference makes backend development and regression testing practical.

## Consequences
CUDA tests are additive and hardware-aware; absence of CUDA must not make the CPU test suite unusable.
