"""Reverse-mode autodiff graph engine.

A ``Node`` is the ``grad_fn`` attached to a non-leaf Tensor: it remembers the
input tensors an operation was applied to and how to turn an upstream
gradient into gradients for those inputs. ``run_backward`` walks the graph
reachable from a root tensor in reverse topological order, accumulating
gradients into leaves.

This module works with Tensor-like objects via duck typing (``is_leaf``,
``requires_grad``, ``device``, ``_grad_fn``, ``_accumulate_grad``) rather
than importing ``forge.tensor.Tensor`` directly, so the graph engine has no
dependency on the concrete Tensor implementation and there is no import
cycle between the two packages. As of Milestone 10, a ``Node``'s
``backward_fn`` itself is backend-aware -- see
``docs/architecture/autograd.md``'s **Backend-aware backward dispatch**
section -- but the graph engine here stays backend-*agnostic*: it never
performs numerical work itself, it only combines whatever gradient values
``backward_fn`` returns (a raw ``numpy.ndarray`` for a CPU tensor, a
``CUDAStorage`` for a CUDA tensor) via ``get_backend(tensor.device).add()``
when more than one consumer contributes to the same tensor's gradient.
"""

from __future__ import annotations

from typing import Any, Callable


class Node:
    """One differentiable operation's backward rule, attached as a Tensor's ``grad_fn``."""

    # `__weakref__` is included (rather than the bare 3-tuple) so lifetime
    # diagnostics/tests can hold a `weakref.ref(node)` to confirm a Node
    # became unreachable without keeping it alive themselves -- see
    # `docs/architecture/autograd.md`'s **Graph teardown and object
    # lifetime** section. Costs one extra pointer per Node, no behavioral
    # change.
    __slots__ = ("inputs", "backward_fn", "name", "__weakref__")

    def __init__(
        self,
        inputs: "tuple[Any, ...]",
        backward_fn: "Callable[[Any], tuple[Any, ...]]",
        name: str,
    ):
        self.inputs = inputs
        self.backward_fn = backward_fn
        self.name = name


def _topological_order(root: Any) -> list:
    """Dependency-ordered (inputs-before-outputs) list of tensors reachable from root.

    Iterative (explicit-stack) post-order DFS -- deliberately not a recursive
    nested function. A recursive `def visit(tensor): ... visit(inp) ...`
    closure that calls itself by name captures a cell referencing its own
    function object (`visit.__closure__` holds a cell whose contents is
    `visit` itself): a genuine Python reference cycle, uncollectible by
    reference counting alone. Because that closure also closes over `order`
    (this function's whole result-so-far) and `visited`, the cycle keeps
    every Tensor already appended to `order` alive until the next
    `gc.collect()` -- this was Forge's M22-discovered cycle (see
    `docs/architecture/autograd.md`'s **Graph teardown and object lifetime**
    section). Each stack entry is `(tensor, expanded)`: an unexpanded entry
    pushes its own post-order marker (`expanded=True`) followed by its
    not-yet-visited `node.inputs` in reverse (so a LIFO stack pops them in
    forward order), reproducing the original recursive traversal's ordering
    and memoization exactly, with no nested function and therefore no
    self-referential closure.
    """
    visited: set[int] = set()
    order: list = []
    stack: list = [(root, False)]

    while stack:
        tensor, expanded = stack.pop()
        if expanded:
            order.append(tensor)
            continue
        tid = id(tensor)
        if tid in visited:
            continue
        visited.add(tid)
        stack.append((tensor, True))
        node = tensor._grad_fn
        if node is not None:
            for inp in reversed(node.inputs):
                if id(inp) not in visited:
                    stack.append((inp, False))

    return order


def run_backward(root: Any, grad_array: Any) -> None:
    """Propagate ``grad_array`` (the upstream gradient for ``root``) through the graph.

    Each non-leaf tensor's ``grad_fn`` is dropped once its gradient has been
    consumed, releasing the graph rather than keeping it alive indefinitely.
    """
    # Deferred import: `forge.backend` never imports `forge.autograd`, so
    # this has nowhere to cycle back to; kept import-local (rather than at
    # module scope) purely to match the lazy-import style the rest of the
    # backend boundary already uses (see `forge/backend/__init__.py`).
    from ..backend import get_backend

    order = _topological_order(root)
    pending: dict[int, Any] = {id(root): grad_array}

    for tensor in reversed(order):
        grad_output = pending.pop(id(tensor), None)
        if grad_output is None:
            continue

        node = tensor._grad_fn
        if node is None:
            # A genuine leaf, or a non-leaf whose graph was already freed by
            # an earlier backward() call -- either way there is nothing
            # further to propagate through, so accumulate here and stop.
            if tensor.is_leaf:
                tensor._accumulate_grad(grad_output)
            continue

        input_grads = node.backward_fn(grad_output)
        for inp, grad in zip(node.inputs, input_grads):
            if grad is None or not inp.requires_grad:
                continue
            if id(inp) in pending:
                # Two consumers contributed a gradient to the same tensor:
                # combine them via that tensor's own backend rather than a
                # bare `+` (a CUDAStorage has no `__add__`; this reuses the
                # existing elementwise-add kernel instead of introducing a
                # second accumulation mechanism).
                pending[id(inp)] = get_backend(inp.device).add(pending[id(inp)], grad)
            else:
                pending[id(inp)] = grad

        tensor._grad_fn = None


# -- no_grad --------------------------------------------------------------

_grad_enabled = True


def is_grad_enabled() -> bool:
    """Whether differentiable Tensor operations currently attach a `grad_fn`."""
    return _grad_enabled


class no_grad:
    """Context manager that suspends autograd graph construction.

    A differentiable Tensor operation normally attaches a `Node` (`grad_fn`)
    to its output whenever any input requires grad (see
    `Tensor._differentiable_wrap`). Inside `no_grad()`, that check is
    skipped regardless of `requires_grad`, so operations still run and
    produce ordinary result values, but no `Node` is created and no graph
    is retained -- the minimal mechanism `Trainer.evaluate()` needs to run
    a forward pass it will never call `backward()` on without accumulating
    graph memory for it. It is one global flag, not a per-tensor or
    per-thread setting, and nests correctly (the previous state is restored
    on `__exit__`, so a nested `no_grad()` is a no-op rather than
    re-enabling grad early).
    """

    def __enter__(self) -> "no_grad":
        global _grad_enabled
        self._previous = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *exc_info: object) -> None:
        global _grad_enabled
        _grad_enabled = self._previous
