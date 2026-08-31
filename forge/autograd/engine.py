"""Reverse-mode autodiff graph engine.

A ``Node`` is the ``grad_fn`` attached to a non-leaf Tensor: it remembers the
input tensors an operation was applied to and how to turn an upstream
gradient into gradients for those inputs. ``run_backward`` walks the graph
reachable from a root tensor in reverse topological order, accumulating
gradients into leaves.

This module works with Tensor-like objects via duck typing (``is_leaf``,
``requires_grad``, ``_grad_fn``, ``_accumulate_grad``) rather than importing
``forge.tensor.Tensor`` directly, so the graph engine has no dependency on
the concrete Tensor implementation and there is no import cycle between the
two packages.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


class Node:
    """One differentiable operation's backward rule, attached as a Tensor's ``grad_fn``."""

    __slots__ = ("inputs", "backward_fn", "name")

    def __init__(
        self,
        inputs: "tuple[Any, ...]",
        backward_fn: "Callable[[np.ndarray], tuple[np.ndarray | None, ...]]",
        name: str,
    ):
        self.inputs = inputs
        self.backward_fn = backward_fn
        self.name = name


def _topological_order(root: Any) -> list:
    """Dependency-ordered (inputs-before-outputs) list of tensors reachable from root."""
    visited: set[int] = set()
    order: list = []

    def visit(tensor: Any) -> None:
        if id(tensor) in visited:
            return
        visited.add(id(tensor))
        node = tensor._grad_fn
        if node is not None:
            for inp in node.inputs:
                visit(inp)
        order.append(tensor)

    visit(root)
    return order


def run_backward(root: Any, grad_array: np.ndarray) -> None:
    """Propagate ``grad_array`` (the upstream gradient for ``root``) through the graph.

    Each non-leaf tensor's ``grad_fn`` is dropped once its gradient has been
    consumed, releasing the graph rather than keeping it alive indefinitely.
    """
    order = _topological_order(root)
    pending: dict[int, np.ndarray] = {id(root): grad_array}

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
                pending[id(inp)] = pending[id(inp)] + grad
            else:
                pending[id(inp)] = grad

        tensor._grad_fn = None
