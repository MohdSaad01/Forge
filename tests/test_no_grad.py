"""Tests for `forge.no_grad()`, the minimal evaluation-time autograd extension.

`no_grad()` is a single global flag checked by
`Tensor._differentiable_wrap`: inside the context, differentiable
operations produce plain, non-grad-requiring results and attach no
`grad_fn`, regardless of whether their inputs require grad. See
`docs/architecture/training-engine.md`.
"""

from __future__ import annotations

import forge
from forge import Tensor
from forge.autograd import is_grad_enabled, no_grad


def test_grad_enabled_by_default():
    assert is_grad_enabled() is True


def test_operation_inside_no_grad_does_not_require_grad():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    with no_grad():
        y = x * 2
    assert y.requires_grad is False
    assert y.grad_fn is None
    assert y.is_leaf is True


def test_operation_outside_no_grad_still_requires_grad():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2
    assert y.requires_grad is True
    assert y.grad_fn is not None


def test_no_grad_is_scoped_and_restores_previous_state():
    x = Tensor([1.0], requires_grad=True)
    with no_grad():
        assert is_grad_enabled() is False
    assert is_grad_enabled() is True
    y = x * 2
    assert y.requires_grad is True


def test_no_grad_restores_state_even_if_body_raises():
    try:
        with no_grad():
            raise ValueError("boom")
    except ValueError:
        pass
    assert is_grad_enabled() is True


def test_nested_no_grad_does_not_reenable_early():
    with no_grad():
        with no_grad():
            pass
        assert is_grad_enabled() is False
    assert is_grad_enabled() is True


def test_forge_top_level_alias():
    assert forge.no_grad is no_grad
