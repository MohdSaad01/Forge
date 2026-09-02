"""Small int-or-(height, width)-pair parsing shared by `Conv2d` and `MaxPool2d`."""

from __future__ import annotations

from ..exceptions import ShapeMismatchError


def pair(value, name: str) -> "tuple[int, int]":
    """Parse a positive int (applied to both dims) or a (height, width) pair of positive ints."""
    if isinstance(value, bool):
        raise ShapeMismatchError(f"{name} must be an int or a (height, width) pair of ints, got {value!r}.")
    if isinstance(value, int):
        if value <= 0:
            raise ShapeMismatchError(f"{name} must be positive, got {value}.")
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        h, w = value
        if isinstance(h, bool) or isinstance(w, bool) or not (isinstance(h, int) and isinstance(w, int)) or h <= 0 or w <= 0:
            raise ShapeMismatchError(f"{name} must be a pair of positive integers, got {value!r}.")
        return (int(h), int(w))
    raise ShapeMismatchError(f"{name} must be an int or a (height, width) pair of ints, got {value!r}.")


def pad_pair(value, name: str) -> "tuple[int, int]":
    """Parse a non-negative int (applied to both dims) or a (height, width) pair of non-negative ints."""
    if isinstance(value, bool):
        raise ShapeMismatchError(f"{name} must be an int or a (height, width) pair of ints, got {value!r}.")
    if isinstance(value, int):
        if value < 0:
            raise ShapeMismatchError(f"{name} must be non-negative, got {value}.")
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        h, w = value
        if isinstance(h, bool) or isinstance(w, bool) or not (isinstance(h, int) and isinstance(w, int)) or h < 0 or w < 0:
            raise ShapeMismatchError(f"{name} must be a pair of non-negative integers, got {value!r}.")
        return (int(h), int(w))
    raise ShapeMismatchError(f"{name} must be an int or a (height, width) pair of ints, got {value!r}.")


__all__ = ["pair", "pad_pair"]
