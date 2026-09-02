"""`Sequential`: an ordered container that chains child modules."""

from __future__ import annotations

from ..exceptions import ModuleError
from ..tensor.tensor import Tensor
from .module import Module


class Sequential(Module):
    """Applies its child modules in construction order: `modules[-1](...modules[0](x))`.

    ```python
    model = Sequential(
        Conv2d(1, 4, 3),
        ReLU(),
        MaxPool2d(2),
        Flatten(),
        Linear(36, 10),
    )
    ```

    Children are registered exactly like any other `Module` attribute
    assignment (`self.<name> = child`, via `Module.__setattr__`) under the
    deterministic names `"0"`, `"1"`, `"2"`, ... in construction order --
    no separate container/registration mechanism. Because `Module._modules`
    is an ordinary `dict` (insertion-ordered), `named_children()`/
    `children()`/`named_modules()`/`modules()`/`parameters()`/
    `named_parameters()`, `train()`/`eval()`, and `Module.to()` all already
    traverse `Sequential`'s children correctly with no overrides needed here
    beyond `forward()` itself.

    Every positional argument must be a `Module` -- anything else raises
    `ModuleError` immediately, before any child is registered, so a
    `Sequential(...)` call either fully succeeds or registers nothing.

    `Sequential()` with zero modules is explicitly supported: it is the
    identity function (`forward(x)` returns `x` unchanged), matching the
    common convention (e.g. PyTorch) that an empty container composes to a
    no-op rather than being rejected.
    """

    def __init__(self, *modules: Module):
        super().__init__()
        for i, m in enumerate(modules):
            if not isinstance(m, Module):
                raise ModuleError(
                    f"Sequential requires Module arguments, got {type(m).__name__} "
                    f"at position {i}."
                )
            setattr(self, str(i), m)

    def forward(self, x: Tensor) -> Tensor:
        for _, child in self.named_children():
            x = child(x)
        return x

    def __repr__(self) -> str:
        inner = ", ".join(repr(child) for _, child in self.named_children())
        return f"Sequential({inner})"


__all__ = ["Sequential"]
