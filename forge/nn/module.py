"""The composable neural-network base class.

`Module` owns two dictionaries -- parameters and child modules -- populated
automatically as attributes are assigned (`self.fc1 = Linear(...)`,
`self.weight = Parameter(...)`). Recursive discovery (`parameters()`,
`named_parameters()`, `modules()`) walks those dictionaries; nothing here
touches gradients directly, that is entirely Tensor/autograd's job (see
`docs/architecture/modules.md`).
"""

from __future__ import annotations

from typing import Any, Iterator

from ..exceptions import ModuleError
from .parameter import Parameter


class Module:
    """A composable neural-network component: parameters, child modules, and a forward pass."""

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_training", True)

    # -- Attribute registration ------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        params = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        if params is None or modules is None:
            raise ModuleError(
                f"Cannot set attribute '{name}' on {type(self).__name__}: "
                "Module.__init__() (via super().__init__()) must be called "
                "before assigning parameters or child modules."
            )

        if isinstance(value, Parameter):
            self.__dict__.pop(name, None)
            modules.pop(name, None)
            params[name] = value
            return
        if isinstance(value, Module):
            self.__dict__.pop(name, None)
            params.pop(name, None)
            modules[name] = value
            return

        # A plain (non-Parameter, non-Module) value assigned over a name that
        # used to hold one un-registers it, e.g. `self.bias = None`.
        params.pop(name, None)
        modules.pop(name, None)
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal attribute lookup fails, so this never
        # shadows a real instance/class attribute.
        params = self.__dict__.get("_parameters", {})
        if name in params:
            return params[name]
        modules = self.__dict__.get("_modules", {})
        if name in modules:
            return modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # -- Discovery ----------------------------------------------------------

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        """Yield `(dotted_name, parameter)` for this module and all descendants.

        A parameter referenced by more than one attribute/module is yielded
        only once, at the first name it is discovered under.
        """
        yield from self._named_parameters(prefix, set())

    def _named_parameters(self, prefix: str, seen: set[int]) -> Iterator[tuple[str, Parameter]]:
        for name, param in self._parameters.items():
            if id(param) in seen:
                continue
            seen.add(id(param))
            yield (name if not prefix else f"{prefix}.{name}"), param
        for name, module in self._modules.items():
            child_prefix = name if not prefix else f"{prefix}.{name}"
            yield from module._named_parameters(child_prefix, seen)

    def parameters(self) -> Iterator[Parameter]:
        """Yield every trainable parameter owned by this module or a descendant."""
        for _, param in self.named_parameters():
            yield param

    def named_children(self) -> Iterator[tuple[str, "Module"]]:
        """Yield `(name, module)` for this module's immediate children only."""
        yield from self._modules.items()

    def children(self) -> Iterator["Module"]:
        for _, module in self._modules.items():
            yield module

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, "Module"]]:
        """Yield `(dotted_name, module)` for this module and every descendant, self first."""
        yield from self._named_modules(prefix, set())

    def _named_modules(self, prefix: str, seen: set[int]) -> Iterator[tuple[str, "Module"]]:
        if id(self) not in seen:
            seen.add(id(self))
            yield prefix, self
        for name, module in self._modules.items():
            child_prefix = name if not prefix else f"{prefix}.{name}"
            yield from module._named_modules(child_prefix, seen)

    def modules(self) -> Iterator["Module"]:
        for _, module in self.named_modules():
            yield module

    # -- Training / evaluation mode -----------------------------------------

    @property
    def training(self) -> bool:
        return self._training

    def train(self, mode: bool = True) -> "Module":
        """Set this module and every descendant to training (`mode=True`) or eval mode."""
        object.__setattr__(self, "_training", mode)
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> "Module":
        """Equivalent to `self.train(False)`."""
        return self.train(False)

    # -- Invocation -----------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise ModuleError(f"{type(self).__name__} does not implement forward().")

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


__all__ = ["Module"]
