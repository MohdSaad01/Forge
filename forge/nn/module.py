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

from ..backend.device import Device
from ..exceptions import ModuleError
from ..tensor.tensor import Tensor
from .parameter import Parameter


class Module:
    """A composable neural-network component: parameters, child modules, and a forward pass."""

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "_training", True)

    # -- Attribute registration ------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        params = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        buffers = self.__dict__.get("_buffers")
        if params is None or modules is None or buffers is None:
            raise ModuleError(
                f"Cannot set attribute '{name}' on {type(self).__name__}: "
                "Module.__init__() (via super().__init__()) must be called "
                "before assigning parameters or child modules."
            )

        if isinstance(value, Parameter):
            self.__dict__.pop(name, None)
            modules.pop(name, None)
            buffers.pop(name, None)
            params[name] = value
            return
        if isinstance(value, Module):
            self.__dict__.pop(name, None)
            params.pop(name, None)
            buffers.pop(name, None)
            modules[name] = value
            return
        if name in buffers:
            # Reassigning an already-registered buffer (e.g. replacing its
            # Tensor wholesale, or `_build_load_node` restoring a saved
            # value) -- goes back into `_buffers`, not a plain instance
            # attribute, so `self.<name>` keeps resolving through
            # `__getattr__` below rather than shadowing it.
            if value is not None and (isinstance(value, Parameter) or not isinstance(value, Tensor)):
                raise ModuleError(
                    f"Cannot assign {type(value).__name__!r} to '{name}': it is a registered "
                    "buffer and only accepts a plain Tensor (or None)."
                )
            buffers[name] = value
            return

        # A plain (non-Parameter, non-Module, non-buffer) value assigned over
        # a name that used to hold one un-registers it, e.g. `self.bias = None`.
        params.pop(name, None)
        modules.pop(name, None)
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal attribute lookup fails, so this never
        # shadows a real instance/class attribute.
        params = self.__dict__.get("_parameters", {})
        if name in params:
            return params[name]
        buffers = self.__dict__.get("_buffers", {})
        if name in buffers:
            return buffers[name]
        modules = self.__dict__.get("_modules", {})
        if name in modules:
            return modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def register_buffer(self, name: str, tensor: "Tensor | None") -> None:
        """Register `name` as a non-trainable, persistent piece of Module state.

        A buffer (e.g. `nn.BatchNorm2d`'s `running_mean`/`running_var`) is a
        plain `Tensor` -- never a `Parameter` -- that `Module` tracks
        alongside parameters/child modules: it moves with `.to(device)`
        (Milestone 9's existing mechanism, since `Tensor._move_storage_` is
        already defined generically on `Tensor`, not `Parameter`), is saved
        and restored by `forge.save_model`/`save_checkpoint`
        (`forge/serialization/model.py`), and survives `train()`/`eval()`
        transitions untouched (neither method touches `_buffers`). It never
        appears in `named_parameters()`/`parameters()` and never receives a
        gradient: nothing in Forge ever includes a buffer in an autograd
        `Node`'s `inputs`, the same "excluded from the graph entirely"
        convention `Tensor.cross_entropy()`'s integer `target` already uses.

        Registration is explicit (unlike `Parameter`/`Module`, which
        auto-register by `isinstance` in `__setattr__`) because a bare
        `Tensor` is not an unambiguous signal of intent -- a module may
        legitimately want a plain, unregistered `Tensor` attribute that
        should not move/persist/traverse (e.g. a cached constant). `tensor`
        may be `None` (a placeholder, mirroring `Conv2d`'s optional `bias`
        pattern) or any `Tensor` with `requires_grad=False`.
        """
        if tensor is not None:
            if isinstance(tensor, Parameter):
                raise ModuleError(
                    f"register_buffer('{name}', ...) received a Parameter; buffers must be "
                    "plain Tensors, distinguishable from trainable Parameters."
                )
            if not isinstance(tensor, Tensor):
                raise ModuleError(
                    f"register_buffer('{name}', ...) requires a Tensor or None, got "
                    f"{type(tensor).__name__}."
                )
            if tensor.requires_grad:
                raise ModuleError(
                    f"register_buffer('{name}', ...) requires a Tensor with requires_grad=False, "
                    "since buffers never participate in autograd."
                )
        self.__dict__.pop(name, None)
        self._parameters.pop(name, None)
        self._modules.pop(name, None)
        self._buffers[name] = tensor

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

    def named_buffers(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        """Yield `(dotted_name, tensor)` for every registered, non-`None` buffer.

        Mirrors `named_parameters()` exactly (dotted names, first-discovered
        de-duplication for a buffer shared by identity across more than one
        attribute), but walks `_buffers` instead of `_parameters`. A buffer
        registered as `None` (e.g. an optional buffer no consumer has set
        yet) is never yielded, matching how an unset optional `Parameter`
        (`self.bias = None`) is never in `_parameters` either.
        """
        yield from self._named_buffers(prefix, set())

    def _named_buffers(self, prefix: str, seen: set[int]) -> Iterator[tuple[str, Tensor]]:
        for name, buf in self._buffers.items():
            if buf is None or id(buf) in seen:
                continue
            seen.add(id(buf))
            yield (name if not prefix else f"{prefix}.{name}"), buf
        for name, module in self._modules.items():
            child_prefix = name if not prefix else f"{prefix}.{name}"
            yield from module._named_buffers(child_prefix, seen)

    def buffers(self) -> Iterator[Tensor]:
        """Yield every registered, non-`None` buffer owned by this module or a descendant."""
        for _, buf in self.named_buffers():
            yield buf

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

    # -- Device movement (Milestone 9) ---------------------------------------

    def to(self, device: "str | Device") -> "Module":
        """Move every Parameter and buffer owned by this module (recursively) to `device`, in place.

        Mutates and returns `self` -- the same convention `train()`/`eval()`
        already use -- rather than constructing a copy of the module tree;
        see `docs/architecture/modules.md` for the full rationale. Every
        `Parameter`/buffer keeps its Python identity, shape, and dtype; only
        its backing storage device changes, via `Tensor._move_storage_()`
        (the in-place counterpart of `Tensor.to()`) -- already defined
        generically on `Tensor`, so a buffer (a plain, non-Parameter Tensor)
        moves through the exact same call a `Parameter` does. No autograd
        graph is built or altered by this call -- every `Parameter` is
        already a leaf, and stays one; a buffer never had one gradient state
        to preserve since it never requires grad. Any previously accumulated
        `.grad` is cleared for a `Parameter` (see `Tensor._move_storage_`'s
        docstring); a buffer has no `.grad` to clear. Raises
        `UnsupportedDeviceError` for an unrecognized device string, exactly
        like `Tensor.to()`.
        """
        target = Device.parse(device)
        for module in self.modules():
            for param in module._parameters.values():
                param._move_storage_(target)
            for buf in module._buffers.values():
                if buf is not None:
                    buf._move_storage_(target)
        return self

    @property
    def device(self) -> "Device | None":
        """The single device shared by every Parameter owned by this module tree.

        `None` if this module (and every descendant) owns no Parameters --
        Forge does not assume every Module has one device (e.g. a bare
        `ReLU`). Raises `ModuleError` if Parameters are found on more than
        one device: normal usage (construct on one device, or call
        `.to(device)`) always leaves a coherent tree, so this signals a
        manually-assembled inconsistency (e.g. reassigning one child's
        Parameter to a different device by hand) rather than silently
        picking one device to report.
        """
        devices = {param.device for param in self.parameters()}
        if not devices:
            return None
        if len(devices) > 1:
            raise ModuleError(
                f"{type(self).__name__} has Parameters on inconsistent devices: "
                f"{sorted(str(d) for d in devices)}. Call .to(device) to make them coherent."
            )
        return next(iter(devices))

    # -- Invocation -----------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise ModuleError(f"{type(self).__name__} does not implement forward().")

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


__all__ = ["Module"]
