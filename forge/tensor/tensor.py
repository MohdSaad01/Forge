"""The Forge Tensor abstraction.

A Tensor is a typed, device-associated, multidimensional numerical value
that can optionally track gradient information. Differentiable operations
attach a ``forge.autograd.Node`` (the tensor's ``grad_fn``) recording enough
information to compute input gradients; ``Tensor.backward()`` triggers
reverse-mode traversal of that graph. See ``docs/architecture/autograd.md``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.exceptions import AxisError

from ..autograd import Node, is_grad_enabled, run_backward
from ..backend import get_backend
from ..backend.device import Device
from ..exceptions import (
    GradientStateError,
    ShapeMismatchError,
    UnsupportedDeviceError,
    UnsupportedDTypeError,
)
from .dtype import DType, dtype_from_numpy, infer_default_dtype, resolve_dtype

Shape = tuple[int, ...]

_GRAD_CAPABLE_DTYPES = (DType.FLOAT32, DType.FLOAT64)


class Tensor:
    """A typed multidimensional numerical value associated with a device."""

    # Milestone 29: set (only by `Tensor.to(..., non_blocking=True)`'s D2H
    # path) to a `forge.backend.cuda.transfer.PendingTransfer` while a CPU
    # tensor's data has been submitted for an asynchronous device-to-host
    # copy that may not have completed yet. `None` (the class-level default,
    # true for every CPU/CUDA tensor built any other way) means "nothing to
    # wait for" -- see the `_data` property below, the single chokepoint that
    # actually waits on it.
    _pending: "Any | None" = None

    def __init__(
        self,
        data: Any,
        dtype: Any = None,
        device: "str | Device" = "cpu",
        requires_grad: bool = False,
    ):
        self._device = Device.parse(device)
        backend = get_backend(self._device)

        if isinstance(data, Tensor):
            if data._device != self._device:
                raise UnsupportedDeviceError(
                    f"Cannot construct a Tensor on device '{self._device}' directly from a "
                    f"Tensor already on device '{data._device}'. Call "
                    f"tensor.to('{self._device}') to move it explicitly first."
                )
            data = data._data

        np_dtype = resolve_dtype(dtype)
        if np_dtype is None and not hasattr(data, "dtype"):
            # Raw Python data (lists/scalars) has no dtype of its own to
            # preserve, so apply Forge's stable default instead of NumPy's
            # platform-dependent inference. Anything that already carries a
            # concrete dtype (a NumPy array/scalar, or another backend's
            # storage, e.g. CUDAStorage) keeps it instead.
            np_dtype = infer_default_dtype(np.asarray(data).dtype)
        try:
            array = backend.from_array(data, np_dtype)
        except (TypeError, ValueError) as exc:
            raise UnsupportedDTypeError(
                f"Could not build a tensor from the given data: {exc}"
            ) from exc

        self._dtype = dtype_from_numpy(array.dtype)
        self._data = array

        self._requires_grad = bool(requires_grad)
        if self._requires_grad and self._dtype not in _GRAD_CAPABLE_DTYPES:
            raise GradientStateError(
                f"requires_grad=True requires a floating-point dtype, got '{self._dtype}'."
            )
        self._grad_fn: "Node | None" = None
        self._is_leaf = True
        self.grad: "Tensor | None" = None

    @classmethod
    def _wrap(
        cls,
        array: np.ndarray,
        device: Device,
        requires_grad: bool = False,
        grad_fn: "Node | None" = None,
    ) -> "Tensor":
        """Build a Tensor directly from backend storage, skipping re-validation."""
        instance = object.__new__(cls)
        instance._data = array
        instance._dtype = dtype_from_numpy(array.dtype)
        instance._device = device
        instance._requires_grad = requires_grad
        instance._grad_fn = grad_fn
        instance._is_leaf = grad_fn is None
        instance.grad = None
        return instance

    def _differentiable_wrap(
        self,
        array: np.ndarray,
        inputs: "tuple[Tensor, ...]",
        backward_fn,
        name: str,
    ) -> "Tensor":
        """Wrap a forward result, attaching a grad_fn if any input requires grad.

        Skipped entirely inside `forge.no_grad()` (`is_grad_enabled()` is
        `False`): the result is a plain, non-grad-requiring Tensor regardless
        of its inputs, so no graph is built for a forward pass that will
        never be differentiated.

        As of Milestone 10, this is not restricted to the 'cpu' device: any
        device whose `Backend` implements the operation's forward method
        (already called by the caller, above) and its `*_backward`
        counterpart (see `forge/backend/base.py`) can attach a `grad_fn`
        here. An operation with no CUDA backward support (`exp`/`log`) never
        reaches this point on a CUDA tensor in the first place -- its
        forward call already raised `CUDAError` before `_differentiable_wrap`
        was invoked, which is the more localized place to fail (see
        `docs/architecture/cuda-backend.md`).
        """
        requires_grad = is_grad_enabled() and any(t._requires_grad for t in inputs)
        grad_fn = Node(inputs, backward_fn, name) if requires_grad else None
        return Tensor._wrap(array, self._device, requires_grad=requires_grad, grad_fn=grad_fn)

    # -- Backend storage access (Milestone 29: pending-transfer sync gate) ------

    @property
    def _data(self) -> Any:
        """The underlying backend storage -- a NumPy array (CPU) or `CUDAStorage` (CUDA).

        A property, not a plain attribute, specifically so `self._pending`
        (set only by `Tensor.to(..., non_blocking=True)`'s device-to-host
        path) has exactly one place to enforce Invariant 4 ("host access
        cannot observe an incomplete D2H transfer"): every existing method
        that reads `self._data` -- `.numpy()`, `__repr__`, every binary/
        reduction op's forward and backward, `backward()`, persistence via
        `Backend.to_numpy(param._data)` -- already passes through here
        unchanged, so none of them needed to learn about pending transfers
        individually. The wait is targeted (only this tensor's own transfer,
        via `PendingTransfer.synchronize()` -- never a device-wide
        synchronization) and paid at most once: `self._pending` is cleared
        immediately after, so a second read is a plain attribute access.
        """
        if self._pending is not None:
            self._pending.synchronize()
            self._pending = None
            # Detach from the transfer's pinned buffer now that it is safe to
            # read: this tensor's *permanent* storage becomes an ordinary
            # pageable copy, so nothing derived from it (further CPU ops,
            # anything this array gets sliced/added/reshaped into) keeps a
            # potentially large pinned allocation alive any longer than the
            # transfer itself needed -- pinned memory is a transfer-time
            # optimization, never permanent Tensor state (Section 21/27/45 of
            # the milestone brief; see docs/architecture/cuda-transfers.md).
            self._storage = np.array(self._storage, copy=True)
        return self._storage

    @_data.setter
    def _data(self, value: Any) -> None:
        self._storage = value
        self._pending = None

    # -- Introspection --------------------------------------------------

    @property
    def shape(self) -> Shape:
        return self._data.shape

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def device(self) -> Device:
        return self._device

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @property
    def is_leaf(self) -> bool:
        """True for tensors not produced by a tracked differentiable operation."""
        return self._is_leaf

    @property
    def grad_fn(self) -> "Node | None":
        return self._grad_fn

    def numpy(self) -> np.ndarray:
        """Return the underlying CPU storage as a NumPy array (shares memory)."""
        if self._device.type != "cpu":
            raise UnsupportedDeviceError(
                f"Cannot access NumPy storage for a tensor on device '{self._device}'."
            )
        return self._data

    def __repr__(self) -> str:
        extra = ", requires_grad=True" if self._requires_grad else ""
        return f"Tensor({self._data!r}, dtype={self._dtype}, device={self._device}{extra})"

    def __len__(self) -> int:
        return len(self._data)

    # -- Elementwise binary ops ------------------------------------------

    def _coerce(self, other: Any) -> "Tensor":
        if isinstance(other, Tensor):
            return other
        return Tensor(other, device=self._device)

    def _binary_op(self, other: Any, backend_method: str, op_symbol: str) -> "Tensor":
        """Apply a broadcasting elementwise op.

        The backward rule is looked up on the same backend as the forward
        call (`backend.<backend_method>_backward`, e.g. `backend.add_backward`)
        so gradient computation for a CUDA operand executes as real CUDA
        kernels, never NumPy math against CUDA storage (see
        `docs/architecture/autograd.md`).
        """
        other_t = self._coerce(other)
        if other_t._device != self._device:
            raise UnsupportedDeviceError(
                f"Cannot apply '{op_symbol}' to tensors on different devices: "
                f"{self._device} and {other_t._device}."
            )
        try:
            np.broadcast_shapes(self.shape, other_t.shape)
        except ValueError as exc:
            raise ShapeMismatchError(
                f"Cannot apply '{op_symbol}' to shapes {self.shape} and {other_t.shape}: "
                "not broadcastable."
            ) from exc

        backend = get_backend(self._device)
        result = getattr(backend, backend_method)(self._data, other_t._data)

        a_data, b_data = self._data, other_t._data
        backward_method = getattr(backend, f"{backend_method}_backward")

        def backward_fn(grad_output):
            return backward_method(grad_output, a_data, b_data)

        return self._differentiable_wrap(result, (self, other_t), backward_fn, op_symbol)

    def __add__(self, other: Any) -> "Tensor":
        return self._binary_op(other, "add", "+")

    def __radd__(self, other: Any) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Tensor":
        return self._binary_op(other, "sub", "-")

    def __rsub__(self, other: Any) -> "Tensor":
        return self._coerce(other)._binary_op(self, "sub", "-")

    def __mul__(self, other: Any) -> "Tensor":
        return self._binary_op(other, "mul", "*")

    def __rmul__(self, other: Any) -> "Tensor":
        return self.__mul__(other)

    # -- Matrix multiplication -------------------------------------------

    def __matmul__(self, other: Any) -> "Tensor":
        other_t = self._coerce(other)
        if other_t._device != self._device:
            raise UnsupportedDeviceError(
                f"Cannot apply '@' to tensors on different devices: "
                f"{self._device} and {other_t._device}."
            )
        if self.ndim not in (1, 2) or other_t.ndim not in (1, 2):
            raise ShapeMismatchError(
                "matmul supports 1D and 2D tensors only, got shapes "
                f"{self.shape} and {other_t.shape}."
            )

        backend = get_backend(self._device)
        try:
            result = backend.matmul(self._data, other_t._data)
        except ValueError as exc:
            raise ShapeMismatchError(
                f"Cannot matrix-multiply shapes {self.shape} and {other_t.shape}: {exc}"
            ) from exc

        a_data, b_data = self._data, other_t._data

        def backward_fn(grad_output):
            return backend.matmul_backward(grad_output, a_data, b_data)

        return self._differentiable_wrap(result, (self, other_t), backward_fn, "@")

    # -- Reductions and reshaping -----------------------------------------

    def sum(self, axis: "int | tuple[int, ...] | None" = None, keepdims: bool = False) -> "Tensor":
        backend = get_backend(self._device)
        try:
            result = backend.sum(self._data, axis, keepdims)
        except AxisError as exc:
            raise ShapeMismatchError(
                f"Cannot sum over axis {axis} for tensor of shape {self.shape}: {exc}"
            ) from exc

        original_shape, ndim = self.shape, self.ndim

        def backward_fn(grad_output):
            return (backend.sum_backward(grad_output, original_shape, ndim, axis, keepdims),)

        return self._differentiable_wrap(result, (self,), backward_fn, "sum")

    def reshape(self, *shape: int) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        backend = get_backend(self._device)
        try:
            result = backend.reshape(self._data, shape)
        except ValueError as exc:
            raise ShapeMismatchError(
                f"Cannot reshape tensor of shape {self.shape} "
                f"({self._data.size} elements) to {shape}: {exc}"
            ) from exc

        original_shape = self.shape

        def backward_fn(grad_output):
            return (backend.reshape_backward(grad_output, original_shape),)

        return self._differentiable_wrap(result, (self,), backward_fn, "reshape")

    def relu(self) -> "Tensor":
        backend = get_backend(self._device)
        result = backend.relu(self._data)

        input_data = self._data

        def backward_fn(grad_output):
            # Dispatches through the backend (`Backend.relu_backward`) so a
            # CUDA input's mask/multiply runs as a real CUDA kernel -- see
            # `docs/architecture/cuda-backend.md`. Never copies `input_data`
            # to CPU to compute the mask.
            return (backend.relu_backward(grad_output, input_data),)

        return self._differentiable_wrap(result, (self,), backward_fn, "relu")

    def exp(self) -> "Tensor":
        backend = get_backend(self._device)
        result = backend.exp(self._data)

        def backward_fn(grad_output):
            # Dispatches through the backend (`Backend.exp_backward`), like
            # `relu`'s backward closure above, rather than a raw `grad_output
            # * result` -- a `CUDAStorage` has no `__mul__` of its own (see
            # `docs/architecture/cuda-backend.md`).
            return (backend.exp_backward(grad_output, result),)

        return self._differentiable_wrap(result, (self,), backward_fn, "exp")

    def log(self) -> "Tensor":
        backend = get_backend(self._device)
        result = backend.log(self._data)

        input_data = self._data

        def backward_fn(grad_output):
            return (backend.log_backward(grad_output, input_data),)

        return self._differentiable_wrap(result, (self,), backward_fn, "log")

    # -- Conv2d / MaxPool2d (Milestone 15) --------------------------------------

    def conv2d(
        self,
        weight: "Tensor",
        bias: "Tensor | None",
        stride: "tuple[int, int]",
        padding: "tuple[int, int]",
    ) -> "Tensor":
        """2D cross-correlation: `self` is NCHW, `weight` is (C_out, C_in, KH, KW).

        Dispatches to `Backend.conv2d`/`Backend.conv2d_backward` (CPU: a
        vectorized im2col-plus-matmul; CUDA: real kernels -- see
        `docs/architecture/cuda-backend.md`), following the same
        Tensor -> grad_fn -> Backend pattern as `matmul`. `bias` may be
        `None`; when given, it is folded into the same backend call (there is
        no general (N,C,H,W)+(C,) broadcasting primitive to compose a
        separate add from) and participates in the same `grad_fn` as
        `weight`.
        """
        if self.ndim != 4:
            raise ShapeMismatchError(
                f"conv2d expects a 4D (N, C_in, H, W) input, got shape {self.shape}."
            )
        if weight.ndim != 4:
            raise ShapeMismatchError(
                f"conv2d expects a 4D (C_out, C_in, KH, KW) weight, got shape {weight.shape}."
            )
        if weight._device != self._device:
            raise UnsupportedDeviceError(
                f"conv2d requires weight on the same device as input; got input on "
                f"'{self._device}' and weight on '{weight._device}'."
            )
        if bias is not None and bias._device != self._device:
            raise UnsupportedDeviceError(
                f"conv2d requires bias on the same device as input; got input on "
                f"'{self._device}' and bias on '{bias._device}'."
            )

        N, Cin, H, W = self.shape
        Cout, Cin_w, KH, KW = weight.shape
        if Cin != Cin_w:
            raise ShapeMismatchError(
                f"conv2d input has {Cin} channels but weight expects {Cin_w} "
                f"(weight shape {weight.shape})."
            )
        SH, SW = stride
        PH, PW = padding
        if KH > H + 2 * PH or KW > W + 2 * PW:
            raise ShapeMismatchError(
                f"conv2d kernel size ({KH}, {KW}) is larger than the padded input "
                f"({H + 2 * PH}, {W + 2 * PW})."
            )
        if bias is not None and bias.shape != (Cout,):
            raise ShapeMismatchError(
                f"conv2d bias must have shape ({Cout},), got {bias.shape}."
            )

        backend = get_backend(self._device)
        bias_data = bias._data if bias is not None else None
        result = backend.conv2d(self._data, weight._data, bias_data, (SH, SW), (PH, PW))

        x_data, w_data = self._data, weight._data

        def backward_fn(grad_output):
            grad_x, grad_w, grad_b = backend.conv2d_backward(
                grad_output, x_data, w_data, bias_data, (SH, SW), (PH, PW)
            )
            return (grad_x, grad_w, grad_b) if bias is not None else (grad_x, grad_w)

        inputs = (self, weight) if bias is None else (self, weight, bias)
        return self._differentiable_wrap(result, inputs, backward_fn, "conv2d")

    def max_pool2d(
        self,
        kernel_size: "tuple[int, int]",
        stride: "tuple[int, int]",
        padding: "tuple[int, int]",
    ) -> "Tensor":
        """2D max pooling over an NCHW tensor.

        Ties within a window break deterministically to the first maximum
        encountered in row-major (top-to-bottom, then left-to-right) scan
        order -- see `Backend.max_pool2d`'s CPU/CUDA implementations. Backward
        recomputes each window's argmax from the saved input `self`, the same
        "recompute from a saved input" convention `relu`/`exp`/`log` already
        use, rather than caching indices from the forward pass.
        """
        if self.ndim != 4:
            raise ShapeMismatchError(
                f"max_pool2d expects a 4D (N, C, H, W) input, got shape {self.shape}."
            )
        N, C, H, W = self.shape
        KH, KW = kernel_size
        SH, SW = stride
        PH, PW = padding
        if KH > H + 2 * PH or KW > W + 2 * PW:
            raise ShapeMismatchError(
                f"max_pool2d kernel size ({KH}, {KW}) is larger than the padded input "
                f"({H + 2 * PH}, {W + 2 * PW})."
            )

        backend = get_backend(self._device)
        result = backend.max_pool2d(self._data, (KH, KW), (SH, SW), (PH, PW))

        x_data = self._data

        def backward_fn(grad_output):
            return (backend.max_pool2d_backward(grad_output, x_data, (KH, KW), (SH, SW), (PH, PW)),)

        return self._differentiable_wrap(result, (self,), backward_fn, "max_pool2d")

    # -- Dropout (Milestone 16) ----------------------------------------------

    def dropout_mask(self, p: float, rng: np.random.Generator) -> "Tensor":
        """A fresh, non-differentiable mask matching this tensor's shape/dtype/device.

        `mask[i]` is `1/(1-p)` with probability `1-p`, `0` otherwise
        (inverted dropout -- the scaling is baked into the mask itself).
        `self`'s values are never read; only its shape/dtype/device are used.
        Dispatches to `Backend.dropout_mask`, which generates every element's
        randomness on this tensor's own device (CPU draws directly from
        `rng`; CUDA draws one integer seed from `rng` and generates the mask
        with a real on-device kernel -- see
        `docs/architecture/cuda-backend.md`'s **CUDA Dropout** section).

        `forge.nn.Dropout` composes this via ordinary multiplication
        (`x * x.dropout_mask(...)`) rather than a dedicated backward rule:
        the mask is a plain `requires_grad=False` leaf, so `mul`'s existing
        autograd already gives the correct forward (`x * mask`) and backward
        (`grad_output * mask`) behavior, and the same mask object -- not a
        freshly redrawn one -- is what backward multiplies against, since it
        is captured by `mul`'s own backward closure.
        """
        backend = get_backend(self._device)
        mask_array = backend.dropout_mask(self._data, p, rng)
        return Tensor._wrap(mask_array, self._device)

    # -- CrossEntropyLoss (Milestone 31) --------------------------------------

    def cross_entropy(self, target: "Tensor") -> "Tensor":
        """Fused mean cross-entropy: `-mean(log_softmax(self)[i, target[i]])`.

        `self` ("logits") must be 2D `(batch, classes)`; `target` must be an
        integer-dtype Tensor of shape `(batch,)` already on `self`'s device,
        holding class indices in `[0, classes)`. This is the fused primitive
        `forge.nn.CrossEntropyLoss` (`forge/nn/loss.py`) dispatches to after
        its own shape/dtype/range validation -- see that module's docstring
        and `Backend.cross_entropy`'s docstring (`backend/base.py`) for why
        this replaced a composed chain of Tensor primitives. `target` never
        requires grad (an integer dtype cannot -- see `_GRAD_CAPABLE_DTYPES`),
        so it is deliberately excluded from `inputs` below: it contributes no
        gradient and never needs an entry in the backward graph.
        """
        if self.ndim != 2:
            raise ShapeMismatchError(
                f"cross_entropy expects logits of shape (batch, classes), got shape {self.shape}."
            )
        if target.ndim != 1 or target.shape[0] != self.shape[0]:
            raise ShapeMismatchError(
                f"cross_entropy expects target of shape ({self.shape[0]},), got shape {target.shape}."
            )
        if target._device != self._device:
            raise UnsupportedDeviceError(
                "cross_entropy requires target on the same device as logits; got logits on "
                f"'{self._device}' and target on '{target._device}'."
            )

        backend = get_backend(self._device)
        result = backend.cross_entropy(self._data, target._data)

        logits_data, target_data = self._data, target._data

        def backward_fn(grad_output):
            return (backend.cross_entropy_backward(grad_output, logits_data, target_data),)

        return self._differentiable_wrap(result, (self,), backward_fn, "cross_entropy")

    # -- Device transfer -----------------------------------------------------

    def to(self, device: "str | Device", non_blocking: bool = False) -> "Tensor":
        """Explicitly move this tensor's data to another device, copying it.

        Never happens implicitly as a side effect of another operation (see
        `_binary_op`'s device-mismatch check) -- this is the one sanctioned
        way to cross devices. The result is always a fresh leaf tensor with
        `requires_grad=False`, regardless of this tensor's own gradient
        state: a device *transfer* is not itself a differentiable operation
        in Forge (there is no gradient rule for "copy across the host/device
        boundary"), independent of whether the source or target device can
        run autograd on its own -- CUDA can, as of Milestone 10 (see
        `docs/architecture/cuda-backend.md`), but `.to()` still never carries
        `requires_grad` across the copy, and never fakes it by quietly
        keeping the source tensor's graph alive.

        `non_blocking` (Milestone 29, default `False`) opts into asynchronous
        CPU<->CUDA transfer semantics -- see `docs/architecture/
        cuda-transfers.md` for the full contract. `False` (the default)
        preserves the exact Milestone 8-28 behavior unchanged: a fully
        host-synchronous copy, regardless of source/target device. `True`
        requires the transfer to actually be a `cpu<->cuda` direction
        (raises `UnsupportedDeviceError` otherwise, including for a CUDA<->
        CUDA or CPU<->CPU no-op -- the latter never reaches this check since
        `target == self._device` already returned above) and:

        - **CPU -> CUDA**: requires `self`'s data to already be pinned host
          memory (`forge.cuda.PinnedMemory(...).numpy()`) -- raises
          `forge.CUDAError` otherwise (Section 13's Option A: fail clearly,
          never silently stage pageable memory through a hidden pinned
          buffer). The transfer is submitted on the current Forge CUDA
          stream (`forge.cuda.current_stream()`) and does not block the host
          before returning.
        - **CUDA -> CPU**: always succeeds; the returned Tensor's backing
          data is a fresh pinned buffer that may not be fully written yet.
          The host is never blocked here -- instead, the very first read of
          the result (`.numpy()`, printing, arithmetic, `.backward()`, ...)
          transparently synchronizes just that transfer (see the `_data`
          property above).
        """
        target = Device.parse(device)
        if target == self._device:
            return self

        if not non_blocking:
            target_backend = get_backend(target)
            source_backend = get_backend(self._device)
            host_array = source_backend.to_numpy(self._data)
            new_storage = target_backend.from_array(host_array, host_array.dtype)
            return Tensor._wrap(new_storage, target, requires_grad=False)

        if self._device.type == "cpu" and target.type == "cuda":
            from ..backend.cuda.backend import get_cuda_backend

            new_storage = get_cuda_backend().from_array_async(self._data, self._data.dtype)
            return Tensor._wrap(new_storage, target, requires_grad=False)

        if self._device.type == "cuda" and target.type == "cpu":
            from ..backend.cuda.backend import get_cuda_backend

            host_array, pending = get_cuda_backend().to_numpy_async(self._data)
            result = Tensor._wrap(host_array, target, requires_grad=False)
            result._pending = pending
            return result

        raise UnsupportedDeviceError(
            f"non_blocking=True is only supported for cpu<->cuda transfers, not "
            f"'{self._device}' -> '{target}'."
        )

    def _move_storage_(self, device: "str | Device") -> None:
        """In-place device transfer, preserving this object's identity (Milestone 9).

        Unlike `Tensor.to()` (value semantics: always returns a *fresh* leaf
        tensor, `self` untouched), this mutates `self._data`/`self._device`
        directly -- the mechanism `Module.to(device)` uses to move
        `Parameter`s so that `model.fc1.weight` is the exact same Python
        object before and after the call, matching the way `SGD.step()`
        already mutates `Parameter._data` in place rather than reassigning
        it (see `docs/architecture/optimization.md`). `requires_grad`,
        `is_leaf` (always `True` for a `Parameter`), and `grad_fn` (always
        `None` for a leaf) are untouched by this move. Any accumulated
        `.grad` is cleared: it was computed for the *previous* device's
        data (and, being a leaf's own `.grad`, is itself a plain tensor with
        no graph to recompute it from), so leaving it in place would
        silently pair a stale gradient with new data.
        """
        target = Device.parse(device)
        if target == self._device:
            return
        source_backend = get_backend(self._device)
        target_backend = get_backend(target)
        host_array = source_backend.to_numpy(self._data)
        new_storage = target_backend.from_array(host_array, host_array.dtype)
        self._data = new_storage
        self._device = target
        self.grad = None

    # -- Autograd ----------------------------------------------------------

    def backward(self, gradient: "Tensor | Any | None" = None) -> None:
        """Run reverse-mode autodiff from this tensor, accumulating into leaf `.grad`s.

        For a non-scalar tensor, ``gradient`` must be supplied explicitly and
        match this tensor's shape; a scalar tensor defaults to an upstream
        gradient of 1. Gradients accumulate into each leaf's ``.grad`` across
        separate ``backward()`` calls (use ``zero_grad()`` to reset). The
        graph feeding this tensor is freed as it is consumed, so calling
        ``backward()`` again on the same non-leaf output raises
        ``GradientStateError``; build a new forward pass instead.

        As of Milestone 10, this runs on any device whose backend
        implements the graph's operations' backward rules -- CPU and CUDA,
        for the operations each supports (see
        `docs/architecture/cuda-backend.md`). An explicit upstream
        ``gradient`` must already be on this tensor's own device (raises
        `UnsupportedDeviceError` otherwise) -- exactly like every other
        binary Tensor operation, a device transfer across ``backward()``'s
        boundary is never performed implicitly.
        """
        if not self._requires_grad:
            raise GradientStateError("backward() called on a tensor that does not require grad.")
        if not self._is_leaf and self._grad_fn is None:
            raise GradientStateError(
                "backward() was already called on this graph and it has been freed. "
                "Run a new forward pass to compute gradients again."
            )

        backend = get_backend(self._device)
        if gradient is None:
            if self.shape != ():
                raise GradientStateError(
                    "backward() requires an explicit upstream gradient for a non-scalar "
                    f"output of shape {self.shape}; only scalar outputs default to a "
                    "gradient of 1."
                )
            grad_array = backend.from_array(np.ones((), dtype=self._data.dtype), self._data.dtype)
        else:
            grad_tensor = gradient if isinstance(gradient, Tensor) else Tensor(gradient, device=self._device)
            if grad_tensor._device != self._device:
                raise UnsupportedDeviceError(
                    f"Upstream gradient device '{grad_tensor._device}' does not match this "
                    f"tensor's device '{self._device}'. Move it explicitly with .to() first."
                )
            if grad_tensor.shape != self.shape:
                raise ShapeMismatchError(
                    f"Upstream gradient shape {grad_tensor.shape} does not match "
                    f"output shape {self.shape}."
                )
            grad_array = grad_tensor._data
            if grad_array.dtype != self._data.dtype:
                if self._device.type == "cpu":
                    grad_array = grad_array.astype(self._data.dtype, copy=False)
                else:
                    raise UnsupportedDTypeError(
                        f"Upstream gradient dtype '{grad_array.dtype}' does not match this "
                        f"tensor's dtype '{self._data.dtype}' on device '{self._device}'."
                    )

        run_backward(self, grad_array)

    def zero_grad(self) -> None:
        """Clear this tensor's accumulated gradient."""
        self.grad = None

    def _accumulate_grad(self, grad_array) -> None:
        if self.grad is None:
            if self._device.type == "cpu":
                grad_array = np.array(grad_array, dtype=self._data.dtype)
            self.grad = Tensor._wrap(grad_array, self._device)
        else:
            backend = get_backend(self._device)
            self.grad = Tensor._wrap(backend.add(self.grad._data, grad_array), self._device)
