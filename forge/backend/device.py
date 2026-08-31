"""Device identity for tensors.

A ``Device`` is a logical execution target. Parsing a device string (even
``"cuda"``) always succeeds here -- that is just naming a target. Whether a
backend actually exists to *execute* on that device is decided separately by
``forge.backend.get_backend``. This keeps "can we name this device" cleanly
separate from "can we run on this device".
"""

from __future__ import annotations

from dataclasses import dataclass

from ..exceptions import UnsupportedDeviceError

SUPPORTED_DEVICE_TYPES = ("cpu", "cuda")


@dataclass(frozen=True)
class Device:
    """A logical execution target, e.g. ``cpu`` or ``cuda:0``."""

    type: str
    index: int | None = None

    @classmethod
    def parse(cls, spec: "str | Device") -> "Device":
        if isinstance(spec, Device):
            return spec
        if not isinstance(spec, str) or not spec.strip():
            raise UnsupportedDeviceError(
                f"Device must be a non-empty string or Device instance, got {spec!r}."
            )

        text = spec.strip().lower()
        if ":" in text:
            type_, _, index_text = text.partition(":")
            try:
                index = int(index_text)
            except ValueError:
                raise UnsupportedDeviceError(
                    f"Invalid device index in '{spec}': expected an integer after ':'."
                ) from None
        else:
            type_, index = text, None

        if type_ not in SUPPORTED_DEVICE_TYPES:
            supported = ", ".join(SUPPORTED_DEVICE_TYPES)
            raise UnsupportedDeviceError(
                f"Unknown device type '{type_}' in '{spec}'. Supported device types: {supported}."
            )

        return cls(type_, index)

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"
