"""The CLI's own user-facing error type.

Distinct from `forge.exceptions.ForgeError`: a `CLIError` is raised only for
problems specific to the command-line adapter itself (a missing file, a bad
output directory, an unavailable dev-only subsystem) -- never for anything
Forge's own APIs already report clearly. `forge/cli/main.py` catches both
`CLIError` and `ForgeError` the same way: print the message, exit non-zero,
no traceback.
"""

from __future__ import annotations


class CLIError(Exception):
    """A user-facing CLI error: print a clean message and exit non-zero."""


__all__ = ["CLIError"]
