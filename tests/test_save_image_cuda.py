"""Milestone 76 CUDA test: `forge.data.save_image` accepts a CUDA-resident Tensor.

`save_image` is a host-only, non-differentiable reporting operation (it
transfers via `Tensor.to("cpu")` exactly like `forge.training.metrics.
_as_numpy`) -- there is no CUDA kernel to test, only that a Tensor living on
the CUDA device is transferred and written correctly rather than raising
`UnsupportedDeviceError`. Skips cleanly when CUDA is unavailable;
hardware-verified on the development machine's GeForce 940MX (CC 5.0) per
`docs/development/development-environment.md`.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.data import save_image

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


def test_save_image_transfers_cuda_tensor_to_cpu_before_writing(tmp_path):
    array = np.zeros((3, 3, 3), dtype=np.float32)
    array[0, :, :] = 1.0
    cuda_tensor = Tensor(array, device="cuda")
    path = tmp_path / "cuda_out.png"

    save_image(cuda_tensor, path)

    assert path.is_file()
    with Image.open(path) as img:
        assert img.mode == "RGB"
        assert img.getpixel((0, 0)) == (255, 0, 0)
