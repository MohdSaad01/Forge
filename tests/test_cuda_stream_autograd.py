"""Milestone 27 tests: autograd, optimizer, and persistence semantics on an explicit CUDA stream.

Every test in this module requires an actual working CUDA backend and is
skipped cleanly otherwise via the module-level `pytestmark`, matching every
other `test_cuda_*.py` file. See Sections 20-23 and 35 of the milestone
brief, and `docs/architecture/cuda-streams.md`'s **Autograd Semantics**/
**Optimizer Semantics**/**Persistence / Checkpoint Semantics** sections.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import is_cuda_available
from forge.nn import Linear, ReLU, Sequential
from forge.optim import SGD, Adam
from forge.serialization import load_model, register_module, save_checkpoint, load_checkpoint, save_model

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


def _model():
    forge.random.seed(0)
    return Sequential(Linear(4, 8), ReLU(), Linear(8, 2))


# -- 1. Forward + backward on one stream matches CPU --------------------------


def test_forward_and_backward_on_same_stream_matches_cpu():
    x_np = np.random.default_rng(0).standard_normal((6, 4)).astype(np.float32)

    def run(device, stream):
        forge.random.seed(1)
        model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2)).to(device)
        x = Tensor(x_np, device=device)
        if stream is None:
            loss = model(x).sum()
            loss.backward()
        else:
            with forge.cuda.stream(stream):
                loss = model(x).sum()
                loss.backward()
            stream.synchronize()
        return float(loss.to("cpu").numpy()) if device == "cuda" else float(loss.numpy()), [
            p.grad.to("cpu").numpy() if device == "cuda" else p.grad.numpy() for p in model.parameters()
        ]

    cpu_loss, cpu_grads = run("cpu", None)
    cuda_loss, cuda_grads = run("cuda", forge.cuda.Stream())

    assert cuda_loss == pytest.approx(cpu_loss, rel=1e-4, abs=1e-5)
    for cg, gg in zip(cpu_grads, cuda_grads):
        np.testing.assert_allclose(gg, cg, rtol=1e-4, atol=1e-5)


# -- 2. Forward on stream A, backward on stream B is automatically safe (Milestone 28) ------


def test_backward_on_a_different_stream_than_forward_matches_same_stream_reference():
    """Milestone 28: forward on stream A, `backward()` on stream B is safe, not a CUDAError (see M27's identical test).

    Every gradient kernel `backward()` launches on `stream_b` reads Tensors
    (activations, weights) last touched on `stream_a` -- `_stream_guard`
    establishes the needed GPU-side dependency automatically for each one
    (Section 16 of the Milestone 28 brief: "Forge must establish the
    dependency between the forward-produced Tensor and the backward
    computation"). Compared against an identical forward+backward run kept
    entirely on one stream.
    """
    x_np = np.random.default_rng(2).standard_normal((5, 4)).astype(np.float32)

    def run(same_stream: bool):
        forge.random.seed(2)
        model = _model().to("cuda")
        x = Tensor(x_np, device="cuda")
        stream_a = forge.cuda.Stream()
        stream_b = stream_a if same_stream else forge.cuda.Stream()

        with forge.cuda.stream(stream_a):
            loss = model(x).sum()
        with forge.cuda.stream(stream_b):
            loss.backward()
        stream_a.synchronize()
        stream_b.synchronize()
        return float(loss.to("cpu").numpy()), [p.grad.to("cpu").numpy() for p in model.parameters()]

    same_loss, same_grads = run(same_stream=True)
    cross_loss, cross_grads = run(same_stream=False)

    assert cross_loss == pytest.approx(same_loss)
    for sg, cg in zip(same_grads, cross_grads):
        np.testing.assert_allclose(cg, sg)


# -- 3. Optimizer step on a stream, observed correctly on that same stream ----


def test_optimizer_step_then_forward_on_same_stream_sees_updated_parameters():
    forge.random.seed(3)
    model = Linear(4, 2).to("cuda")
    optimizer = SGD(model.parameters(), lr=1.0)  # large lr: update is easy to detect
    x = Tensor(np.random.default_rng(4).standard_normal((4, 4)).astype(np.float32), device="cuda")

    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        out_before = model(x)
        loss = out_before.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        out_after = model(x)  # stream-ordered: must observe the updated parameters
    s.synchronize()

    assert not np.allclose(out_before.to("cpu").numpy(), out_after.to("cpu").numpy())


def test_adam_step_on_stream_matches_cpu_adam_step():
    x_np = np.random.default_rng(5).standard_normal((6, 4)).astype(np.float32)
    y_np = np.random.default_rng(6).standard_normal((6, 2)).astype(np.float32)

    def run(device, stream):
        forge.random.seed(9)
        model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2)).to(device)
        optimizer = Adam(model.parameters(), lr=1e-2)
        x = Tensor(x_np, device=device)
        y = Tensor(y_np, device=device)

        def step():
            optimizer.zero_grad()
            pred = model(x)
            loss = ((pred - y) * (pred - y)).sum()
            loss.backward()
            optimizer.step()

        if stream is None:
            step()
        else:
            with forge.cuda.stream(stream):
                step()
            stream.synchronize()
        return [p.to("cpu").numpy() if device == "cuda" else p.numpy() for p in model.parameters()]

    cpu_params = run("cpu", None)
    cuda_params = run("cuda", forge.cuda.Stream())

    for cp, gp in zip(cpu_params, cuda_params):
        np.testing.assert_allclose(gp, cp, rtol=1e-4, atol=1e-5)


# -- 3b. Cross-stream optimizer/parameter dependency safety (Milestone 28, Sections 38-40) --


def test_optimizer_step_on_a_different_stream_than_backward_matches_same_stream_reference():
    """Gradients computed on Stream A, `optimizer.step()` issued on Stream B (Section 38).

    `SGD.step()` reads `param.grad` (produced on A) from Stream B --
    `_stream_guard` must establish the dependency automatically, with no
    host synchronization required in between, and the result must exactly
    match an identical run kept entirely on one stream.
    """
    x_np = np.random.default_rng(13).standard_normal((6, 4)).astype(np.float32)

    def run(same_stream: bool):
        forge.random.seed(13)
        model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2)).to("cuda")
        optimizer = SGD(model.parameters(), lr=0.5)
        x = Tensor(x_np, device="cuda")

        stream_backward = forge.cuda.Stream()
        stream_step = stream_backward if same_stream else forge.cuda.Stream()

        with forge.cuda.stream(stream_backward):
            model(x).sum().backward()
        with forge.cuda.stream(stream_step):
            optimizer.step()

        stream_backward.synchronize()
        stream_step.synchronize()
        return [p.to("cpu").numpy() for p in model.parameters()]

    same_stream_params = run(same_stream=True)
    cross_stream_params = run(same_stream=False)

    for same, cross in zip(same_stream_params, cross_stream_params):
        np.testing.assert_allclose(cross, same)


def test_parameter_read_and_update_across_streams_are_never_racy_in_either_order():
    """Section 39: "the most important correctness tests in the milestone."

    Forward+backward (a *read* of `model.weight`) on Stream A, then
    `optimizer.step()` (a *write* to `model.weight`) on Stream B, then a
    second forward (another read) back on Stream A -- three ops touching the
    same parameter storage, alternating streams, with no explicit
    synchronization anywhere in between. `_stream_guard` must order the
    write strictly after the first read and the second read strictly after
    the write (both directions of the read/update race), matching an
    identical sequence kept entirely on one stream exactly.
    """

    def build(seed: int):
        forge.random.seed(seed)
        model = Linear(4, 4).to("cuda")
        optimizer = SGD(model.parameters(), lr=1.0)
        x = Tensor(np.random.default_rng(seed).standard_normal((6, 4)).astype(np.float32), device="cuda")
        return model, optimizer, x

    # Cross-stream: read (A) -> update (B) -> read (A) again.
    model, optimizer, x = build(14)
    stream_a = forge.cuda.Stream()
    stream_b = forge.cuda.Stream()
    with forge.cuda.stream(stream_a):
        out_before = model(x)
        out_before.sum().backward()
    with forge.cuda.stream(stream_b):
        optimizer.step()
    with forge.cuda.stream(stream_a):
        out_after = model(x)
    stream_a.synchronize()
    stream_b.synchronize()

    # Reference: the identical sequence, kept entirely on one stream.
    ref_model, ref_optimizer, ref_x = build(14)
    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        ref_before = ref_model(ref_x)
        ref_before.sum().backward()
        ref_optimizer.step()
        ref_after = ref_model(ref_x)
    s.synchronize()

    assert not np.allclose(out_before.to("cpu").numpy(), out_after.to("cpu").numpy())
    np.testing.assert_allclose(out_before.to("cpu").numpy(), ref_before.to("cpu").numpy())
    np.testing.assert_allclose(out_after.to("cpu").numpy(), ref_after.to("cpu").numpy())


# -- 4. Persistence after async work (Section 35) -------------------------------


class _StreamMLP(forge.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 6)
        self.relu = ReLU()
        self.fc2 = Linear(6, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


register_module("_StreamMLP_M27", _StreamMLP, get_config=lambda m: {})


def test_save_model_after_async_work_on_a_stream_round_trips_correctly(tmp_path):
    """A parameter last touched inside a `with forge.cuda.stream(s):` block must serialize correctly.

    `save_model()` reads every parameter via `Backend.to_numpy()`, which
    (Milestone 27) synchronizes that storage's own last-use stream before
    the device-to-host copy -- so this must be safe with no explicit
    `stream.synchronize()`/`forge.cuda.synchronize()` call from the test
    itself.
    """
    forge.random.seed(11)
    model = _StreamMLP().to("cuda")
    optimizer = SGD(model.parameters(), lr=0.1)
    x = Tensor(np.random.default_rng(7).standard_normal((5, 4)).astype(np.float32), device="cuda")

    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

    path = tmp_path / "model_m27.forge"
    save_model(model, str(path))  # no explicit synchronize() before this call

    loaded = load_model(str(path), device="cuda")
    for original, restored in zip(model.parameters(), loaded.parameters()):
        np.testing.assert_allclose(original.to("cpu").numpy(), restored.to("cpu").numpy())


def test_save_checkpoint_after_async_work_on_a_stream_round_trips_correctly(tmp_path):
    forge.random.seed(12)
    model = _StreamMLP().to("cuda")
    optimizer = Adam(model.parameters(), lr=1e-2)
    x = Tensor(np.random.default_rng(8).standard_normal((5, 4)).astype(np.float32), device="cuda")

    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

    path = tmp_path / "checkpoint_m27.forge"
    save_checkpoint(str(path), model, optimizer, epoch=1)  # no explicit synchronize() first

    checkpoint = load_checkpoint(str(path), device="cuda")

    for original, restored in zip(model.parameters(), checkpoint.model.parameters()):
        np.testing.assert_allclose(original.to("cpu").numpy(), restored.to("cpu").numpy())
