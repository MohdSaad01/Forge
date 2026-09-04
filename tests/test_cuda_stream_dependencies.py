"""Milestone 28 tests: automatic GPU-side cross-stream Tensor dependencies.

Every test in this module requires an actual working CUDA backend and is
skipped cleanly otherwise via the module-level `pytestmark`, matching every
other `test_cuda_*.py` file. See `forge/backend/cuda/backend.py`'s
`CUDABackend._stream_guard` for the mechanism these tests verify, and
`docs/architecture/cuda-streams.md`'s **Automatic cross-stream dependencies**
section for the full contract. `tests/test_cuda_stream_allocator.py` and
`tests/test_cuda_stream_autograd.py` (Milestone 27, extended in Milestone 28)
cover the allocator and autograd/optimizer angles this module does not
repeat in full.
"""

from __future__ import annotations

import numpy as np
import pytest

import forge
from forge import Tensor
from forge.backend.cuda import stream as cuda_stream
from forge.backend.cuda.backend import get_cuda_backend, is_cuda_available

pytestmark = pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available on this machine")


@pytest.fixture
def wait_event_calls(monkeypatch):
    """Records every GPU-side dependency wait `_stream_guard` inserts, real CUDA calls included.

    Spies on `CUDAStream.wait_event` and `stream.wait_event_on_default_stream`
    specifically -- not `CUDAEvent.__init__` -- because `allocator.py`'s
    pending-block release (`release_pending`, Milestone 27) also creates
    `CUDAEvent`s for an unrelated reason (block-reuse safety), which would
    otherwise contaminate a same-stream-fast-path assertion whenever an
    intermediate Tensor happens to be garbage-collected mid-test. Only
    `_stream_guard` ever calls `wait_event`/`wait_event_on_default_stream`,
    so this list is exactly the set of cross-stream dependencies Forge
    actually established -- each real `cudaStreamWaitEvent` call still runs
    (the spies call through to the original), so results stay correct.
    """
    calls: "list[str]" = []

    original_wait_event = cuda_stream.CUDAStream.wait_event

    def spy_wait_event(self, event):
        calls.append("stream")
        return original_wait_event(self, event)

    monkeypatch.setattr(cuda_stream.CUDAStream, "wait_event", spy_wait_event)

    original_default = cuda_stream.wait_event_on_default_stream

    def spy_default(lib, event):
        calls.append("default")
        return original_default(lib, event)

    monkeypatch.setattr(cuda_stream, "wait_event_on_default_stream", spy_default)

    return calls


# -- 1. Same-stream fast path: no dependency machinery at all ------------------


def test_same_stream_operations_establish_no_dependency(wait_event_calls):
    s = forge.cuda.Stream()
    with forge.cuda.stream(s):
        a = Tensor(np.ones((8,), dtype=np.float32), device="cuda")
        b = a + a
        c = b + a
        d = c * b
    s.synchronize()

    assert wait_event_calls == []
    np.testing.assert_allclose(d.to("cpu").numpy(), np.full((8,), 6.0, dtype=np.float32))


def test_default_stream_operations_establish_no_dependency(wait_event_calls):
    a = Tensor(np.ones((8,), dtype=np.float32), device="cuda")
    b = a + a
    c = b + a

    assert wait_event_calls == []
    np.testing.assert_allclose(c.to("cpu").numpy(), np.full((8,), 3.0, dtype=np.float32))


# -- 2. Cross-stream dependency establishment and deduplication ----------------


def test_cross_stream_read_establishes_exactly_one_dependency(wait_event_calls):
    stream_a = forge.cuda.Stream()
    stream_b = forge.cuda.Stream()

    with forge.cuda.stream(stream_a):
        x = Tensor(np.ones((8,), dtype=np.float32), device="cuda")

    with forge.cuda.stream(stream_b):
        y = x + x  # both operands are the same storage -> one distinct producer

    stream_a.synchronize()
    stream_b.synchronize()

    assert wait_event_calls == ["stream"]
    np.testing.assert_allclose(y.to("cpu").numpy(), np.full((8,), 2.0, dtype=np.float32))


def test_two_inputs_from_the_same_producer_stream_dedupe_to_one_dependency(wait_event_calls):
    stream_p = forge.cuda.Stream()
    stream_c = forge.cuda.Stream()

    with forge.cuda.stream(stream_p):
        a = Tensor(np.full((8,), 2.0, dtype=np.float32), device="cuda")
        b = Tensor(np.full((8,), 3.0, dtype=np.float32), device="cuda")

    with forge.cuda.stream(stream_c):
        c = a + b  # two distinct storages, one shared producer stream

    stream_p.synchronize()
    stream_c.synchronize()

    assert wait_event_calls == ["stream"]
    np.testing.assert_allclose(c.to("cpu").numpy(), np.full((8,), 5.0, dtype=np.float32))


def test_multi_producer_streams_each_get_their_own_dependency(wait_event_calls):
    """`C = A + B`, `A` on stream A and `B` on stream B -- must wait for *both* (Invariant 4)."""
    stream_a = forge.cuda.Stream()
    stream_b = forge.cuda.Stream()
    stream_c = forge.cuda.Stream()

    with forge.cuda.stream(stream_a):
        a = Tensor(np.full((8,), 2.0, dtype=np.float32), device="cuda")
    with forge.cuda.stream(stream_b):
        b = Tensor(np.full((8,), 3.0, dtype=np.float32), device="cuda")

    with forge.cuda.stream(stream_c):
        c = a + b

    stream_a.synchronize()
    stream_b.synchronize()
    stream_c.synchronize()

    assert sorted(wait_event_calls) == ["stream", "stream"]
    np.testing.assert_allclose(c.to("cpu").numpy(), np.full((8,), 5.0, dtype=np.float32))


def test_multi_consumer_streams_each_see_correct_producer_data(wait_event_calls):
    """`x` produced on stream P, then read by two independent consumer streams A and B."""
    stream_p = forge.cuda.Stream()
    stream_a = forge.cuda.Stream()
    stream_b = forge.cuda.Stream()

    with forge.cuda.stream(stream_p):
        x = Tensor(np.full((8,), 5.0, dtype=np.float32), device="cuda")

    with forge.cuda.stream(stream_a):
        y1 = x + x
    with forge.cuda.stream(stream_b):
        y2 = x * x

    stream_p.synchronize()
    stream_a.synchronize()
    stream_b.synchronize()

    assert wait_event_calls == ["stream", "stream"]
    np.testing.assert_allclose(y1.to("cpu").numpy(), np.full((8,), 10.0, dtype=np.float32))
    np.testing.assert_allclose(y2.to("cpu").numpy(), np.full((8,), 25.0, dtype=np.float32))


# -- 3. All four stream directions (Section 14 of the milestone brief) ---------


def _cross_stream_double(producer, consumer):
    if producer is None:
        w = Tensor(np.full((8,), 4.0, dtype=np.float32), device="cuda")
    else:
        with forge.cuda.stream(producer):
            w = Tensor(np.full((8,), 4.0, dtype=np.float32), device="cuda")

    if consumer is None:
        y = w + w
    else:
        with forge.cuda.stream(consumer):
            y = w + w

    for s in (producer, consumer):
        if s is not None:
            s.synchronize()
    return y.to("cpu").numpy()


@pytest.mark.parametrize(
    "direction", ["default_to_explicit", "explicit_to_default", "explicit_a_to_explicit_b", "explicit_b_to_explicit_a"]
)
def test_all_four_stream_directions_produce_correct_results(direction):
    stream_a = forge.cuda.Stream()
    stream_b = forge.cuda.Stream()
    producer, consumer = {
        "default_to_explicit": (None, stream_a),
        "explicit_to_default": (stream_a, None),
        "explicit_a_to_explicit_b": (stream_a, stream_b),
        "explicit_b_to_explicit_a": (stream_b, stream_a),
    }[direction]

    result = _cross_stream_double(producer, consumer)
    np.testing.assert_allclose(result, np.full((8,), 8.0, dtype=np.float32))


# -- 4. No host blocking / no accidental cudaDeviceSynchronize (Sections 4, 27, 47) --


def test_cross_stream_dependency_between_two_explicit_streams_never_calls_device_synchronize(monkeypatch):
    """`cudaDeviceSynchronize()` must never be the mechanism behind a cross-stream dependency.

    Spies on the real `cf_synchronize` C entry point (still calling through,
    so behavior elsewhere is unaffected): establishing and using a
    cross-stream dependency between two *explicit* streams (never the
    default stream, which still synchronizes for its own, unrelated M26
    reason) must call it zero times.
    """
    backend = get_cuda_backend()
    calls = {"n": 0}
    original = backend._lib.cf_synchronize

    def spy():
        calls["n"] += 1
        return original()

    monkeypatch.setattr(backend._lib, "cf_synchronize", spy)

    stream_a = forge.cuda.Stream()
    stream_b = forge.cuda.Stream()
    with forge.cuda.stream(stream_a):
        x = Tensor(np.ones((8,), dtype=np.float32), device="cuda")
    with forge.cuda.stream(stream_b):
        y = x + x

    assert calls["n"] == 0
    stream_a.synchronize()
    stream_b.synchronize()
    np.testing.assert_allclose(y.to("cpu").numpy(), np.full((8,), 2.0, dtype=np.float32))


# -- 5. empty_cache() remains safe once cross-stream dependencies exist (Section 24) --


def test_empty_cache_remains_safe_after_cross_stream_dependencies():
    forge.cuda.empty_cache()
    shape = (4096,)
    stream_a = forge.cuda.Stream()
    stream_b = forge.cuda.Stream()

    with forge.cuda.stream(stream_a):
        x = Tensor(np.full(shape, 7.0, dtype=np.float32), device="cuda")
    with forge.cuda.stream(stream_b):
        y = x + x  # cross-stream dependency, then x is released while still pending
    del x

    forge.cuda.empty_cache()  # must not free anything still-referenced, nor corrupt y

    stream_a.synchronize()
    stream_b.synchronize()
    np.testing.assert_allclose(y.to("cpu").numpy(), np.full(shape, 14.0, dtype=np.float32))
    forge.cuda.empty_cache()


# -- 6. Stress: many streams, random cross-stream reads, allocator churn (Sections 42-43) --


def test_cross_stream_stress_random_workload_matches_reference():
    rng = np.random.default_rng(123)
    n_streams = 4
    streams = [forge.cuda.Stream() for _ in range(n_streams)]
    shape = (256,)

    values_np = rng.standard_normal(shape).astype(np.float32)
    current = Tensor(values_np, device="cuda")  # default stream
    reference = values_np.copy()

    for i in range(60):
        s = streams[i % n_streams]
        with forge.cuda.stream(s):
            op = i % 3
            if op == 0:
                current = current + current
                reference = reference + reference
            elif op == 1:
                other = Tensor(np.full(shape, float(i % 7), dtype=np.float32), device="cuda")
                current = current - other
                reference = reference - float(i % 7)
            else:
                current = current * Tensor(np.full(shape, 0.5, dtype=np.float32), device="cuda")
                reference = reference * 0.5

    forge.cuda.synchronize()
    np.testing.assert_allclose(current.to("cpu").numpy(), reference, rtol=1e-4, atol=1e-4)


# -- 7. Event lifetime stress: repeated cross-stream dependencies leak nothing (Section 44) --


def test_repeated_cross_stream_dependencies_do_not_grow_cuda_allocation():
    forge.cuda.empty_cache()
    stream_a = forge.cuda.Stream()
    stream_b = forge.cuda.Stream()
    shape = (128,)

    def one_round():
        with forge.cuda.stream(stream_a):
            x = Tensor(np.ones(shape, dtype=np.float32), device="cuda")
        with forge.cuda.stream(stream_b):
            y = x + x
        del x
        return y.to("cpu").numpy()  # to("cpu") synchronizes stream_b before reading

    for _ in range(10):
        one_round()
    stream_a.synchronize()
    stream_b.synchronize()
    forge.cuda.empty_cache()
    steady1 = forge.cuda.memory_stats().allocated_bytes

    for _ in range(100):
        one_round()
    stream_a.synchronize()
    stream_b.synchronize()
    forge.cuda.empty_cache()
    steady2 = forge.cuda.memory_stats().allocated_bytes

    assert steady2 == steady1
