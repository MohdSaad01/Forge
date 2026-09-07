# M53 — Module Buffers and BatchNorm2d

## 1. Motivation from M52

M52 (`docs/development/m52-product-direction.md`) surveyed Forge's current
capability set and evaluated five candidate next directions. It selected a
normalization-based CNN (`nn.BatchNorm2d` added to the existing MNIST CNN)
as the strongest candidate, on evidence rather than novelty: a direct probe
against the real Forge API confirmed a genuine, reusable, precisely-bounded
gap -- one that three prior milestones (M16's `docs/architecture/modules.md`,
M50's own model-selection section) had already flagged as a deliberate
omission without anyone building it, because no workload had needed it
until now.

The gap is architectural, not a missing op: `Module` had no concept of a
**buffer** -- non-trainable, non-differentiable, persistent, device-moved,
train/eval-mode-relevant state. `running_mean`/`running_var` are exactly
this. M52's probe additionally found `Tensor.sqrt()` and division
(`__truediv__`) missing on CPU, and confirmed CUDA has no general
multi-axis reduction or `(1,C,1,1)`-vs-`(N,C,H,W)` broadcast to compose
BatchNorm's math from at all.

M53 builds exactly this: the buffer mechanism, the two general elementwise
ops, and `nn.BatchNorm2d` itself (CPU composed from general ops, CUDA via
one dedicated fused kernel group).

## 2. Module buffer design

`forge/nn/module.py` gains a third per-module dict, `_buffers: dict[str,
Tensor | None]`, alongside the existing `_parameters`/`_modules`.

**Registration is explicit** (`Module.register_buffer(name, tensor)`),
unlike `Parameter`/`Module`, which auto-register in `__setattr__` purely by
`isinstance` check. This is a deliberate asymmetry: `Parameter` and `Module`
are distinctive types that unambiguously signal "register me," but a bare
`Tensor` is not -- a module may legitimately want a plain, unregistered
`Tensor` attribute that should never move/persist/traverse (e.g. a cached
constant), and auto-registering every bare-`Tensor` attribute as a buffer
would silently break that. `register_buffer` rejects a `Parameter` (buffers
must stay distinguishable from trainable state) and rejects
`requires_grad=True` (a buffer never participates in autograd, enforced at
the one place it can be created).

`__setattr__` gained one new branch: reassigning an **already-registered**
buffer name with a plain `Tensor` (or `None`) updates `_buffers[name]`
in place rather than falling through to a plain instance attribute --
otherwise `self.running_mean = new_tensor` after construction would create
a stale, disconnected attribute while `_buffers['running_mean']` kept the
old value. `__getattr__` gained the matching lookup. `named_buffers()`/
`buffers()` mirror `named_parameters()`/`parameters()` exactly (dotted
names, first-discovered de-duplication for a buffer shared by identity,
recursive over children); a buffer registered as `None` is never yielded
(mirroring how an unset `self.bias = None` never appears in
`named_parameters()`).

`Module.to(device)`'s existing per-Parameter walk gained one more line:
call `Tensor._move_storage_()` on every non-`None` buffer too. No new
device-movement mechanism was needed -- `_move_storage_` was already
defined generically on `Tensor`, not `Parameter`-specific, so a buffer (an
ordinary, non-Parameter `Tensor`) moves through the identical call.

`Module.train()`/`.eval()` needed **no change at all**: neither method
touches `_buffers`, so a buffer's value trivially survives every
train/eval transition.

No gradient-exclusion mechanism was added to autograd itself: a buffer
never receives a gradient purely because nothing in Forge ever includes one
in an autograd `Node`'s `inputs` tuple -- the same "excluded from the graph
entirely" convention `Tensor.cross_entropy()`'s integer `target` already
established at M31. This is provably correct by construction, not by a
runtime check.

Buffers are scoped to exactly what `BatchNorm2d` needs -- a flat
`name -> Tensor | None` dict, moved/persisted, never differentiated. No
buffer-level `requires_grad` toggle, no lazy/partial buffer initialization
machinery, per M53's own scope boundary (Section 21 of the milestone brief).

## 3. Tensor operations added

- **`Tensor.sqrt()`** -- ordinary elementwise op, CPU (`np.sqrt`) and CUDA
  (`k_sqrt`, the same `UNARY_LAUNCHER` pattern `relu`/`exp`/`log`/`tanh`
  use). Backward: `grad_output * 0.5 / result` (`result` is sqrt's own saved
  output), the same "derivative from the saved output" shape `tanh`/`exp`
  already use.
- **`Tensor.__truediv__`/`__rtruediv__`** -- ordinary elementwise op
  (`Backend.div`/`div_backward`), following the exact `_binary_op`
  dispatch every other binary operator (`+`/`-`/`*`) already uses. CPU
  supports full NumPy broadcasting, matching `+`/`-`/`*`'s existing CPU
  generality. **CUDA supports only exact-matching shapes** -- no consumer
  needs a broadcasting division on CUDA (`nn.BatchNorm2d`'s own CUDA path
  uses the dedicated fused kernel below, never composed `Tensor.__truediv__`
  calls), so no `cf_div_bcast_*` kernel was added, matching the
  already-established "CPU is more general than CUDA" broadcasting gap
  `add`/`sub`/`mul` have had since M9/M14. `div_backward` on CUDA is
  composed from `div`/`mul`/`_neg` (already-tested forward kernels), the
  same style `mul_backward` already uses -- no dedicated backward kernel.

Both are added as ordinary, generally reusable ops (like `tanh` at M50),
not restricted to BatchNorm's use -- but no other new elementwise op
(`mean`, `transpose`, `softmax`, `sigmoid`, `pow`) was added; `mean` is
expressed as `x.sum(axis=..., keepdims=True) * (1.0 / count)` and variance
via `(diff * diff).sum(...)`, reusing `sum`/`mul` rather than adding new
primitives, per the "smallest reusable primitive" principle M49/M52
established.

## 4. BatchNorm2d API

```python
nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, dtype=None, device="cpu")
```

- `num_features`: required, the channel count `C`.
- `eps`: added inside the normalization square root (`sqrt(var + eps)`),
  never inside the running-variance update itself.
- `momentum`: the exponential-moving-average weight given to each new
  batch's statistics (`new = (1 - momentum) * old + momentum * batch`) --
  the same meaning most normalization implementations use for this name
  (**not** PyTorch's `1 - momentum` weighting on the *old* running value,
  which is the same formula, just worth stating explicitly since "momentum"
  is not a self-evidently-directional term).
- `affine`: `True` registers `weight`/`bias` `Parameter`s (shape `(C,)`,
  initialized to `1`/`0`); `False` sets both to `None` and the module has no
  trainable parameters.
- `running_mean`/`running_var`: buffers (`register_buffer`), shape `(C,)`,
  initialized to `0`/`1`.

This mirrors `Conv2d`'s existing `bias: bool` optional-parameter-pair
convention (`self.bias = Parameter(...) if bias else None`) rather than
inventing a new "optional parameter group" idiom.

## 5. Training/evaluation semantics

Per-channel statistics over `(N, H, W)`:

```text
training: mu_c  = mean(x[:, c, :, :])                       (biased, over N*H*W elements)
          var_c = mean((x[:, c, :, :] - mu_c)^2)             (biased)
          xhat  = (x - mu) / sqrt(var + eps)
          running_mean <- (1 - momentum) * running_mean + momentum * mu
          running_var  <- (1 - momentum) * running_var  + momentum * var_unbiased
                           where var_unbiased = var * m / (m - 1), m = N*H*W (m > 1)
eval:     xhat  = (x - running_mean) / sqrt(running_var + eps)
output:   affine=True  -> weight * xhat + bias
          affine=False -> xhat
```

**Normalization uses the biased variance estimator** (divide by `m`); **the
running-variance update uses the unbiased estimator** (divide by `m - 1`).
This is the conventional split most normalization implementations use: an
unbiased estimate of the population variance for the value that will be
reused at inference time on new data, but the standard biased estimator for
in-batch normalization itself. `m <= 1` skips the unbiased correction
(falls back to the biased value) rather than dividing by zero.

Reduction axes are always `(0, 2, 3)` (batch and spatial, per channel) --
never over the channel axis itself.

## 6. CPU implementation

`nn.BatchNorm2d.forward()`'s CPU path is a **pure composition** of general
Tensor primitives:

```python
mean = x.sum(axis=(0, 2, 3), keepdims=True) * (1.0 / count)
diff = x - mean
var = (diff * diff).sum(axis=(0, 2, 3), keepdims=True) * (1.0 / count)
std = (var + self.eps).sqrt()
xhat = diff / std
# affine: xhat * self.weight.reshape(1, C, 1, 1) + self.bias.reshape(1, C, 1, 1)
```

This is possible because CPU already supported everything this needs before
M53 (confirmed by M52's own probe): `sum(axis=(0,2,3), keepdims=True)`
passes straight through to `np.sum`, and `(1,C,1,1)`-vs-`(N,C,H,W)`
broadcasting in `Tensor._binary_op` uses `np.broadcast_shapes` generally.
The only two missing pieces were `sqrt`/`div` themselves (Section 3).

The direct, load-bearing consequence: **CPU BatchNorm2d has no
BatchNorm-specific backward rule anywhere in the codebase.** Every gradient
(`dx`, `dgamma`, `dbeta`) falls out of ordinary reverse-mode autograd
composing `sum_backward`/`sub_backward`/`mul_backward`/`sqrt_backward`/
`div_backward`/`reshape_backward` -- each already independently tested.
This was verified against finite differences directly (`tests/
test_batchnorm.py`), not assumed from the individual ops' own correctness.

Running-statistics update happens inside a `with forge.autograd.no_grad():`
block (via `forge.autograd.no_grad`, `forge/nn/batchnorm.py` imports it the
same way `Trainer.evaluate()` does) so the update computation itself never
attaches to the main forward graph -- `running_mean`/`running_var`'s
`._data` (raw backend storage) is overwritten in place at the end, the same
in-place-mutation idiom `Tensor._move_storage_`/`SGD.step()` already use for
Parameter/buffer state.

## 7. CUDA implementation

CUDA has no general multi-axis reduction (`CUDABackend.sum()` supports only
`axis=None` or `axis=1` on a strictly-2D tensor) or general broadcast
(scoped to `Linear`'s two shapes) to compose BatchNorm2d's math from the way
CPU does. Rather than build a general N-D reduction/broadcast engine
(explicitly out of scope -- see **Candidate designs**, below), this adds one
dedicated, narrowly-scoped kernel group, reached through a single new
CUDA-only `Tensor` method, `Tensor.batch_norm2d(...)` -- **not** part of the
`Backend` ABC every other backend method implements on both devices, since
`CPUBackend` genuinely never needs it (mirrors how `_scale`/`_transpose`/
`_reduce_rows` are CUDA-only helper methods, not ABC members).

`nn.BatchNorm2d.forward()` branches on `x.device.type`: CPU takes the
composed path above; CUDA calls `x.batch_norm2d(weight, bias, running_mean,
running_var, training, momentum, eps)`.

**Forward** (`CUDABackend.batch_norm2d`, `kernels.cu`'s "BatchNorm2d"
section):

1. **`k_bn_mean_var_reduce`** (training only) -- one block per channel, the
   same one-block-per-channel shared-memory tree-reduce shape M15/M34's
   `k_conv2d_backward_bias_reduce` already established for "reduce a
   `(N,C,H,W)`-shaped tensor to one value per channel." Accumulates
   `sum(x)`/`sum(x*x)` in one pass, then `mean = sum/count`,
   `var = sum(x*x)/count - mean^2` (clamped at `0` against the rare
   float-cancellation case of a near-zero true variance).
2. **`k_bn_update_running_stats`** (training only) -- one thread per
   channel (`C` is always small at Forge's target scale), updates
   `running_mean`/`running_var` in place.
3. **`k_bn_normalize`** -- one thread per `(n,c,h,w)` output element, fuses
   normalize + affine into a single pass. `weight`/`bias` may be null
   pointers (`affine=False`), dereferenced only inside that branch.

Eval mode skips kernels 1-2 entirely and uses `running_mean`/`running_var`
directly as `mean`/`var` for kernel 3 -- no separate "eval kernel" exists.

**Backward** (`CUDABackend.batch_norm2d_backward`):

1. **`k_bn_backward_reduce`** -- the same one-block-per-channel shape as the
   forward reduction, computing `sum(dy)`/`sum(dy*xhat)` per channel
   (`xhat` recomputed from the saved `x`/`mean`/`var`, the "recompute from a
   saved input" convention `conv2d_backward`/`max_pool2d_backward` already
   use). These two per-channel sums **are** `dbeta`/`dgamma` directly --
   returned with no further kernel.
2. **`k_bn_backward_dx`** -- one thread per element:
   - training: `dx = (invstd/m) * (m*dy*w - sum_dy*w - xhat*sum_dy_xhat*w)`
     (the standard batchnorm backward identity, `w = weight[c]` or `1` if
     `affine=False`).
   - eval: `dx = dy * w * invstd` (no cross terms -- `mean`/`var` are
     constants there, so no batch-statistic dependency to differentiate
     through).

`invstd` is never materialized as a separate buffer: both `k_bn_normalize`
and both backward kernels recompute `rsqrt(var[c] + eps)` inline from the
saved `var`, avoiding one more kernel/allocation for a value cheaper to
recompute than to store and re-read.

`running_mean`/`running_var` are mutated in place by `CUDABackend.
batch_norm2d` exactly like `sgd_step`/`adam_step` already mutate optimizer
state in place -- triggered by a forward pass instead of an optimizer step,
but the same convention.

## 8. Autograd behavior

Validated directly (finite differences on CPU, CPU/CUDA parity on CUDA --
Section 12):

- Gradient w.r.t. input: correct in both training (full cross-term formula)
  and eval (simplified constant-statistics formula) modes.
- Gradient w.r.t. `weight`/`bias`: correct, and entirely absent when
  `affine=False` (no `weight`/`bias` `Tensor`s exist to gradient-check).
- No gradient ever reaches `running_mean`/`running_var` -- structurally
  guaranteed (Section 2), verified by an explicit test asserting
  `.grad is None` after `backward()`.
- Repeated use (multiple forward/backward calls without `zero_grad()`)
  accumulates correctly -- ordinary `Tensor._accumulate_grad()` behavior,
  no BatchNorm-specific accumulation logic.
- Train -> eval -> train transitions produce correctly-shaped output at
  every step, with no additional graph-lifetime interaction.

## 9. Serialization behavior

`forge/serialization/model.py`'s `_build_save_node`/`_build_load_node`
gained a `"buffers"` branch alongside the existing `"parameters"` branch --
same shape (dotted name -> `{shape, dtype}` metadata plus a values array),
minus `requires_grad` (buffers never have one) plus explicit `None` support
(an unset optional buffer serializes as `null`, not an array). Buffer
arrays share the same `PARAMETERS_DIR` ("parameters/") archive namespace as
parameter arrays -- dotted names can never collide between the two within
one module (a name resolves to at most one of parameter/buffer/module,
enforced by `__setattr__`/`register_buffer`), so this needed no new archive
directory.

`FORMAT_VERSION` bumped `1 -> 2` (a new required key changes the archive
shape -- per `docs/architecture/decisions/ADR-003-persistence-format.md`'s
existing policy, "a version bump is a deliberate, documented breaking
change," never silently tolerated). `CHECKPOINT_FORMAT_VERSION` bumped
`1 -> 2` in lockstep, since `save_checkpoint`/`load_checkpoint` embed the
same model-node shape via the shared `_build_save_node`/`_build_load_node`
functions.

`nn.BatchNorm2d` registered in `forge/serialization/registry.py`
(`get_config` reports `num_features`/`eps`/`momentum`/`affine`; buffer/
parameter *values* round-trip via the generic mechanism above, not through
config).

Verified: `state_dict` (the save archive) includes both buffers; saving
preserves affine parameters and running statistics; loading restores them
exactly (`atol=1e-6` on CPU, `atol=1e-5` on CUDA); `eval()` after loading
produces the same output as before saving, for both `save_model`/
`load_model` and `save_checkpoint`/`load_checkpoint` (the latter alongside
real Adam optimizer state, on both CPU and CUDA).

## 10. Device movement

`Module.to()`'s extended walk (Section 2) covers `nn.BatchNorm2d` for free
-- no BatchNorm-specific device-movement code exists. Verified: CPU -> CUDA
and CUDA -> CPU movement of `running_mean`/`running_var` (identity-preserving,
matching `Parameter`'s existing move semantics), `nn.BatchNorm2d` nested
inside `Sequential` alongside `Conv2d`, and that a moved buffer never
acquires `requires_grad=True` or a stray `.grad`.

## 11. Candidate designs considered

**CPU: Candidate A (composed primitives) — adopted.** CPU already supports
the exact reduction axes and broadcast shape BatchNorm needs (M52's probe);
composing from general ops means zero BatchNorm-specific backward code,
automatic correctness from already-tested primitives, and `sqrt`/`div`
gain a genuine consumer (Section 3's requirement) rather than existing only
for hypothetical future use.

**CUDA: Candidate A (composed primitives) — rejected.** Would require
building general CUDA multi-axis reduction and general CUDA broadcasting
first -- both explicitly out of scope (M14/M9 already scoped CUDA reduction/
broadcast narrowly to their own real consumers, and nothing else in Forge
needs the general versions yet). This is the same "let a concrete model
determine the op, don't build a general engine speculatively" principle
M49/M52 established.

**CUDA: Candidate B (dedicated fused kernel group) — adopted.** Two
reduction kernels (forward mean/var, backward sum_dy/sum_dy_xhat) plus two
fused elementwise kernels (normalize+affine, backward dx), reusing the
existing one-block-per-channel reduction idiom from M15/M34's
`k_conv2d_backward_bias_reduce` rather than inventing a new one.
Deliberately **not** one single monster kernel: splitting reduction from
elementwise transform keeps each kernel's correctness surface small and
independently testable, and Section 7 of the milestone brief explicitly
prioritizes correctness/maintainability over minimizing launch count.
Exposed as a CUDA-only `Tensor.batch_norm2d()` method rather than a
`Backend`-ABC member both backends implement, since a CPU implementation
of the same fused kernel would be genuinely dead code (CPU's own path never
calls it) -- the ABC is for capabilities every backend actually provides,
not a place to add unused symmetry.

## 12. Validation results

- **CPU forward** matches an independent NumPy reference exactly
  (`tests/test_batchnorm.py::test_training_forward_matches_numpy_reference`).
- **CPU/CUDA forward and backward parity**: `<1e-4` relative tolerance
  across training mode, eval mode, affine and non-affine configurations,
  input/weight/bias gradients, and running-statistics updates
  (`tests/test_batchnorm_cuda.py`), hardware-verified on the reference
  940MX.
- **Finite-difference gradient checks** (float64, central difference,
  `eps=1e-6`): input, weight, and bias gradients, both affine and
  non-affine, `<1e-4` relative tolerance (`tests/test_batchnorm.py`).
- **Training/eval semantics**: batch stats used and running stats updated
  while training; running stats frozen and used (not batch stats) in eval;
  eval output proven independent of the rest of the current batch; momentum
  and unbiased-variance-scaling verified against a hand-computed expected
  value; `eps` proven to actually change the output.
- **No gradient reaches buffers**, on both CPU and CUDA, verified directly.
- **Serialization/checkpointing**: buffers present in the saved archive;
  values round-trip exactly; `eval()`-after-load output matches
  pre-save output; BatchNorm state and Adam optimizer state restore
  together correctly.
- **Device movement**: CPU<->CUDA buffer movement, identity-preserving,
  nested inside `Sequential`.
- **Real training workload**: `examples/mnist/model.py::build_model_bn()`
  (`Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Conv2d -> ReLU ->
  MaxPool2d -> Flatten -> Linear -> ReLU -> Linear`, exactly M52's proposed
  architecture) trains and reaches >=80% accuracy on the same synthetic
  labeled-stripe dataset `tests/test_mnist_example_integration.py` already
  uses for the BatchNorm-free baseline, on both CPU
  (`tests/test_mnist_bn_example_integration.py`) and CUDA
  (`tests/test_mnist_bn_example_cuda_integration.py`); train-mode and
  eval-mode outputs on the same input are proven to actually differ (the
  behavioral property that matters, not just that both modes run without
  error).
- **Memory safety**: 100 repeated forward/backward iterations (plus
  interleaved eval-mode passes) through a `Conv2d -> BatchNorm2d` CUDA model
  showed `allocated_bytes` returning to its pre-loop baseline after
  `gc.collect()` (matching exactly the model's own parameter/buffer bytes),
  with all further allocations served from the M25 caching allocator
  (`cache_hit_count` climbing, `cache_miss_count` flat) rather than new
  driver allocations -- no persistent growth.
- **Full suite**: 1,715 passed, single process, zero skips (this session's
  hardware has a working CUDA backend) -- 1,627 pre-M53 plus 88 new tests,
  no regressions.

## 13. Performance results

Not the primary goal of M53 (per the milestone brief's explicit Performance
Rule) and no dedicated optimization pass was run. One measurement to
confirm the baseline fused-kernel implementation is not pathological: 200
synthetic MNIST-shaped samples, batch size 128, one `Trainer.fit()` epoch
(after a warmup epoch) on the reference 940MX --
`build_model()` (no BatchNorm): **0.049s**;
`build_model_bn()` (with BatchNorm2d): **0.058s** (**1.18x**).
This is a reasonable, unsurprising overhead for two additional per-batch
kernel-launch groups (forward reduction+normalize, backward reduction+dx)
on top of an already-small CNN -- adequate for Forge's target workload, so
no further profiling or optimization was pursued, per Section 13's explicit
"if adequate, stop" instruction.

## 14. Limitations

- The forward/backward mean/variance reductions accumulate `sum(x)`/
  `sum(x*x)` in the tensor's own compute dtype (never a separate
  always-double accumulator) -- standard, and adequate at Forge's target
  scale, but a channel with an extremely large element count and a small
  true variance could in principle lose precision to cancellation in
  `E[x^2] - E[x]^2`; the `var >= 0` clamp only guards the sign, not the
  precision loss itself. Not observed at any tested shape.
- CUDA `div`/`Tensor.batch_norm2d` broadcasting/device scope is narrower
  than CPU's, by design (Section 3/7) -- documented, not a bug, but a real
  capability gap a future CUDA-broadcast-needing consumer would have to
  close on its own terms.
- `BatchNorm2d`'s CUDA reduction kernels launch with a fixed `C` blocks
  (one per channel); at very large channel counts relative to the GPU's SM
  count this under-occupies the device -- not measured as a problem at any
  shape this milestone's validation used (`C` in the single/low
  double-digits), consistent with M52's own risk note.

## 15. Future work explicitly deferred

Per Section 18 of the milestone brief, none of the following were built,
and none are implied by anything in this milestone: `BatchNorm1d`/
`BatchNorm3d`, `LayerNorm`/`GroupNorm`/`InstanceNorm`, `SyncBatchNorm`, a
generic normalization-framework abstraction, distributed BatchNorm,
mixed-precision-specific infrastructure, additional optimizers, unrelated
Tensor operators (`mean`, `transpose`, `softmax`, `sigmoid`), or new CUDA
performance infrastructure. The buffer mechanism itself is generic enough
to serve any of these later, but nothing here was built in anticipation of
that -- consistent with M49-M52's "concrete workload -> demonstrated gap ->
minimal reusable capability" principle.
