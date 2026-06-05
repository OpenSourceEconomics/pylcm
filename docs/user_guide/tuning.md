---
title: Performance and Memory Tuning
---

# Performance and Memory Tuning

Two questions decide how a model runs on accelerators: *does it fit in memory*, and *are
the devices used well*. pylcm keeps them separate as two independent knobs on every grid
— `batch_size` (splay) and `distributed` (shard) — plus a forward-simulation chunk size
and a handful of XLA environment flags. This page explains what each does, when it
helps, and the trade-offs that are easy to get backwards.

The one-line model:

- **`batch_size` (splay) is a memory knob. It is time-neutral.**
- **`distributed` (shard) is a speed knob. It applies only to discrete,
  non-transitioning axes.**

Keeping these straight is the whole game: splaying never speeds anything up, and
sharding is the only knob that does.

## The two grid knobs

Every grid — `DiscreteGrid` and every continuous grid (`LinSpacedGrid`, `LogSpacedGrid`,
`IrregSpacedGrid`, the piecewise variants) — takes both:

```python
from lcm.grids import DiscreteGrid, LinSpacedGrid

# A permanent (never-transitioning) discrete state, sharded one block per device (speed):
pref_type = DiscreteGrid(PrefType, distributed=True)

# A continuous state, scan-chunked into pieces to save memory (time-neutral):
assets = LinSpacedGrid(start=0.0, stop=1_000.0, n_points=200, batch_size=50)
```

| knob                       | what it does                                                             | what it buys      | applies to                            |
| -------------------------- | ------------------------------------------------------------------------ | ----------------- | ------------------------------------- |
| `batch_size=k` (splay)     | `lax.scan` the per-period work over chunks of `k` points along that axis | lower peak memory | any axis                              |
| `distributed=True` (shard) | place that axis's blocks on separate devices                             | parallel speedup  | discrete, non-transitioning axes only |

`batch_size=0` (the default) means "no splay" — one kernel per period over the full
axis. `distributed=False` (the default) means "not sharded".

## `batch_size`: splay for memory, time-neutral

At each period, backward induction builds the value array over every (state, action)
combination and maximises over actions. `batch_size=k` only changes how that work is
*tiled*: instead of one big `vmap`, it runs a `lax.scan` over chunks of `k` points along
the chosen axis. **The total FLOPs are identical** — every combination is still
evaluated exactly once — so the wall-clock barely moves. What drops is peak memory,
because only one chunk's intermediate is live at a time.

Splay stays time-neutral as long as each chunk still has enough parallel work to
saturate the device — and in a real model it does, because the other grid dimensions
(assets × savings × shocks × …) provide ample parallelism inside every chunk.

It stops being free only at the extremes:

- **Over-chunking** (very small `batch_size` → many tiny chunks): per-launch overhead
  piles up, and a chunk can get too small to saturate the device. This bites hardest
  when CUDA graphs are off (see [Environment flags](#environment-flags)), because every
  chunk is then launched individually.
- **Under-chunking** (`batch_size=0`, batch the whole axis): the full intermediate must
  fit at once. If that forces the allocator to spill or to shrink fusion tiles, batching
  can be *slower* than splaying — which is the whole reason the knob exists.

**Which axis to splay.** Prefer a large, *uniform* axis:

- Continuous axes (savings, assets, accumulated earnings) are ideal: they have many
  points (fine control over the chunk count) and are full-size in every regime, so the
  relief is uniform.
- A discrete axis that *collapses* in some regimes — for example a lagged choice that is
  fixed when the agent is forced out of the labour market — gives lumpy relief: splaying
  it does nothing in the regimes where it is already a singleton.

**Rule: use the fewest chunks that fit.** Halving memory needs only two chunks
(`batch_size = n_points / 2`), not `batch_size = 1`.

## `distributed`: shard for speed (discrete, non-transitioning axes)

`distributed=True` places the blocks of an axis on separate devices and solves them in
parallel. It is the only knob that reduces wall-clock — but it is legal only for a
narrow class of axes, and pylcm enforces the boundaries at construction time.

**It runs communication-free only for axes the agent never transitions along.** If an
agent's position on the axis is fixed for life (a permanent type, a fixed group), each
block's value function is independent of the others, so the blocks sit on different
devices with *zero* cross-device traffic. An axis the agent *moves along* (health,
wealth, a lagged choice) couples the blocks: every period would need an all-to-all
exchange, and the communication swamps the compute.

Two guards make this concrete — both raise `GridInitializationError` at construction:

- **Continuous axes cannot be sharded.** `distributed=True` on any continuous grid is
  rejected. (Continuous-axis sharding would require the solved value array to carry an
  explicit output sharding; that path is not enabled.)
- **You cannot splay and shard the same axis.** `batch_size > 0` together with
  `distributed=True` is rejected: each batch is its own dispatch, and on a sharded axis
  every dispatch carries a per-period cross-device collective, so batching multiplies
  the synchronisation count (`×ceil(n_per_device / batch_size)`) and inverts the
  compute/communication ratio. Keep `batch_size=0` on the sharded axis. When a device's
  chunk is too big, shed memory by splaying a *different*, non-sharded axis — usually
  the practical fix, since it needs no extra devices. If you do have spare devices,
  shard the same axis across more of them: that helps precisely when a device holds more
  than one block (`n_points / n_devices > 1`), the only case where splaying the sharded
  axis would have helped anyway, and it shrinks the per-device chunk *and* adds
  parallelism with no extra collectives.

```{note}
Sharding divides the state space across devices, so it also *reduces* per-device memory — a
sharded model often needs no splay at all. Reach for splay only if a single device still
cannot hold its share.
```

## Forward simulation: `subject_batch_size`

Solving is one memory profile; simulating a large panel forward is another.
`Model.simulate(..., subject_batch_size=k)` chunks the simulated subjects so only one
chunk is resident at a time:

- `subject_batch_size=0` (the default) simulates all subjects in a single pass.
- `subject_batch_size=k` walks the panel in chunks of `k`.

Like grid `batch_size`, this is a time-neutral memory knob — raise the chunk count if
the simulated panel does not fit, and otherwise leave it at a single pass.

## Worked example

Measured on 80 GB A100s, one six-regime lifecycle model:

- **One GPU, every axis batched** — full solve + simulate ≈ **1 h 37 m**.
- **Three GPUs, the permanent-type axis sharded one block per device** — a *heavier*
  policy-overlay variant of the same model ≈ **59 m**. The shard more than offsets the
  extra per-regime work: three devices beat one even on a bigger problem.
- **Two single-GPU runs that differ only in which axis is chunked for memory** finished
  within about a minute of each other (≈ 1 h 37 m vs ≈ 1 h 38 m) — direct confirmation
  that the choice of splay axis is time-neutral; only the device count moved the wall.

The takeaway is the one-line model: the multiplicative speedup comes from *sharding*
across devices, not from any choice of `batch_size`.

## Environment flags

pylcm sets two JAX defaults at import and leaves the rest to the environment.

**Set by pylcm (override before importing `lcm`):**

- `XLA_PYTHON_CLIENT_PREALLOCATE=false` — allocate GPU memory on demand instead of
  grabbing a fixed fraction up front. This plays nicely with other processes and makes
  `nvidia-smi` and memory benchmarks reflect real usage.
- `JAX_COMPILATION_CACHE_DIR=~/.cache/jax` — persist the JIT cache so repeated runs of a
  large (many-regime) model skip the multi-minute compile.

**Knobs you set yourself**, with the trade-off each carries:

- `XLA_PYTHON_CLIENT_PREALLOCATE=true` — preallocate a single pool. At production scale
  a stable pool avoids fragmentation and reduces allocator churn across the solve; pair
  it with `XLA_PYTHON_CLIENT_MEM_FRACTION`.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90` — the fraction of device memory the preallocated
  pool claims. The remainder stays as non-pool headroom that the driver, collectives,
  and CUDA graphs draw on; leave enough for them on a multi-GPU run.
- `XLA_PYTHON_CLIENT_ALLOCATOR=default` — keep JAX's pooled BFC allocator. The
  `platform` setting (per-op `cudaMalloc`/`cudaFree`) is dramatically slower; avoid it.
- `XLA_FLAGS=--xla_gpu_autotune_level=0` — disable kernel autotuning. Off gives a
  deterministic, lower-memory compile; on searches for faster GEMM/conv kernels but
  reserves the largest candidate's scratch at compile time, which can re-trigger an OOM
  on a model that barely fits. **Default to off.** Backward induction is dominated by
  gather/scatter and interpolation over the state-action grid, not dense GEMMs, so
  autotuning has little to optimize: head-to-head, the per-period execution time is
  unchanged on/off (matched to logging precision), while compile time and peak memory
  both rise. Turn it on only if a measurement on your model shows an actual per-period
  speedup.
- `XLA_FLAGS=--xla_gpu_enable_command_buffer=` (empty, i.e. disabled) — turn off CUDA
  graphs. Command buffers batch kernel launches but consume non-pool driver memory;
  disabling them frees that headroom at the cost of per-launch overhead. That overhead
  lands hardest on splay-heavy configs (many small kernels), so a heavily-splayed model
  pays more for disabling them.

```{warning}
Sharding only helps if the devices are actually visible *and* exclusively yours. If your
launcher grants N GPUs but `CUDA_VISIBLE_DEVICES` exposes only one, a model declared
`distributed=True` silently runs on a single device — the classic "allocated 3, saw 1";
assert `jax.device_count()` matches what you sharded for at startup, before the solve.
And with `PREALLOCATE=true`, a GPU that another job or a leaked process is already using
fails the pool preallocation outright — an `OUT_OF_MEMORY` on device 0 within seconds of
startup (not mid-solve) — so request GPUs exclusively.
```

**A stable multi-GPU configuration.** One environment that holds up at production scale,
trading compile-time kernel search and launch batching for memory headroom:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_ALLOCATOR=default          # pooled BFC
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer='
```

Command buffers are the one knob to revisit once a model fits comfortably: re-enabling
them amortizes launch overhead, at the cost of the non-pool driver memory they consume.
Autotuning, by contrast, has not been observed to speed these gather-bound solves, so
leaving it off costs nothing and keeps the memory headroom.

## Checklist

- Shard a never-transitioning discrete axis across devices for speed
  (`distributed=True`).
- Keep `batch_size=0` on a sharded axis — never batch and shard the same axis.
- If a single device still can't hold its share, splay a large continuous axis, using
  the fewest chunks that fit.
- Never splay a sharded axis, and never expect splay to speed anything up — it only buys
  memory.
- Chunk the forward pass with `subject_batch_size` if the simulated panel doesn't fit.
- Verify `jax.device_count()` matches your sharding before the solve.
