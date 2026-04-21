---
title: Dispatch Strategies
---

# Dispatch Strategies

When pylcm compiles the Bellman kernel for a model, it has to decide **how each discrete
state dimension is swept through the compiled program**. That choice — pylcm's *dispatch
strategy* — determines three things you can feel:

1. How much GPU memory the compiled program needs.
1. How much parallelism it exposes on one GPU.
1. Whether the axis can be sharded across multiple GPUs.

The `DispatchStrategy` enum, passed to `DiscreteGrid(..., dispatch=...)`, makes these
choices explicit on a per-dimension basis. This page explains what each strategy does,
when you'd pick each, and how to migrate if you were relying on the old implicit
behaviour.

## The four strategies

```python
from lcm import DiscreteGrid, DispatchStrategy
```

### `FUSED_VMAP` — the default

```python
DiscreteGrid(Cat)
# or explicitly
DiscreteGrid(Cat, dispatch=DispatchStrategy.FUSED_VMAP)
```

The dim stays inside the state-action space. `jax.vmap` fuses it with the other state /
action axes into one XLA kernel. Maximum single-GPU parallelism; memory scales with the
full Cartesian product of *all* state / action grid sizes.

Pick this when everything fits in GPU memory and you want the GPU to chew through all
value-function evaluations in a single fused program — which, for small-to-medium
models, is the fastest option by a wide margin.

### `CHUNKED_LAX_MAP` — memory-bounded scan inside the state space

```python
DiscreteGrid(Cat, dispatch=DispatchStrategy.CHUNKED_LAX_MAP, batch_size=4)
# shorthand:
DiscreteGrid(Cat, batch_size=4)  # dispatch derived when you only pass batch_size
```

The dim stays in the state-action space but is swept by `jax.lax.map` with a user-chosen
`batch_size`. That means the kernel processes `batch_size` consecutive slices of this
dim at a time and serialises across chunks — a memory-bounded alternative to
`FUSED_VMAP`.

Two regimes of this strategy matter:

- `batch_size=1` is a fully serial scan — the working-set size is `S`, the product of
  all *other* state-action grid sizes.
- `batch_size=k ≥ 2` gives chunked parallelism: `k` slices fuse into one subkernel,
  which then runs `ceil(N/k)` times.

Pick this when the full product doesn't fit in HBM but the dim has a non-identity
transition (so partition lifting, below, doesn't apply). It's pylcm's knob for trading
parallelism for memory.

### `PARTITION_SCAN` — lifted out of the state space, serial at the top level

```python
DiscreteGrid(Cat, dispatch=DispatchStrategy.PARTITION_SCAN)
```

The dim is **lifted out** of the state-action space and swept by `jax.lax.scan` at the
kernel's top level. Pylcm treats the dim as *fixed for each subject's lifetime* —
per-subject value over this dim doesn't change, so the Bellman update can be computed
independently per partition point.

This requires the dim's transition to be the identity in every regime
(`state_transitions[name] = None` where applicable, or simply absent on terminal
regimes). Pylcm raises `ModelInitializationError` if you mix `PARTITION_*` with a
non-identity transition.

The axis stays JAX-visible at the wrap site, so a future multi-GPU release can swap the
`jax.lax.scan` for `shard_map` to distribute partition points across devices without
touching your model code.

Pick this when the Cartesian product of the partition dim and all *other* states is too
big for one GPU, or when you plan to shard across GPUs soon.

### `PARTITION_VMAP` — lifted out, parallel at the top level

```python
DiscreteGrid(Cat, dispatch=DispatchStrategy.PARTITION_VMAP)
```

Same lifting as `PARTITION_SCAN`, but the top-level sweep is `jax.vmap` — so all
partition points are fused into one compiled XLA program. On a single GPU this is
basically `FUSED_VMAP` with partition bookkeeping on top: same memory, same parallelism,
no throughput advantage over the default.

The reason `PARTITION_VMAP` exists is multi-GPU. `shard_map` can shard the partition
axis across devices and, inside each shard, `jax.vmap` keeps per-device parallelism.
Picking `PARTITION_VMAP` today describes the future multi-device behaviour you want at
each shard boundary.

## Trade-off table

Notation: `N` is the cardinality of the dim, `S` is the product of all *other* state
grid sizes on the regime, and `P` is the product of other partition dims already lifted
on the model.

|                                        | `FUSED_VMAP`                | `CHUNKED_LAX_MAP`, `batch_size=k` | `CHUNKED_LAX_MAP`, `batch_size=1` | `PARTITION_SCAN`  | `PARTITION_VMAP`            |
| -------------------------------------- | --------------------------- | --------------------------------- | --------------------------------- | ----------------- | --------------------------- |
| Dim in state-action space?             | yes                         | yes                               | yes                               | no — lifted       | no — lifted                 |
| GPU working-set memory                 | `N * S * P`                 | `k * S * P`                       | `S * P`                           | `S * P`           | `N * S * P`                 |
| Single-GPU parallelism across this dim | N-wide, fused               | k-wide per chunk                  | 0 — serial                        | 0 — serial        | N-wide, fused               |
| XLA compile cost                       | super-linear in `N * S * P` | ~constant                         | tiny                              | tiny              | super-linear in `N * S * P` |
| Non-identity transitions?              | yes                         | yes                               | yes                               | no — must be None | no — must be None           |
| Multi-device via `shard_map`?          | no                          | no                                | no                                | yes               | yes                         |
| Arrayed-params user API?               | no                          | no                                | no                                | yes               | yes                         |

## Decision tree

```{mermaid}
flowchart TD
    start[Discrete state dim] --> trans{Does the state's<br/>transition change its value?}
    trans -- yes --> fits{Does full product<br/>`N * S * P` fit in one GPU?}
    trans -- no --> multi{Do you plan to scale<br/>across multiple GPUs?}
    fits -- yes --> fused[FUSED_VMAP<br/>default]
    fits -- no --> chunked[CHUNKED_LAX_MAP<br/>batch_size ≥ 1]
    multi -- no --> lift{Does full product<br/>fit in one GPU?}
    multi -- yes --> scanstrat[PARTITION_SCAN if<br/>per-device memory-bound<br/>PARTITION_VMAP otherwise]
    lift -- yes --> fused2[FUSED_VMAP<br/>if transition-change is not needed later]
    lift -- no --> scanmem[PARTITION_SCAN]
```

Shortcut: **if you never plan to add multi-GPU, and the transition is identity, you have
no reason to use `PARTITION_*` over `FUSED_VMAP` or `CHUNKED_LAX_MAP`**. The
partition-lifted strategies earn their keep either (a) when the full product doesn't fit
on one GPU and the transition is identity (`PARTITION_SCAN`), or (b) when multi-GPU is
on the horizon (`PARTITION_*`).

## Examples

### Small single-GPU model — use the default

```python
from lcm import DiscreteGrid, categorical


@categorical(ordered=True)
class Edu:
    low: int
    high: int


education = DiscreteGrid(Edu)  # FUSED_VMAP, fuses with other state axes
```

### Memory-bounded state dim with a non-trivial transition

```python
health = DiscreteGrid(
    HealthStatus,
    dispatch=DispatchStrategy.CHUNKED_LAX_MAP,
    batch_size=4,
)
# or the shorthand that derives dispatch from batch_size:
health = DiscreteGrid(HealthStatus, batch_size=4)
```

Stays in the state-action space because `health` transitions (e.g. via a Markov
transition matrix), but processes 4 slices at a time to stay under a memory bound.

### Preference-type partition, single GPU, memory-bound

```python
pref_type = DiscreteGrid(PrefType, dispatch=DispatchStrategy.PARTITION_SCAN)
```

Lifted out of the state-action space because `pref_type` doesn't change over a subject's
lifetime (`state_transitions["pref_type"] = None`). Swept by `jax.lax.scan`, so working
memory is `S` (one slice) rather than `N * S`.

### Same model, prepared for multi-GPU

```python
pref_type = DiscreteGrid(PrefType, dispatch=DispatchStrategy.PARTITION_VMAP)
```

Functionally identical on one GPU (same memory and parallelism as `FUSED_VMAP`), but the
top-level `jax.vmap` over the partition axis is exactly the shape `shard_map` needs to
shard across devices when multi-GPU lands. Inside each shard, the vmap keeps per-device
parallelism.

## Caveats and invariants

- **All partition-lifted dims in a model must share one `DispatchStrategy`.** Mixing
  `PARTITION_SCAN` and `PARTITION_VMAP` raises `ModelInitializationError` at model
  construction. This restriction will be lifted once a workload motivates it.
- **Partition-lifted dims require the identity transition in every regime.** Declare
  `state_transitions[name] = None` on each non-terminal regime that includes the state,
  or omit it entirely on terminal regimes. Any non-None transition on a partition-lifted
  state is an error.
- **Partition-lifted V-arrays carry a leading axis per partition dim.**
  `model.solve(...)` returns `V[period][regime]` with shape
  `partition_shape + state_shape`. The user-visible shape convention is "partitions
  first" so `shard_map` / `pmap` can plug in at the leading axis.
- **`batch_size` only has meaning for `CHUNKED_LAX_MAP`.** Any other dispatch paired
  with `batch_size > 0` is an error.

## Migration

If you previously used `state_transitions[name] = None` on a `DiscreteGrid` state to get
partition-like memory savings, opt into it explicitly:

```python
# before (implicit auto-lift heuristic, no longer active):
pref_type = DiscreteGrid(PrefType)

# after (explicit opt-in):
pref_type = DiscreteGrid(PrefType, dispatch=DispatchStrategy.PARTITION_SCAN)
```

If you were using `batch_size=k` on `DiscreteGrid`, nothing changes — the shorthand
still resolves to `CHUNKED_LAX_MAP` with your chosen `k`.
