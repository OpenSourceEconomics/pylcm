---
title: Runtime, results, and persistence
---

# Runtime, results, and persistence

## Solving

`model.solve(params=..., log_level=...)` returns one immutable `SolutionResult` for
every built-in or external solver. Its `values` store is indexed as
`period -> regime -> value array`; replay and collective-dissolution data stay in the
addressed artifact stores instead of changing the return type.

Optional arguments:

- `max_compilation_workers` caps parallel XLA compilation;
- `log_path` and `log_keep_n_latest` control diagnostic snapshots;
- `retention` selects which post-solve artifacts remain available;
- `execution_config` carries hardware-local controls, currently the
  [compiler workspace budget](#compiler-workspace-budgets).

There are no flag-selected tuple returns. Pass the complete result to
`model.simulate(solution=...)`; omitting `solution` asks simulation to solve first.

(compiler-workspace-budgets)=

### Compiler workspace budgets

`ExecutionConfig(device_memory_bytes=...)` declares a per-device byte ceiling for the
compiler-reported peak workspace of every compiled solve core:

```python
from lcm import ExecutionConfig

solution = model.solve(
    params=params,
    log_level="debug",
    execution_config=ExecutionConfig(device_memory_bytes=40 * 2**30),
)
```

Without a budget (the default), every streamed action product is lowered at its
bootstrap width — the largest power of two below the product's extent, capped at 64 — or
at the width a solver requests, such as `GridSearch(action_block_width=...)`, and
compiler memory reports are not consulted. The whole product is lowered only when a
budget shows it fits or a solver requests it. With a budget, the planner enumerates a
deterministic width frontier for each streamed axis (one, the powers of two below the
extent, and the full extent; a requested width is the only candidate) and walks it
widest-first — descending width product, ties broken toward the lexicographically
largest width tuple in axis declaration order. Each candidate is lowered and compiled,
its compiler-reported peak is read, and the first candidate that fits is dispatched; a
narrower candidate is compiled only after every wider one exceeded the budget. That
selects the same candidate an exhaustive search would, at the cost of one extra lowering
per rejected width: a core whose full extent fits compiles exactly one candidate, and a
core that fits at no width compiles its whole frontier before the error. Compilation is
scheduled in waves across regime-period cells — every cell's widest candidate first,
then the next candidate of only those cells still over budget — so parallel compilation
and the deduplication of identical lowerings are unchanged. A dense program has exactly
one candidate.

The budget is compile-only and fail-closed:

- no candidate is executed to measure it — the compiler's peak is the planning signal,
  because the runtime high-water mark of a run that dies is a truncated underestimate;
- a budget that no candidate meets raises `ExecutionPlanningError` before backward
  induction starts, naming the smallest reported peak;
- a budget requires JIT compilation and cannot accompany an already-solved result passed
  to `model.simulate(solution=...)`;
- the selected widths are execution choices: they enter neither the model nor the
  parameter fingerprint, and a period capture records them so `replay_period` lowers the
  same executable without planning again.

The ceiling bounds each compiled program's reported per-device peak, not the device-wide
footprint of a solve: retained values, continuation arrays, executable caches, and
allocator overhead lie outside it. Among feasible candidates the choice is by width, not
by measured runtime.

(api-solution-result)=

### Solution results

`model.solve(...)` keeps values, metadata, replay artifacts, diagnostics, and explicit
omission reasons in one `SolutionResult`:

```python
from lcm.solver_api import ResultRetention

solution = model.solve(
    params=params,
    log_level="debug",
    retention=ResultRetention.VALUES_AND_REPLAY,
)

V_working = solution.value(period=0, regime="working")
```

The result and its supporting types live in the lightweight `lcm.solver_api` submodule;
they are not re-exported from the top-level `lcm` namespace. The retention modes are:

| Mode                        | Retention-controlled result data                                             |
| --------------------------- | ---------------------------------------------------------------------------- |
| `VALUES`                    | Values; no replay artifacts                                                  |
| `VALUES_AND_REPLAY`         | Values plus applicable simulation-policy and dissolution artifacts (default) |
| `ALL_PERSISTABLE_ARTIFACTS` | Values plus every applicable model-verifiable artifact                       |

Values live in a `ValueStore`; artifacts are addressed by an
`ArtifactRef(period=..., regime=..., key=...)` and kept in immutable `ArtifactStore`
instances. Both stores expose the same eager values after a solve and independently lazy
entries after restoration. Inspecting coordinates, metadata, omissions, or
`load_state(...)` does not load numerical data. After restoration, `solution.value(...)`
and `solution.values[period][regime]` load only the requested value entry and verify its
checksum first. Whole-store numerical traversal is explicit through
`solution.values.materialize()`; per-coordinate access preserves independent laziness.
For ordinary array artifacts, `store[ref]` does the same. A plugin-defined PyTree
requires `store.materialize(ref, template=...)` (as used by model replay) so its
non-executable archive leaves can be rebuilt safely. `LoadState.UNLOADED` is storage
state, not an omission reason.

Built-in key identities include `SIMULATION_POLICY`, `DISSOLUTION_FLAG`,
`EGM_CONTINUATION`, and `SOLVER_DIAGNOSTICS`. The `omissions` mapping distinguishes an
artifact that is not applicable, not requested, unsupported, or not persisted.

Metadata carries a durable SHA-256 model fingerprint, the exact digest of canonical
parameters that can affect the solution, solver and replay-route identities, artifact
descriptors, and the solution and solver-interface schema versions. The fingerprint
covers the mathematical model and the facts needed to interpret stored arrays, including
period/regime topology, state and action names, grid support and category order,
solver/replay/artifact versions, numerical conventions, and those solution-relevant
parameter values. It excludes execution-only details such as devices, JIT selection,
tiling, sharding, and compiler versions, as well as parameters and callable semantics
used exclusively by simulation-phase transitions. Separately, an in-memory result keeps
its model-instance token; that same-instance check is not applied to a restored archive.
`metadata.source` records that distinction as `IN_MEMORY` or `PERSISTED`.

Each `(period, regime)` value has a lightweight `ValueArraySchema` recording its exact
shape, dtype, and canonical named axes. Artifact descriptors play the corresponding
descriptive role for retained payloads. Neither authenticates returned data. Simulation
rebuilds immutable authority from the canonical model, canonical parameters, and the
installed consuming route, then checks values, repeated metadata, and materialized
artifacts independently.

Solver diagnostics follow `log_level`, independently of retention. A continuation is
always available to the backward graph that requires it, regardless of result retention.
It remains in the returned result only under `ALL_PERSISTABLE_ARTIFACTS` and only when
its model-built authority declares `MODEL_VERIFIABLE`; otherwise the result records
`NOT_REQUESTED` or `NOT_PERSISTED`.

A retention also selects what a solve computes. Every built-in kernel publishes its
programs with a scope and, for replay or additive artifact programs, exact
`retained_artifact_keys` and an exact `retained_artifact_payload_types` entry for every
key. All programs retaining the same key must agree on its type. The type describes the
final artifact in the period kernel's `KernelOutput`, after any composite or adapter
transformation; it does not assert that the artifact is present in every invocation.
This lets even an inapplicable omitted cell carry an exact descriptor. The solve
compiles and runs only the programs selected for that period/regime cell. Each replay
program explicitly names the values-only program it replaces, so selecting one replay
artifact cannot suppress an unrelated values program in the same multi-core graph.
`VALUES` runs the values-only programs everywhere, so DCEGM and NB-EGM publish their
values and carries without assembling a policy, and a nested NNBEGM solve folds its
candidates without building replay banks or the adaptive nested policy. A
replay-retaining solve runs replay programs only where a declared replay route consumes
them. `ALL_PERSISTABLE_ARTIFACTS` selects replay alternatives and additive artifact-only
programs per exact model-authoritative artifact address; it does not widen a
regime-level boolean. A standalone case-piece NB-EGM regime has no replay consumer, so
it runs its values-only program under every retention and its policy is recorded as
`NOT_APPLICABLE`. Values and carries agree across retentions to the working format's
spacing.

A values-only solve is the cheap way to obtain value functions from a case-piece or
piecewise-affine budget model, for example to compare solvers or to sweep parameters:

```python
from lcm.solver_api import ResultRetention

values_only = model.solve(
    params=params,
    log_level="warning",
    retention=ResultRetention.VALUES,
)

V_alive = values_only.value(period=0, regime="alive")
```

Such a result simulates only where every decision is recoverable from values; a model
whose simulation reads an NB-EGM or NNBEGM policy needs the default retention before it
can be simulated.

`ALL_PERSISTABLE_ARTIFACTS` keeps only artifacts whose model-built authority declares
`PersistencePolicy.MODEL_VERIFIABLE`. The `NNBEGM` replay policy of an
`AdaptiveOuterMesh` search is replayed against the exact mesh the solve generated, a
fact the solving model instance holds privately beside the result rather than inside it;
that policy is retained under `VALUES_AND_REPLAY` and omitted as `NOT_PERSISTED` under
`ALL_PERSISTABLE_ARTIFACTS`. Simulating from such a result is refused before forward
execution with the omission reason named. The finite candidate bank of a
`FiniteOuterGrid` search is self-contained and retained under both modes; it can
therefore be written to the complete archive. A built-in `EGMCarry` continuation is also
model-verifiable and is retained only by `ALL_PERSISTABLE_ARTIFACTS`, making that mode
strictly broader than replay-only retention for an EGM regime.

`save_solution(solution=..., path=...)` atomically writes the complete labelled result
to a versioned archive; `solution.save(path=...)` is the equivalent convenience method.
The archive contains JSON metadata and independently addressed numerical datasets with
SHA-256 checksums. It contains no model, Python class, callable, pickle, or executable
code. See [Standalone persistence](#api-standalone-persistence) for loading and version
compatibility.

## Simulation

`model.simulate(...)` accepts parameters, initial conditions, an optional complete
`SolutionResult` as `solution=...`, and a required `log_level`. Omitting `solution`
solves first. Bare value mappings and separate policy or dissolution-flag inputs are not
accepted.

Before consuming a `SolutionResult`, simulation checks its durable model and
solution-parameter fingerprints, exact solver/plugin and replay-route identities, schema
versions, period count, regime order, solver types, and exact active period/regime value
coverage. An in-memory result additionally has to come from that model instance. It
unconditionally checks every required value and its descriptive schema independently
against the model-owned shape, canonical dtype, and named axes, including at
`log_level="off"`.

All artifact stores and omission records must address active result cells with the exact
key version and channel; one reference cannot appear in multiple stores or be both
present and omitted. Required lazy entries are materialized during preflight and
checksum verified. Numerical leaves are copied into private owned buffers; a
plugin-defined PyTree is rebuilt as a fresh exact tuple or structurally closed dataclass
record from the model authority's sealed construction plan, without another plugin
flatten or unflatten callback. PyTree-represented static metadata is checked against the
plan, while callback-injected instance state is replaced by its declared canonical
value. The consuming route receives an owned replay snapshot rather than the caller's
container or array objects. Built-in EGM/NNBEGM policies and collective dissolution
flags retain their specialized validation. An installed external route additionally
validates solver-specific invariants and builds a JAX-transformable replay reader; this
does not sandbox installed plugin code. No forward-simulation step runs until all
required entries pass.

A values-only result is therefore sufficient for a model whose decisions are fully
recoverable from values, but fails closed before forward simulation when a required
replay artifact is absent or invalid. Use the default
`ResultRetention.VALUES_AND_REPLAY` when the model may require such artifacts.

`subject_batch_size` streams subjects without changing results. `seed` controls random
draws. A collective model may require an addressed dissolution replay artifact and
`own_stakeholder`; see [Collective regimes](collective_regimes.md).

Initial conditions are a mapping of state names plus `regime_id` to equal-length arrays,
or a DataFrame with a `regime_name` column.

## Validation and logging

`log_level` controls both output and runtime validation:

| Level        | Behavior                                                      |
| ------------ | ------------------------------------------------------------- |
| `"off"`      | Silent; runtime probability and non-finite checks skipped     |
| `"warning"`  | Validate, warn, continue                                      |
| `"progress"` | Warning behavior plus timings                                 |
| `"debug"`    | Validate and raise at first failure; include value statistics |

Start model development at `"debug"`. Reduce validation only after the model is trusted
and the cost matters.

pylcm enables a persistent JAX compilation cache by default. Set
`JAX_COMPILATION_CACHE_DIR` to choose the full directory or `LCM_COMPILATION_CACHE_NAME`
to choose the project-specific leaf. Set `XLA_PYTHON_CLIENT_PREALLOCATE=true` before
importing pylcm to restore JAX's device preallocation; pylcm otherwise requests
on-demand allocation.

(api-simulation-result)=

## `SimulationResult`

`to_dataframe(additional_targets=None, use_labels=True, terminal_rows="first")`
materializes a flat DataFrame. `additional_targets` accepts selected DAG outputs or
`"all"`. `terminal_rows="all"` retains every frozen absorbing row; the default keeps
only terminal entry.

Inspection properties include `regime_names`, `state_names`, `action_names`,
`n_periods`, `n_subjects`, `available_targets`, `raw_results`, `flat_params`, and
`period_to_regime_to_V_arr`.

`SimulationResult.save(directory=...)` writes array checkpoints, value functions,
metadata, and a Feather table. `SimulationResult.load(directory=...)` restores it.

(api-standalone-persistence)=

## Standalone persistence

- `save_solution(solution=solution, path=path)` writes a complete `SolutionResult`
  atomically. A failed write leaves no partially published replacement.
- `load_solution(path=path)` reads metadata and constructs independently lazy value and
  artifact stores. Every entry is checksum verified when materialized.
- `load_solution(path=path, verify_checksums=True)` verifies every payload without
  changing any entry from `UNLOADED` to `LOADED`.
- `load_legacy_solution(path=path)` explicitly reads the old value-only HDF5 format for
  migration. Legacy values have no durable model fingerprint, descriptors, omissions, or
  checksums and are not a complete replay result.
- `SolveSnapshot` and `SimulateSnapshot` describe diagnostic snapshots.
- `load_snapshot(path, exclude=...)` loads a snapshot, with optional components omitted.

The solution archive requires exact matches for its format, solution-schema, and public
solver-interface versions. Replay additionally requires the matching plugin, route, and
artifact-schema versions. pylcm reports incompatible versions rather than guessing a
migration. Without an external plugin installed, `load_solution` can still read standard
metadata and omissions, verify all checksums, and inspect values or array payloads
lazily. A plugin-defined PyTree cannot be interpreted or replayed until the matching
route supplies its model-authoritative template.

Workflow: [Solving and simulating](../user_guide/solving_and_simulating.md),
[DataFrame interoperability](../user_guide/pandas_interop.md), and
[Debugging](../user_guide/debugging.md).
