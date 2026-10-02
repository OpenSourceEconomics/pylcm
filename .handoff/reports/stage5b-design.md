# Stage 5B design: block-major lifetime execution with explicit retention

Worktree `/home/user/pylcm-5b`, branch `feat/invariant-block-major-lifetime`, base
`9dc2145a` (Stage 3 + Pro round-0 F1/F2 + Stage 5A). Authoritative plan:
`handoff/.handoff/plan.md` §1, §2, §6, §8 (5B), §11 (8A), §12, §13.

Reading limitation: `agent-guide/execution-and-audits.md` routes ownership/lifetime
work to `/home/hmg/sciebo/pro-audits/pylcm-architecture-performance-plan` (documents
01–19). That path does not exist in this container (`ls /home/hmg` fails). The
handoff `plan.md` is the authoritative plan the parent supplied; documents 01–19 were
not read. This is recorded, not reconstructed.

## 0. What 5B changes, in one paragraph

Stage 3 solves period-major: every period runs every code's bound program and writes
each block into a full value, and the solve result keeps every period's full value
(all codes) on device until the call ends; Stage 5A then simulates grouped by code from
those full values. 5B adds an opt-in **component schedule**: for each code, run the
*existing* backward-induction engine over that code's component only (all periods),
simulate that code's subjects from the still-resident blocks (combined execution), copy
the blocks to the host, and delete the device blocks before the next code starts. The
logical result is still one complete `ValueStore`; its entries are lazy and assemble
one `(period, regime)` value from the retained host blocks only when read.

## 1. Public opt-in surface

```python
from lcm import ExecutionConfig, InvariantBlockSchedule

ExecutionConfig(
    invariant_block_widths={"pref_type": 1},
    invariant_block_schedule=InvariantBlockSchedule.BLOCK_MAJOR,
)
```

- `InvariantBlockSchedule` is an `Enum` (`PERIOD_MAJOR`, the default and the Stage 3
  route unchanged bit for bit; `BLOCK_MAJOR`). Keyword-only field of the frozen
  `ExecutionConfig`; a non-member raises `TypeError` in `__post_init__`.
- `BLOCK_MAJOR` is refused at `Model` construction with `ExecutionPlanningError`
  (condition + remedy, every failed condition listed) when:
  - `invariant_block_widths` is empty (nothing to schedule);
  - some active regime does not carry the blocked state (a type-free regime would
    be re-solved per code; shared-node injection is deferred, see §9);
  - a regime age-specializes the blocked state's grid (period axes would overlay
    the component grid).
  Everything Stage 3 already refuses stays refused.
- Forward simulation from a block-major result (split or combined) is refused at the
  `simulate` call with `ExecutionPlanningError` when:
  - the simulate phase does not certify grouping (5A route absent), since a
    block-major result can only be read one code at a time without assembling full
    values;
  - simulation is budgeted (`device_memory_bytes` resolves to a budget). Budgeted
    block-major *solve* is supported; budgeted chunk admission against per-code
    residency is deferred (§9). Remedy named: `device_memory_bytes=None`, or
    `PERIOD_MAJOR`.

Alternatives considered:
- a `solve(..., schedule=...)` keyword: rejected — the schedule must also govern the
  automatic solve inside `simulate`, and execution choices live in `ExecutionConfig`
  (not in the durable fingerprint) by repository convention;
- a string `Literal`: rejected for an `Enum` (AI standards: enums for categorical
  values; `WidthSearch` precedent);
- a retention sub-object (`retention="host"|"archive"`): deferred. v1 has exactly one
  retained representation (host); the disk-fragment representation is 8A's
  (§7), and adding the field then is backwards compatible.

## 2. How the schedule plugs into the existing engine

New module `src/_lcm/solution/block_major.py` (the component schedule). It is not a
second scheduler: within a component the existing period loop, wave planner,
liveness ledger, donation, transfer caches, admission and diagnostics run unchanged.

1. **Component.** `InvariantComponent(state_name, start, code)` — one code of the
   blocked state, at its position on the full grid. Width 1 only (Stage 3 limit).
   Designed as one element of a component *selection*; 8A passes a subset.
2. **Component view of the canonical regimes** (`component_regimes`). For every
   regime carrying the state, a `dataclasses.replace` copy of its `SolutionPhase`:
   - `_base_state_action_space` with the state's grid narrowed to `[code]`, built
     on the host in the grid's dtype and placed on the grid's sharding — exactly
     `block_state_action_space`, so no program is compiled to bind a code and the
     original code is preserved (code 2 stays 2);
   - `period_kernels` with each bound GridSearch kernel narrowed to its program for
     `code`, rebound with `start=0` (the position *within the component's stored
     value*), keeping its name `main[pref_type=<code>]`, family and function.
   Functions, params, transitions, grids specs, fingerprints, labels and the Model are
   untouched; no Model is constructed. Every engine call that derives shapes from
   `regime.solution.state_action_space(...)` (templates, layouts, materialization,
   diagnostics) therefore sees the component consistently from one place.
3. **Engine call.** `backward_induction.solve(..., regimes=component_regimes,
   executable_cache=cache)`. New keyword-only `executable_cache` (a dict the schedule
   owns for the call) is threaded to `_compile_all_functions` as its `compiled`
   and compiler-memory maps. Component programs share lowering keys (family name,
   period signature, abstract shapes, donations), so the first component compiles and
   the others hit the cache: **no compile per code**. Planning (materialization and
   width selection) runs per component and is host time, measured not hidden.
4. **Stage 3 dispatch.** `_run_dispatch_unit` writes blocks into a fresh full value.
   With one bound program covering a one-code template it returns the block as is
   (a whole-array `dynamic_update_slice` is a byte copy; skipping it removes one
   allocation and the cold `_write_block` helper per component).
5. **Reads.** Inside a component, a continuation carrying the state is stored as the
   component block (state axis length 1). The bound program's selected view selects
   position 0 with `codes=(code,)` — an explicit coverage, never inferred from length.
   Values without the state cannot exist (refused, §1).
6. **Per-component outputs.** The engine returns an ordinary `BackwardInductionResult`
   whose values are component blocks on the solve layout (same sharding as the full
   value, state axis length 1). GridSearch on this route publishes no replay
   policies, continuations or dissolution flags; the schedule refuses a component
   result carrying any (assertion of the route, not a silent drop).

### Simulation (Stage 5A route)

`simulate()` gains keyword-only `component_values: ComponentValueSource | None`. When
set (block-major, grouped), the chunk loop runs code by code in grid order:

- `acquire(code)` returns that code's `period -> regime -> block` on device:
  - combined (`simulate(solution=None)`): runs the component solve for `code`;
  - split (`simulate(solution=block_major_result)`): uploads the retained host blocks
    onto the component layout (`device_put`, a byte copy);
- every chunk of the code reads through `PeriodSimulationReads(stored_codes=(code,))`;
- `release(code)` waits for the chunk outputs (offloaded to host), then retains
  (combined) or drops (split) the blocks and deletes the device buffers;
- a code with no subjects is still acquired and released in combined execution, so
  the published solution is complete.

Contract extension (flagged): 5A's `type_local_view` selects `start =
route.codes.index(code)`, i.e. it assumes the stored value covers every code. It gains
an explicit `stored_codes` (default `route.codes`), so the selection start is the
code's position in the stored artifact's declared coverage. RNG keys, chunk rows and
order restoration are untouched: subjects are planned over the full population once.

## 3. Store and persistence types backing block entries

- `RetainedComponentValues` (block_major.py): owner of the retained representation.
  Holds, per `(code, period, regime)`, one host `numpy` block (exact bytes from
  `jax.device_get`), plus per `(period, regime)` the full shape, dtype, stored axis
  index of the state and the solve layout (from `_get_regime_V_shapes_and_shardings`
  on the canonical regimes — shapes and shardings only, nothing allocated). It also
  carries the completion manifest (§4) and the retention record (§6).
- `_ComponentValueEntry(_LazyEntry)` per `(period, regime)`: `load_state` is
  `UNLOADED` (nothing device-resident); `materialize()` concatenates the K host blocks
  along the state axis in grid order and `device_put`s the result onto the value's
  solve layout — the same representation an eager `Model.solve` returns, so
  `solution.values[p][r]` keeps its type, dtype, shape and sharding. Each read is a
  fresh owned array; nothing is cached on device.
- `own_value_store` (engine-owned stores) passes lazy entries through instead of
  wrapping them in `_CanonicalValueEntry` (private store contract, flagged).
- Metadata inspection (`len`, `in`, iteration, `load_state`, `metadata.value_schemas`)
  never touches a block.
- Engine consumers read selectively: simulation (split and combined) reads component
  blocks via `ComponentValueSource`, never a full value.
- `SolutionResult.save` uses the existing versioned archive. Save preparation reads
  a component entry's host assembly directly (optional `_LazyEntry.host_value()`
  hook; default `None` keeps every other entry type on its current path), so saving
  never uploads to the device. `load_solution` returns the standard lazy HDF5 result;
  simulating it on a block-major model takes the foreign-solution path (full values,
  5A grouped reads) — correct and complete, without the lifetime benefit.
- `SimulationResult.period_to_regime_to_V_arr` holds the block-backed `ValueStore`
  (annotation widened to a non-traversing `Mapping` boundary, flagged); its `save`
  writes host-assembled CPU arrays to the orbax checkpoint so it never needs every
  full value on the accelerator at once.

## 4. Ownership, completion manifest, cleanup, interrupted runs

- **Ownership.** Device component blocks are owned by the schedule for the duration of
  one component; host blocks are owned by `RetainedComponentValues`, referenced only by
  the result's lazy entries and its engine view. Dropping the last `SolutionResult` /
  `SimulationResult` reference frees them (Python ownership; no files, no global
  registry, no cache).
- **Completion manifest.** `ComponentCoverage`: state name, full code tuple, covered
  codes in completion order, and per covered code the exact `(period, regime)` key set
  with shapes/dtypes. `RetainedComponentValues.complete()` publishes a result only when
  coverage is complete (every code), disjoint (no code twice) and each code's key set
  equals the model's solved domain with the declared component shape; otherwise it
  raises (never a partial or summary-only result). 8A reuses the same manifest for
  fragments (adds checksums and fingerprints, §7).
- **Release point.** A component's device blocks are deleted only after (a) the
  engine's own loop finished (all in-component consumers committed; the engine already
  drains its outputs), (b) in combined execution the code's simulation chunks have
  returned and their outputs are on the host, and (c) the blocks have been copied to the
  host (`device_get` blocks until complete). Only then is `Array.delete()` called.
- **Cleanup on failure.** Any exception inside a component (engine refusal, NaN raise,
  simulation error) deletes that component's device blocks in a `finally` and
  propagates; retained host blocks of earlier codes are unreachable once the schedule
  object goes out of scope. No `SolutionResult` or `SimulationResult` is created.
- **Interrupted-run validity.** v1 has no on-disk state, so an interrupted run leaves
  nothing to resume or mistake for a result. (Disk fragments, atomic renames and
  resume belong to 8A, §7.)

## 5. Admission and liveness

- Within a component: unchanged engine admission. The component's templates and
  retained values are the component's, so a budgeted component solve admits against
  the component's own residency, and previous components' blocks are already deleted
  before the next component plans — the accounting is truthful.
- No change to `PlannedInputLiveness`, `PeriodTransferCache`, donation or the wave
  planner. Each component builds its own ledger (a fresh engine call).
- New explicit accounting (`ComponentRetentionRecord`, logged at debug as JSON like
  the core plan record): per code, retained host bytes, device-to-host bytes, and
  host-to-device bytes uploaded for simulation; totals.
- **Full materialization refusal.** `ValueStore.materialize()` over component entries
  sums the per-device bytes of every full value on its solve layout; with a device
  budget (the model's resolved `device_memory_bytes`) a total above the budget raises
  `ExecutionPlanningError` before any upload, naming the need, the budget and the
  remedies (read entries one at a time, save to an archive, raise the budget).
  Implemented as an optional joint-admission hook on `_LazyEntry`
  (`_admit_joint_materialization`), called once by `ValueStore` before it loads; the
  default is a no-op for every existing entry type (flagged).

## 6. Test matrix (independent references)

New file `tests/solution/test_block_major_lifetime.py` (registered in the CI manifest),
run at `--precision=64` and `--precision=32`. Fixtures: `independent_types` (3 codes,
typed terminal; enumeration oracle independent of pylcm), the Stage 3 sector model
(state axis behind a moving discrete axis, typed terminal), and the 5A life-cycle
model (typed dead, type-dependent survival/health, unbalanced/empty groups).

| Test | Reference |
| --- | --- |
| default schedule is period-major; malformed value refused | config |
| refusals: no widths, type-free regime, budgeted simulate, ungrouped simulate | construction / call |
| block-major values == period-major blocked == unblocked, bytes | Stage 3 / unblocked solves |
| values match the enumeration oracle (`DECIMAL_PRECISION`) | `solve_by_enumeration` |
| metadata inspection assembles nothing | assembly counter |
| one `value()` assembles one entry, on the solve layout | counter + sharding |
| split simulate panel == Stage 3+5A panel == unblocked panel, bytes | panels |
| split simulate assembles no full value | counter |
| combined `simulate()` panel and `result.solution` values == references | panels, values |
| save → load → values and simulated panel == references | archive round trip |
| result usable after the producing model is deleted; device blocks deleted after solve; host owner freed when the result is dropped | weakrefs / `is_deleted` |
| `values.materialize()` above an explicit budget refused before upload; accounted bytes equal Σ blocks | record |
| compile count == period-major blocked compile count | `_compile_and_log` counter |
| a failure in the second component raises, publishes nothing, deletes its blocks | fault injection |
| manifest refuses a missing and a repeated code | unit |
| changed params: equals unblocked at the new params, differs from the base | solves |

Comparators are byte-level (`tobytes()`, dtype, shape) per Pro H2; a positive control
(changed seed/params) shows the comparator can fail.

Multi-device: the existing 8-device file `tests/test_continuous_assets_sharding.py`
(pinned by path in CI topology jobs) gains a block-major vs period-major parity case
under sharded `wealth`, so no `cpu.yml` change is needed.

## 7. Surfaces 8A reuses (not implemented here)

- `InvariantComponent` selection and `solve_component(...)`: a launcher runs a subset
  of codes through the same engine and the same executable cache.
- `RetainedComponentValues` + `ComponentCoverage`: a fragment is one node's covered
  codes; 8A adds model and params fingerprints, pylcm version and per-block SHA-256
  checksums to the manifest, writes blocks with the archive's checksummed HDF5 payload
  writer, publishes each fragment by atomic rename after its manifest, and the collector
  merges manifests with the same complete/disjoint/schema checks before exposing one
  `ValueStore` of lazy component entries. Archive directories are caller-owned (pylcm
  never deletes them); an interrupted node leaves only complete fragments, and the
  collector refuses an incomplete set.

## 8. Contract changes to flag (owner: parent)

1. Public: `ExecutionConfig.invariant_block_schedule`, `InvariantBlockSchedule`.
2. Engine: `backward_induction.solve(executable_cache=...)` keyword-only; Stage 3
   `_run_dispatch_unit` single-block path.
3. Value-view/transfer: explicit `stored_codes` coverage in 5A's grouped reads.
4. Solver-API private: `own_value_store` passes lazy entries; `_LazyEntry` optional
   `host_value()` and `_admit_joint_materialization()` hooks; `ValueStore.materialize`
   calls the latter.
5. `SimulationResult` value mapping annotation widened to a non-traversing boundary;
   `simulate(component_values=...)` keyword-only.
6. Persistence: save reads a component entry's host assembly.

## 9. Deferred (explicit)

- Budgeted block-major simulation (per-code residency in chunk admission).
- Type-free regimes (shared-node injection across components).
- Width > 1 / several codes per component, several blocked states.
- Disk-fragment retention and 8A launcher/collector.
- Blueprint cache for per-component planning host time (Pro R3).
- GPU measurements (Marvin requests in the report).
