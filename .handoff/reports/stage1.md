# Stage 1 report: placement-independent invariant-component analysis

## Identity
- Commit: `748d0682` on `feat/invariant-state-analysis`, parent `4674c038` (#482 head at branch time). Not pushed.
- Diff stat vs 4674c038:
  - `src/_lcm/regime_building/invariant_components.py` +452 (new)
  - `tests/regime_building/test_invariant_components.py` +578 (new)
  - `tests/ci/ci-workloads.json` +6 (new test file registered by `python -m tests.ci.generate_ci_workloads`)
- No certificate-sealed source and no pinned test was touched. `processing.py`, `model.py` and `fixed_components.py` are unchanged. `check_seals.py` exits 0 (`check_seals.log`), and the commit hooks "candidate certificate seals match the tree" and "mutation anchors resolve once" passed.

## Private interfaces (`_lcm.regime_building.invariant_components`)
- `analyze_invariant_components(*, user_regimes, regimes, reachability, initial_nodes, ages, fixed_component_splits) -> MappingProxyType[StateName, InvariantComponent]`. On a built model, call it with `model._engine_user_regimes`, `model._regimes`, `model.reachability`, `model.initial_nodes`, `model.ages` and `model._fixed_component_splits`. It is pure, reads no `ExecutionConfig` or `sharded_states`, and never raises for an ineligible model; failures go into the record.
- `fail_if_invariant_blocking_is_unsafe(*, components, block_widths, phase)` raises `ExecutionPlanningError` listing every failed condition, plus remedies: remove the state from the request, or declare `fixed_transition` on every carrier edge and remove reads across codes. It refuses:
  - more than one blocked state;
  - a state with no identity law;
  - a width that is not an exact positive int (bool is rejected) or is larger than the number of codes;
  - no carrier in the phase;
  - every analysis refusal.
- Records, all frozen dataclasses:
  - `RegimeEdge(period, source, target)`.
  - `InvariantPhaseAnalysis(phase, carrying_periods: MappingProxyType[RegimeName, tuple[int,...]], preserving_edges, shared_dependencies, refusals)`, with the property `eligible = bool(carrying_periods) and not refusals`.
  - `InvariantComponent(state_name, codes, labels, original_state_name, original_codes_by_code, initial_nodes: tuple[(period, regime)], solve, simulate)`.

## Design decision: the opt-in control
`ExecutionConfig(invariant_block_widths=...)` is **not** added. Plan §2 says to "add fields incrementally, only with their implementing stage", and the implementing stage is 3. Stage 1 therefore exposes:
- the analysis as private metadata, computed on demand from a built model;
- the private pre-dispatch check, which Stage 3 calls from its config validation.

The analysis is also not wired into `Model.__init__`. Doing so would edit `src/lcm/model.py`, which is byte-sealed and has an AST-pinned `Model.__init__`. That would mean a repin/reseal and likely conflicts with the parallel Stage 0 work. Stage 3 can cache the record on the model when it wires the control (see below).

## Semantics (per phase, over `model.reachability.{solution,simulation}`)
- **Candidates:** every state that some phase slice (`normalize_regime_phases`) gives the `_IdentityTransition` law, bare or per target. This covers the generated `<state>_fixed` group states, which the existing `fixed_component` normalization declares with `fixed_transition`. A state with no identity law is no candidate, even if its law is a probability-one diagonal or a deterministic rotation: invariance is never inferred from values.
- **Carriers:** regimes holding the state in that phase's `grid_states` (after broadcast pruning). A carrier's periods are its active periods in that phase's graph.
- **Per retained edge (period, source, target):**
  - carrier to carrier: the source slice's law toward the target must be the identity. A per-target dict names the target's law; a joint kernel that outputs the state owns the cell, so it counts as a non-identity law. Otherwise the edge is refused as a reset or cross-type read.
  - carrier to non-carrier: recorded as a shared dependency. This is the type-free terminal, which stays one untyped node.
  - non-carrier to carrier: refused. Entry, or re-entry after a drop, has no established binding, and the source would read every code's value.
- **Grid:** all carriers must hold one `DiscreteGrid` with identical categories and codes; this is the canonical code mapping. A continuous or age-specialized grid is refused.
- **Other value channels**, refused whenever they touch a carrier:
  - `same_period_ref_regimes`;
  - `gated_edges` together with `edge_reference_regimes`;
  - in simulate only, a carrier whose `simulation.external_replay_route` is neither `None` nor `GridRecomputationRoute`, or whose `replay_unsupported` is set.
- **Simulate roots:** `initial_nodes` keeps every admissible `(period, regime)` start whose regime carries the state in simulate. Nothing is added and nothing is dropped.

## Shape of the record Stage 3 consumes
`components["pref_type"].solve` gives:
- `carrying_periods`: the (regime, period) scheduler units that are blocked per code;
- `preserving_edges`: each continuation read that must resolve to the *selected* view of the target's V for the same original code;
- `shared_dependencies`: each read of an *unsliced shared* target V (type-free child, computed once).

`codes`/`labels` and `original_codes_by_code` keep global codes. For example, interleaved `fixed_component=(0,1,0,1)` gives group 1 = original codes `(1, 3)`, so a block never renumbers to 0. Stage 3 should call `fail_if_invariant_blocking_is_unsafe(components=..., block_widths=config.invariant_block_widths, phase="solve")` during config resolution, before lowering.

## ACA target
- **Route:** the real pinned model, not a stand-in fixture. aca-model `ad38653696ec366e318ac61b9a81b597a4ecb700` (git clone in the session scratchpad; `aca rev` is printed in the log), `create_benchmark_model(pref_type_grid=DiscreteGrid(BenchmarkPrefType))`, CPU, benchmarks-cuda12 env.
- **Script:** `aca_check.py`; log `aca_check.log`; rc=0.
- **Result:** `pref_type` codes (0,1). In both solve and simulate: eligible, 19 carriers including `dead`, 388 preserving edges, 0 shared dependencies, no refusals, 50 initial roots. A width-1 blocking request is accepted. `aca_probe.log` holds the structural probe (every regime carries `pref_type` with `_IdentityTransition`; no same-period refs or gated edges).
- **Repository test:** the suite has a matching minimal fixture, `_model()` in the test file: a model-level `pref_type` with `fixed_transition`, kept in the terminal bequest. The ACA check is evidence only. It is not a repo test, because aca-model is in no test lane and a model build takes about 47 s.

## Commands and exit status (host hmg-office, everything under `zsh -ic "cap ..."`, `PYTHONPATH=<worktree>/src`, `pixi run --as-is`)
| Step | Evidence | rc | junit (tests/fail/err/skip) |
|---|---|---|---|
| Red (stubs raising NotImplementedError) | red.log/xml | 1 | 29/29/0/0: 28 NotImplementedError plus 1 fixture error (wrong regime-id class), fixed and re-run as red_entry |
| Red, entry fixtures | red_entry.log/xml | 1 | 2/2/0/0, both NotImplementedError |
| Green 1 | green1.log/xml | 1 | 29/4/0/0. Three tests expected a reset-only state to be a candidate, which contradicts the candidate definition; those tests were corrected to "no candidate" plus a per-edge reset/cross-type refusal. The fourth was the blocking test that used the same fixture. |
| Green 2 | green2.log/xml | 0 | 30/0/0/0 |
| Final, new file, `--precision=64` | final_new_p64.log/xml | 0 | 30/0/0/0 |
| Final, new file, `--precision=32` | final_new_p32.log/xml | 0 | 30/0/0/0 |
| Related: slot-read ledger, CI manifest, fixed-component, same-period-ref and gated self-loop fixtures (`-n 2`) | final_related.log/xml | 0 | 477/0/0/0 |
| `prek run --files <changed>` | prek3.log (prek1/2 show the lint/ty findings that were fixed) | 0 | |
| `prek run ty --files <changed>` | ty.log | 0 | |
| `generate_ci_workloads --check` | ci_workloads_check.log | 0 | |
| `check_seals.py` | check_seals.log | 0 | |
| ACA check | aca_check.log | 0 | |

- No test failed in the final runs.
- `lcm.__file__` resolved to `<worktree>/src/lcm/__init__.py`; it is printed in the aca logs and in the battery agent's import probe.
- The final batteries ran through a battery subagent, which parsed the junit XML. The counts were then re-parsed independently from the XML.
- Limitation: the p64 and p32 runs overlapped in time (both are small single-process runs).
- The full suite was not run (per the handoff). Execution code is untouched, so existing behaviour is unchanged by construction. `test_analysis_leaves_the_solution_unchanged` checks bitwise-equal V before and after analysis.

## Status per plan §4 item
| Item | Status |
|---|---|
| Eligibility only from accepted declarations (identity law); reuse fixed_component normalization | implemented, validated |
| Record per candidate: name/code mapping, regimes, active periods, solve/simulate, preserving edges, shared type-free deps | implemented, validated |
| Value channels: continuation, same-period refs, edge refs/gated, replay | implemented (refusal when a channel touches a carrier); validated for continuation, same-period refs, gated edges. The replay-payload refusal has no dedicated test: no small GridSearch fixture publishes a replay payload, and that refusal reads EGM-family routes. |
| Type-preserving chain; type-free terminal kept shared | implemented, validated |
| Type-dependent child keeps its view; drop/re-entry and entry refused; reset/cross-type refused | implemented, validated |
| Solve and simulate analysed separately (phase mismatch); initial roots preserved | implemented, validated |
| Independent of `sharded_states` | validated (records are equal with and without `sharded_states=("pref_type",)`) |
| Explicit unsafe blocking fails before dispatch with ExecutionPlanningError | implemented as a private check, validated; the public control is deferred to Stage 3 |
| Labels/slices not starting at type zero | validated by mapping the generated group to its original codes; selection itself is Stage 2/3 |
| Disabled-optimization parity | validated (analysis-free execution path unchanged; bitwise V check) |
| GridSearch-only route gate | deferred to Stage 3 (solver eligibility belongs to the execution route, not the structural analysis) |

## Side effect to note
- My first ACA probe used `pixi run -e benchmarks-cuda12` without `--as-is`. pixi re-synced the **shared** env (`.pixi` symlinks to the main checkout) to this branch's lock. That changed aca-model in `benchmarks-cuda12` from 735a250 (main's lock) to ad38653 (#482's pin). No other package differs between the two locks.
- The env now matches #482/Stage 0's pin. A `pixi run -e benchmarks-cuda12` from the main checkout would re-sync it back.
- Every later command used `--as-is`.

## Evidence integrity
`SHA256SUMS` in this directory covers every file here except itself.
