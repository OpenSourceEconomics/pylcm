---
title: Internal Architecture
---

# Internal Architecture

This page describes how pylcm's source tree is organised and *why* it is laid out that
way. The audience is contributors and advanced users who want to find code, add a
feature, or understand which module they should be editing. End-users only ever write
`from lcm import Model, Regime, ...` and never need anything here.

## The `lcm` / `_lcm` split

pylcm's source is two packages, and the split is a hard binary:

- **`src/lcm/`** — the public surface. Everything a user constructs or consumes lives
  here, physically: the user-facing classes, the `@categorical` decorator, the `as_leaf`
  helper, the public type aliases, and the exception classes. `lcm/__init__.py`
  re-exports the public symbols so users write `from lcm import Model`.
- **`src/_lcm/`** — the private implementation. The build pipeline, the canonical engine
  dataclasses, the JAX-traced solve / simulate machinery, validators, I/O plumbing, and
  the engine-side type aliases. The leading underscore on the *package* carries the
  entire "private" signal — modules inside `_lcm/` are plainly named.

The package boundary is the primary signal: a module is either in `lcm/` (public) or in
`_lcm/` (private). Within `lcm/` a leading underscore marks the two places that carry
implementation the public surface happens to need at its own import time —
`lcm/_compilation_cache.py` and the `lcm/_solver_api/` package, whose contents are
re-exported through the plainly named `lcm/solver_api.py`. Internal code reaches the
user-facing classes through `from lcm.regime import Regime as UserRegime` etc.;
`lcm/__init__.py` registers the beartype claw over both packages before any of their
submodules load.

```
lcm/
├── __init__.py          ← runtime perimeter + re-export façade for the public symbols
├── ages.py              ← AgeGrid
├── branch_aggregation.py ← outer branch-aggregation configuration (re-export façade)
├── case_piece.py        ← case_boundary, piece, piecewise_affine, affine_breakpoint
├── certainty_equivalent.py ← certainty-equivalent classes (re-export façade)
├── collective.py        ← collective and value-dependent choice declarations
├── condition.py         ← lcm.ref and the declared-condition vocabulary
├── consumption_savings_regime.py ← specialized consumption-savings regime declarations
├── execution.py         ← ExecutionConfig
├── fixed_forms.py       ← the conventional accounting forms of a one-asset regime
├── grids.py             ← LinSpacedGrid, LogSpacedGrid, IrregSpacedGrid, DiscreteGrid,
│                          PiecewiseLinSpacedGrid, PiecewiseLogSpacedGrid,
│                          GridBreakpoint, and the @categorical decorator
├── koopmans_aggregation.py ← LinearAggregator, CESAggregator, KoopmansAggregator
├── model.py             ← Model
├── outer_search.py      ← outer-search strategy configuration (re-export façade)
├── params.py            ← as_leaf + the MappingLeaf / SequenceLeaf re-exports
├── persistence.py       ← SolveSnapshot, SimulateSnapshot, load_snapshot,
│                          complete-result save/load and legacy value reader
├── phased.py            ← Phased
├── processes.py         ← the seven *Process classes
├── regime.py            ← Regime
├── result.py            ← SimulationResult
├── solver_api.py        ← versioned solver, artifact, replay, and solution contracts
├── solvers.py           ← built-in solvers plus the out-of-tree re-export façade
├── taste_shocks.py      ← ExtremeValueTasteShocks
├── transition.py        ← fixed_transition, MarkovTransition, JointTransition
├── typing.py            ← user-facing type aliases, string labels, UserFunction
├── exceptions.py        ← every project-specific exception class
├── _compilation_cache.py ← this project's slice of the persistent JIT cache
└── _solver_api/         ← the implementation behind lcm/solver_api.py
```

```
_lcm/
├── __init__.py            ← bootstraps `lcm`, so the claw is installed either way
├── ages.py                ← AgeGrid validators and step parsing
├── axis_boundaries.py     ← ownership rules for one-dimensional interior boundaries
├── beartype_conf.py       ← the beartype configurations
├── certainty_equivalent.py ← certainty-equivalent classes and engine helpers
├── coarse_transition.py   ← the shared-evaluation cell of a coarse regime transition
├── config.py              ← build-time configuration constants
├── continuation.py        ← ContinuationSpec / EGMContinuationSpec and the artifact key
├── dtypes.py              ← canonical-dtype resolution
├── engine.py              ← canonical / engine-side dataclasses
├── gated_edge.py          ← what a ValueDependentTransition decomposes into
├── identity_transition.py ← the identity law behind lcm.fixed_transition
├── logsum.py              ← EV1 smoothed maximum and choice probabilities
├── model_processing.py    ← Model.__init__ build pipeline
├── pandas_utils.py        ← pd.Series ↔ JAX array bridge
├── post_decision_bound.py ← the checkable lower-bound declaration
├── power_mean.py          ← stable weighted power mean
├── probability.py         ← one reading of a probability's bits, shared by consumers
├── reachability.py        ← construction-time solve/simulate regime graphs
├── state_action_space.py  ← materialize a regime's state / action grids
├── transition_checks.py   ← pre-solve regime + state transition prob checks
├── transition_plans.py    ← canonical target-edge transition plans
├── typing.py              ← engine-side aliases and protocols
├── user_regime_validation.py ← validators for the user-facing Regime
├── variables.py           ← factories that build `Variables` from `Regime`
├── version.py             ← generated version string (hatch-vcs), untracked
├── zero_safe.py           ← arithmetic treating an exactly-zero weight as a null event
├── constraints/           ← declared constraints: normalization, routes, materialization
├── docs/                  ← render reference tables from public declarations
├── egm/                   ← the EGM family's kernels, envelopes and outer search
├── execution/             ← placement, transfers, liveness, admission, width selection
├── grids/                 ← grid infrastructure
├── optimization/          ← safeguarded scalar optimization primitives
├── params/                ← params templating and processing
├── persistence/           ← snapshot I/O and versioned solution-archive internals
├── processes/             ← stochastic-process infrastructure
├── regime_building/       ← per-regime canonicalisation
├── simulation/            ← forward sampling (simulate) + its planning and admission
├── solution/              ← backward induction (solve), the shipped solvers, admission
└── utils/                 ← small, dependency-light helpers
```

Names cross the boundary in exactly one direction (user → canonical form) and only twice
— once when `Model(regimes={...})` is called (which triggers
`model_processing.build_regimes_and_template` →
`regime_building.processing.process_regimes`), and once for `flat_params` at every
`solve` / `simulate` call.

## The public surface — `lcm/`

`lcm/` is the canonical home for every class users construct or consume. Keeping all of
them in one package makes the public surface easy to find, review, and keep stable —
anything in `_lcm/` is fair game for refactoring. **`lcm/` modules are shallow by
design**: each file holds class definitions and the small number of public top-level
functions that round out the surface. Validators, I/O plumbing, DataFrame assembly, and
similar implementation detail live in `_lcm/` and are imported back in.

The mapping of public names to files:

| File                                                                  | What lives there                                                                                                                                                                                                                                           |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model.py`                                                            | `Model`                                                                                                                                                                                                                                                    |
| `regime.py`                                                           | `Regime`, including its `koopmans_aggregator` slot — left empty, `finalize_regimes` injects the model-level default `lcm.LinearAggregator`. Validators live in `_lcm/user_regime_validation.py`; the phase normalizer in `_lcm/regime_building/phases.py`. |
| `ages.py`                                                             | `AgeGrid`. Step parser and validators live in `_lcm/ages.py`.                                                                                                                                                                                              |
| `grids.py`                                                            | `LinSpacedGrid`, `LogSpacedGrid`, `IrregSpacedGrid`, `DiscreteGrid`, `PiecewiseLinSpacedGrid`, `PiecewiseLogSpacedGrid`, `GridBreakpoint`, and the `@categorical` decorator                                                                                |
| `processes.py`                                                        | The seven `*Process` classes — `UniformIIDProcess`, `NormalIIDProcess`, `LogNormalIIDProcess`, `NormalMixtureIIDProcess`, `TauchenAR1Process`, `RouwenhorstAR1Process`, `TauchenNormalMixtureAR1Process`.                                                  |
| `persistence.py`                                                      | `SolveSnapshot`, `SimulateSnapshot`, `load_snapshot`, complete-result `save_solution` / `load_solution`, and `load_legacy_solution`. Archive and snapshot writers live in `_lcm/persistence/`.                                                             |
| `result.py`                                                           | `SimulationResult`. DataFrame assembly, metadata, and additional-targets computation live in `_lcm/simulation/result_*.py` and `_lcm/simulation/additional_targets.py`.                                                                                    |
| `solver_api.py`                                                       | Lightweight versioned contracts for solver identity, kernel output, artifacts, replay, lazy solution stores, and descriptive result metadata.                                                                                                              |
| `solvers.py`                                                          | Built-in solver configurations and the complete public re-export façade used by an out-of-tree solver.                                                                                                                                                     |
| `params.py`                                                           | `as_leaf` plus the `MappingLeaf` / `SequenceLeaf` re-exports. The leaf-class definitions and the engine params machinery live in `_lcm/params/`.                                                                                                           |
| `typing.py`                                                           | The model-authoring aliases (`FloatND`, `ScalarInt`, `Period`, `Age`, ...), the domain string labels (`RegimeName`, `StateName`, ...), the `UserFunction` protocol, and the `User*` boundary aliases.                                                      |
| `exceptions.py`                                                       | Every project-specific exception class.                                                                                                                                                                                                                    |
| `execution.py`                                                        | `ExecutionConfig` — devices, sharded states, planner-owned axis widths, the device-memory budget and its headroom fraction.                                                                                                                                |
| `phased.py`                                                           | `Phased`, the container for phase-specific variants of a regime-slot value.                                                                                                                                                                                |
| `transition.py`                                                       | `fixed_transition`, `MarkovTransition`, `JointTransition`.                                                                                                                                                                                                 |
| `collective.py`                                                       | `ValueDependentTransition`, `StakeholderRoute`, `ProjectedRegimeValue`, `ValueDependentConstraint`, `CollectiveUtility`, `ParetoObjective`.                                                                                                                |
| `koopmans_aggregation.py`                                             | `KoopmansAggregator`, `LinearAggregator`, `CESAggregator`.                                                                                                                                                                                                 |
| `certainty_equivalent.py`, `branch_aggregation.py`, `outer_search.py` | Re-export façades for classes whose definitions live in `_lcm/`.                                                                                                                                                                                           |
| `condition.py`                                                        | `lcm.ref` and the declared-condition vocabulary every solver reads the same way.                                                                                                                                                                           |
| `case_piece.py`                                                       | `case_boundary`, `piece`, `piecewise_affine`, `affine_breakpoint`, `smooth_helper`.                                                                                                                                                                        |
| `taste_shocks.py`                                                     | `ExtremeValueTasteShocks`.                                                                                                                                                                                                                                 |
| `fixed_forms.py`, `consumption_savings_regime.py`                     | The conventional accounting nodes and the specialized regime declarations a one-asset consumption-savings model can reuse.                                                                                                                                 |
| `_compilation_cache.py`, `_solver_api/`                               | Underscore-private inside `lcm/`: the persistent-cache location `lcm/__init__.py` needs before anything else imports, and the implementation `solver_api.py` re-exports.                                                                                   |

### Why a package boundary, not just naming?

A naming convention (a `_private_` prefix on every internal module) tells *readers* what
is internal. A package boundary makes it visible to *tools*: code-search, auto-import,
public-API audits, and the linter can all key off `_lcm/`. The boundary is enforced by
the absence of imports — `lcm/` modules import from `_lcm/` only at well-defined wiring
points, and `_lcm/` reaches the user-facing classes through aliased imports
(`from lcm.regime import Regime as UserRegime`).

## Grid and process infrastructure: `_lcm/grids/` and `_lcm/processes/`

```
_lcm/grids/
├── base.py            ← Grid, ContinuousGrid, UniformContinuousGrid (ABCs)
├── continuous.py      ← LinSpacedGrid, LogSpacedGrid, IrregSpacedGrid
├── discrete.py        ← DiscreteGrid
├── piecewise.py       ← PiecewiseLinSpacedGrid, PiecewiseLogSpacedGrid,
│                         GridBreakpoint
├── categorical.py     ← @categorical decorator + validators
└── coordinates.py     ← coordinate lookup helpers used by interpolation

_lcm/processes/
├── base.py            ← _ContinuousStochasticProcess + Gauss-Hermite / mixture helpers
├── iid.py             ← UniformIIDProcess, NormalIIDProcess, LogNormalIIDProcess,
│                        NormalMixtureIIDProcess
├── ar1.py             ← TauchenAR1Process, RouwenhorstAR1Process,
│                        TauchenNormalMixtureAR1Process
├── grid_resolution.py ← admission of a support only runtime params fix
└── state_conditioned.py ← direct-CDF rows when a shock parameter varies with a state
```

The leaf classes are surfaced through `lcm/grids.py` and `lcm/processes.py`; the ABCs
(`Grid`, `_ContinuousStochasticProcess`, etc.) are used by internal code but are not
part of the documented public API.

Two design points worth knowing:

- **Process classes bundle both a discretization grid AND a transition mechanism**,
  unlike ordinary grids which are pure outcome-space. Users place
  `UniformIIDProcess(...)` in `Regime(states=...)` directly — the transition is invoked
  automatically. Putting a process class in `state_transitions` is a bug.
- **All vocabulary in the engine speaks of `process`, not `shock`.** Use `is_process`
  (on `VariableInfo`), `process_names` (on `Variables`), and `ProcessName` (typing
  alias). `shock` is reserved for the colloquial meaning and never appears as an
  identifier.

## Private siblings of the public modules

Several `lcm/` modules have a private counterpart in `_lcm/` that holds their
implementation detail. The pattern is the same throughout: `lcm/` keeps the class
definitions and the public top-level functions; the `_lcm/` counterpart holds
validators, helpers, and I/O plumbing that internal code is free to refactor.

```
_lcm/ages.py            ← STEP_UNITS, _parse_step,
                          _validate_age_grid / _validate_range / _validate_values
_lcm/user_regime_validation.py  ← the validators called from Regime.__post_init__

_lcm/simulation/initial_conditions.py
                        ← MISSING_CAT_CODE, PSEUDO_STATE_NAMES, and the
                          build / validate helpers for initial conditions

_lcm/persistence/
├── io.py               ← _atomic_dump, _save_pkl, _save_h5, _load_h5,
│                          _get_platform, _next_counter, _enforce_retention,
│                          _write_metadata, _write_environment_files
└── snapshots.py        ← _save_solve_snapshot, _save_simulate_snapshot,
                          _strip_V_arr_from_result, _bind_forward_refs

_lcm/simulation/result_metadata.py
                        ← ResultMetadata + _compute_metadata, _get_output_dtypes
_lcm/simulation/result_dataframe.py
                        ← _create_flat_dataframe and the per-regime / per-period
                          assembly helpers, plus categorical conversion
_lcm/simulation/additional_targets.py
                        ← _resolve_targets, _compute_targets, and DAG helpers
                          for to_dataframe(additional_targets=...)
```

Why split these out? Two reasons:

- **The public surface is easier to audit.** `regime.py`, `persistence.py`, and
  `result.py` each contain only the dozen-or-so symbols users actually touch. A reader
  looking for "what is the public contract of a Regime?" sees that contract directly,
  without scrolling past validator bodies.
- **Internal helpers can move freely.** Anything in `_lcm/` is internal — its location,
  signature, and existence can change without bumping the user surface.

A note on shadowing: the canonical `Regime` lives in `_lcm/engine.py`. The validators in
`_lcm/user_regime_validation.py` operate on the user-facing `lcm.regime.Regime` and
reach it through TYPE_CHECKING-guarded imports to break the circular dependency at
import time; beartype resolves the forward references at first call.

## Engine-side: `_lcm/engine.py`

`engine.py` holds the **canonical** post-processing dataclasses — the form the DP
machinery operates on:

- `Regime` — the canonical regime (distinct from the user-facing `lcm.regime.Regime`; in
  source files that import both we alias the user-facing one as `UserRegime`).
- `StateActionSpace` — pre-built state and action grids for a regime.
- `SolutionPhase` / `SimulationPhase` — the canonical `Regime`'s two frozen phase
  namespaces, reached as `regime.solution` and `regime.simulation`. Each holds that
  phase's variables, grids and compiled function sets, so every phase-dependent read
  names its phase in the access path. `SolutionPhase.state_action_space(params)` is the
  method that fills a `StateActionSpace` with runtime-supplied grid points.
- `Variables` / `VariableInfo` — name + kind + topology metadata for every state and
  action in a regime.
- `PeriodRegimeSimulationData` — raw simulation output for one (regime, period) pair,
  before `SimulationResult` materialises a DataFrame.

The file name `engine.py` reflects what's inside: the engine's view of a model.

## Build pipeline: `model_processing.py` and `regime_building/`

A user `Regime` is finalized at model build — model-level slots merged, broadcast
variables pruned, the Koopmans aggregator and certainty equivalent injected,
completeness validated — into the plain, complete `Regime`s exposed as
`model.user_regimes`. The params template reads this user-vocabulary form, while
`process_regimes` internally splits each regime into canonical per-phase slices and
compiles the engine `Regime`.

```
_lcm/model_processing.py  ← top-level pipeline:
                            user regimes + params → canonical Model

_lcm/regime_building/
├── broadcast.py          ← model-level slot merge (exactly-one-level rule,
│                            `None` masking) + DAG-reachability pruning of
│                            broadcast states and actions
├── finalize.py           ← finalize_regimes: derived-categorical merge,
│                            Koopmans-aggregator and certainty-equivalent
│                            injection, completeness validation; output
│                            stays a plain lcm.regime.Regime
├── phases.py             ← normalize_regime_phases: expand every regime
│                            slot into per-phase RegimePhaseSpec slices
│                            (the Phased grammar boundary)
├── age_normalization.py  ← model-level normalization of age specialization
├── age_specialization.py ← per-age-specialized node resolution and
│                            grid-shape validation
├── canonicalize.py       ← canonicalize_regimes: rewrite every phase
│                            slice's laws and regime transition into the
│                            canonical target-granular form over exactly
│                            the reachable targets
├── processing.py         ← per-regime canonicalisation:
│                            UserRegime → engine.Regime
├── transitions.py        ← collect_state_transitions: walk user-supplied
│                            state_transitions into per-target callables
├── stochastic_state_transitions.py
│                         ← process-time AST + n_outcomes derivation for
│                            stochastic state transitions (raises
│                            InvalidStateTransitionProbabilitiesError on
│                            subscript-order mismatches)
├── Q_and_F.py            ← build (Q, F) closure for solve / simulate;
│                            also resolves the utility/feasibility DAG,
│                            whose two targets share one upstream chain
├── argmax.py             ← argmax helpers over action grids
├── max_Q_over_a.py       ← argmax / max over action grids
├── V.py                  ← value-function interpolation info
├── w_dag.py              ← user-DAG resolution for the Koopmans
│                            aggregator's *extra* params — those beyond
│                            utility and CE, which the Bellman step wires
│                            directly
├── next_state.py         ← compose per-state transitions into a single
│                            next_state function for simulation
├── ndimage.py            ← map-coordinates wrapper for continuous interp
├── collective.py         ← the stakeholder value gather at the household
│                            argmax
├── gated_edges.py        ← the gated-edge objects behind mutual-consent
│                            marriage / dissolution routing
├── fixed_process_laws.py ← bind process laws supplied through fixed_params
│                            into the process grids
├── transition_invariants.py
│                         ← construction-time checks that the two transition
│                            namespaces stayed separate
├── zero_safe.py          ← zero-weight-safe arithmetic for the collective
│                            solve core
└── diagnostics.py        ← cold-path machinery invoked by validate_V to
                            pinpoint *which* intermediate produced a NaN
```

### Broadcast pruning is one fixed point over both phases

`prune_broadcast_variables` weeds each regime's *broadcast* states and actions — the
ones declared at model level — by DAG reachability: a broadcast variable survives in a
regime only if a root computation of that regime (utility, the Koopmans aggregator,
constraints, derived categoricals, the regime transition, or a law of motion toward a
target that keeps the state) transitively reads it. Regime-level declarations are never
pruned, and `model.pruned_variables` records the outcome per regime.

The two phase slices are not pruned independently. They feed each other — a target that
keeps a state only because its *simulation* slice reads it makes the *solution*-side
entry law toward that target a pruning root, and whatever that law reads then has to
survive in the source — so `_joint_phase_closure` alternates the two slice operators
until the kept-sets stop growing. The retained set is the least fixed point of both
operators taken jointly, not one application of each.

Pruning a state from a regime does not delete its whole law of motion.
`_retained_state_transition` keeps the part that survives: a law keyed by target regime
is an *entry* law, placing a value on the support of a target that carries the state and
saying nothing about the source's own copy, so the cells aimed at targets that retain
the state stand. The entry is dropped only when nothing the regime reaches can receive
the value. That is what makes promoting a state from regime level to model level a
declaration move rather than a change of transition structure.

The two-step name (`model_processing` at the model level, `regime_building` per regime)
reflects what each layer actually does — the top level merges regimes and resolves fixed
params; each regime is then canonicalised independently.

### Co-mapped landing coordinates

A continuation whose target is read at a landing point co-mapped with the value array
needs the landing coordinates named, not guessed. `_co_mapped_landing_names`
(`Q_and_F.py`) asks the continuation's interpolator — gated or plain — which arguments
it declares, and returns the `next_<state>` names among them for the states co-mapped
with the continuation's `V`. Only what the interpolator actually names is supplied, so
adding a co-mapped state does not silently start feeding a coordinate no reader
consumes.

The numerical checks fired at solve / simulate time live outside `regime_building/`:

- `regime_building/stochastic_state_transitions.py` runs at `Model(...)` construction
  time and can fail the build before any params are involved. It catches malformed user
  functions (e.g., `probs_array[health, age]` where the signature is `(age, health)`)
  via AST analysis. Always on, never gated.
- `_lcm/transition_checks.py` runs from `Model.solve()` / `Model.simulate()` before
  backward induction starts. It evaluates the regime and state transition functions on
  the regime's grid Cartesian product and verifies output shape, [0, 1] range, and
  sum-to-1. State checks are gated by `log_level != "off"` because the Cartesian product
  can blow up on models with many continuous-grid-dependent stochastic states.
- `_lcm/solution/validate_V.py` runs *during* backward induction (after each period in
  `backward_induction.py`, and once on the V handed to `simulate.py`). On NaN it invokes
  the diagnostic-intermediates closure built in `regime_building/diagnostics.py` to
  pinpoint which intermediate (`U`, `F`, `E[V]`, `Q`) produced the NaN.

## Reachability: `_lcm/reachability.py`

`build_model_reachability` builds the model's static solve and simulate graphs once, at
model construction, from the single canonical `active_periods_by_regime` mapping
(`regime_building.processing.compute_active_periods_by_regime`) and the declared regime
transitions. There is no runtime topology pass — the graph never changes after
construction, and no runtime probability value narrows or widens it.

Every retained edge is `EdgeStatus.CONDITIONAL`; there is no `TRUE` status, because no
declaration form (not even a per-target dict with one key) proves unconditional positive
probability independently of state, action, and free runtime parameters. A coarse (bare
callable / bare `MarkovTransition`) regime transition is therefore conservative: it
retains an edge to every regime active in the next period, and every such edge's state
handoff is checked at model build — a carried state, a deterministic/stochastic law, or
an explicit target-local/entry law must supply each target state's next-period value. A
per-target dict narrows support to its declared key set instead.

The solve and simulate phases build independent graphs (`ModelReachability.solution` /
`.simulation`), because a regime transition's `Phased` sides can differ between them —
so the two graphs may retain different edges for the same source period.

Solver and simulation runtime code (`_lcm/solution/`, `_lcm/simulation/`) consume this
graph — `PhaseReachability.targets`, `.union_targets`, `.edge_status`, ... — but never
infers reachability itself: no runtime module calls an activity predicate, inspects a
declared transition's raw mapping keys, or derives continuation-target membership from
state-law-bundle keys. `regime_building/processing.py` and `diagnostics.py` read the
graph to decide which targets a period's `Q_and_F` (or diagnostic) closure needs to
build; they do not re-derive it.

## Solve and simulate

`_lcm/solution/` and `_lcm/simulation/` are the JAX-traced hot paths, and each has grown
a planning layer around its numerical core. Both packages are large enough that the
useful map is by *family* rather than by file; the responsibility of every individual
module is in its own module docstring.

```
_lcm/solution/
├── backward_induction.py  ← the loop: V[T], V[T-1], ..., V[0], driving whichever
│                            solver each regime declares
├── contract.py            ← the seam a solver meets the loop through
├── grid_search.py, egm.py, dcegm.py, negm.py, nbegm.py, nnbegm.py
│                          ← the shipped solvers; shipped_solvers.py lists them
├── action_reduction.py, action_streaming.py, logsumexp_action_reduction.py,
│   collective_action_reduction.py
│                          ← the blockwise reductions over the action product
├── model_authority.py, replay_validation.py, native_values.py, period_capture.py,
│   period_replay.py       ← what a replay route may assume, and the preflight of it
├── solve_inputs.py, continuation_reads.py, continuation_target.py,
│   continuation_arguments.py, undeclared_reads.py, retained_buffers.py
│                          ← which artifact each dispatch reads, and who still owns it
├── v_topology.py, kernel_output.py, kernel_attribution.py, artifacts.py,
│   result_snapshot.py, fingerprint.py, model_seal.py
│                          ← what a period publishes, and how it is identified
├── preconditions.py, periodization.py, solver_diagnostics.py, diagnostics.py
└── validate_V.py          ← per-period NaN / Inf validation

_lcm/simulation/
├── simulate.py         ← the forward loop over periods, regimes and subjects
├── programs.py, program_types.py, program_arguments.py, policy_programs.py
│                       ← the per-regime work, declared as planner-owned programs
├── runtime.py, compile.py, unit_executor.py, subject_devices.py, subject_parallel.py
│                       ← lowering, the prepared-route probe, and device-local dispatch
├── chunk_*.py, memory.py, residency.py, value_reads.py, value_placement.py,
│   entry_*.py, operand_placement.py, host_operations.py, solution_copies.py
│                       ← budgeted chunking and the admission of every payload
├── gated_routing.py    ← the forward value router for gated edges
├── random.py           ← PRNG-key handling for the sampling draws
├── transitions.py      ← per-state transition composition for simulation
├── plan_summary.py     ← the diagnostic record of the resolved plan
├── result_metadata.py, result_dataframe.py, additional_targets.py
└── initial_conditions.py
                        ← canonicalize / validate the user's
                          initial_conditions kwarg
```

The DP and sampling logic is the only *numerical* thing here; everything that constructs
the inputs (parameters, grids, transitions, compiled callables) lives in
`regime_building/` and is read out of the canonical `Regime` instances. Everything that
decides *where* those inputs live and *whether* they fit lives in `_lcm/execution/`.

## Execution planning: `_lcm/execution/`

`ExecutionConfig` is a user declaration; `_lcm/execution/` is what resolves it against
what a model actually declares, before anything is compiled. The package owns five
concerns that the solve and simulate loops only consume:

- **Placement.** `plan_submesh_placement` (`placement.py`) decides the devices each
  regime's nodes run on. Sharding follows pruning: a regime that prunes the sharded
  state runs single-device, and only a state *every* regime prunes is refused.
- **The transfer catalogue.** `value_transfer.py` resolves each declared value read into
  exactly one operator (see below).
- **Liveness and donation.** `liveness.py` counts the declared consumers of each planned
  input and `donation.py` decides which arguments a dispatch may hand to its executable,
  so a buffer is released only after its last declared consumer returns.
- **Scheduling.** `scheduler.py` turns the resolved programs into waves and fixes
  physical buffer lifetime once the ledger closes a count.
- **Budget and width.** `execution_plan.py` resolves the device-memory budget and
  `workspace_planning.py` selects the widest workspace widths that fit inside it.

### The transfer catalogue has six kinds

A stored value is written on one regime's layout and read by another's core, and the two
need not agree. `classify_value_transfer` is a total function from (stored layout,
required layout) onto one `ValueTransferKind`:

- `ALIGNED_LOCAL` — already resident on the source core's mesh; passed through with its
  own partitioning intact.
- `COPY_TO_SOURCE_LAYOUT` — an explicit copy into the required layout on that mesh.
- `ALL_GATHER` — a partitioned value the reader needs whole.
- `LOCAL_SLICE` — a replicated value the reader needs only its own shard of.
- `RESHARD` — a change of partitioning within one mesh.
- `CROSS_MESH_COPY` — a copy between two different device sets, for disjoint or nested
  meshes only.

The one pair no single collective can serve is two meshes that *partially* overlap with
neither containing the other; that is refused with an `ExecutionPlanningError` rather
than approximated. A value stored off the source mesh is delivered as a replica
(`jax.P()`) on the source mesh, and a request for partitioned cross-mesh delivery is
refused on the same route. The reference table is in
[Custom solvers](../reference/custom_solvers.md).

Each planned read also authenticates its declared
`(source_regime, source_period, core_key)` against the compiled core it claims to come
from, so agreement among declarations cannot make a different node authoritative.

### Continuous-route transfer scratch

Every non-`ALIGNED_LOCAL` transfer needs somewhere to land, and that scratch is charged
against the budget only on the narrow continuous sharded-state route
(`_period_transfer_scratch_reservations`, armed when
`execution.continuous_sharded_state is not None`). The reservation is a declared
conservative envelope, not measured allocator scratch: every copy is assumed pending
simultaneously and overlapped with every core's compiler peak, while a copy several
consumers share is counted once. It requires each transfer's endpoints to equal the
admission devices, which is what keeps the route inside the devices whose resident and
concurrent-output burdens were planned.

### Device memory: the headroom, and the note

`ExecutionConfig(device_memory_bytes=...)` is a request, not the ceiling admission uses.
`_effective_device_memory_bytes` takes the minimum of the request and every selected
device's allocator pool limit less `device_memory_headroom_fraction` of it (0.15 by
default), so an already-conservative request is never reduced twice and a device that
reports no limit contributes no cap. A `None` request applies no pool-derived cap,
though public model construction still reads each visible device's pool statistics once.
The result is floored at one byte: a pool small enough that its headroom consumes all of
it still yields a budget, one that refuses every width — which is the honest outcome,
where a non-positive budget would not be a budget at all.

The resolved budget is logged as a summary line naming the request, the headroom
fraction, the per-device limits and the effective ceiling — at warning level when the
devices capped the request, at info level otherwise. `device_memory_cap_note()` returns
the same fact as a clause that admission refusals append, so a refusal always says which
of the two budgets it was measured against.

Workspace admission combines a represented compiler allocation reservation with external
resident storage. The reservation enforces both the raw peak and
`argument + output - alias + temporary` bytes for each complete device record. Profiled
simulation operations place all dynamic arguments on the execution devices and compile
with `keep_unused=True`, so their reported argument storage includes inputs used only
for shape or dtype.

Solve and simulation cores allow the compiler to eliminate unused arguments. Each
candidate's public `Compiled.input_shardings` tree identifies the surviving dynamic
input occurrences; eliminated arguments remain in external residency while their owners
are live. Simulation subtracts only surviving inputs' actual buffer spans, preserving
uncovered portions of larger aliased owners. Solve queries its schedule inventory for
the exact regime, period, core, and widths, excluding only surviving declared reads on
their stored layout. Logical artifact aliases determine solve residency; template arrays
do not establish aliases between future outputs. These exclusions change neither the
declared reads nor their consumer lifetimes, and do not require moving unused inputs to
the core's devices.

Solve also charges its concrete fixed owners throughout the solve, reserves shared
transfer destinations for the whole period, and reserves internal producer outputs
within their regime-period cell. Fixed-owner aliases are unioned on each actual device;
abstract shape templates own no device storage. Known internal output shardings
determine their per-device payload; missing layouts use a full-payload bound. These
reservations can overlap compiler-counted inputs, intentionally over-counting storage.
Width selection is widest under this declared bound, rather than an allocator-optimal
choice.

Compiler peaks remain the actual executable reports. Complete allocation counters must
include retained input payloads; unavailable or mismatched input metadata refuses
budgeted core planning. Executable reuse still requires a fresh residency check. The
remaining limits of whole-call simulation accounting are recorded in the
[architecture transition ledger](../development/architecture_transition_ledger.md).

Budgeted simulation entry retains the original caller arrays and existing model grids,
fixed parameters, regime IDs and ages while canonicalizing plain numeric parameters and
initial conditions. Host dtype conversion and the existing range checks run first; each
upload admits its destination payload and declared transfer scratch before allocating on
the first selected device. This staging device is explicit even when device zero is
excluded. Padding profiles the unchanged last-row repeat and concatenate operation
against its represented compiler reservation. Completed leaves remain owned and charged
before the next leaf is admitted; the full canonical input mapping remains live
throughout padding. Subsequent subject placement uses the ordered simulation devices
after padding. Series and DataFrame values are assembled on the host and use the same
upload admission. Automatic solves retain and charge these simulation inputs alongside
their solve inputs.

Foreign eager value stores use an explicit call-local allocator for every private JAX
copy. Each copy preserves the source shape, dtype, sharding and device order, and admits
the exact compiled copy's memory report against the originals and earlier copies. The
entry owner retains intermediate copies until validation commits the resolved values;
failure releases that transient bank. A remembered validated view needs no new copies.
Source devices outside the simulation subset still count when they use the same backend.
Mixed CPU/accelerator foreign copies are refused before copying; an accelerator device
ceiling is not a host-memory budget. Trusted native GridSearch value archives use a
call-local loader after metadata, authority and coordinate validation. Each host leaf is
checked against its archived address, shape, dtype and checksum before an admitted
upload; dtype narrowing is refused before upload. Unloaded values stage on the first
selected device because archives do not serialize source sharding. Preloaded cache
arrays keep their actual source layout. Both detached copies are admitted while the
original cache and earlier copies remain live. A failed upload leaves the entry
unloaded; failure during a later copy preserves its already published private cache.

Native cache construction has a separate serialization lock from its brief cache
peek/publication lock. Admission snapshots never hold a cache lock across loading or
allocation, so simultaneous entry loads can inspect the complete cached bank. These
locks protect entry memoization; they do not coordinate budgets across concurrent
mutations of an entire result. Loader callbacks remain call-local and do not enter the
archive cache or consumed-view memo. Budgeted foreign artifact authorities, native
artifact payloads and arbitrary lazy decoders remain unprofiled and are refused before
their copying or upload callbacks. Their unbudgeted behavior is unchanged.

## Forward simulation: subjects, gates and the prepared route

Simulation's own vocabulary is about the *population* axis, which solve does not have.

### Subject parallelism is declared, not inferred

`SubjectShardable` (`simulation/subject_parallel.py`) is an explicit capability, not a
shape coincidence: a program declaring it promises that its named argument subtrees
carry the independent subject axis, that every output leaf retains that axis, and that
no operation crosses subjects. Merely having an array whose leading extent equals the
population is *not* this declaration — parameter arrays and solution reads are shared
even when their first dimension happens to match.

`declared_subject_shard_arg_names` reads the declaration off the innermost callable,
seeing through the `functools.partial` wrappers the model layer puts around a body to
bind fixed params, and drops every name a binding has already consumed, so only
arguments still supplied per dispatch are offered for partitioning. A positional binding
is refused, because it renames nothing and would shift the keyword contract the
declaration is written in.

The two halves of a gated edge use this to split differently:

- The **gate fold** (`_GateFoldBody`) reads the next period's value and dissolution
  arrays on each target's own regime-level grids and writes the substituted continuation
  over those same grids. No operand and no output carries a subject axis, so it declares
  `()` — the empty tuple — and is *replicated* wherever the population is spread over
  several devices.
- The **gate route** declares `_GATE_ROUTE_SUBJECT_ARG_NAMES` — `next_states`,
  `new_subject_regime_ids`, `subjects_in_regime`, `own_stakeholder`,
  `new_own_stakeholder` — and is *partitioned* across the subject devices. The grids the
  fold published stay shared operands of it.

### The prepared unbudgeted dispatch route

An unbudgeted repeat of an exact abstract signature does not re-walk the full
materialize-and-plan route. `SimulationRuntime.dispatch` probes a prepared-route record
*before* anything is materialized, keyed on the complete abstract signature of the
caller's arguments, and taken only when both `execution.device_memory_bytes is None` and
no residency context is supplied. A hit binds this call's own leaves onto the cached
static preparation; a miss — budgeted, first-seen, or freshly bound operands that no
longer match — takes the validated route, which then publishes the record an exact
repeat may reuse. The record holds immutable abstract description and compiled code
only: never a caller array, and never an admission or validation verdict. The runtime's
own configuration is immutable for its lifetime and therefore deliberately absent from
the key — a different execution config, device order, width or JIT disposition is a
different runtime with its own empty cache.

Without a declared budget the runtime still derives a subject tile width rather than
using a fixed one: `_unbudgeted_subject_width` caps one tile's argument slice at a
constant byte block, weighing only the operands the materialized program already holds
abstractly, so the result depends on nothing but the model's shapes, dtypes and the
population size and is identical on every backend. It never falls below the fixed
default width, and it passes through the same admissible-width check, so axis alignment,
the floor and the extent clamp are unchanged. Width is a lowering specialization only: a
wider tile moves no value and no RNG stream.

## The solver seam: keys and routes

Two declarations connect a solver to the engine without either reading the other's
concrete types.

**A continuation is keyed, not concrete.** `_lcm/continuation.py` defines
`ContinuationSpec`, pairing an all-finite template with the `ArtifactKey` under which
its kernels publish it; `EGMContinuationSpec` is the EGM family's specialization, adding
the layout a reading parent needs. A solver declares what it *reads* as
`Solver.required_continuation_keys`, and `process_regimes` matches every declared key
against what each reachable target publishes before anything compiles, so a version
mismatch is a build error naming both regimes rather than a failure inside the first
rolled period. `ContinuationPayload` is the `ContinuationArtifact` protocol — one
property, `artifact_key` — so backward induction stores and rolls a payload of any type.
The shipped EGM family still exchanges concrete `EGMCarry` fields between its own
producers and readers; the transition ledger records what closing that gap requires.

**A replay route is declared, not discovered.** Every canonical regime answers
`simulation.replay_route` with one object carrying a `ReplayMode`, the exact payload
class it retains, whether a solve owes one, and the reader that consumes it.
`EGMPolicyRead` and `NNBEGMPolicyRead` are the built-in routes for the EGM and nested
NB-EGM families; a regime that retains nothing declares grid recomputation. An external
solver instead returns an `ExecutableReplayRoute` from `SolutionKernels`. Its stable
plugin and route identities, period-specific artifact requirements, per-cell artifact
authorities, mathematical validator, and JAX-transformable reader all use types
re-exported by `lcm.solvers`.

`model_authority.py` builds authority from the canonical model and consuming route. A
restored or caller-supplied result is canonicalized once; required lazy entries are
materialized and checksum checked; then pylcm and plugin validation preflight every
required cell before forward execution. Each period acquires its own placed payload and
coordinate copies. The plugin validates that exact immutable snapshot and context again
immediately before reader construction; the reader receives those same objects.
Authority and descriptive metadata retain their validated identities through placement.
Descriptors transported in `SolutionMetadata` remain descriptive and cannot authenticate
their own payloads.

## Solution identity and persistence

`Model.solve()` returns one `SolutionResult` containing a `ValueStore`, addressed
artifact stores, descriptive metadata, and explicit omission reasons. In-memory entries
are loaded; entries restored by `load_solution` are independently lazy. Loading one
period/regime value or artifact does not project or materialize its siblings, and an
unloaded entry is present rather than omitted.

The model fingerprint in `SolutionMetadata` is a deterministic digest of the canonical
mathematical declarations and parameters needed to interpret a result. It includes grid
support, category order, solver/replay/artifact identities, and callable semantics while
excluding device, compiler, tiling, sharding, JIT, and other execution-only choices. An
in-memory result additionally carries a process-local model-instance guard; a restored
archive is accepted by a separately constructed compatible model through the durable
fingerprint.

`_lcm/persistence/solution.py` writes one HDF5 archive through an atomic sibling file.
Its manifest is JSON; every value or artifact leaf is a separately addressed numerical
dataset whose checksum binds its logical address, shape, dtype, and bytes. The archive
serializes no Python implementation. `load_solution` checks the manifest and exact
format/solution/solver-interface versions, then returns lazy handles that reopen the
archive and verify a leaf before caching it. A whole-archive checksum pass deliberately
does not change load state. Plugin PyTrees are reconstructed only from an installed
route's model-authoritative template.

Persistence is a per-artifact decision. `ArtifactDescriptor.persistence` is
`MODEL_VERIFIABLE` only when another model process can independently rebuild the
corresponding `ArtifactAuthority`; otherwise saving records `NOT_PERSISTED` and omits
the payload. This keeps adaptive solve-generated coordinates out of the trust root. The
same artifact identities drive computation: `REPLAY` programs are value-producing
alternatives and `ARTIFACT` programs are additive, while each names its exact
`CoreProgram.retained_artifact_keys` and the exact final `KernelOutput` payload type for
every retained key. Every producer of one key must agree, including programs republished
through a composite kernel; every replay program also names the exact values-only
program it replaces. The engine selects them independently for every period/regime cell
from model-authoritative `ArtifactRef` values, so requesting one persistable auxiliary
or replay artifact does not suppress an unrelated values program.

## Params: boundary form vs. canonical form

```
_lcm/params/
├── processing.py       ← cast_params_to_canonical_dtypes:
│                          User-supplied dicts (with int/float/np.array
│                          leaves) → flat MappingProxyType keyed by
│                          qualified names with JAX-array leaves.
├── regime_template.py  ← per-regime template construction:
│                          inspect Regime functions to derive what
│                          parameters they need.
├── mapping_leaf.py     ← UserMappingLeaf / MappingLeaf — wrapper that
│                          carries an immutable dict through a JAX pytree
│                          without becoming a Mapping itself.
└── sequence_leaf.py    ← UserSequenceLeaf / SequenceLeaf — same for
                           sequences.
```

The public `lcm/params.py` module exposes `as_leaf` and re-exports the four leaf
classes; their definitions and the engine params machinery live in `_lcm/params/`.

Two leaf types exist because params dicts can contain heterogeneous leaves (scalars,
arrays, named tuples, etc.). Wrapping them in `MappingLeaf` / `SequenceLeaf` lets the
pytree machinery treat them as opaque leaves rather than walking into them — important
when a "leaf" is itself a dict mapping named arguments to JAX arrays.

The `User*` types accept the wide boundary form (`int`, `float`, `np.ndarray`,
`pd.Series`, etc.). After `cast_params_to_canonical_dtypes` runs, only canonical
JAX-array leaves and canonical-narrow `MappingLeaf` / `SequenceLeaf` instances survive.
The downstream `solve` / `simulate` code only ever sees the canonical form.

## Pandas bridge: `_lcm/pandas_utils.py`

A single module for converting between user-friendly `pd.Series` / `pd.DataFrame`
representations and the JAX arrays the engine expects. `array_from_series` is the
workhorse: it inspects the function source via AST helpers (in
`utils/ast_inspection.py`) to determine the expected multi-index order, then
materialises a properly-shaped JAX array.

This file gets used both at params processing (for any `pd.Series` leaves in user
params) and at simulation output (for building the result DataFrame).

## Utilities: `_lcm/utils/`

Small, dependency-light helpers grouped by topic:

- `ast_inspection.py` — Parse a function body to find `probs_array[a, b]` subscript
  patterns. Used by the static AST check and by `pandas_utils`.
- `containers.py` — `ensure_containers_are_immutable`, `first_non_none`,
  `invert_regime_ids`.
- `dispatchers.py` — `productmap`, `vmap_1d`, `simulation_spacemap`. See
  [Dispatchers](dispatchers.ipynb).
- `error_messages.py` — `format_messages`, which collapses a list of validation errors
  into a single string.
- `functools.py` — `all_as_kwargs`, `get_union_of_args`.
- `logging.py` — `get_logger`, `format_duration`, log-formatting helpers.
- `namespace.py` — `flatten_regime_namespace` / `unflatten_regime_namespace` for the
  qualified-name pytree keys.

## Type aliases: `lcm/typing.py` vs `_lcm/typing.py`

```
lcm/typing.py    ← user-facing aliases: jaxtyping array shapes (FloatND,
                   ScalarInt, ...), Period, Age, the domain string labels
                   (RegimeName, StateName, ...), the UserFunction Protocol,
                   and the User* boundary aliases (UserParams,
                   UserInitialConditions, ...)
_lcm/typing.py   ← engine-side aliases and protocols: the compound mapping
                   aliases, the canonical post-processing forms (Params,
                   InitialConditions, ...), and the structural Protocol
                   classes (EconFunction, TransitionFunction, ...)
```

The split mirrors the public / private package boundary. `lcm/typing.py` holds the
aliases a user needs to annotate model functions and the `User*` aliases that accept
wide boundary types; it imports nothing from `_lcm`. `_lcm/typing.py` holds the
engine-internal aliases — including the post-canonicalization forms (`Params`,
`InitialConditions`) — and builds on the public aliases it imports from `lcm.typing`.

The domain string labels (`RegimeName`, `StateName`, `ActionName`, `ProcessName`, ...)
are PEP 695 aliases of `str` that exist purely to make signatures self-documenting. They
are *defined* in `lcm/typing.py` and re-exported from `_lcm/typing.py`, so
`from _lcm.typing import RegimeName` keeps working while there is only one definition.

## Exceptions: `lcm/exceptions.py`

Every project-specific exception class lives here, all inheriting from `PyLCMError`.
They split into two categories:

- **Initialization errors** — raised at `Model(...)` / `Regime(...)` /
  `LinSpacedGrid(...)` construction time. These map beartype violations on user-facing
  constructors to a project-typed error (so users see e.g. `ModelInitializationError`,
  not a `BeartypeCallHintViolation`).
- **Runtime errors** — `InvalidValueFunctionError`,
  `InvalidRegimeTransitionProbabilitiesError`,
  `InvalidStateTransitionProbabilitiesError`, `InvalidParamsError`,
  `InvalidInitialConditionsError`, `SolutionIntegrityError`, and
  `IncompatibleSolutionError`. These fire from transition/value checks, restored-result
  preflight, or the solution-archive reader during solve / simulate / load.

The exception classes are public, and `lcm.exceptions` is the one place to reach them:
`from lcm.exceptions import InvalidParamsError`. They are deliberately not re-exported
on the top-level package, so `lcm.InvalidParamsError` does not resolve.
`ExecutionPlanningError` — every budget, width, device and sharding refusal — is the one
a user tuning a model meets most often. `format_messages`, the helper that assembles a
list of validation errors into one string, is internal validation plumbing and lives in
`_lcm/utils/error_messages.py`.

## Bootstrap modules

A few `_lcm/` modules exist for ordering reasons rather than for any conceptual
grouping:

- `beartype_conf.py` — Holds the beartype configurations used in the package (the
  internal-claw conf + the user-facing constructor-decorator confs). `lcm/__init__.py`
  registers `beartype_package` over both `_lcm` and `lcm` with `INTERNAL_CONF` before
  either package's submodules load, and `_lcm/__init__.py` bootstraps `lcm` so importing
  the private package first still installs the claw.
- `config.py` — Build-time configuration constants (paths to test data, etc.).
- `dtypes.py` — Canonical-dtype resolution (`canonical_float_dtype()`), which depends on
  the JAX x64 setting.

## Reading order for new contributors

If you're reading the codebase for the first time, the path of least confusion is:

1. **`lcm/regime.py`** to see what users supply.
1. **`lcm/model.py`** to see what `Model.__init__` triggers.
1. **`_lcm/model_processing.py`** for the top-level pipeline.
1. **`_lcm/regime_building/processing.py`** for per-regime canonicalisation — the
   longest single file and the heart of the build.
1. **`_lcm/engine.py`** for the canonical dataclasses the DP machinery consumes.
1. **`_lcm/reachability.py`** for the static solve/simulate graphs solve and simulate
   consume but never infer.
1. **`_lcm/solution/backward_induction.py`** and **`_lcm/simulation/simulate.py`** for
   the actual DP and sampling.

By the time you reach (7), the canonical form should feel familiar and the JAX-traced
code becomes easy to read.
