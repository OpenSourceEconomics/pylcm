---
title: Model and Regime
---

# Model and Regime

(api-model)=

## `Model`

`Model(...)` assembles regimes over one lifecycle:

```python
import lcm

model = lcm.Model(
    ages=lcm.AgeGrid(start=25, inclusive_stop=80, step="Y"),
    regimes={"working": working, "retired": retired},
    regime_id_class=RegimeId,
    edges={"working": {"retired": 25}},
    initial_nodes=lcm.InitialNodes(by_age={25: "working"}),
    enable_jit=True,
)
```

Required arguments are exactly one of `ages` and `n_periods`, plus `regimes`, `edges`,
`initial_nodes`, and `regime_id_class`, a class created with
`@categorical(ordered=False)` whose fields match the regime names. A model must contain
at least one terminal regime; a model whose starts are all terminal needs no other.

`n_periods` is a positive integer excluding booleans and counts every computational
slot, including an explicitly represented terminal slot. In this mode `model.ages` is
`None`, graph selectors use `PeriodRange` or `Periods`, starts use
`InitialNodes(by_period={0: "working"})`, and simulation inputs carry a `period` column.
The model does not assign an age or elapsed duration to a period. See
[Periods and temporal parameters](../user_guide/period_time.md).

`edges` declares every regime transition, structure and law. Each source maps to a plain
`{target: source_ages}` mapping when every source age has one destination (the graph is
the law), or to `Transition(targets={target: source_ages, ...}, law=..., gates=...)`
when some source age has several or a destination is gated. `targets` may be omitted
when the law names its targets, and is then derived from it. A regime with no outgoing
edges is terminal. See
[Regime transitions and graph support](transitions.md#api-regime-transitions).

The mapping-valued slots `functions`, `constraints`, `states`, `state_transitions`,
`actions`, and `derived_categoricals` broadcast declarations to regimes. A name is
defined at model or regime level, never both. A regime-level `None` masks a broadcast
entry. Broadcast variables unused by every root computation in both phases are pruned;
inspect `model.pruned_variables`.

`output_only_functions=(...)` names functions kept solely for simulation reporting.
Declare a name on the model or on a regime, never both. Model-level names apply to each
regime defining them. Missing names, declarations of already-read functions, and unread
functions without a declaration raise at build. See
[Output-only functions](../user_guide/defining_models.md#output-only-functions).

`koopmans_aggregator` and `certainty_equivalent` are single broadcast values. Declare
each at model level or in every non-terminal regime, not a mixture. `fixed_params` binds
parameters when the model is built. Simulation takes its population from each call's
initial conditions; compiled runtime programs are reused when their shapes and other
compilation inputs match.

Public inspection attributes include:

- `ages`, `n_periods`, and `regime_names_to_ids`;
- `user_regimes`, the finalized declarations in user vocabulary;
- `edges`, the edges exactly as declared, `Transition` laws included;
- `initial_nodes`, a normalized immutable `InitialNodes` whose `by_age` or `by_period`
  mapping contains exact coordinates and sorted, unique tuples of regime names;
- `graph`, immutable declared edges, effective phase graphs, valued/visited nodes,
  pruning reasons and `laws`, each regime's law as the solver and simulator evaluate it:
  bound to the graph, pruned of fixed-zero cells and lowered to the demanded ages
  (`laws[name].terminal` is true for a regime without outgoing edges,
  `laws[name].gated_edges` holds the edges its `Transition.gates` declare);
- `graph.coordinate_kind`, the meaning of the first coordinate in graph nodes and edges;
- `pruned_variables`;
- `get_params_template()`, which returns a mutable nested template.

`model.validate_transitions(params=params)` runs the transition preflight used by
`solve(runtime_checks=True)` without backward induction. It returns `None` on success
and raises the same first diagnostic for invalid regime probabilities, stochastic state
laws or joint transition supports and probabilities. It checks active periods and phase
grids with numerical checks always enabled and no console output. Utilities, solver
outputs and general deterministic next-state finiteness are outside this preflight.

Two methods check initial conditions without solving or simulating. Both accept the same
`initial_conditions` forms as `simulate` — a mapping of arrays with `"regime_id"` codes,
or a DataFrame with a `"regime_name"` column, ages and categorical labels — and the same
`params`:

- `validate_initial_conditions(*, initial_conditions, params) -> None` raises
  `InvalidInitialConditionsError` for a malformed structure (unknown regime, missing or
  extra state, unequal lengths, off-grid age, inactive regime, invalid categorical code)
  or for a subject admitting no jointly feasible action combination, with the same
  diagnostics `simulate(log_level="debug")` reports.
- `initial_conditions_feasibility(*, initial_conditions, params) -> Bool1D` returns a
  one-dimensional boolean mask, `True` exactly for the feasible subjects in the order
  supplied. A malformed structure raises rather than being reported infeasible;
  `validate_initial_conditions` succeeds exactly when the mask is all `True`.

A subject is feasible when at least one combination of its regime's declared action-grid
points satisfies every constraint jointly or, in a regime without actions, when every
state-only constraint holds. Both methods apply every initial-condition check `simulate`
applies, including the `own_stakeholder` declaration a collective start needs, and take
no `log_level`: they always raise. The verdict does not depend on how the check is
executed; at present both run in one eager pass on the default device, without
device-memory admission or subject padding. When a constraint depends on an
age-specialized function, each subject is checked against the constraints and functions
resolved at its own start period.

`model.initial_state_grids(regime_name=...)` returns an immutable mapping of finalized
public state declarations for initial-condition recipes. A generated `CompositeState`
entry is replaced by its original member grids, preserving their category labels and
codes. Ordinary grids, `Phased` declarations, and named joint-normal coordinate
declarations keep their existing finalized form. `initial_state_names(regime_name=...)`
gives physical per-subject names, expanding both composite members and joint-normal
coordinates. These differ from `state_names` / `state_grid`, which describe value-array
axes.

`model._regimes` is private canonical engine state.

(api-initial-nodes)=

## `InitialNodes`

Declare admissible starts with exactly one of `InitialNodes(by_age={25: "working"})` or
`InitialNodes(by_period={0: "working"})`. The keyword declares the coordinate kind; a
model rejects a declaration for the other clock. Values are regime names or nonempty
collections of names. Selectors contribute their Cartesian product with the names, and
overlapping selectors are unioned.

Age keys accept numeric ages, tuples, Python ranges, and `AgeRange`. Period keys accept
integers, tuples of integers, Python ranges, `Periods`, and `PeriodRange`. Booleans and
fractional period keys are invalid. `AgeRange` and `PeriodRange` intersect the model's
grid; exact keys, tuples, Python ranges, and `Periods` must name existing coordinates. A
selector that admits no node is an error. Initial-node selectors do not use the
parameter-table rule that silently discards surplus coordinates.

The declaration copies and freezes its mappings and name collections. After model
construction, `model.initial_nodes` contains exact coordinates and sorted, unique regime
tuples under the same keyword; the unused mapping is `None`. Pass this object directly
back into `Model(initial_nodes=...)`. `model.graph.initial_nodes` exposes expanded
coordinate–regime pairs, whose meaning is given by `graph.coordinate_kind`.

Legacy bare pairs and selector mappings remain accepted in age models. Period models
require `InitialNodes(by_period=...)`. `ByAge` and `ByPeriod` select transition laws;
they do not supply initial-node regime names.

(api-composite-state)=

## `CompositeState`

`Model(composite_states={"history": CompositeState(members=(...), start_codes={...})})`
replaces the axes of deterministic discrete member states by one generated discrete
state, `history`, whose codes index the member tuples reachable from the start codes.
Solve and simulate run on those tuples only, so the value function no longer spans the
full product of the members.

- Members are discrete states that every carrying regime holds on the same grid. Their
  laws must be deterministic and read only discrete variables, the period, the age and
  fixed parameters; a law that reads anything else is refused by name. A constraint that
  reads a continuous variable or a free parameter is not used to prune tuples.
- User functions keep reading the member names. Initial conditions name the members, as
  code arrays or as labels in a DataFrame; a start tuple the declaration never reaches
  raises `InvalidInitialConditionsError`. Members reach `to_dataframe` through
  `additional_targets`.
- The code set holds every tuple reachable at some period. A generated constraint,
  `<name>_is_reachable`, makes each code infeasible at the periods it cannot be reached
  in, so its value there is `-inf` and no reachable cell reads it.
- Member laws and grids are the same in both phases; a `Phased` member is refused.
- Members are held on category grids. A member declared by its values
  (`DiscreteGrid(values=...)`) is refused, and so is a member law that reads a variable
  declared by its values.
- Initial conditions are encoded only for subjects whose starting regime carries the
  members; member codes must be whole numbers.
- A model declares at most one composite state.

(api-period-candidate-lowering)=

## Lower one production candidate

`model.lower_period_candidate(params=..., log_level=..., candidate=..., retention=...)`
returns a frozen `lcm.LoweredPeriodCandidate` containing raw unoptimized StableHLO bytes
and an immutable descriptor manifest. Select an exact primary candidate with
`lcm.PeriodCandidate(regime=..., period=..., core=..., widths=...)`; periods are
zero-based. The core must belong to the ordinary solve's retention-selected graph, and
widths must occur in its ranked frontier. No donation-free fallback or off-frontier
width is substituted.

The method shares solve's sealed bindings, parameter and transition validation, solution
authority, solver checks and fullgraph structural preparation. It lowers only the
requested primary, including a later ranked or budgeted primary. It does not compile,
submit, dispatch, or evaluate memory admission. Preparation can initialize a backend and
allocate zero templates. Block-major and eager schedules are unsupported.

The manifest records exact rank, lowering key, donation, retention, layouts and input
shapes. `dedup_fanout` is the actual-wave count only for an unbudgeted rank-zero
candidate: shared preparation and the production ranked source determine the complete
initial pending map without admission. For later ranks or a configured device-memory
budget it is `None`, because compiled admissions determine actual wave membership.
`None` means unavailable, not zero and not a count over the full structural frontier.
Compiler memory, optimized HLO, buffer assignment and physical residency are also
unavailable. No allocation or performance claim follows from lowering.

Identity profiles require source checkouts with matching installed native libraries. The
bounded Linux CUDA12 profile requires one physical CUDA device, CPU retention support,
identifiable installed CUDA packages, and matching loaded PJRT, native CUDA and driver
libraries. Other GPU profiles fail closed. Current native build inputs, tool bytes and
native/JAX FFI header bytes are recorded; observed header bytes do not attest the inputs
of a cached native build. Source, native and runtime hashes are collected only on this
diagnostic path; an unknown descriptor or inconsistent identity fails closed. Returned
bytes preserve the exact raw IR, with no source-location normalization. The result keeps
no model, device array, JAX `Lowered`, executable or future alive. A compile API is not
provided.

(api-regime)=

## `Regime`

A general regime declares:

| Field                   | Contract                                                                      |
| ----------------------- | ----------------------------------------------------------------------------- |
| `states` / `actions`    | Name-to-grid mappings                                                         |
| `functions`             | Named DAG functions; a finalized regime needs utility                         |
| `output_only_functions` | Tuple of unread function names retained for simulation reporting              |
| `constraints`           | Ordinary predicates or structured `Condition` objects                         |
| `state_transitions`     | Ordinary target-state producers for cells not supplied by `joint_transitions` |
| `joint_transitions`     | Target-local shared-draw laws that jointly produce one or more next states    |
| `derived_categoricals`  | Discrete grids for categorical DAG outputs                                    |
| `solver`                | `lcm.solvers.GridSearch()` by default                                         |
| `taste_shocks`          | Optional EV1 taste-shock configuration                                        |
| `shocks`                | Model-level IID shock groups; read as `x` or, from transitions, as `next_x`   |
| `koopmans_aggregator`   | Optional regime-level continuation aggregator, `Phased` or `Decoupled`        |
| `certainty_equivalent`  | Optional regime-level lottery reduction                                       |
| `exits`                 | Named current-period moves with mapped policy queries                         |
| `description`           | Human-readable description                                                    |

Use `Regime.replace(...)` to derive a modified immutable declaration.

A regime declares no regime transition; its outgoing edges and their law live in
`Model(edges=...)`, and a regime without outgoing edges is terminal. Terminal regimes
declare no `state_transitions`, `joint_transitions`, Koopmans aggregator, or certainty
equivalent because they have no continuation.

`ConsumptionSavingsRegime` and `NestedConsumptionSavingsRegime` add the economic roles
required by EGM-family solvers. See
[Consumption-saving regimes and margins](consumption_savings.md). Collective fields on
`Regime` are documented separately in [Collective regimes](collective_regimes.md).

(api-current-exits)=

## Current-period exits

`Regime(exits={name: Exit(...)})` declares a move before the period's action. Scheduled
next-period transitions remain in `Model(edges=...)`.

- `Exit(targets=..., rule=..., available=None, observes=())` names each source
  stakeholder's destination. A singleton uses the key `"self"`.
- `ExitTarget(regime=..., projection=..., role=None, utility_adjustment=None)` names a
  real destination and maps its complete current state. A collective destination
  requires its stakeholder `role`. The adjustment reads source states and parameters and
  enters the exit payoff once. A projection may use `Phased(solve=..., simulate=...)`
  settlements; it does not change information.
- `Consent(counterparty=Counterparty(regime=..., role=..., projection=...))` compares
  each party's payoff at the destination's common Participating policy strictly against
  that party's Raw stay value. Equality keeps the original state.
- Built-in `rule="unilateral"` or `"mutual"` declares dissolution participation. A
  callable rule explicitly declares Boolean `reads_outside_values` metadata and returns
  `RuleResult(action=...)`, a scalar integer global alternative ID.

A custom rule can request `Q_<stakeholder>` (`Q_self` for a singleton), `exit_payoffs`
(exit name to member to value), `availability`, `alternative_ids`, `pareto_weights`, and
ordinary named source states, helpers and parameters. Stay IDs follow the flattened
C-order action product; exit IDs follow in declaration order. Invalid IDs are errors.
Custom multi-exit choices use one shared authored rule; competing rules do not define an
arbitration order.

`available` reads states, parameters and observed shocks, and precedes unsafe benchmark
evaluation. `observes` adds target information available at the exit choice. Hidden
target shocks are priced before the source choice and revealed when a mover executes the
destination's Participating action. Partial observation of a correlated group is
refused; see the adopted [IID and exit contract](../development/iid-exits-redesign.md).

When an exit observes every shock of a target group, all of that group's quadrature rows
are the same physical point, so the target's policy is evaluated once and that answer is
weighted by every row's original weight. Values equal the row-by-row reduction. A group
the exit observes only partly keeps its row-by-row evaluation. Model build refuses a
fully observed group of the caller's own shocks whose continuous law (`Normal`,
`LogNormal`, `Uniform`) can be degenerate at the caller but not at the target, for
example a spread read from a state that the exit changes, naming what would have to be
implemented to support it.

The cells of a query chunk and the rows of an information support run as batched lanes.
A lane whose exit is unavailable is still computed and then discarded; its target reads
its first grid point instead of the projected state, so a projection that is meaningless
when the exit is unavailable cannot reach user functions there.

Native current-period exits use each queried regime's declared `GridSearch` solver.
Endogenous-grid query reconstruction is not implemented. Query admission also refuses
invariant or co-mapped state layouts, action partitioning, stored Full-value references
and runtime-bound query status adapters.

Collective current-IID decisions select one common action and route exits at each node,
then integrate each member's selected value. Stored values retain the stakeholder axis
and carry no current-shock axis. This also applies to collective regimes without exits.
Collective EV1 tastes and custom source selectors combined with source EV1 tastes are
explicitly refused.

(api-participation-checks)=

## Scheduled participation checks

Attach `ParticipationCheck(predicate=..., routes=..., reads=..., observes=())` to an
incoming `Transition(participation={target_name: ...})` in `Model(edges=...)`. It
belongs to that incoming source, scheduled target, original source role and actual
participation period. The scheduled transition produces the landing state before the
comparison, and the final regime acts after it. Initial occupants and subjects arriving
through another edge cannot borrow this decision or its fallback. A scheduled target
cannot declare both a `Gate` and a `ParticipationCheck`.

The check is the per-period limited-commitment participation constraint. It also
re-checks a continuing household on its scheduled self-transition: on a couple → couple
edge, a failed check is a divorce.

`predicate` returns a scalar Boolean from the landed target's states and helpers,
parameters, and the explicitly named `reads`. Declare every value operand, including
`V_target`, with a `ParticipationRead`. `D_target` is refused; request typed policy
support instead. The predicate's threshold, window or other comparison determines
acceptance; it is not replaced by a maximization over the compared values. Acceptance
also requires the target's Participating policy to have support. `observes` names the
target physical shocks visible at this comparison.

`routes` maps every original source stakeholder to a
[`StakeholderRoute`](collective_regimes.md#api-stakeholder-route); a singleton has its
single route. On acceptance, `target_stakeholder` selects the role in the scheduled
target. On rejection, `fallback` names the destination, role and complete state
projection from the landed target state. A collective destination requires its exact
stakeholder name; a singleton has no stakeholder role. A
`Phased(solve=..., simulate=...)` fallback prices the source decision using the solve
settlement and routes simulation using the simulate settlement. The incoming source and
role remain attached to this settlement, so another edge's fallback cannot replace it.

Participation callbacks use the target's actual participation `period` and `age`. Their
runtime parameters belong under `params["edges"][source][target]`: the predicate uses
`["predicate"]`, read projections use `["reads"][read_name][state]`, and fallback
projections use `["routes"][source_role]["fallback"]`, with a phase key for a phased
fallback. Use `model.get_params_template()` for the complete paths. Queried regime
functions and target helpers retain their own regime parameter namespaces; temporal
values are selected for the participation date, rather than the incoming source's
decision date.

A source prices its next participation check exactly at every landing. A participation
check at a later date is taken again at each landing from a stored per-period surface of
its branches and operands, each interpolated between grid nodes: exact where landings
fall on nodes, approximate between them; see
[Participation checks over several periods](../user_guide/collective_regimes.md#participation-check-depth).

(api-participation-reads)=

### `ParticipationRead`

`ParticipationRead(regime=..., projection=..., kind=..., stakeholder=None, field="value", observes=())`
names the represented object supplied to the predicate. `projection` provides every
state of the referenced regime from the landed target state, and `stakeholder` selects a
collective reference's exact role. The `reads` mapping key is the argument name the
predicate receives.

- `kind="Raw"` and `kind="Participating"` reconstruct their canonical `GridSearch`
  action problems at the projected state, including the common action for a collective
  regime. `field="value"` reads the selected value; `field="support"` reads Boolean
  policy support. A read's `observes` adds the referenced physical shocks visible for
  this comparison to the decision's declared information.
- `kind="IntegratedFull"` reads the stored, shock-integrated Full value representation
  at the projected state. It exposes only `field="value"`, with empty `observes`; policy
  reconstruction, realized information and support are unavailable from this read.

(api-extreme-value-taste-shocks)=

## `ExtremeValueTasteShocks`

Declare IID extreme-value shocks on the Cartesian product of a regime's discrete actions
with:

```python
regime = Regime(
    ...,
    taste_shocks=lcm.ExtremeValueTasteShocks(),
)
```

The scale is a strictly positive runtime parameter at
`params[regime_name]["taste_shocks"]["scale"]`. Use no declaration for a hard maximum;
zero is not a valid scale.

For each discrete-action combination, the solve first maximizes the feasible `Q` values
over all continuous-action axes. It then applies the EV1 expected-maximum formula to
those choice-specific values. Simulation forms the feasible choice-specific values on
its policy's action candidates, adds one independent mean-zero Gumbel shock per
discrete-action combination, and takes the argmax. For any fixed candidate values, the
expected latent perturbed maximum equals their smoothed log-sum. The shock affects the
choice, while simulation publishes the selected unshocked value. This describes a regime
whose Koopmans aggregator also chooses. With a `Decoupled` aggregator the continuous
actions and the choice probabilities come from the objective, and the solve stores the
expected value profile at the noisy choice; see
[preferences](../methods/preferences.md). DCEGM simulation uses grid-restricted
candidates and therefore need not reproduce its off-grid solve value or choice
probabilities; see [Solvers and capabilities](solvers.md#api-dcegm).

This feature requires at least one discrete action and is implemented by `GridSearch`
and `DCEGM`. It is rejected for `NEGM`, `NBEGM`, and `NNBEGM`; on a collective regime;
on the source regime of a gated transition; and together with a nonlinear certainty
equivalent. These are semantic boundaries, not ignored options: the declaration is
rejected during `Regime` declaration or `Model` construction, before solve.

(api-euler-gamma)=

### Location-0 shocks and `EULER_GAMMA`

pylcm's taste shocks are mean-zero. Many papers use the standard Gumbel with location
zero, whose mean is `scale * EULER_GAMMA`, the scale times the Euler-Mascheroni
constant. To match such a model, add that mean to the utility of every action in each
regime that declares taste shocks:

```python
import jax.numpy as jnp

from lcm import EULER_GAMMA


def utility(*, consumption, work, kappa, taste_shock_scale):
    return jnp.log(consumption) - kappa * work + taste_shock_scale * EULER_GAMMA
```

Pass `taste_shock_scale` with the same value as the regime's `taste_shocks` scale.
`lcm.EULER_GAMMA` equals `numpy.euler_gamma`.

Apply the shift only to the Bellman utility. Keep feasibility, transition laws, and
other consumers of the original utility unchanged. If `utility` also feeds a constraint
or another DAG node, factor the original expression into `base_utility`, point those
consumers at `base_utility`, and add the shift in the `utility` wrapper. For example,
`base_utility >= 0` must not become `base_utility + scale * EULER_GAMMA >= 0`.

With these other uses preserved, the recipe reproduces the location-0 model, including
solved values and choice probabilities, when the Koopmans aggregator satisfies
`W(u + c, CE, ...) = W(u, CE, ...) + c` for `c = scale * EULER_GAMMA`.
`LinearAggregator` (the default) and custom aggregators additive in utility with unit
slope have this property. pylcm adds the shock to the choice-specific value after
aggregation; for aggregators without this property, such as `CESAggregator` in general,
shifting utility does not reproduce location-0 shocks.

## Derived categoricals

Use `derived_categoricals={"name": DiscreteGrid(category_class=Category)}` when a
parameter is indexed by a categorical function output rather than by a state or action.
The function must return an integer code, not a Boolean, because it is used as an array
index under JIT.

## Parameter ownership

A free function argument becomes a model parameter unless another state, action, DAG
function, context value, or fixed parameter supplies it. Values may be given at model,
regime, or function level, but each parameter value has one unambiguous source.
Parameters of callables declared in `Model(edges=...)` live under `params["edges"]` at
their declaration path, and a regime-level value never reaches them; see
[Edge parameter paths](transitions.md#api-edge-parameters). Start from
`model.get_params_template()` rather than constructing a nested parameter mapping from
memory.

Workflow: [Defining models](../user_guide/defining_models.md) and
[Parameters](../user_guide/parameters.md).
