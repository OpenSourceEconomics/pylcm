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

`stages={stage_name: (regime_name, ...), ...}` declares several decisions within every
period, in order; every regime is in exactly one stage. Subjects start in their regime's
stage, an edge lands at its target's next occurrence, the first later step of the
target's stage, and an exit enters a regime of its own or a later stage of its period. A
subject has no row and no transition at a step its regime sits out, and parameters,
regime codes, values and simulated results name only declared regimes. Stages of one
period are linked without discounting, so crossing into the next period discounts once,
and a regime whose edges land in its period declares no Koopmans aggregator and uses no
endogenous-grid solver. `period` and `age` mean the period in user functions,
time-indexed parameters and the simulated frame, which gains a `stage` column.
`model.n_periods`, the solution and the raw simulation results count steps, one per
period and stage. `stages=None` gives every period one stage. See
[Several decisions in one period](../user_guide/period_time.md#several-decisions-in-one-period).

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
state-only constraint holds. A subject starting where its regime's current-period exit
decides is accepted without a feasible stay action, since only the solution says whether
the exit moves it; `simulate` raises `NoFeasibleActionError` if it then neither acts nor
moves. Both methods apply every initial-condition check `simulate` applies, including
the `own_member` declaration a collective start needs, and take no `log_level`: they
always raise. The verdict does not depend on how the check is executed; at present both
run in one eager pass on the default device, without device-memory admission or subject
padding. When a constraint depends on an age-specialized function, each subject is
checked against the constraints and functions resolved at its own start period.

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
| `exit`                  | Current-period move with mapped policy queries                                |
| `description`           | Human-readable description                                                    |

Use `Regime.replace(...)` to derive a modified immutable declaration.

(api-function-returns)=

### What a function returns

Each user function returns the kind its role reads. A Python scalar (`0`, `1.5`, `True`)
is accepted wherever an array of its kind is, and is converted where the engine receives
it:

| Role                                                                                                                                    | Returns                                                        |
| --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Constraint, gate, exit availability                                                                                                     | A Boolean, e.g. `wealth > 0` or `work == 1`                    |
| Law of a discrete state, regime selector                                                                                                | An integer code, or a Boolean for the codes 0 and 1            |
| Utility, law of a continuous state, transition probability, joint-transition weight or output, exit utility adjustment, borrowing bound | A number: an integer or a float, read at the model's precision |

Only the returned dtype is checked, when the function is first evaluated in `solve()`. A
return of the wrong kind raises `InvalidFunctionOutputError`, naming the function. A
predicate returning `1` is refused rather than read by truthiness: no check can tell `1`
from `2` inside a compiled function. A code returned as a float is refused rather than
truncated onto a code, and a Boolean utility is usually a comparison returned in place
of its consequence.

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

`Regime(exit=Exit(...))` declares a move before the period's action. A regime has at
most one exit; its parameters live under `exit`. Scheduled next-period transitions
remain in `Model(edges=...)`.

- `Exit(targets=..., rule=..., available=None, observes=(), nodes=None)` names each
  source member's destination. A singleton uses the key `"self"`. `nodes`, a
  `MonteCarlo` or `QuasiMonteCarlo` rule, applies when the regime is solved: every shock
  group of the choice that the regime's own stay problem does not read, observed target
  information included, is drawn `n` times from its law at the current state, and the
  same draws serve every row of the groups the stay problem reads, which keep their
  declared nodes. `None` integrates over the full product. Simulation uses the realized
  shocks. `observes` names every shock of a group or none of them; naming some of a
  group raises `ShockDeclarationError` when the model is built.
- `ExitTarget(regime=..., projection=..., role=None, utility_adjustment=None)` names a
  real destination and maps its complete current state. A collective destination
  requires its member `role`. The adjustment reads source states and parameters and
  enters the exit payoff once. A projection may be `Phased(solve=..., simulate=...)`; it
  does not change information, and under `MutualConsent` a `Phased` projection that
  reads the settlement, directly or through other functions, is refused when the model
  is built. A projection names every state its destination declares, model-level ones
  included. Where the destination does not read a model-level state and pruning drops
  it, the projection onto it is dropped as well, as a law of motion toward that
  destination is, so the same projection builds either way. A projected continuous state
  follows its destination grid's `off_grid` rule, as a realised next state does, in the
  solve and in the simulation alike. Under `"clip_state"` the destination's problem is
  priced, its shock laws read and the mover landed at the state capped to the grid's
  outer nodes. Under `"raise"` a read the exit choice evaluates outside the grid raises
  `OffGridStateError`, even where the household then stays; a read where the exit is
  unavailable, or at a shock node of zero mass, does not. `"extrapolate"` and
  `"clip_lookup"` keep the projected state. A `"clip_state"` or `"raise"` grid whose
  points are supplied at run time is refused when the model is built.
- `Consent(counterparty=Counterparty(regime=..., role=..., projection=...))` compares
  each party's payoff at the destination's common Participating policy strictly against
  that party's Raw stay value. Equality keeps the original state. A party with no
  feasible action has the value minus infinity, so it gains exactly where its value at
  the destination is finite. The counterparty's projection follows the same rule as a
  target's. A formation pays nothing: its target declares no `utility_adjustment`.
- Built-in `rule=Unilateral()` keeps the match only if every member is willing to stay.
  A member is willing when its stay value is above its exit payoff, or equal to it under
  the default `tie="stay"`; `tie="leave"` makes an indifferent member leave. Among the
  kept stay actions the household takes its Pareto-weighted argmax; with none kept, the
  exit is taken. `rule=MutualConsent(settle=...)` needs a settlement, below.
- `compares` says where that comparison is made. `"each_action"` tests every stay
  action, as just described. `"optimum"` tests only the household's unconstrained
  optimum: `Unilateral` separates when any member refuses there, and a household that
  stays acts at its optimum. Where the exit is not available the household stays at its
  optimum; with no feasible action and the exit available it separates. A separation
  needs a feasible action in every destination where the exit puts its member; without
  one the cell has no value, as under `"each_action"`. The default `None` is `"optimum"`
  with `renegotiate` and `"each_action"` otherwise. `MutualConsent` always compares at
  the optimum; `"each_action"` together with `renegotiate` or `settle` is refused.

A subject moves at most once per period, so an exit payoff is the value of the
destination's problem at the projected state, without the destination's own exit. The
destination's rule for leaving does not constrain that value, and two regimes may exit
into each other. A formation by `Consent` is the exception: the formed couple applies
its own rule for staying, and the outside values that rule compares are again problem
values. Within one step (one stage of one period), a rule for staying that is read
through such a value reads problem values only; a longer chain is refused. A mover acts
in the problem its move was priced at, the formed couple's Participating policy after
consent and the destination's problem otherwise. A regime with an available exit that
needs no consent, entered in one period both by consent and by another exit, would act
in two different problems; the model refuses it when it is built.

`Unilateral(renegotiate=Renegotiate(state="mu"))` first optimizes the Raw stay policy at
every node of the ordered discrete state. It supports a collective pair: increasing node
order favors the first member in the utility mapping, decreasing order the second. If
both accept the incumbent, it stays. If one refuses, walk one node toward that member
and stop at the first node both accept. Simultaneous refusal, reaching the grid edge or
encountering the other member's refusal separates the pair; a later feasible island does
not reopen the path. Equality follows the rule's `tie`. Outside payoffs are those at the
original source state, its incumbent node included; they are not evaluated again along
the path.

`MutualConsent(settle=Settlement(name="share", candidates=grid, default=f))` lets a pair
of members shift the division on separation before deciding. Every exit target's
projection functions and utility adjustment may read the settlement as the argument
`name`; a settlement no target reads, and a name that is a state, action, function or
shock of the source, are refused when the model is built. `candidates` is a continuous
grid with fixed points; increasing values favour the first member. `default` is
evaluated at the source node before the action and returns one number there: it may read
states, parameters, the period, functions of the regime and shocks the regime itself
draws, and it reads neither an action nor the settlement. Its value is clipped to the
candidates' range and need not be a candidate. With `O_i(s)` member `i`'s outside value
at settlement `s` (its destination's problem value at the projections evaluated with
`s`, plus its adjustment evaluated with `s`), `W_i` its value at the household optimum
and `d` the default:

1. A member prefers leaving at `s` when `O_i(s)` is above `W_i`, or equal to it under
   `tie="leave"`. `tie` decides every comparison of the rule.
1. A member refuses to stay when it prefers leaving at `d`. Nobody refuses: stay at the
   optimum. Both refuse: separate at `d`; the candidates are not searched, and where a
   destination has no feasible action at its landing state at `d` the cell has no value.
1. Exactly one refuses: among the candidates strictly on the other member's side of `d`
   at which every destination is supported, take the one nearest `d` at which that
   member prefers leaving. It never does at `d` itself. Separate there if the member who
   refuses to stay prefers leaving there too; otherwise, or without such a candidate,
   stay at the optimum.

The engine does not check that the accepting member's outside value rises in that
member's direction; the rule takes the nearest accepting candidate and tests the member
who refuses to stay at that candidate only, never at a farther one. Each member's
outside value at a candidate is compared with its value at the household optimum, not
with its outside value at the default. A member whose projection does not read the
settlement is read once, and only its adjustment varies; a candidate read whose
projections land on the destination's grid takes the stored table. A household that
separates lands at the projections evaluated with the selected settlement and receives
the adjustments evaluated with it. The settlement is not a column of the simulated
output; the landing states and values carry it. Two members only; a shifted settlement
is searched on the candidates, never between them; a default that is random and
unobserved at the decision cannot be declared.

An outside policy can have a gated transition into the next period. Its nominal target
components and each projected personal fallback enter as explicit future value reads;
the gate is applied at the landing point. This also holds when the outside regime is
only queried and publishes no own Full value bank.

The selected node supplies the action and spouse values and becomes the current state
used by forward transitions. Declare `fixed_transition("mu")` when the state continues,
so its selected node is also next period's node. A conflicting state law or joint update
is refused. The feature uses the declared finite grid; it does not solve for continuous
indifference. Other same-period Full recursion remains forbidden. Use `Model.simulate`
for the routed decision: `lookup_policy` refuses an active renegotiating exit because it
cannot return changed state and separation destinations.

`available` reads states, parameters and observed shocks, and precedes unsafe benchmark
evaluation. `observes` adds target information available at the exit choice. Hidden
target shocks are priced before the source choice and revealed when a mover executes the
destination's action. Partial observation of a correlated group is refused; see the
adopted [IID and exit contract](../development/iid-exits-redesign.md).

A `utility_adjustment`, a settlement's `default` and `available` are the same in the
solve and in the simulation. One that reads, directly or through other functions, a
function or law of motion declared `Phased` or a carried state is refused when the model
is built, with the regime, the target and the name it reads.

When `available` reads nothing but discrete shocks of one group that the stay problem
does not read, as an offer flag does, the solve enumerates that group's nodes where the
exit is available and one node where it is not, weighted by the whole mass of those
nodes at the value of staying; the values equal the full enumeration's to rounding.

When an exit observes every shock of a target group, all of that group's quadrature rows
are the same physical point, so the target's policy is evaluated once and that answer is
weighted by every row's original weight. Values equal the row-by-row reduction. A group
the exit observes only partly keeps its row-by-row evaluation.

When the source draws the group of a shock its exit observes, the source and the
destination see one draw, so its law is the same at the source's state and at the
landing state: it may read parameters, and states that the projection returns unchanged
under their own name. Model build raises `ShockDeclarationError` for such a law that
reads a state the projection changes or is not proven to keep, for every law family and
every rule, naming the regimes, the group, the shock and the state. To let such a shock
depend on a changing state, draw it in the source and carry it to the destination as a
state. A group only the destination draws has its law at the landing state: the source's
decision, the destination's node and the simulated draw all evaluate it there, so it may
read any state the projection sets.

When solving, an outside value is read from a table rather than evaluated again, where
the model build can prove that the read lands on the table's nodes:

- the destination is solved in that period, with `GridSearch`, on a grid that is fixed
  when the model is built and not specialized to the period's age;
- every projected state is provably a point of the destination's grid, after a
  `"clip_state"` grid's cap: the projection returns a source state unchanged between
  identical grids, or each of its finitely many inputs (grid points of the source's
  states, the period, its age, codes of a discrete shock) lands on a grid point;
- each shock group of the destination is either observed whole by the exit, on the
  destination's own nodes, or not observed at all.

The source regime tabulates the destination's problem value at every point of its grid
and every node of the observed groups, integrating the others, once per period; each
read gathers from that table. A cell whose problem fails its numerical checks fails only
the reads that take it. A read the build cannot prove aligned, for example a projection
that reads a parameter or a continuous shock, or lands between grid points, is evaluated
at its exact point; no stored value is interpolated. An outside read of a destination
solved with an endogenous-grid method is refused when the model is built. Simulation
always evaluates the destination at each subject's realized state and shocks.

The cells of a query chunk and the rows of an information support run as batched lanes.
Unless `ExecutionConfig(axis_widths={"query_cell": ...})` fixes the chunk, it holds as
many cells as fill one step with the rows of the widest query it evaluates rather than
reads from a table, and at least 64. A lane whose exit is unavailable is still computed
and then discarded; its target reads its first grid point instead of the projected
state, so a projection that is meaningless when the exit is unavailable cannot reach
user functions there.

Native current-period exits use each queried regime's declared `GridSearch` solver,
whether the regime is evaluated at a point or tabulated for stored reads.
Endogenous-grid query reconstruction is not implemented. Query admission also refuses
invariant or co-mapped state layouts, action partitioning, stored Full-value references
and runtime-bound query status adapters.

Collective current-IID decisions select one common action and route exits at each node,
then integrate each member's selected value. Stored values retain the member axis and
carry no current-shock axis. A collective regime without an exit runs the ordinary grid
search, integrating each member's value at the household's choice over the nodes. A
regime with EV1 taste shocks that declares an exit is refused; see
[`ExtremeValueTasteShocks`](#api-extreme-value-taste-shocks).

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

The scale is a runtime parameter at `params[regime_name]["taste_shocks"]["scale"]`: one
strictly positive, finite number per regime. It cannot vary with a state, a period or an
age, and regimes may differ in it. Use no declaration for a hard maximum; zero is not a
valid scale.

For each discrete-action combination, the solve first maximizes the feasible `Q` values
over all continuous-action axes. It then applies the EV1 expected-maximum formula to
those choice-specific values. Simulation forms the feasible choice-specific values on
its policy's action candidates, adds one independent mean-zero Gumbel shock per
discrete-action combination, and takes the argmax. For any fixed candidate values, the
expected latent perturbed maximum equals their smoothed log-sum. The shock affects the
choice, while simulation publishes the selected unshocked value. The solved value
therefore includes the expected taste shock: it is the log-sum
$\sigma \log \sum_j e^{Q_c(j)/\sigma}$. The simulated `value` column records $Q_c(j)$ of
the action taken, without its shock, so its mean over subjects at one state estimates
$\sum_j P(j)\, Q_c(j)$ and lies below the solved value wherever more than one action has
positive probability. This describes a regime whose Koopmans aggregator also chooses.
With a `Decoupled` aggregator the continuous actions and the choice probabilities come
from the objective, and the solve stores the expectation of `value` at the noisy choice
including the chosen action's shock, $\sum_j P(j)\,[v(j) - \sigma \log P(j)]$ with
$v(j)$ the `value` at combination $j$; simulation again records $v(j)$ of the action
taken. See [preferences](../methods/preferences.md). DCEGM simulation uses
grid-restricted candidates and therefore need not reproduce its off-grid solve value or
choice probabilities; see [Solvers and capabilities](solvers.md#api-dcegm).

This feature requires at least one discrete action and is implemented by `GridSearch`
and `DCEGM`. A collective regime admits it when exactly one member has a positive Pareto
weight; that member decides, and every other member's value is the mean of its action
values under the decider's choice probabilities (see
[Collective regimes](collective_regimes.md)). It is rejected for `NEGM`, `NBEGM`, and
`NNBEGM`; on a collective regime with more than one positive weight; on the source
regime of a gated transition; on a regime that declares an `exit`, whether a dissolution
or a formation; and together with a nonlinear certainty equivalent. Another regime's
exit may read a regime with taste shocks as its target or its `Consent` counterparty, at
the decider's smoothed expected maximum and every other member's mean value. These are
semantic boundaries, not ignored options: the declaration is rejected during `Regime`
declaration or `Model` construction, before solve, except that a weight declared as a
function is checked at every solve.

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
