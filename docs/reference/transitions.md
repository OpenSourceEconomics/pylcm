---
title: Transitions and phase specialization
---

# Transitions and phase specialization

(api-state-transitions)=

## State transitions

For every reachable target that carries an ordinary non-process state, a non-terminal
regime needs exactly one producer for that `(target, state)` cell: an ordinary
`state_transitions` law or a `JointTransition` output. In `state_transitions`:

- an ordinary callable is deterministic;
- `DeterministicTransition(func=func)` explicitly marks the same deterministic law;
- `StochasticTransition(func=func)` wraps a probability-vector function;
  `fixed_component=` declares a component the law never changes (see
  [tuning](../user_guide/tuning.md));
- `fixed_transition("state_name")` declares the identity law;
- a per-target mapping gives different laws for different reachable target regimes.

Stochastic process states already own their transitions and must not appear in
`state_transitions`. A terminal regime has no state transitions.

Per-target state-transition mappings must cover exactly the reachable targets that carry
the state. Reachability comes from `Model.edges`; extra or missing target handoffs are
errors.

(api-regime-transitions)=

## Regime transitions and graph support

`Model` snapshots its edge mappings, and `Transition`, `ByAge`, and `ByPeriod` snapshot
their nested law mappings, including phase-specific mappings. These published mappings
are read-only. Reusing a source dictionary for another model cannot change an existing
model's declarations. Callables keep their identity; their captured data must remain
unchanged while the model is in use.

`Model(edges=...)` declares every regime transition, structure and law; a `Regime`
declares none. A source maps to either

- a plain `{target: source_ages}` mapping, when it has exactly one outgoing edge at
  every source age — the graph is the law; or
- `Transition(targets={target: source_ages, ...}, law=..., gates=...)`, required when
  some source age has several outgoing edges, or when a destination is gated.

A regime with no outgoing edges is terminal. A `Transition` law is one of

- a regime name, a deterministic destination;
- a plain function or `DeterministicTransition(func=func)` returns a global regime code;
- `StochasticTransition(func=func)` returns probabilities in full global regime-code
  order;
- a per-target mapping supplies scalar `StochasticTransition` probability functions;
- `ByAge(...)` selecting one of these per source age, or
  `Phased(solve=..., simulate=...)`.

A `Transition` law is evaluated at every source age with outgoing edges, including ages
with a single declared destination or a single one left after fixed-zero pruning; there
it must put unit mass on that destination. Write such a law horizon-aware, or let
`ByAge` (e.g. `ByAge.until`) leave those ages unselected: a `ByAge` law must select
every age with several outgoing edges, and an age it does not select uses its one edge.
A law short of unit mass is caught by the probability check at `log_level="debug"`,
which names the cells dropped for lack of an edge; run a model at that level at least
once.

`targets` is optional when the law names its targets: a per-target mapping, a regime
name, or a `ByAge` / `Phased` whose every case and side is one of these. The
destinations are then derived from the law against the model's age grid:

- a plain mapping or name reaches each of its keys at every non-final source age;
- a `ByAge` case reaches its keys at the non-final ages that case selects, and a
  `default` covers every age no other case selects;
- the cases and sides of `ByAge` and `Phased` laws contribute their union;
- each route fallback regime of a gate is reached at the ages of its gated target.

Supplied anyway, `targets` must equal the derived destinations and source ages exactly;
otherwise `Model(...)` raises a `ModelInitializationError` that lists both the supplied
and the derived targets. A law over all targets — a function, a
`DeterministicTransition`, a full-vector `StochasticTransition`, or a `ByAge` with any
such case — names none, so it requires `targets`.

`gates={target: Gate(...)}` makes the transition into a target value-dependent: the law
supplies the probability of reaching the target as for any other destination, and the
target's [`Gate`](collective_regimes.md#api-gate) decides whether a row stays there or
takes its route's fallback. One gate per target holds at every age the target is reached
and in both phases; a gate is never wrapped in `ByAge` or `Phased`.

The targetless factories `@deterministic_transition()` and `@stochastic_transition()`
produce the same wrappers for state and regime laws and preserve DAG signatures.
`stochastic_transition` also accepts the state-law option `fixed_component`.

```python
# Fragment: use these edges with the corresponding regimes and age grid.
edges = {
    "working": Transition(
        targets={
            "working": AgeRange(start=25, exclusive_stop=62),
            "retired": (61, 62),
        },
        law=DeterministicTransition(func=retire_if_eligible),
    ),
    "retired": {"dead": AgeRange(start=62, exclusive_stop=75)},
}
```

Each destination appears once, paired with its permitted **source ages**. An edge lands
at the next grid coordinate. A deterministic law must select an available destination; a
full vector must be exactly zero outside graph support. Scalar probability mappings
supply graph-selected cells without declaring topology themselves. Terminal regimes have
no outgoing edges. A law that differs between the phases is phased inside one
`Transition`, `Transition(targets=..., law=Phased(solve=..., simulate=...))`, on targets
both phases share. `Model(edges=Phased(solve={...}, simulate={...}))` gives the model
different perceived and realized edges, each phase's mapping with its own `Transition`;
state handoffs stay phase-specific on the source regime. A lone edge in one phase,
paired with a per-target probability mapping in the other, counts as a probability-one
cell for its destination.

(api-edge-parameters)=

### Edge parameter paths

The regime-transition law belongs to the edges, and so do its parameters: the parameter
path of an edge-declared callable is its declaration path under `params["edges"]`.

```text
params["edges"][source][arg]                       # a law over all targets
params["edges"][source][target][arg]               # a per-target cell, gated or not
params["edges"][source][target]["predicate"][arg]  # the target's Gate
params["edges"][source][target]["references"][reference][state][arg]
params["edges"][source][target]["routes"][route]["fallback"][state][arg]
params["edges"][source][target]["routes"][route]["fallback"]["solve" | "simulate"][state][arg]
```

The last line is a `Phased` fallback. A law over all targets has no `law` segment;
`ByAge` cases and `Phased` sides of one law share its slot and union their arguments. A
source without a law has no `params["edges"]` entry. `get_params_template()` lists each
slot at this declaration path, the most specific of three levels; a value may instead be
given once at `params["edges"][source][arg]`, which covers every callable below the
source that reads `arg`, or at the model level. Each slot takes its value from exactly
one level. A value under the source regime, `params[source][arg]`, never reaches an edge
callable. Per-target state laws belong to the source regime and keep their paths under
`params[source][target]`. The slots are read off the declared `Transition`, so they do
not depend on the horizon or on fixed-zero pruning. See
[Move transition parameters under `edges`](../user_guide/migrating_dated_regimes.md#migrating-edge-parameters).

(api-dated-regime-transitions)=

### Age-indexed laws

For a period model, the corresponding declaration is
`ByPeriod(cases={selector: law, ...}, default=...)`. Case selectors are integer periods,
tuples or ranges of integers, `Periods(values=...)`, or
`PeriodRange(start=..., exclusive_stop=...)`. `ByPeriod.until` takes
`start_period_inclusive` and `stop_period_exclusive`; the last selected source period
uses `then`. Plain graph edge selectors require `Periods` or `PeriodRange`. Age and
period declarations cannot be mixed. See
[Periods and temporal parameters](../user_guide/period_time.md).

`ByAge(cases={selector: law, ...}, default=...)` selects complete numerical laws by
source age inside a `Transition`. Its selectors are exact ages, tuples, integer ranges,
or half-open `AgeRange(start=..., exclusive_stop=...)` intervals. `ByAge.until` uses
`law` before `stop_age_exclusive`, except that the last selected source age uses `then`.
It does not declare topology or initial nodes. `AgeGrid(inclusive_stop=...)` includes
its final coordinate, where no effective edge originates; a declared selector may still
name it as dormant metadata alongside earlier source ages.

Ordinary scalar cells can be pruned from the effective graph when their complete DAG
uses only construction-fixed leaves and yields exactly represented zero. Dynamic leaves
and coordinate-indexed fixed parameters retain edges; positive subnormals also remain
live. Invalid or all-zero laws retain probability validation, and pruning never
renormalizes mass. `model.graph.edges` retains declared support;
`model.graph.pruned_edges` records the phase-specific proof reasons. See
[Age-indexed regimes](../user_guide/dated_regime_graph.md).

(api-joint-transitions)=

## Joint transitions

`JointTransition(*, support_size, support, probabilities, outputs)` declares one or more
next states driven by one shared draw; all four are keyword-only. `support` is a literal
pytree of joint nodes or a callable returning one, `probabilities` returns a vector of
length `support_size`, and each `outputs` entry projects a sampled joint node into one
target state.

Joint laws occupy the separate `Regime.joint_transitions` slot. Its public shape is a
mapping from target regime, to local joint-node name, to the `JointTransition`:

```python
source = Regime(
    joint_transitions={
        "target_regime": {
            "joint_draw": JointTransition(
                support_size=2,
                support={
                    "wealth": wealth_nodes,
                    "health": health_nodes,
                },
                probabilities=joint_probabilities,
                outputs={
                    "wealth": next_wealth,
                    "health": next_health,
                },
            )
        }
    },
    functions={"utility": utility},
)
```

The source's edge into `target_regime` and its law are declared in `Model(edges=...)`,
as for any regime transition. The outer key names the reachable target regime. The inner
key names the sampled joint node that output functions may read. Each output owns one
`(target, state)` producer cell. A bare `state_transitions[state]` law may coexist and
broadcasts only to other, unclaimed reachable targets. An explicit ordinary law on the
same target-state cell, or a second joint kernel claiming that cell, is rejected.

Transition-local joint lotteries are currently implemented only by `GridSearch`.
Selecting an EGM-family solver for a regime that declares one is rejected when the model
is built.

Use `Phased` around the entire `JointTransition` for perceived and realized variants;
both variants keep the same output names and support size. When both supports are
literal, they must also have the same pytree structure, leaf event shapes, and dtypes.
Support values, probability functions, and output-law implementations may differ between
phases.

### What each part may read

- a callable `support` reads only the source `period`, the source `age` when the model
  declares an age grid, and parameters; period models reject an `age` argument at
  construction, including in either `Phased` variant;
- `probabilities` may also read source states, actions, and helpers;
- an output law may transform the shared node using source values, and may read
  `next_<state>` outputs already resolved on the same target edge.

Support shapes and probability vectors are checked in the params-bound runtime preflight
for every active period and both phases. Callable supports may change values, but their
pytree structure, leaf event shapes, and dtypes must stay fixed across periods and
phases. Each support leaf has leading axis `support_size` and contains finite numeric or
Boolean values. Probability rows have exactly `support_size` entries, are finite and in
`[0, 1]`, and sum to one. `runtime_checks=True` rejects invalid mass at every log level;
`runtime_checks=False` skips numerical preflight. Construction validation always runs.
Any path that continues into aggregation normalizes the probability mass it receives.

The solve variant is validated on solve grids. The simulation variant is validated on
simulation grids, including the domain of a carried-only state that its probability
function reads. With runtime checks enabled, a phase law that cannot be evaluated and
checked is refused rather than treated as valid.

### Parameter paths

A joint law belongs to the source regime, like a per-target state law, and so do its
parameters. Support and probability parameters live below the kernel name; output
parameters keep the ordinary target-local `next_<state>` paths:

```text
params[source][target][kernel]["support"]
params[source][target][kernel]["probabilities"]
params[source][target]["next_wealth"]
```

### Outputs onto a stochastic process

An output may target a stochastic process as well as an ordinary grid, which is how
correlated innovations land on a grid pylcm discretized rather than one discretized by
hand. The output law still names a physical value. Because the target's value function
is stored on the process's nodes, that value reaches the continuation as its
coefficients in the node basis — the hat weights of linear interpolation. Naming a node
reads that node alone; naming a point between nodes reads the linear interpolation of
the target's value function, which is the only reading its nodes support.

The output law displaces the process's own law on that edge, so the correlation the
kernel imposes is what the target is entered at. The support is the contract: a value
outside the process's grid has no representation in that basis and yields `NaN`, which
the caller's value function reports rather than extrapolating.

(api-age-specialization)=

## Age specialization

`AgeSpecializedFunction(build, signature)` and `AgeSpecializedGrid(build, signature)`
produce age-specific declarations during model construction. `build(age)` returns the
function or continuous grid for that age. `signature(age)` returns a stable hashable
key; equal keys must mean identical resolved behavior because those periods may share
one compiled program. Because model construction may resolve the same age multiple
times, `build(age)` must also be deterministic and side-effect-free.

Exact function placement:

- accepted in `functions` and `constraints` of non-terminal regimes;
- not accepted as the regime transition, inside `StochasticTransition`, or directly as a
  state-transition value;
- a state law may be a plain function that reads an age-specialized helper;
- terminal regimes do not accept age-specialized functions;
- additional DataFrame targets may not depend on them because published target functions
  use a representative age.

`AgeSpecializedGrid` is accepted only as a top-level continuous-state grid. It is not an
action, discrete/process grid, runtime-points grid, or a member of a carried
`Phased(solve=callable, simulate=Grid)` state. Grid class, node count, shape, and dtype
remain constant across ages.

Factories run while the model is built. Solve, simulation, compilation, and diagnostics
select the already-resolved period objects and never call `build(age)`.

(api-solve-and-simulation-phases)=

## Solve and simulation phases

`Phased(solve=..., simulate=...)` is the outermost wrapper for declarations that may
differ by phase:

- `Model.edges` accepts perceived and realized source–destination topology, each phase
  with its own `Transition` laws; ordinary per-target mappings may declare different
  solve and simulate target keys, and a `Transition` law may itself be `Phased` with
  matching transition forms;
- `functions` and `state_transitions` accept phase-specific variants;
- `koopmans_aggregator` accepts one callable per phase;
- `states` accepts the special carried-state form
  `Phased(solve=callable, simulate=Grid)`.
- `joint_transitions[target][kernel]` accepts `Phased` around the whole
  `JointTransition`.

Constraints, actions and derived categoricals are phase-invariant and reject `Phased`.
Ordinary nested phase wrappers and wrappers inside per-target transition mappings are
invalid. Structured declarations own two additional, explicit seams:
`CollectiveUtility.utilities[stakeholder]` may hold a phase-specific utility, and a
`StakeholderRoute` may use a phase-specific `fallback`. These field-specific seams and
the whole-joint-kernel seam above are not permission to place `Phased` arbitrarily
inside mappings.

A carried state is derived during backward induction, so it adds no solve-grid axis, but
is seeded and evolved as a genuine simulation state. Its law of motion still belongs in
`state_transitions`.

Decisions use the solve law, while realized transitions use the simulate law. Every
simulation-visited node is solved with its own perceived continuation dependencies;
value-only nodes do not create realized visits. Each phase supplies valid probabilities
and state handoffs for its own edges. Use outer `Phased` state-transition mappings when
the two phases need different destination handoffs. A `Gate` is shared by both phases: a
target reached in both phases carries the equal `Gate` in both or none, and only the law
supplying its probability may differ.

Workflow: [Transitions](../user_guide/transitions.ipynb) and
[Age-specialized functions and grids](../user_guide/age_specialized.md). Rationale:
[Phase-dependent model structure](../explanations/phase_grammar.ipynb).
