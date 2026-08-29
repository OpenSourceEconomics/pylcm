---
title: Collective regimes
---

# Collective regimes

A collective regime carries one value function per stakeholder and takes a single shared
action for all of them. Two further capabilities travel with it: feasibility that reads
values rather than only states, and a transition whose branch depends on values at the
target regime.

Six declarations express this, and each one goes inside a slot `Regime` already has, so
a collective model has no extra constructor arguments to learn.

| Declaration                | Where it is declared                                                              | What it expresses                                   |
| -------------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------- |
| `CollectiveUtility`        | `functions={"utility": ...}`                                                      | the regime's stakeholders and their flow utilities  |
| `ParetoObjective`          | `CollectiveUtility(objective=...)`                                                | how stakeholder action values are scalarized        |
| `ValueDependentConstraint` | `constraints={"name": ...}`                                                       | a feasibility predicate that may read values        |
| `ValueDependentTransition` | `transition={"target": ...}`                                                      | a transition into one target, gated on values there |
| `StakeholderRoute`         | `ValueDependentTransition(routes=...)`                                            | where one source stakeholder goes on each branch    |
| `ProjectedRegimeValue`     | a constraint's `references`, an edge's `gate_references`, or a route's `fallback` | another regime's current-period value, at a mapping |

All six are frozen, keyword-only dataclasses defined in `src/lcm/collective.py` and
exported from `lcm`. A type violation on any of them raises `RegimeInitializationError`.

For the modelling narrative — why a household declares its weights instead of summing
them, what the dissolution flag means economically — see
[Households and value-dependent choice](../user_guide/collective_regimes.md).

## `CollectiveUtility`

```python
utilities: Mapping[str, UserFunction]
objective: ParetoObjective | None = None
```

Declared as `functions={"utility": CollectiveUtility(...)}`. That declaration is what
makes a regime collective: the `utilities` keys **are** the regime's stakeholders, in
insertion order, and that order fixes the trailing stakeholder axis of the regime's
value function and of every published array.

`objective=None` (the default) weights the stakeholders equally.

Construction rewrites the regime: each `utilities` entry becomes a `functions` entry
named `utility_<stakeholder>`, the key tuple becomes the regime's stakeholders, and
`objective` becomes the regime's Pareto objective. The household maximizes the objective
over the feasible action product and reads off each stakeholder's own action value at
that common choice.

```{math}
a^*(x) = \arg\max_{a\,:\,F(x,a)} \sum_s \lambda_s(x)\, Q^s(x, a),
\qquad V^s(x) = Q^s(x, a^*(x)).
```

Rejected at `Regime` construction:

- a regime that declares both a `CollectiveUtility` and a conflicting `stakeholders`
  tuple;
- a stakeholder whose `utility_<s>` is also declared directly in `functions` as a
  different function;
- a `ParetoObjective` given both as `CollectiveUtility.objective` and as the regime's
  own `pareto_objective`, unless the two are equal.

## `ParetoObjective`

```python
weights: Mapping[str, UserFunction | float]
normalization: str = "pointwise"
```

`weights` holds one weight $\lambda_s$ per stakeholder, keyed by stakeholder name. A
`float` is a constant. A callable is a function of the regime's **states** and of
`period` / `age`; every other argument it names becomes a free scalar parameter under
the regime's `pareto_objective` key (see [Parameters](#collective-parameters)). An
argument spelled like one of the regime's other functions does **not** receive that
function's output — it becomes a parameter you must supply.

A weight may not read an action. A weight that varies with the choice states a different
objective per candidate, whose maximizer is a Pareto optimum of no fixed weighting.

`normalization` is annotated as a plain `str` and validated against two values:

- `"pointwise"` (the default) divides by the total at each cell, so the weights sum to
  one wherever the objective is evaluated and a state-dependent declaration keeps one
  scale across the grid;
- `"none"` uses the declared weights as they stand. The scalarization is then not on the
  stakeholders' own scale, and comparing values across cells whose totals differ
  compares different objectives.

Admissibility of a weighting — one finite, non-negative weight per stakeholder with a
strictly positive total — is checked in two places, because a constant is knowable at
construction while a function of a parameter is not. See
[Where each rule is enforced](#collective-where-enforced).

## `ValueDependentConstraint`

```python
predicate: UserFunction
references: Mapping[str, ProjectedRegimeValue] = field(
    default_factory=lambda: MappingProxyType({})
)
```

Declared inside `constraints`, beside the ordinary callables, so a regime has one
constraint slot rather than two. The regime's mask is the AND of both kinds, and a state
cell whose mask is empty publishes the regime's dissolution flag `D`.

`predicate` returns `True` where the cell is feasible. It may read:

- `Q_<s>` for each stakeholder of the declaring regime — that stakeholder's action value
  at the candidate;
- each key of **this constraint's own** `references`, bound to that reference's
  interpolated value;
- ordinary states, actions, functions and parameters.

`references` maps the name a reference enters the predicate under to the
`ProjectedRegimeValue` supplying it. Two constraints of one regime may share a reference
name only if they declare the identical reference.

A `ValueDependentConstraint` is only meaningful on a collective regime: `Q_<s>` exists
only where stakeholders do. Declaring one on a singleton regime is rejected at `Regime`
construction.

## `ProjectedRegimeValue`

```python
regime: RegimeName
projection: Mapping[StateName, UserFunction]
stakeholder: str | None = None
```

Another regime's **current-period** value, read at mapped state coordinates. The
reference regime is solved earlier in the same period — the solver orders each period's
active regimes topologically by these declarations — and its value function is
interpolated at the projected coordinates: linear on continuous axes, lookup on discrete
axes.

Reading the current period rather than the continuation is what a within-period
participation constraint needs: a couple's period-$t$ decision is checked against the
values its members would have as singles in that same period $t$.

`regime` names another regime of the model, active in every period the declaring regime
is active. No transition edge between the two is required — a reference read works
across otherwise unconnected regime islands.

`stakeholder` names whose value to read from a **collective** reference regime. It is
required there and must be `None` for a singleton reference.

Where the declaration sits fixes what its `projection` may read and which states it owes
a coordinate function for:

| Position                                   | Projects from               | May introduce free params | Owes one coordinate per                          |
| ------------------------------------------ | --------------------------- | ------------------------- | ------------------------------------------------ |
| `ValueDependentConstraint.references`      | the DECLARING regime's cell | no                        | state of the reference regime's value function   |
| `ValueDependentTransition.gate_references` | the TARGET regime's grid    | yes, as edge params       | state of the reference regime's value function   |
| `StakeholderRoute.fallback`                | the TARGET regime's grid    | yes, as edge params       | state the reference regime carries in simulation |

The fallback owes the larger set because a route does not only price the closed branch,
it writes the routed row into the fallback regime, and forward simulation carries every
one of that regime's simulate states per subject. A state left unprojected would keep
whatever the row held before the edge routed it there. The solve states are a subset, so
the same projection still serves the fold's value read.

A `ProjectedRegimeValue` on an age-specialized reference regime is measured against the
grid of the period whose value is being folded; see
[Gate references and leg fallbacks on an age-specialized regime](../user_guide/age_specialized.md#gate-references-and-leg-fallbacks-on-an-age-specialized-regime).

## `StakeholderRoute`

```python
fallback: ProjectedRegimeValue | Phased
target_stakeholder: str | None = None
```

Where one source stakeholder goes on each branch of a gated transition. A route owns
four destinations, and simulation carries all four: the open branch's regime (the
transition's own key) and role (`target_stakeholder`), and the closed branch's regime
and role (`fallback.regime` and `fallback.stakeholder`). A row landing in a singleton
regime carries no role, so `target_stakeholder=None` means a singleton target.

A bare `ProjectedRegimeValue` fallback is both what the closed branch is worth and where
it puts a row. `Phased(solve=..., simulate=...)` separates them, because what a
household expects from leaving and what a settlement hands it are two objects:

- the **solve** leg prices the source's decision;
- the **simulate** leg supplies the regime, the role and the state coordinates a routed
  row actually lands on.

Both sides of a `Phased` fallback must be `ProjectedRegimeValue`, and each is validated
against the phase that reads it.

Three read-only properties resolve the declaration:

| Property             | Returns                                                                     |
| -------------------- | --------------------------------------------------------------------------- |
| `solve_fallback`     | the `ProjectedRegimeValue` the closed branch is priced at                   |
| `simulate_fallback`  | the `ProjectedRegimeValue` a routed row's regime, role and states come from |
| `fallback_is_phased` | whether the two branches were declared separately                           |

## `ValueDependentTransition`

```python
probability: UserFunction | MarkovTransition
gate: UserFunction
routes: Mapping[str, StakeholderRoute]
gate_references: Mapping[str, ProjectedRegimeValue] = field(
    default_factory=lambda: MappingProxyType({})
)
off_grid: Literal["pointwise", "reject"] = "pointwise"
```

Declared inside `transition`, keyed by target regime name, so target selection and
value-dependent routing are one declaration of one semantic transition.

**The key is always the gate-open target** — the regime a row enters when the gate is
true. A dissolution edge is therefore keyed by the *continuing* collective regime under
`gate = ~D_target`, with each partner's own regime as that partner's route fallback.
Keying it by one partner's regime would send both partners there whenever the couple
stays together.

`probability` accepts exactly what a plain per-target `transition` entry accepts. It and
`gate` are two distinct operations: `probability` selects whether this target edge is
attempted at all, `gate` keeps that target or takes the route's stakeholder-specific
fallback.

`routes` holds one route per **source** stakeholder, keyed by stakeholder name. A
singleton source declares exactly one route, under any key.

### Gate operands

`gate` is a Boolean predicate evaluated pointwise on the **target** regime's grid, in
the target fold's context. It may read:

| Operand                                | Available when                                            |
| -------------------------------------- | --------------------------------------------------------- |
| `V_target`                             | the target regime is a singleton                          |
| `V_target_<s>`                         | the target regime is collective, one per its stakeholders |
| `D_target`                             | the target regime is collective — its dissolution flag    |
| each `gate_references` key             | always, bound to that reference's interpolated value      |
| target states, params, `period`, `age` | always                                                    |

Mutual consent is the strict, unanimous gate
`(V_target_f > V_single_f) & (V_target_m > V_single_m)`; "no dissolution this period" is
`~D_target`.

The whole `V_target` vocabulary is reserved to the engine, so a `gate_references` key
spelled `V_target`, `V_target_<s>` or `D_target` is rejected rather than silently
preempted by the built-in operand.

At the end of each period's solve, the engine folds one gated continuation per declared
edge and source stakeholder $s$ on the target regime's grid,

```{math}
\bar W^s(x) = \operatorname{where}\big(\text{gate}(x),\;
V^{\text{route}_s}_{\text{target}}(x),\;
V^s_{\text{fallback}}(\pi_s(x))\big),
```

and the source's continuation reads $\bar W$ in place of the raw target value.

### `off_grid`

What the edge promises about a landing point between the target's nodes.

- `"pointwise"` (the default) reads every operand at the landing point and applies the
  gate there, in both phases. The operands are interpolated, so the value carries the
  ordinary interpolation error of any continuation — but it is a value one branch really
  delivers, and the branch the solve priced is the branch simulation routes down.
- `"reject"` demands that no such point exists: the model refuses to build unless the
  target regime's grid is reached exactly, i.e. it carries no continuous state. Declare
  it where a straddled gate would be an economic error rather than an approximation.

(collective-where-enforced)=

## Where each rule is enforced

The enforcement point is not uniform, and the distinction is load-bearing: a rule
checked at model build cannot be repaired by an argument to `solve`, while a rule
checked at evaluation only fires once the model runs.

| Rule                                                                                                | Enforced at                                                       | Exception                   |
| --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------- |
| `normalization` is `"pointwise"` or `"none"`                                                        | `ParetoObjective` construction                                    | `ValueError`                |
| weight keys match the regime's stakeholders                                                         | `Regime` construction                                             | `RegimeInitializationError` |
| a **constant** weight is finite and non-negative; the constants leave a positive total              | `Regime` construction                                             | `RegimeInitializationError` |
| a `ValueDependentConstraint` on a singleton regime                                                  | `Regime` construction                                             | `RegimeInitializationError` |
| a regime-level reference projection introduces no free parameter                                    | `Regime` construction                                             | `RegimeInitializationError` |
| a gate is a plain callable, not a `MarkovTransition`                                                | `Regime` construction                                             | `RegimeInitializationError` |
| `routes` covers the source's stakeholder structure                                                  | `Regime` construction                                             | `RegimeInitializationError` |
| taste shocks, a nonlinear certainty equivalent, or a non-`GridSearch` solver on a collective regime | `Regime` construction                                             | `NotImplementedError`       |
| the same three on the SOURCE regime of a `ValueDependentTransition`                                 | `Regime` construction                                             | `NotImplementedError`       |
| a reference or fallback regime exists, and `stakeholder` matches its structure                      | model build                                                       | `ModelInitializationError`  |
| a projection covers exactly the states its position owes                                            | model build                                                       | `ModelInitializationError`  |
| the same-period reference graph is acyclic                                                          | model build                                                       | `ModelInitializationError`  |
| a gate reads `D_target` on a **singleton** target                                                   | model build                                                       | `ModelInitializationError`  |
| a gate or projection argument names a node of the target's own DAG                                  | model build                                                       | `ModelInitializationError`  |
| a `gate_references` key aliases `V_target` / `V_target_<s>` / `D_target`                            | model build                                                       | `ModelInitializationError`  |
| `off_grid="reject"` on a target carrying a continuous state                                         | model build                                                       | `ModelInitializationError`  |
| an ungated transition between regimes of different stakeholder structure                            | model build                                                       | `NotImplementedError`       |
| **the gate's realized return dtype is Boolean**                                                     | **evaluation — every `solve()`, and again in the simulate phase** | `RegimeInitializationError` |
| a **callable** weight is finite, non-negative and positively totalled on the grid                   | evaluation — every `solve()`                                      | `InvalidParamsError`        |

The last two are the ones easily mistaken for build-time checks.

A gate's return annotation cannot constrain what a user function returns, so the
realized dtype is checked where the gate is evaluated. A model whose gate returns a
float builds without complaint and fails on the first `solve()`, naming the target
regime and the phase. This matters because a gate selects its branch with a strict
`where`, in which every nonzero value is true: a numeric gate would open the edge on
every cell rather than express a probability.

A Pareto weighting is likewise a property of *values* once a weight is a function of a
parameter or a state. A declaration admissible for one parameter draw need not be for
the next, so the check runs on every solve, on the regime's own grid, at every age the
regime is active — not only the first time.

(collective-parameters)=

## Parameters

Free arguments of these declarations reach `get_params_template()` in three places.

**Per-stakeholder utilities** appear under `utility_<stakeholder>`, one entry per
stakeholder, at the top level of the regime's template.

**A value constraint's predicate** appears under the name the constraint has in
`constraints`, at the top level of the regime's template branch — for example
`template["couple"]["participation_f"]["slack"]`. Its `references` contribute nothing,
because a regime-level reference projection may introduce no free parameter.

**Pareto weights** appear under the pseudo-function key `pareto_objective`. Every
stakeholder's weight reads one shared namespace, so a parameter named in two weights is
one parameter and appears once. A weight argument that names a state, `period` or `age`
is wired at call time and never surfaces. The key is present only when some weight is a
callable with a free argument.

**Every callable of a gated transition** nests under the **target** regime's name,
beside that target's `next_regime` cell:

| Template entry                                    | Callable                                 |
| ------------------------------------------------- | ---------------------------------------- |
| `gate`                                            | the `gate` predicate                     |
| `gate_ref_<reference key>_<state>`                | one `gate_references` projection         |
| `leg_fallback_<fallback regime>_<state>`          | one route fallback projection (solve)    |
| `simulate_leg_fallback_<fallback regime>_<state>` | the simulate side of a `Phased` fallback |

A fallback entry is named by the regime it falls back to rather than by its `routes`
key, because that is the identity both sides of the solve/simulate seam can spell. Two
routes of one edge falling back to the same regime therefore share one entry, and their
parameters are unioned there.

For a source regime with a gate parameter `marriage_bonus`, a parameterized gate
reference, and a `Phased` fallback whose simulate side takes `settlement_share`:

```python
template["source"]["target"] == {
    "next_regime": {},
    "gate": {"marriage_bonus": "float"},
    "gate_ref_V_outside_x": {"ref_share": "float"},
    "leg_fallback_fallback_x": {},
    "simulate_leg_fallback_fallback_x": {"settlement_share": "float"},
}
```

## Solve and simulate

A gate that reads `D_target` needs the dissolution flags at simulate time. Ask `solve`
to return them and pass them back:

```python
solution, flags = model.solve(
    params=params,
    log_level="debug",
    return_dissolution_flags=True,
)

result = model.simulate(
    params=params,
    initial_conditions=initial_conditions,
    period_to_regime_to_V_arr=solution,
    period_to_regime_to_dissolution_flags=flags,
    log_level="debug",
)
```

`period_to_regime_to_dissolution_flags` is required only for a model with a gate that
reads `D_target`; such a gate raises `NotImplementedError` at simulate time if it is
left `None`. It is a no-op for every other model. Letting `simulate` solve for itself
(`period_to_regime_to_V_arr=None`) threads the flags through automatically.

### Roles are carried per row

A row's own stakeholder identity is what picks its route, and it moves with the row: it
is set from a route's `target_stakeholder` on entering a collective regime, cleared when
the row lands in a singleton one.

Seed it in `initial_conditions` under `own_stakeholder`, in the model-wide vocabulary
`model.stakeholder_names_to_ids`:

```python
initial_conditions = {
    "wealth": jnp.array([1.0, 2.0, 3.0]),
    "regime_id": jnp.full(3, model.regime_names_to_ids["couple"]),
    "own_stakeholder": jnp.full(
        3, model.stakeholder_names_to_ids["f"], dtype=jnp.int32
    ),
}
```

The seed is demanded exactly where the answer turns on it, which is a property of the
starting regime rather than of the model. A cohort starting in a collective regime is
refused without it — rather than defaulted to whichever stakeholder happens to be
declared first — when some collective regime in that regime's forward closure declares a
gated transition with more than one route. Because a row keeps its role across an
ordinary regime transition, the closure is what decides: a two-leg transition the cohort
runs into later demands the seed, and one in a regime the cohort can never reach demands
nothing. A cohort starting in a singleton regime occupies no role and needs none. A
declared code outside the model's role vocabulary, or one naming a role the starting
regime does not have, is rejected as an `InvalidInitialConditionsError`.

Simulation carries one fixed-size cohort. Dissolution does not split one row into two
linked people.

### Published columns

`SimulationResult.to_dataframe()` publishes:

- `value_<stakeholder>`, one column per entry of the collective regime's stakeholders,
  in declaration order;
- `own_stakeholder`, labelled from `model.stakeholder_names_to_ids`. A model with any
  collective regime publishes it for every regime, since a row that leaves a household
  still has to say it now occupies no role; a row in a singleton regime carries no value
  there. A model with no collective regime publishes no such column.

## Capabilities a collective regime does not have

Each of these raises at `Regime` construction, naming the regime slot to change:

- **EV1 taste shocks.** The collective argmax is the hard household maximum of the
  Pareto-weighted objective, not a smoothed one.
- **A nonlinear certainty equivalent.** The per-stakeholder continuation is the linear
  expectation $E[V'^s]$, so `certainty_equivalent=LinearExpectation()` is the only
  admissible declaration.
- **Any solver other than `GridSearch`.** The household argmax and per-stakeholder value
  readout run over the full action product.

The **source** regime of a `ValueDependentTransition` carries the same three
restrictions, whether or not it is collective: it reads the folded continuation through
the grid-search machinery, which a DC-EGM, taste-shock or certainty-equivalent source
does not have.

A `fold=True` IID process is a further restriction, and a different one: a fold
integrates the shock's node axis away immediately after the period's collective readout,
so no same-period gate, value-constraint predicate or reference projection may read the
shock's realized value. That is rejected at `Regime` construction for a regime's own
declarations, and at model build for a folded regime read as another regime's
same-period endpoint.

Finally, an ungated transition between regimes of different stakeholder structure stays
rejected at model build. Mixed singleton/collective topologies go through
`ValueDependentTransition`, which is what lets a row change household structure without
mixing values across it.

## The lowered form

Each of the declarations above is decomposed at `Regime` construction into the fields
the engine threads. Those fields stay on `Regime` as **derived, read-only values**: they
are recomputed from the declarations on every construction, and they cannot be passed to
`Regime(...)` or to `Regime.replace`.

| Declaration                | Derives                                                                                                   |
| -------------------------- | --------------------------------------------------------------------------------------------------------- |
| `CollectiveUtility`        | `stakeholders`, `pareto_objective`, and one `utility_<s>` entry per stakeholder in `decomposed_functions` |
| `ValueDependentConstraint` | `value_constraints[name]`, `same_period_refs[reference key]`                                              |
| `ValueDependentTransition` | `decomposed_transition[target]` (the `probability`) and `gated_edges[target]`                             |

The declaration objects themselves stay where the author wrote them, in `functions`,
`constraints` and `transition`. The engine reads the decomposed views
(`decomposed_functions`, `decomposed_constraints`, `decomposed_transition`) rather than
the raw slots, so reading a derived field tells you what a declaration produced without
that field ever being a way to declare it.

The lowered edge type is `_lcm.gated_edge.GatedEdge`. It is engine-internal and not part
of the public API: there is exactly one way to declare a gated edge, and it is
`ValueDependentTransition`.

Model code reads these fields; model *authors* do not need them. Declare a household
with the six objects above.

## See also

- [Households and value-dependent choice](../user_guide/collective_regimes.md) — the
  guide, with a worked marriage market.
- [Model and Regime](model_and_regime.md) — the slots these declarations sit in.
- [Transitions and phase specialization](transitions.md) — what `probability` accepts,
  and what `Phased` means elsewhere.
- [Runtime, results, and persistence](runtime_and_results.md) — `solve` and `simulate`
  arguments in general.
