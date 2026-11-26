---
title: Collective regimes
---

# Collective regimes

A collective regime carries one value function per member and takes a single shared
action for all of them. A further capability travels with it: a transition whose branch
depends on the target regime.

Five declarations express this, and each one goes inside an argument `Regime` or `Model`
already has — a regime slot, or a `Transition` in `Model(edges=...)` — so a collective
model has no extra constructor arguments to learn.

| Declaration            | Where it is declared                                           | What it expresses                                    |
| ---------------------- | -------------------------------------------------------------- | ---------------------------------------------------- |
| `CollectiveUtility`    | `functions={"utility": ...}`                                   | the regime's members and their flow utilities        |
| `ParetoObjective`      | `CollectiveUtility(objective=...)`                             | how member action values are scalarized              |
| `Gate`                 | `Transition(gates=...)` in `Model(edges=...)`, keyed by target | a transition into one target, gated on its state     |
| `MemberRoute`          | `Gate(routes=...)`                                             | where one source member goes on each branch          |
| `ProjectedRegimeValue` | a route's `fallback`                                           | another regime's value in the landing period, mapped |

All five are frozen, keyword-only dataclasses defined in `src/lcm/collective.py` and
exported from `lcm`. A type violation on any of them raises `RegimeInitializationError`.

For the modelling narrative — why a regime with several members declares its Pareto
weights instead of summing them, and where the members go when no action is feasible —
see the [user guide on collective regimes](../user_guide/collective_regimes.md).

(api-collective-utility)=

## `CollectiveUtility`

```python
utilities: Mapping[str, UserFunction | Phased | None]
objective: ParetoObjective | None = None
```

Declared as `functions={"utility": CollectiveUtility(...)}`. That declaration is what
makes a regime collective: the `utilities` keys **are** the regime's members, in
insertion order, and that order fixes the trailing member axis of the regime's value
function and of every published array.

`objective=None` (the default) weights the members equally.

The declaration itself stays in the raw `functions["utility"]` slot. Construction
derives the regime's member tuple and Pareto objective from it, while the engine-facing
`decomposed_functions` view exposes every nondelegated body and any delegated body
already supplied. After model-level declarations are merged, successful finalization
guarantees one `utility_<member>` entry per member. A utility entry may be:

- a callable, used in both phases;
- `Phased(solve=..., simulate=...)`, for different solve and simulation bodies; or
- `None`, which delegates that body to a `utility_<member>` entry supplied on the regime
  or by a model-level `functions` declaration.

The members maximize the objective over the feasible action product. Each member's own
action value is read off at that common choice.

```{math}
a^*(x) = \arg\max_{a\,:\,F(x,a)} \sum_\kappa \lambda_\kappa(x)\, Q^\kappa(x, a),
\qquad V^\kappa(x) = Q^\kappa(x, a^*(x)).
```

The following are rejected:

- an empty `utilities` mapping, when the `CollectiveUtility` is constructed;
- a member whose `utility_<member>` is also declared directly in `functions` as a
  different function, when the `Regime` is constructed; and
- a delegated (`None`) member for which no `utility_<member>` body arrives after
  model-level declarations are merged, when the model finalizes the regime.

`members` and `pareto_objective` are derived, read-only fields. They are not parallel
declaration routes and cannot be passed to `Regime(...)` or `Regime.replace`. Every
regime has `members`: the `utilities` keys for a collective regime, and `("self",)` for
a regime declared with a plain utility function. A collective regime may not name a
member `"self"`.

(api-pareto-objective)=

## `ParetoObjective`

```python
weights: Mapping[str, UserFunction | float]
normalization: str = "pointwise"
```

`weights` holds one Pareto weight $\lambda_\kappa$ per member, keyed by member name. A
`float` is a constant. A callable is a function of the regime's **states** and of
`period` / `age`; every other argument it names becomes a free scalar parameter under
the regime's `pareto_objective` key (see [Parameters](#collective-parameters)). An
argument spelled like one of the regime's other functions does **not** receive that
function's output — it becomes a parameter you must supply.

A Pareto weight may not read an action. A Pareto weight that varies with the choice
states a different objective per candidate, whose maximizer is a Pareto optimum of no
fixed weighting.

`normalization` is annotated as a plain `str` and validated against two values:

- `"pointwise"` (the default) divides by the total at each cell, so the Pareto weights
  sum to one wherever the objective is evaluated and a state-dependent declaration keeps
  one scale across the grid;
- `"none"` uses the declared Pareto weights as they stand. The scalarization is then not
  on the members' own scale, and comparing values across cells whose totals differ
  compares different objectives.

Admissibility of a weighting — one finite, non-negative Pareto weight per member with a
strictly positive total — is checked in two places, because a constant is knowable at
construction while a function of a parameter is not. See
[Where each rule is enforced](#collective-where-enforced).

(api-projected-regime-value)=

## `ProjectedRegimeValue`

```python
regime: RegimeName
projection: Mapping[StateName, UserFunction]
member: str | None = None
```

Another regime's value in the period a gated row lands in, read at mapped state
coordinates. The reference regime's value function of that period is already solved when
the source decides, and it is interpolated at the projected coordinates: linear on
continuous axes, lookup on discrete axes.

`regime` names another regime of the model. A fallback read requires its value in every
period the gated target is reached, so the reference regime is solved there too without
being declared as a start. No transition edge between the two is required — a reference
read works across otherwise unconnected regime islands.

`member` names whose value to read from a **collective** reference regime. It is
required there and must be `None` for a singleton reference.

The `projection` of a route's fallback projects from the TARGET regime's grid, may
introduce free parameters, which become edge parameters, and owes one coordinate per
state the reference regime carries in simulation. That is more than the value function's
states, because a route does not only price the closed branch, it writes the routed row
into the fallback regime, and forward simulation carries every one of that regime's
simulate states per agent. A state left unprojected would keep whatever the row held
before the edge routed it there. The solve states are a subset, so the same projection
still serves the fold's value read.

A `ProjectedRegimeValue` on an age-specialized reference regime is measured against the
grid of the period whose value is being folded; see
[Leg fallbacks on an age-specialized regime](../user_guide/age_specialized.md#leg-fallbacks-on-an-age-specialized-regime).

(api-member-route)=

## `MemberRoute`

```python
fallback: ProjectedRegimeValue | Phased
target_member: str | None = None
```

Where one source member goes on each branch of a gated transition. A route owns four
destinations, and simulation carries all four: the open branch's regime (the
transition's own key) and role (`target_member`), and the closed branch's regime and
role (`fallback.regime` and `fallback.member`). A row landing in a singleton regime
carries no role, so `target_member=None` means a singleton target.

A bare `ProjectedRegimeValue` fallback is both what the closed branch is worth and where
it puts a row. `Phased(solve=..., simulate=...)` separates them, because what a member
expects from leaving and what a transfer hands it are two objects:

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

(api-gate)=

## `Gate`

```python
predicate: UserFunction
routes: Mapping[str, MemberRoute]
off_grid: Literal["pointwise", "reject"] = "pointwise"
```

Declared in the source's `Transition` in `Model(edges=...)`, beside the law and keyed by
the gated target:

```python
edges = {
    "couple": Transition(
        law={"couple": StochasticTransition(func=probability_one)},
        gates={"couple": Gate(predicate=above_wealth_floor, routes=...)},
    ),
}
```

The law supplies the probability of reaching the gated target, exactly as for any other
destination; the gate decides, once there, whether a row stays or takes its route's
fallback. The two are distinct operations: the law selects whether this target edge is
attempted at all, the predicate keeps that target or takes the route's member-specific
fallback.

The `Transition`'s destinations include the gate-open target and every route's fallback
regime. A law that names its targets derives them: each route fallback regime is reached
wherever its gated target is. A gated source declares a `Transition` even where every
age has a single outgoing edge, because the gate adds routing that the edge alone does
not state.

**The key is always the gate-open target** — the regime a row enters when the predicate
is true. An edge that may dissolve a couple is therefore keyed by the *continuing*
collective regime, with each partner's own regime as that partner's route fallback.
Keying it by one partner's regime would send both partners there whenever the couple
stays together.

`routes` holds one route per **source** member, keyed by member name. A singleton source
declares exactly one route, under any key.

One `Gate` per target holds at every age the target is reached and in both phases. A
gate is never wrapped in `ByAge` or `Phased`; only a route's `fallback` may be `Phased`.
With `Transition(law=Phased(solve=..., simulate=...), gates=...)` the law differs by
phase and the gates do not, so the perceived and the actual transition probabilities can
differ without changing the edge. With `Phased` edges, a target reached in both phases
carries the equal `Gate` in both or none. A gate on a target its `Transition` never
reaches is refused.

### Predicate operands

`predicate` is a Boolean predicate evaluated pointwise on the **target** regime's grid,
in the target fold's context. It reads target states, params, `period` and `age`.

A gate routes on states, never on a value. A predicate argument named `V_target`,
`V_target_<member>` or `D_target` raises `GateReadsValueError`, a
`RegimeInitializationError`, when the `Gate` is constructed. A decision on the
composition that compares values is a `CompositionAction` in
`Regime(composition_action=...)` of the regime that decides. Formation uses a `Consent`
rule on the single's regime. Separation uses `PCOnActions`, `PCOnWeight` or
`MutualConsent` on the regime with several members.

At the end of each period's solve, the engine folds one gated continuation per declared
edge and source member $\kappa$ on the target regime's grid,

```{math}
\bar W^\kappa(x) = \operatorname{where}\big(\text{predicate}(x),\;
V^{\text{route}_\kappa}_{\text{target}}(x),\;
V^\kappa_{\text{fallback}}(\pi_\kappa(x))\big),
```

and the source's continuation reads $\bar W$ in place of the raw target value.

### `off_grid`

What the edge promises about a landing point between the target's nodes.

- `"pointwise"` (the default) applies the predicate at the landing point, in both
  phases, and interpolates each branch's value there. That value carries the ordinary
  interpolation error of any continuation — but it is a value one branch really
  delivers, and the branch the solve priced is the branch simulation routes down.
- `"reject"` demands that no such point exists: the model refuses to build unless the
  target regime's grid is reached exactly, i.e. it carries no continuous state. Declare
  it where a straddled gate would be an economic error rather than an approximation.

(collective-where-enforced)=

## Where each rule is enforced

The enforcement point is not uniform, and the distinction is load-bearing: a rule
checked at model build cannot be repaired by an argument to `solve`, while a rule
checked at evaluation only fires once the model runs.

| Rule                                                                                                                                                           | Enforced at                                                       | Exception                   |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------- |
| `normalization` is `"pointwise"` or `"none"`                                                                                                                   | `ParetoObjective` construction                                    | `ValueError`                |
| Pareto weight keys match the regime's members                                                                                                                  | `Regime` construction                                             | `RegimeInitializationError` |
| a **constant** Pareto weight is finite and non-negative; the constants leave a positive total                                                                  | `Regime` construction                                             | `RegimeInitializationError` |
| a gate's `predicate` reads no `V_target`, `V_target_<member>` or `D_target`                                                                                    | `Gate` construction                                               | `GateReadsValueError`       |
| a gate's `predicate` is a plain callable, not a `StochasticTransition`                                                                                         | model build, binding the edge law to its source                   | `RegimeInitializationError` |
| `routes` covers the source's member structure                                                                                                                  | model build, binding the edge law to its source                   | `RegimeInitializationError` |
| a gate sits on a target its `Transition` reaches; a target reached in both phases carries one equal `Gate` in both or none                                     | model build                                                       | `ModelInitializationError`  |
| taste shocks (unless exactly one constant or callable Pareto weight can be positive), or a non-`GridSearch` solver, on a collective regime                     | `Regime` construction                                             | `NotImplementedError`       |
| taste shocks, a nonlinear certainty equivalent, or a non-`GridSearch` solver on the SOURCE regime of a gated transition                                        | model build, binding the edge law to its source                   | `NotImplementedError`       |
| taste shocks on a regime that declares a `composition_action`, whether for a separation or a formation                                                         | model build                                                       | `RegimeInitializationError` |
| a `Decoupled` aggregator on a regime that declares a `composition_action`, whether for a separation or a formation                                             | model build                                                       | `RegimeInitializationError` |
| a reference or fallback regime exists, and `member` matches its structure                                                                                      | model build                                                       | `ModelInitializationError`  |
| a projection covers exactly the states its position owes                                                                                                       | model build                                                       | `ModelInitializationError`  |
| a predicate or projection argument names a node of the target's own DAG                                                                                        | model build                                                       | `ModelInitializationError`  |
| `off_grid="reject"` on a target carrying a continuous state                                                                                                    | model build                                                       | `ModelInitializationError`  |
| the law of a shock a composition action observes in a group the source also draws reads only parameters and states its projection returns unchanged            | model build                                                       | `ShockDeclarationError`     |
| a transition between regimes of different member structure with neither a `Gate` nor, from a singleton source, a `Transition(roles=...)` entry for that target | model build                                                       | `NotImplementedError`       |
| **the predicate's realized return dtype is Boolean**                                                                                                           | **evaluation — every `solve()`, and again in the simulate phase** | `RegimeInitializationError` |
| a **callable** Pareto weight is finite, non-negative and positively totalled on the grid                                                                       | evaluation — every `solve()`                                      | `InvalidParamsError`        |
| with taste shocks, exactly one member's Pareto weight is positive at every cell of the grid                                                                    | evaluation — every `solve()`                                      | `InvalidParamsError`        |

The last three are the ones easily mistaken for build-time checks.

A gate's return annotation cannot constrain what a user function returns, so the
realized dtype is checked where the gate is evaluated. A model whose gate returns a
float builds without complaint and fails on the first `solve()`, naming the target
regime and the phase. This matters because a gate selects its branch with a strict
`where`, in which every nonzero value is true: a numeric gate would open the edge on
every cell rather than express a probability.

A Pareto weighting is likewise a property of *values* once a Pareto weight is a function
of a parameter or a state. A declaration admissible for one parameter draw need not be
for the next, so the check runs on every solve, on the regime's own grid, at every age
the regime is active — not only the first time.

## Preferences per member

A collective regime declares `koopmans_aggregator` and `certainty_equivalent` as a
regime with one member does. A bare declaration is one aggregator, or one certainty
equivalent, with one set of parameters, applied to every member. Either slot may instead
map every member name to that member's own declaration, so that members discount or
weigh risk differently:

```python
couple = Regime(
    functions={"utility": CollectiveUtility(utilities={"f": u_f, "m": u_m})},
    koopmans_aggregator={"f": CESAggregator(), "m": LinearAggregator()},
    certainty_equivalent={"f": PowerMean(), "m": LinearExpectation()},
    # states, actions and the rest as for any regime
)
```

Member $\kappa$'s continuation is aggregated under its own certainty equivalent,
$CE^\kappa = g_\kappa^{-1}(\sum_r p_r \, E_w[g_\kappa(V'^\kappa_r)])$, and combined with
its utility by its own aggregator, $Q^\kappa = W^\kappa(u^\kappa, CE^\kappa)$. Linear
and nonlinear members may be mixed. An additive shock transition is averaged into every
member's continuation as into a single's. A slot keyed by member needs a solver that
supports several members, such as `GridSearch()`; an endogenous-grid solver takes either
slot as one bare value only.

`Decoupled(objective=..., value=...)` is declared for the regime as a whole, never per
member. Without taste shocks the members choose the action that maximizes the
Pareto-weighted sum of their objectives, and store each member's `value` there. Ties in
that sum go to the members' stored values, so the stored values do not depend on the
order of the actions. When exactly one member has a positive Pareto weight, that member
decides, and a tie goes to every other member's stored value in member order and then to
the deciding member's; with several positive Pareto weights, to every member's stored
value in member order. Actions that are still tied give every member the same value. The
first of them in grid order is simulated, so the simulated action can depend on the
order of the actions. With taste shocks the one deciding member chooses on its own
objective, as a regime with one member would. Its stored value is the expectation of its
`value` at the noisy choice including the chosen action's shock. Every other member's
stored value is the mean of its own `value` under the deciding member's choice
probabilities, without a shock term. Ties in the deciding member's objective among the
continuous actions of one discrete action go to the stored values in the same order. A
regime with a `Decoupled` aggregator declares no composition action. Another regime's
composition action reads it at the values its solve publishes (see
[The composition action](model_and_regime.md#api-current-exits)).

`Model.lookup_policy(..., keep_intermediates=...)` and the NaN diagnostics of a failed
solve work for a collective regime as for a regime with one member. `Q`, `U` and `CE`
end in the member axis, in member order; the diagnostics count a cell as NaN where any
member's value is.

(collective-parameters)=

## Parameters

Free arguments of these declarations reach `get_params_template()` in four places.

**Per-member utilities** appear under `utility_<member>`, one entry per member, at the
top level of the regime's template.

**Per-member preferences** appear under `koopmans_aggregator_<member>` and
`certainty_equivalent_<member>` when the slot is keyed by member. A bare declaration
keeps the single keys `koopmans_aggregator` and `certainty_equivalent`, shared by every
member.

**Pareto weights** appear under the pseudo-function key `pareto_objective`. Every
member's Pareto weight reads one shared namespace, so a parameter named in two Pareto
weights is one parameter and appears once. An argument of a Pareto weight that names a
state, `period` or `age` is wired at call time and never surfaces. The key is present
only when some Pareto weight is a callable with a free argument.

**Every callable of a transition law and its gates** belongs to the edges, so its
parameters sit at its declaration path under `params["edges"][source][target]`, not
under the source regime:

| Template path below `params["edges"][source][target]`   | Callable                                                           |
| ------------------------------------------------------- | ------------------------------------------------------------------ |
| `[arg]`                                                 | the law's `StochasticTransition` cell for the target, gated or not |
| `["predicate"][arg]`                                    | the gate's `predicate`                                             |
| `["routes"][route]["fallback"][state][arg]`             | one route fallback projection                                      |
| `["routes"][route]["fallback"]["solve"][state][arg]`    | the solve side of a `Phased` fallback                              |
| `["routes"][route]["fallback"]["simulate"][state][arg]` | the simulate side of a `Phased` fallback                           |

A fallback is keyed by its `routes` key and each projection by the state it projects.
Only callables that read a free argument appear. A value may also be supplied once for
every edge callable of the source, at `params["edges"][source][arg]`, or at the model
level; a value under the source regime never reaches a gate or projection. See
[Edge parameter paths](transitions.md#api-edge-parameters).

For a source regime whose gate predicate takes `wealth_floor`, and a `Phased` fallback
whose simulate side takes `transfer_share`:

```python
template["edges"]["source"]["target"] == {
    "predicate": {"wealth_floor": "float"},
    "routes": {"f": {"fallback": {"simulate": {"x": {"transfer_share": "float"}}}}},
}
```

## Solve and simulate

A solution carries no dissolution artifact. A row that lands in a gated target cell
where no action is feasible takes its route's fallback when simulated;
`SimulationResult.dissolution_fallbacks` records those rows when runtime checks are on.
Pass the complete solution to `simulate(solution=...)`, or omit it to let simulation
solve first:

```python
solution = model.solve(
    params=params,
    log_level="debug",
)

result = model.simulate(
    params=params,
    initial_conditions=initial_conditions,
    solution=solution,
    log_level="debug",
)
```

### Roles are carried per row

A row's own member identity picks its route, and the row carries it. On entering a
collective regime, the row takes the role that its route, transition or destination
names: a route's `target_member`, a `Transition(roles=...)` entry, or a `Destination`'s
`role`. The role is cleared when the row lands in a singleton regime.

Seed it in `initial_conditions` under `own_member`, in the model-wide vocabulary
`model.member_names_to_ids`:

```python
initial_conditions = {
    "wealth": jnp.array([1.0, 2.0, 3.0]),
    "regime_id": jnp.full(3, model.regime_names_to_ids["couple"]),
    "own_member": jnp.full(3, model.member_names_to_ids["f"], dtype=jnp.int32),
}
```

The seed is demanded exactly where the answer turns on it, which is a property of the
starting regime rather than of the model. A cohort starting in a collective regime is
refused without it — rather than defaulted to whichever member happens to be declared
first — when some regime in that regime's forward closure, the regime itself included,
routes by member: it declares a gated transition with more than one route, or an
`CompositionAction` with destinations for more than one member. Because a row keeps its
role across an ordinary regime transition, the closure decides: such a transition or
composition action that the cohort reaches later demands the seed, and one in a regime
the cohort can never reach demands nothing. A cohort starting in a singleton regime
occupies no role and needs none. A declared code outside the model's role vocabulary, or
one naming a role the starting regime does not have, is rejected as an
`InvalidInitialConditionsError`.

Simulation carries one fixed-size cohort. Dissolution does not split one row into two
linked people.

### Published columns

`SimulationResult.to_dataframe()` publishes:

- `value_<member>`, one column per entry of the collective regime's members, in
  declaration order;
- `own_member`, labelled from `model.member_names_to_ids`. A model with any collective
  regime publishes it for every regime, since a row that leaves a collective regime
  still has to say that it occupies no role; a row in a singleton regime carries no
  value there. A model with no collective regime publishes no such column.

## Capabilities a collective regime does not have

Each of these raises at `Regime` construction, naming the regime slot to change:

- **EV1 taste shocks with more than one deciding member.** A collective regime admits
  taste shocks when exactly one member has a positive Pareto weight, and that member
  decides. Its value is its own smoothed expected maximum
  $\sigma \log \sum_j e^{Q^d_c(j)/\sigma}$ over the discrete actions $j$, where
  $Q^d_c(j)$ is its maximum over the feasible continuous actions, attained at $c^*(j)$.
  Every other member $\kappa$'s value is $\sum_j P(j)\, Q^\kappa(j, c^*(j))$, its own
  action value at the deciding member's best continuous action, weighted by the deciding
  member's choice probabilities $P(j)$. A discrete action of probability zero
  contributes nothing. Simulation draws the deciding member's choice as it draws the
  choice of a regime with one member. `Model.lookup_policy` returns $Q^d_c(j)$ as `Qc`,
  $P(j)$, $c^*(j)$ and the member values, without a draw. A cell where every feasible
  alternative is worth $-\infty$ to the deciding member is worth $-\infty$ to every
  member and has no choice probabilities, also with a `Decoupled` aggregator. A
  simulation that reaches it raises `NoFeasibleActionError`, as at a cell without a
  feasible action. Constant Pareto weights with more than one positive entry are refused
  here; a Pareto weight declared as a function is checked at every solve, at every cell.
  A regime with taste shocks, with one member or several, cannot declare a composition
  action and is refused when the model is built. Another regime's composition action
  reads its member values as the solve publishes them, as destination or as `Consent`
  counterparty.
- **Any solver other than `GridSearch`.** The members' joint argmax and the per-member
  value readout run over the full action product.

The **source** regime of a gated transition, whether or not it is collective, declares
no taste shocks, no nonlinear certainty equivalent and no solver other than
`GridSearch`: it reads the folded continuation through the grid-search machinery, which
a DC-EGM, taste-shock or certainty-equivalent source does not have. A source restriction
raises at model build, when the edge law is bound to its source, since a `Regime` on its
own declares no edge.

Finally, a transition between regimes of different member structure is rejected at model
build unless it declares, for that target, a `Gate` or, from a singleton source, a role.
`Transition(roles={"couple": "f"})` says that a row that enters the collective target
`couple` from the singleton source continues there as its member `f`: its continuation
toward `couple` is the couple's value for `f`, and a simulated row entering `couple`
carries the role `f`. A role is the always-open `Gate` whose fallback is the target
itself in the same role, so a row landing where the target has no feasible action raises
`NoFeasibleActionError`, as on a plain edge. A target takes a role or a `Gate`, never
both, and a collective source takes no role. A collective source changes member
structure through a `Gate` or through its own `CompositionAction`.

## Derived engine views

Each declaration above stays in the raw slot where the author wrote it. `Regime`
construction derives stored, read-only fields from the declaration in `functions`. A
`Gate` lives in a `Transition` in `Model(edges=...)`, so its derived fields belong to
the law `Model` binds for the source, published as `model.graph.laws[source]`. The
`decomposed_*` properties separately compute engine-facing views from the current raw
declarations whenever they are read. Neither kind of output is a declaration route: none
can be passed to `Regime(...)` or to `Regime.replace`.

| Declaration         | Construction-derived fields                    | On-access engine view                                                                                          |
| ------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `CollectiveUtility` | `members`, `pareto_objective`                  | `decomposed_functions`: nondelegated and already-supplied bodies; complete after successful model finalization |
| `Gate`              | `model.graph.laws[source].gated_edges[target]` | `model.graph.laws[source].decomposed_transition[target]`: the law's cell for the target                        |

The declaration objects themselves stay where the author wrote them, in `functions`,
`constraints` and the `Model(edges=...)` transition. The engine reads the decomposed
views (`decomposed_functions`, `decomposed_constraints`, `decomposed_transition`) rather
than the raw slots. Reading either a stored derived field or a decomposed view therefore
reveals what a declaration produced without creating a second way to declare it.

The derived edge type is `_lcm.gated_edge.GatedEdge`. It is engine-internal and not part
of the public API: there is exactly one way to declare a gated edge, and it is a `Gate`
in `Transition(gates=...)`.

Model code reads these fields; model *authors* do not need them. Declare a collective
regime with the five objects above.

## See also

- [User guide on collective regimes](../user_guide/collective_regimes.md) — the guide,
  with a worked marriage market.
- [Model and Regime](model_and_regime.md) — the slots these declarations sit in.
- [Transitions and phase specialization](transitions.md) — what a `Transition` law
  accepts, and what `Phased` means elsewhere.
- [Runtime, results, and persistence](runtime_and_results.md) — `solve` and `simulate`
  arguments in general.
