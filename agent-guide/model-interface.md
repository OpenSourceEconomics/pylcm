## Model and Regime Interface

### Regime Definition

The `Regime` class defines a single regime in the model. The regime name is specified as
the key in the `regimes` dict passed to `Model`:

```python
# Non-terminal regime (its outgoing edges live in `Model(edges=...)`)
Regime(
    states={  # Pure outcome-space grids
        "wealth": LinSpacedGrid(...),
        "education": DiscreteGrid(EduStatus),
    },
    state_transitions={  # How states evolve over time
        "wealth": next_wealth,  # Deterministic transition
        "education": fixed_transition("education"),  # Fixed state (identity law)
    },
    actions={"action_name": Grid},  # Action grids (can be empty)
    functions={  # Must include "utility"; other functions optional
        "utility": utility_function,
        "name": helper_func,
    },
    constraints={"name": constraint_func},  # Optional: constraint functions
    koopmans_aggregator=CESAggregator(),  # Optional: overrides the model-level one
    certainty_equivalent=PowerMean(),  # Optional: overrides the model-level one
)

# Terminal regime (no outgoing edges, no state_transitions)
Regime(
    functions={"utility": terminal_utility},
    states={"wealth": LinSpacedGrid(...)},
)

# Target-dependent transitions (keyed by target regime name)
Regime(
    states={"health": DiscreteGrid(Health)},
    state_transitions={
        "health": {
            "working": StochasticTransition(func=health_probs_working),
            "retired": StochasticTransition(func=health_probs_retired),
        },
    },
    # Additional configuration may follow.
)

# Regime transitions: structure and law, declared on the model
Model(
    edges={
        # Several destinations at some source age: a `Transition` carries the law
        "working": Transition(
            targets={"working": AgeRange(start=25, exclusive_stop=65), "retired": 64},
            law=next_regime_func,
        ),
        # One destination at every source age: the graph is the law
        "retired": {"dead": AgeRange(start=65, exclusive_stop=90)},
        # "dead" has no outgoing edges, so it is terminal
    },
    # Additional configuration may follow.
)
```

**Regime Requirements:**

- `Regime` takes no regime transition. `Model(edges=...)` declares every regime
  transition, structure and law. A source maps to a plain `{target: source_ages}`
  mapping when every source age has exactly one destination (the graph is the law), or
  to `Transition(targets={target: source_ages, ...}, law=..., gates=...)` when some
  source age has several or a destination is gated. A regime with no outgoing edges is
  terminal. A `Transition` law is a regime
  name, plain deterministic function, `DeterministicTransition(func=...)`, full-vector
  `StochasticTransition(func=...)`, per-target scalar probability mapping, `ByAge`
  selecting complete laws, or `Phased` of those. `targets` is optional when the law
  names its targets (per-target mapping, regime name, or `ByAge` / `Phased` of those):
  it is then derived — each key at the non-final ages its case covers, each gate's route
  fallback regimes at the gated target's ages — and a supplied `targets` must equal it
  exactly (`ModelInitializationError` otherwise). A law over all targets requires
  `targets`. A `ByAge` law must select every age with several outgoing edges. A law is
  evaluated at every source age with outgoing edges, including single-destination ages,
  where it must put unit mass on that destination; its parameter slots do not depend
  on the horizon. Ordinary scalar probability cells require `StochasticTransition`.
- Edge parameters belong to the edges: the parameter path of an edge-declared callable
  is its declaration path under `params["edges"]` — `[source][arg]` for a law over all
  targets, `[source][target][arg]` for a per-target cell (gated or not), and
  `[source][target]["predicate"][arg]`,
  `[source][target]["references"][reference][state][arg]`,
  `[source][target]["routes"][route]["fallback"][state][arg]` for a target's `Gate`.
  Values resolve from that path, `params["edges"][source][arg]` or the model level; a
  regime-level value never reaches an edge callable. Per-target state laws stay under
  their source regime.
- `koopmans_aggregator` and `certainty_equivalent` are optional: `None` means the regime
  takes the model-level value. Declaring either at the regime level requires declaring
  it in *every* non-terminal regime — no mixing with the model-level broadcast. Terminal
  regimes take neither and declaring one is an error.
- `functions` must contain a `"utility"` entry (the utility function); checked when the
  model finalizes its regimes, not at `Regime` construction
- `state_transitions` maps state names to transition functions. Every non-process state
  in a non-terminal regime must have an entry (checked at model build).
  `fixed_transition(state_name)` marks a fixed state (identity law; its argument must
  match the dict key). `None` is rejected. Wrap in `StochasticTransition` for stochastic
  transitions.
- Per-target dicts in `state_transitions` map target regime names to transition
  functions — every reachable target carrying the state must be listed, and no
  unreachable or unknown target may be (checked at model build; narrow reachability with
  `Model.edges`). Within a per-target dict, stochasticity must be
  consistent (all `StochasticTransition` or none).
- Stochastic processes have intrinsic transitions and must NOT appear in
  `state_transitions`.
- Terminal regimes must have empty `state_transitions`.
- Regime, state, action, function, constraint, stakeholder, route and gate-reference
  names, and parameter arguments of model functions, cannot contain the reserved
  separator `__` and cannot start or end with `_`. `edges` is reserved as a regime, function
  and argument name

### Model Creation

```python
from lcm import AgeGrid, categorical


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    retired: ScalarInt


Model(
    regimes={  # Required: dict mapping names to Regime instances
        "working": working_regime,
        "retired": retired_regime,
    },
    ages=AgeGrid(start=25, inclusive_stop=75, step="Y"),  # Or n_periods=...; exactly one
    regime_id_class=RegimeId,  # Required: dataclass mapping names to indices
    edges={"working": {"retired": 25}},  # Source → destination → source ages
    initial_nodes=((25, "working"),),  # Required: admissible starting age-regime pairs
    description="Optional description",
    enable_jit=True,  # Control JAX compilation (default: True)
)
```

**Model-level regime slots (broadcast):**

`Model(functions=..., constraints=..., states=..., state_transitions=..., actions=...)`
accepts exactly what the regime-level slot accepts (incl. `Phased`, stochastic
processes, per-target dicts, `fixed_transition`). Each entry is merged into every regime
under the exactly-one-level rule — a name is defined at model level or regime level,
never both (ambiguity errors); a regime-level `None` masks the model entry (masking a
state also drops its broadcast law; an unbound mask errors). Broadcast laws are not
merged into terminal regimes.

**Model-level continuation slots:**

`Model(koopmans_aggregator=..., certainty_equivalent=...)` default to
`LinearAggregator()` and `LinearExpectation()`. Unlike the mapping slots above these are
single values, so the rule is all-or-nothing rather than per-name: declare each here, or
in every regime that has a continuation, never some of each (a mixed declaration
errors). Terminal regimes receive neither, and declaring either on one is an error.

Broadcast states and actions are pruned per regime by DAG reachability: a broadcast
variable survives only where a root computation (utility, the Koopmans aggregator,
constraints, derived categoricals, the regime transition, or a law of motion toward a
reachable target that keeps the state) transitively reads it. The two phase slices are
closed *jointly* to one least fixed point, not pruned one slice at a time, so a
simulation-side read can keep a solution-side entry law alive. Regime-level declarations
are never pruned. `model.pruned_variables` records the result per regime.

Declare device axes with `Model(execution_config=ExecutionConfig(sharded_states=...))`,
using model-level discrete states or the narrow continuous GridSearch route described in
`docs/user_guide/tuning.md`: one concrete `LinSpacedGrid`, sole continuous and sharded
state, retained in every regime. Sharding follows pruning, regime by regime: a regime
that prunes the sharded state simply runs single-device, and only a state that *every*
regime prunes is refused. The narrow continuous route keeps the stricter rule — there
the state must be retained in every regime. Grids define outcome spaces only;
planner-owned widths live in `ExecutionConfig(axis_widths=...)` and name axes the
model's actual core programs declare.

**Model Requirements:**

- `edges` is required: `{source: {target: source_age_selector}}` for a source with one
  destination per source age, `{source: Transition(targets={target: selector, ...},
  law=..., gates=...)}` for a source with several at some age or a gated destination,
  or `Phased(solve=..., simulate=...)` with one such mapping per phase. `model.edges`
  keeps them as declared, laws included; `model.graph.edges` is law-free. Selectors are
  exact ages, nonempty tuples, integer ranges, or
  `AgeRange(start=..., exclusive_stop=...)`. An edge lands at the next `AgeGrid`
  coordinate. Terminal regimes have no outgoing edges.
- `initial_nodes` is required with no default. Prefer explicit pairs such as
  `((25, "working"),)`. A mapping from age selectors to a regime name or nonempty
  sequence of names is also accepted. The normalized immutable pairs are admissible
  roots; solve demand includes realized visits and their perceived value dependencies.
- Wrappers and decorator factories are targetless and shared by state and regime laws:
  `DeterministicTransition(func=...)`, `StochasticTransition(func=...)`,
  `@deterministic_transition()`, and `@stochastic_transition()`.
- `ByAge` selects numerical laws; it never declares roots or connectivity.
- `model.graph` is an immutable inspection result, not a user-constructed graph class.
  `graph.edges.solve` / `.simulate` retain declared edges with exact source ages;
  `.solution` / `.simulation` expose effective period-indexed graphs. `.nodes` contains
  valued age–regime pairs and `.visited_nodes` contains realized reachable pairs.
  `.pruned_edges["solve" or "simulate"]` maps `(source_age, source, target)` to
  `"fixed_zero_probability"` for construction-fixed zero proofs.
- `regime_id_class` must be a dataclass with fields matching regime names (use
  `@categorical`)
- Field values are consecutive `ScalarInt` (0-d `jnp.int32`) scalars starting from 0,
  auto-assigned by `@categorical`

### Core Methods

- `model.solve(params=params, log_level="debug")` - Solve the model and return a
  `SolutionResult`; value arrays live in `solution.values`
- `model.simulate(params=params, initial_conditions=initial_conditions, solution=solution, log_level="debug")`
  \- Simulate forward from the complete result. Omit `solution` to solve automatically.
- Collective dissolution flags are addressed replay artifacts in `SolutionResult`;
  `simulate(solution=...)` validates and projects them. Automatic simulation threads
  them directly.
- `log_level` is **required** on both `solve()` and `simulate()`
  (`off < warning < progress < debug`). It governs all runtime validation: `"off"` skips
  it, `"warning"` / `"progress"` warn and continue, `"debug"` raises. Start projects at
  `"debug"`.
- `model.validate_initial_conditions(initial_conditions=..., params=...)` and
  `model.initial_conditions_feasibility(initial_conditions=..., params=...)` check a
  population without solving or simulating; the second returns a per-subject boolean
  mask in caller order. They take the `simulate` input forms (mapping or DataFrame), no
  `log_level`, apply every initial-condition check `simulate` applies (including the
  `own_stakeholder` declaration of a collective start) and always raise on malformed
  structure. At present they run in one eager pass on one device without memory
  admission or padding; the verdict does not depend on that. Unsupported checks
  (age-specialized constraint ancestors with subjects at several ages) raise
  `UnsupportedOperationError`.

### Derived Categoricals

When parameters are indexed by a DAG function output (not a model state/action), declare
`derived_categoricals={"name": DiscreteGrid(CategoryClass)}` on the `Regime` that uses
it. For convenience, model-level `derived_categoricals` on `Model(...)` are broadcast to
all regimes under the exactly-one-level rule — a name is declared at model level or
regime level, never both (ambiguity errors, also when the grids match). Functions used
as derived categoricals must return **integer** types, not booleans — JAX cannot use
booleans as array indices inside JIT. Use `jnp.int32(...)` to cast.

### Collective regimes and value-dependent choice

A regime whose utility is a `CollectiveUtility` has **stakeholders** — the `utilities`
keys, in insertion order, which fix the trailing axis of `V` and of every published
array. Everything is declared in a slot that already exists: the regime's utility and
constraints, and the `Transition` on its `Model(edges=...)` entry:

```python
Regime(
    functions={"utility": CollectiveUtility(utilities={"f": u_f, "m": u_m})},
    constraints={
        "participation_f": ValueDependentConstraint(  # goes in `constraints`
            predicate=participation_f,
            references={
                "V_alone_f": ProjectedRegimeValue(
                    regime="single_f", projection={"wealth": half_of_wealth}
                )
            },
        )
    },
)

# The gate lives on the source's edges, beside the law, keyed by TARGET
Model(
    edges={
        "couple": Transition(
            # targets derived: the law's keys plus each route's fallback regime
            law={"couple": StochasticTransition(func=stays_married)},
            gates={
                "couple": Gate(
                    predicate=no_dissolution,  # Boolean, on the target's grid
                    routes={
                        "f": StakeholderRoute(target_stakeholder="f", fallback=alone_f)
                    },
                    references={
                        "V_alone_f": ProjectedRegimeValue(
                            regime="single_f", projection={"wealth": half_of_wealth}
                        )
                    },
                    off_grid="pointwise",  # or "reject"
                )
            },
        ),
    },
    # Additional configuration may follow.
)
```

- **The transition key is always the GATE-OPEN target.** A dissolution edge is keyed by
  the *continuing* collective regime under `predicate = ~D_target`. Keying it by the
  singleton would send both partners there whenever the couple stays together.
- `routes` is keyed by **source** stakeholder; a singleton source declares exactly one
  route. Each route owns four destinations: the open regime (the dict key) and role
  (`target_stakeholder`), and the closed regime and role (`fallback.regime`,
  `fallback.stakeholder`).
- One `Gate` per target, valid at every age the target is reached and in both phases;
  never inside `ByAge` / `Phased` (only a route `fallback` may be `Phased`). With
  `Phased` edges, a target reached in both phases carries the equal `Gate` in both or
  none. A gate on a target the `Transition` never reaches is refused.
- Predicate operands: `V_target` (singleton target) / `V_target_<s>` (collective
  target), and `D_target`, which is **collective-only** — reading it on a singleton
  target is refused at model build. The Boolean-dtype requirement is checked **at
  evaluation**, i.e. on the first `solve()`, not at build.
- `ParetoObjective(weights=...)` scalarizes the household; a weight may read the
  regime's states and `period`/`age`, never an action. Its other arguments become free
  parameters under the `pareto_objective` key. Omit it for equal weights.
- A constraint-local projection may introduce **no** free parameter (refused when the
  `Regime` is constructed); an edge projection's free arguments become that edge's
  params, under `params["edges"][source][target]`.
- Stakeholder identity is **per row**: seed `initial_conditions["own_stakeholder"]`
  whenever the starting regime's forward closure contains a collective regime declaring
  a transition with more than one route. A row keeps its role across an ordinary regime
  transition, so a two-leg route it runs into later demands the seed, while one in a
  regime the cohort can never reach demands nothing. It is published as an
  `own_stakeholder` column, missing for singleton rows.
- `GatedEdge`, `stakeholders`, `value_constraints`, `same_period_refs` and `gated_edges`
  are the **lowered** form these declarations decompose into. Write the declarations.

See `docs/user_guide/collective_regimes.md` and `docs/reference/collective_regimes.md`.

### Case-piece solver (NB-EGM)

`NBEGM` (from `lcm.solvers`) is the endogenous-grid solver for a 1-D consumption-saving
regime whose budget is split by a binary case boundary on the liquid state (e.g. a
Medicaid asset test). The model author exposes the split with metadata-only decorators
(`lcm.case_boundary`, `lcm.piece`); the solver runs EGM per case,
NaN-dead masks each case to the region where its predicate is consistent with the
recovered state, and merges the cases on the liquid grid with the branch-aware upper
envelope.

```python
import jax.numpy as jnp

import lcm
from lcm.typing import FloatND

# Medicaid asset test: eligible while liquid wealth is below the limit.
medicaid_eligible = lcm.case_boundary(
    condition=lcm.ref("liquid") < lcm.ref("medicaid_asset_limit"),
    kind="jump",
)


@lcm.piece(output="subsidy", when=medicaid_eligible)
def subsidy_medicaid(subsidy_high: float) -> FloatND:
    """Subsidy into market resources for the Medicaid-eligible (low-asset) case."""
    return jnp.asarray(subsidy_high)


@lcm.piece(output="subsidy", otherwise=medicaid_eligible)
def subsidy_private(subsidy_low: float) -> FloatND:
    """Subsidy into market resources for the private (high-asset) case."""
    return jnp.asarray(subsidy_low)


# The kernels form cash-on-hand themselves, so the regime declares pylcm's own
# node rather than a local spelling of the same arithmetic.
resources = lcm.cash_on_hand_with_subsidy
```

- `lcm.case_boundary(*, condition, kind)` declares one executable, inspectable split.
  `condition` is exactly one `<`, `<=`, `>` or `>=` comparison built from `lcm.ref`;
  conjunctions, unions, equality tests and opaque callables are rejected, because they
  do not identify one ordered split with unambiguous ownership. Ownership of the exact
  boundary point falls out of the operator — `<` leaves equality to the `otherwise`
  side, `<=` gives it to `when` — and is not a separate argument. `kind` is
  `"continuous_kink"`, `"jump"`, or `"hard_constraint"`.
- `lcm.piece(output=…, when=…|otherwise=…)` marks the smooth formula for one side of an
  output. The decorator only attaches metadata and returns the function unchanged, so
  the model still solves identically under `GridSearch`.
- The case-piece route is scoped narrowly, and everything outside it is refused at model
  build. A case-piece regime must split exactly one output, named `subsidy`, on a
  boundary that is `equality="otherwise"`, `kind="jump"`, and declared on the liquid
  state; each piece reads only flat params, never a state or action; and the regime
  declares no discrete action and no taste shocks. Kinks, floors, and every other
  bracket shape go through `lcm.piecewise_affine` instead — see
  `docs/methods/nonconvex_budgets.md` for the choice between the two declaration forms
  and `docs/reference/piecewise_affine.md` for the decorator contract.
- The case-piece kernels form cash-on-hand themselves rather than calling the regime's
  budget node, so the route accepts pylcm's own declaration of that form by identity:
  `lcm.cash_on_hand_with_subsidy`. A budget they cannot form goes through a
  `lcm.piecewise_affine` schedule with a `post_decision_function`, or `GridSearch`.
- The liquid law is read, not assumed: NBEGM takes the landing points the declared law
  reaches on the savings grid and their derivative, so any term the modeller writes is
  solved. The law states them as a function of a post-decision savings node — named
  `savings` by default, otherwise named to `NBEGM(post_decision_function=...)` — and a
  law in displacement form (`next_liquid(resources, consumption, ...)`) is refused at
  build. `lcm.liquid_law_from_savings` is the conventional form and stays an ordinary
  executable function, so the same model solves identically under `GridSearch`.
- The solver's `validate` runs an AST + JAXPR smoothness gate over the user economic
  nodes reachable in each case (rejecting hidden `if`/`where`/`searchsorted` branching);
  mark a reviewed numerical `clip`/`max`/`abs` helper with `@lcm.smooth_helper` to
  exempt it.

### SimulationResult

`simulate()` returns a `SimulationResult` object:

```python
result = model.simulate(
    params=params,
    initial_conditions=initial_conditions,
    log_level="debug",
)

# Convert to DataFrame (deferred computation)
df = result.to_dataframe()

# With additional computed targets (utility, functions, constraints)
df = result.to_dataframe(additional_targets=["utility", "consumption"])

# All available targets
df = result.to_dataframe(additional_targets="all")

# Integer codes instead of categorical labels
df = result.to_dataframe(use_labels=False)

# Keep the frozen post-entry terminal-regime rows (absorbing representation);
# the default (terminal_rows="first") emits each subject's terminal entry row only
df = result.to_dataframe(terminal_rows="all")

# Access metadata
result.regime_names  # list[str]
result.state_names  # list[str]
result.action_names  # list[str]
result.n_periods  # int
result.n_subjects  # int
result.available_targets  # list[str] - computable additional targets

# Access raw data for advanced users
result.raw_results  # dict[RegimeName, dict[int, PeriodRegimeSimulationData]]
result.flat_params  # FlatParams
result.period_to_regime_to_V_arr  # dict[int, dict[RegimeName, FloatND]]

# Persistence: writes `arrays/` (orbax), `V_arr/` (the solved values),
# `metadata.pkl` (cloudpickle), and `simulated_data.arrow` (feather of
# `to_dataframe`).
# `directory` is a pathlib.Path, not a str.
from pathlib import Path

result.save(directory=Path("path/to/dir"))
loaded = SimulationResult.load(directory=Path("path/to/dir"))
```

### SolutionResult

`model.solve()` returns an `lcm.solver_api.SolutionResult` that keeps values, artifacts,
metadata, and explicit omission reasons separate. Its default retention keeps replay
artifacts, while persistence-oriented retention keeps exactly those artifacts whose
declared policy permits serialization. `model.simulate(solution=...)` checks the durable
model fingerprint, solution-relevant canonical parameters, compatibility versions, and
all required artifact descriptors before consuming the result. Descriptors are owned by
the canonical model and include exact container structure, leaf dtypes and shapes,
ordered axes, state and action roles, categorical domains, and consumer routing.
`SolutionResult.save()` and `lcm.persistence.load_solution()` use the versioned,
non-executable solution archive format; non-persisted artifacts remain explicit
omissions. Solver plugins can be written against `lcm.solvers`, `lcm.solver_api`,
`lcm.typing`, and `lcm.grids`, and can validate their boundary implementation with the
published out-of-tree conformance suite. The solver API and archive format remain
explicitly versioned experimental interfaces rather than stable compatibility promises.

### Initial Conditions Format

Initial conditions use a flat dictionary with state names plus `"regime_id"`:

```python
initial_conditions = {
    "wealth": jnp.array([1.0, 2.0, 3.0]),
    "health": jnp.array([0.5, 0.8, 0.3]),
    "regime_id": jnp.array([RegimeId.working, RegimeId.working, RegimeId.retired]),
}
```

### Key Attributes

- `model.get_params_template()` - Mutable copy of the parameter template (dict by regime
  name, plus an `"edges"` root for edge-declared callables)
- `model.user_regimes` - Immutable mapping of regime names to plain `Regime` objects:
  the regimes as the model runs them, finalized at model build (model-level slots
  merged, Koopmans aggregator injected, completeness validated), still in user
  vocabulary
- `model._regimes` - Immutable mapping of regime names to canonical `Regime` objects
  (`_lcm.engine.Regime`) produced by `process_regimes`. Private — the canonical form is
  engine-internal; user code should read `user_regimes`.
- `model.pruned_variables` - Immutable mapping of regime names to the broadcast
  states/actions pruned from that regime by DAG reachability
- `model.ages` - The AgeGrid defining the lifecycle, or `None` in period mode
- `model.n_periods` - Number of computational slots, including explicit terminal slots
- `model.regime_names_to_ids` - Immutable mapping from regime names to integer indices

Period models use `InitialNode(period=..., regime=...)`, `PeriodRange` and `Periods`
in graph declarations, and `period` in simulation inputs. `ByPeriod` and
`PeriodSpecializedFunction` / `PeriodSpecializedGrid` preserve their age counterparts'
contracts while taking integer periods. Never infer an age from an integer period.
See [period time](../docs/user_guide/period_time.md) for temporal parameters and the
mandatory distinction between computational stages and economic time.
