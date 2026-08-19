"""The user-facing `Regime` definition.

The validators and the identity transition live behind a leading underscore in
`_lcm.user_regime_validation` and `_lcm.regime_building.transitions`. This
module is intentionally thin: the public class definition. A non-terminal
regime that declares no `koopmans_aggregator` takes the model-level one at
model build.

"""

import dataclasses
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, cast

from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.grids import ContinuousGrid, DiscreteGrid, Grid
from _lcm.regime_building.phases import normalize_regime_phases
from _lcm.regime_building.transitions import collect_state_transitions
from _lcm.solution.contract import _BoundLiquidMargin, _BoundOuterContinuousMargin
from _lcm.typing import ActionName, ActiveFunction, FunctionName, RegimeName, StateName
from _lcm.user_regime_validation import (
    _validate_collective_regime,
    _validate_fold_declarations,
    _validate_gated_edges,
    _validate_logical_consistency,
    _validate_mapping_contents,
)
from _lcm.utils.containers import (
    ensure_containers_are_immutable,
)
from lcm.certainty_equivalent import CertaintyEquivalent
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.solvers import (
    GridSearch,
    OneMarginSolver,
    Solver,
    TwoMarginSolver,
)
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.transition import AgeSpecializedGrid, MarkovTransition
from lcm.typing import UserFunction


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class SamePeriodRef:
    """Declaration of a same-period cross-regime reference value.

    A collective regime's `same_period_refs` maps a *reference-value name* (the
    named argument under which the interpolated value enters the regime's
    `value_constraints` predicates) to one of these declarations: WHICH other
    regime's same-period value function is read, HOW the reading regime's state
    cell maps into the reference regime's state coordinates, and — when the
    reference regime is itself collective — WHOSE stakeholder value is read.

    The reference regime is solved earlier in the same period (the solver
    orders each period's active regimes topologically by these declarations),
    and its value function is linearly interpolated at the projected
    coordinates with the same machinery the continuation uses — but with the
    CURRENT period's arrays. Reading the current period rather than the
    continuation is what a within-period participation constraint needs: a
    couple's period-$t$ decision is checked against the values its members
    would have as singles in that same period $t$.
    """

    regime: RegimeName
    """Name of the reference regime whose same-period V is read.

    Must be another regime of the model, active in every period the declaring
    regime is active. No transition edge between the two regimes is required —
    same-period reference reads work across otherwise unconnected regime
    "islands", which is what lets a value constraint compare this regime's
    value against a regime it never transitions into.
    """

    projection: Mapping[StateName, UserFunction]
    """How the declaring regime's state cell maps to the reference coordinates.

    One entry per state of the *reference* regime: `state name -> function`
    returning that coordinate. Each function resolves through the declaring
    regime's DAG, so it may read the declaring regime's states, actions, and
    functions (plus `period` / `age`); it may not introduce new free
    parameters. The reference V is interpolated at the resulting coordinates
    (linear on continuous axes, lookup on discrete axes).
    """

    stakeholder: str | None = None
    """Which stakeholder's value to read from a collective reference regime.

    Required when the reference regime is collective (its V carries a
    stakeholder axis); must be `None` when the reference regime is a singleton
    (its V has no stakeholder axis).
    """

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "projection",
            ensure_containers_are_immutable(self.projection),
        )


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class EdgeLeg:
    """One source-stakeholder leg of a gated edge.

    A gated edge carries one leg per SOURCE stakeholder (a singleton source
    declares exactly one leg; a collective source one per stakeholder). Each leg
    says, for that source stakeholder's continuation object `Wbar^s` on the
    target regime's grid:

    - `target_stakeholder` — which component of the target regime's value the
      OPEN (gate-True) branch takes. For a collective target it names one of the
      target's stakeholders; for a singleton target it is `None`.
    - `fallback` — a `SamePeriodRef` giving the value the CLOSED (gate-False)
      branch takes: a same-period reference regime's V at a projection from the
      TARGET regime's grid coordinates — typically the source stakeholder's own
      single regime, at the projection back to its single state. The mixture is the
      strict `jnp.where(gate, V_target, V_fallback)` — NEVER a linear
      `gate*V_target + (1-gate)*V_fallback` (the target value is `-inf` in a
      dissolution cell, and `0 * -inf = NaN`).
    """

    fallback: SamePeriodRef
    """The gate-closed branch: a reference regime's same-period V at a projection."""

    target_stakeholder: str | None = None
    """The gate-open branch's target-value component, or `None` for a singleton
    target."""


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class GatedEdge:
    """A gated edge routing a source regime's continuation into a target.

    The construct that unlocks MIXED singleton/collective
    regime topologies: a singleton regime may reach a collective regime (mutual
    consent marriage) and a collective regime may route per-stakeholder to
    singleton regimes (dissolution) — but only THROUGH a declared gated edge. Direct
    raw transitions between different-stakeholder regimes stay rejected.

    A source regime declares `gated_edges` as a mapping of TARGET regime name
    to `GatedEdge`. At the end of each period's solve, the engine folds, for
    each declared edge and each source stakeholder `s`, a gated continuation
    object on the target regime's grid:

        Wbar^s(x) = jnp.where(gate(x), V_target^{leg_s}(x), V_fallback^s(pi_s(x)))

    The source's continuation then reads `Wbar` in place of the raw target V,
    threaded through the ordinary transition machinery.

    - `gate` — a BOOLEAN user function evaluated pointwise on the target
      regime's grid. It may read the target regime's per-stakeholder value
      components under the names `V_target_<s>` (one per target stakeholder),
      the target's dissolution flag `D_target` (a collective target only), each
      key of `gate_refs` (a same-period reference value at its projection), and
      ordinary target states / params. Mutual consent to marriage is the strict,
      unanimous gate
      `gate = (V_target_f > V_single_f_ref) & (V_target_m > V_single_m_ref)`;
      "no dissolution this period" is `gate = ~D_target`. Stochastic gates —
      a gate returning a probability in `(0, 1)` rather than a boolean — are
      not supported.
    - `legs` — one `EdgeLeg` per SOURCE stakeholder (keyed by source
      stakeholder name; a singleton source declares exactly one leg under any
      key).
    - `gate_refs` — extra same-period reference values the `gate` reads,
      exactly like a regime's `same_period_refs` but projected from the TARGET
      regime's grid.
    """

    gate: UserFunction
    """Boolean gate function on the target grid (see the class docstring)."""

    legs: Mapping[str, EdgeLeg]
    """One `EdgeLeg` per source stakeholder (single leg for a singleton source)."""

    gate_refs: Mapping[str, SamePeriodRef] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Same-period reference values the `gate` reads (projected from the target
    grid)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "legs", ensure_containers_are_immutable(self.legs))
        object.__setattr__(
            self, "gate_refs", ensure_containers_are_immutable(self.gate_refs)
        )


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Regime:
    """User-facing regime definition.

    `Model` processes instances of this class into the canonical regime form
    (`_lcm.engine.Regime`) used internally by the solver and simulator.

    State transitions are specified via `state_transitions`, mapping state names to
    transition functions. A bare callable is deterministic; wrap in `MarkovTransition`
    for stochastic transitions. `fixed_transition(state_name)` marks a fixed state
    (identity law). Stochastic processes have intrinsic transitions and must not
    appear in `state_transitions`.

    The `transition` field on the regime itself is the *regime* transition function.
    A regime with `transition=None` is terminal — no separate `terminal` flag is
    needed.

    """

    # `UserFunction`/`Phased` inside the per-target dict pass the type check
    # so the validator can reject them with an explanation.
    transition: (
        UserFunction
        | MarkovTransition
        | Phased
        | Mapping[RegimeName, MarkovTransition | UserFunction | Phased]
        | None
    )
    """Regime transition, or `None` for terminal regimes.

    Three forms:

    - bare callable ⇒ deterministic, returns the target regime id
    - `MarkovTransition` ⇒ stochastic, returns a probability vector over all
      regimes
    - per-target dict ⇒ stochastic, maps target regime names to
      `MarkovTransition`-wrapped functions returning that target's
      probability. The key set declares the regime's reachable targets;
      omitted regimes are structurally unreachable.

    A bare callable or bare `MarkovTransition` declares conservative support
    over every regime active in the next period — every temporally
    compatible candidate must therefore have a valid state handoff (a
    carried state, a deterministic/stochastic law, or an explicit
    target-local/entry law). Use a per-target mapping to declare narrower
    support instead. Runtime-zero transition probabilities do not narrow
    this topology; only the declared form does.

    `Phased` gives each phase its own variant (matching form required; for
    per-target dicts, identical key sets).
    """

    active: ActiveFunction = lambda _age: True
    """Callable that takes age (float) and returns True if regime is active."""

    # `None` masks a model-level entry of the same name.
    states: Mapping[StateName, Grid | Phased | AgeSpecializedGrid | None] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of state variable names to grids or phase-variant declarations.

    A plain `Grid` value is a state shared by both phases.
    `Phased(solve=callable, simulate=Grid)` declares a carried state: a
    derived function (no grid axis) in the solve phase and a seeded, evolved
    state in the simulate phase, whose law of motion is its regular
    `state_transitions` entry.
    An `AgeSpecializedGrid` value is a continuous state whose grid bounds vary
    with age (fixed `n_points`); it is resolved to a concrete grid per period at
    model build.
    """

    state_transitions: Mapping[
        StateName,
        UserFunction
        | MarkovTransition
        | Phased
        # `Phased` inside a per-target dict passes the type check so the
        # validator can reject it with the outermost-only explanation.
        | Mapping[RegimeName, UserFunction | MarkovTransition | Phased]
        | None,
    ] = field(default_factory=lambda: MappingProxyType({}))
    """Mapping of state names to transition functions or per-target dicts.

    Every non-process state must have an entry — omitting a state raises an error.
    `fixed_transition(state_name)` marks a fixed state (identity law). Wrap in
    `MarkovTransition` for stochastic transitions. Per-target dicts map target
    regime names to transition functions — every reachable target must be listed.
    `Phased` gives each phase its own law of motion; it wraps the whole entry
    (outermost only, never inside a per-target dict).
    """

    actions: Mapping[ActionName, Grid | None] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of action variable names to grid objects."""

    functions: Mapping[FunctionName, UserFunction | Phased | None] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of function names to callables; must include 'utility'.

    `Phased` gives each phase its own implementation.
    """

    # `Phased` passes the type check so the validator can reject it with an
    # explanation (constraints are phase-invariant).
    constraints: Mapping[FunctionName, UserFunction | Phased | None] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of constraint names to constraint functions.

    Constraints are phase-invariant: a phase-specific feasible set would let
    the simulated argmax range over actions the value function was never
    computed for, so `Phased` is rejected here.
    """

    derived_categoricals: Mapping[FunctionName, DiscreteGrid] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Categorical grids for DAG function outputs not in states/actions."""

    solver: Solver = field(default_factory=GridSearch)
    """Solution algorithm for this regime during backward induction.

    - `GridSearch()` (default): grid search over the full state-action product.
    - `EGM(...)`: envelope-free one-asset endogenous grid method.
    - `DCEGM(...)`: discrete-continuous endogenous grid method.
    - `NEGM(...)`: an outer continuous search nesting an inner `DCEGM`.

    The endogenous-grid solvers validate their structural contracts during
    `Model(...)`. `ConsumptionSavingsRegime` may own their shared state, action,
    resources, and post-decision role names so those names need not be repeated
    on each solver configuration.
    """

    taste_shocks: ExtremeValueTasteShocks | None = None
    """EV1 taste shocks on the regime's discrete-action combinations.

    When set, the shock scale becomes the runtime param
    `{"taste_shocks": {"scale": ...}}` and the solve aggregates discrete
    actions via the smoothed expected maximum instead of the hard maximum.
    Requires at least one discrete action.
    """

    koopmans_aggregator: UserFunction | Phased | None = None
    """Combines current-period utility with the certainty equivalent into `Q`.

    Signature `W(utility, CE, ...)`; further arguments are runtime params
    under the pseudo-function name `koopmans_aggregator`, or outputs of
    regime functions of the same name. `Phased(solve=..., simulate=...)`
    gives the two phases different aggregators (a naive/sophisticated
    beta-delta split, say). `None` means the regime takes the model-level
    aggregator (`lcm.LinearAggregator` unless the `Model` says otherwise). Terminal
    regimes have no continuation and take none.
    """

    certainty_equivalent: CertaintyEquivalent | None = None
    """Nonlinear certainty equivalent over the next-period value distribution.

    When set, the solve aggregates the continuation as
    `g⁻¹(Σ_r p_r · E_w[g(V')])` instead of the linear expectation, and the
    transform parameters become runtime params under the pseudo-function
    name `certainty_equivalent`. Only non-terminal regimes solved by
    `GridSearch`, `NBEGM`, or `NNBEGM` support it, and it cannot be combined with
    `taste_shocks`.
    """

    description: str = ""
    """Description of the regime."""

    stakeholders: tuple[str, ...] | None = None
    """Names of the stakeholders whose individual values this regime carries.

    `None` (the default) is the singleton case: the regime has one implicit
    stakeholder and one value function. A non-`None` tuple declares a
    *collective regime*: a couple (or other multi-party household) that solves
    one household argmax but reads off a per-stakeholder value at that common
    argmax, with value-aware feasibility and value-gated regime routing
    (consent / dissolution).

    A collective regime carries a per-stakeholder utility
    `functions["utility_<s>"]` for each stakeholder `<s>` and household Pareto
    `weights`. Its solve reads off each stakeholder's own value at the shared
    household argmax, and a non-terminal one aggregates the per-stakeholder
    continuation `Q^s = H(u^s, E[V'^s])`. A non-terminal collective regime's
    transition targets must all be collective regimes with the identical
    `stakeholders` tuple — per-stakeholder routing to different regimes goes
    through `gated_edges`. EV1 taste shocks, nonlinear certainty equivalents,
    and non-GridSearch solvers on a collective regime raise
    `NotImplementedError`.

    A shock declared `fold=True` is refused when the model is built, naming the
    regime and the state. A collective regime writes `-inf` where no action
    satisfies every stakeholder's participation constraint — a sentinel a gated
    edge resolves to the outside option, not a value on the household's own
    scale — and quadrature over that sentinel is not an expectation: a
    household dissolving at one node would be stored as dissolving at all of
    them. The same shock folds normally in a singleton regime.

    Three things to know before simulating one:

    - The population is one fixed-size cohort of independent rows. A
      dissolution does not split a row into two independently tracked
      households; each row records where every stakeholder would land and then
      continues as one of them.
    - `simulate` requires `own_stakeholder` for a collective source, and it
      names the role every row in that call carries.
    - The off-grid value gate is approximate: it interpolates the target's
      already-maximized value rather than recomputing the household maximum at
      the realized off-grid point (see `get_edge_simulate_gate_evaluator`).
    """

    weights: Mapping[str, float] | None = None
    """Household Pareto weights `λ_s` per stakeholder for a collective regime.

    Used only when `stakeholders is not None`: the collective solve maximizes the
    household scalarization `O = Σ_s λ_s Q^s` over the feasible action set. When
    omitted (the default), equal weights `1/len(stakeholders)` are used — the
    symmetric-couple case, `λ = 0.5` on each partner. Supply an explicit mapping
    to give one stakeholder more weight in the household decision. Ignored —
    and must be `None` — for a singleton regime.
    """

    value_constraints: Mapping[FunctionName, UserFunction] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Value-aware feasibility predicates for a non-terminal collective regime.

    Each entry maps a constraint name to a predicate returning `True` where the
    (state, action) combination is feasible. Unlike ordinary `constraints`
    (which are evaluated before and independently of `Q`), a value constraint
    is evaluated AFTER the per-stakeholder action values and may read, as named
    arguments:

    - `Q_<s>` for each stakeholder `<s>` — that stakeholder's own action value
      `Q^s(x, a)` (felicity plus discounted continuation) at the cell;
    - each key of `same_period_refs` — the reference regime's same-period value
      interpolated at the projected state (e.g. the dissolved single's value);
    - ordinary states, actions, regime functions, and parameters via the DAG
      (a predicate's own parameter surfaces in the params template under the
      constraint's name).

    The final action mask is the AND of ordinary constraints and all value
    constraints; the household argmax runs over the masked set, and a state
    cell whose mask is empty publishes the dissolution flag `D = True` (returned by
    the solve alongside V — never conflated with a numeric `-inf` value, which
    can occur on-path). A participation constraint takes the form
    `Q_j >= V_single_j(pi_j(x)) - Delta_j` for each stakeholder `j`.

    Only non-terminal collective regimes may declare value constraints.
    """

    gated_edges: Mapping[RegimeName, GatedEdge] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Gated edges routing this regime's continuation into a target regime.

    Maps a TARGET regime name to a `GatedEdge`. A gated edge lets this regime
    reach a target of a DIFFERENT stakeholder layout (a singleton regime into a
    collective one for mutual-consent marriage, or a collective regime into
    singleton regimes for dissolution) — the only way to cross the mixed-topology
    fence. When declared, the engine folds a gated continuation object
    `Wbar^s = jnp.where(gate, V_target, V_fallback)` on the target regime's
    grid at each period's end, and this regime's continuation reads `Wbar` in
    place of the raw target V. See `GatedEdge`. Only meaningful together
    with the corresponding `transition` / `state_transitions` into the target's
    state space; a target reached by a gated edge is exempt from the mixed-
    stakeholder rejection.
    """

    same_period_refs: Mapping[str, SamePeriodRef] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Same-period cross-regime reference values read by `value_constraints`.

    Maps each reference-value name (the argument name under which the
    interpolated value enters the predicates) to a `SamePeriodRef` declaring
    the reference regime, the state projection, and — for a collective
    reference — the stakeholder. Reference regimes are solved earlier within
    the same period (topological order; cycles are rejected at model build).
    Only collective regimes that also declare `value_constraints` may declare
    references.
    """

    @property
    def terminal(self) -> bool:
        """Whether this is a terminal regime (derived from transition being None)."""
        return self.transition is None

    @property
    def stochastic_regime_transition(self) -> bool:
        """Whether the regime transition is stochastic.

        A `MarkovTransition` and a per-target dict are both stochastic.
        `Phased` variants must have matching forms, so the solve variant is
        representative.
        """
        transition = (
            self.transition.solve
            if isinstance(self.transition, Phased)
            else self.transition
        )
        return isinstance(transition, MarkovTransition | Mapping)

    def __post_init__(self) -> None:
        self._fail_if_egm_solver_has_no_margin_declaration()
        # A collective regime's own declaration is validated here (the
        # `stakeholders` tuple, `weights`, the value-constraint grammar;
        # out-of-scope features — taste shocks, nonlinear certainty
        # equivalents, non-GridSearch solvers — are rejected). What a
        # model-level slot may still supply — the per-stakeholder
        # `utility_<s>` functions and at least one discrete action — is
        # checked when the model finalizes its regimes. The default `None`
        # (singleton) path never enters this branch.
        if self.gated_edges:
            _validate_gated_edges(self)
        if self.stakeholders is not None:
            _validate_collective_regime(self)
        elif self.weights is not None:
            raise RegimeInitializationError(
                "`weights` is a household Pareto-weight declaration for a "
                "collective regime; it is only meaningful together with "
                "`stakeholders`. Omit it for a singleton regime."
            )
        elif self.value_constraints:
            raise RegimeInitializationError(
                "`value_constraints` are value-aware feasibility predicates for "
                "a collective regime; they read the per-stakeholder action "
                "values `Q_<s>`, which only exist when `stakeholders` is set. "
                "Use ordinary `constraints` for a singleton regime."
            )
        elif self.same_period_refs:
            raise RegimeInitializationError(
                "`same_period_refs` declares same-period reference values for a "
                "collective regime's `value_constraints`; it is only "
                "meaningful together with `stakeholders`. Omit it for a "
                "singleton regime."
            )

        _validate_mapping_contents(self)
        _validate_logical_consistency(self)
        _validate_fold_declarations(self)

        def make_immutable(name: str) -> None:
            value = ensure_containers_are_immutable(getattr(self, name))
            object.__setattr__(self, name, value)

        # Completeness (a `utility` entry, aggregator injection, transition
        # coverage) is validated when the model finalizes its regimes
        # — model-level slots may still satisfy it after merging.
        make_immutable("functions")
        make_immutable("states")
        make_immutable("state_transitions")
        make_immutable("actions")
        make_immutable("constraints")
        make_immutable("derived_categoricals")
        make_immutable("value_constraints")
        make_immutable("gated_edges")
        make_immutable("same_period_refs")
        # `weights` is optional; a singleton regime declares none at all.
        if self.weights is not None:
            make_immutable("weights")

        # The phase grammar (states matrix, carried laws, regime-transition
        # variants) is validated by the normalizer; the per-phase spec it
        # builds is consumed during model processing.
        normalize_regime_phases(self)

    def _fail_if_egm_solver_has_no_margin_declaration(self) -> None:
        if isinstance(self, _EGMFamilyRegime):
            return
        if isinstance(self.solver, OneMarginSolver | TwoMarginSolver):
            raise RegimeInitializationError(
                "EGM-family solvers require regime-owned margin declarations: use "
                "ConsumptionSavingsRegime for a OneMarginSolver or "
                "NestedConsumptionSavingsRegime for a TwoMarginSolver."
            )

    def _validate_finalized_structure(self, *, regime_name: RegimeName) -> None:
        """Validate subclass-owned structure after model-level slots are merged."""
        _ = regime_name

    def get_koopmans_aggregator(
        self,
        phase: Literal["solve", "simulate"] = "solve",
    ) -> UserFunction | None:
        """Get the Bellman aggregator this phase runs.

        Args:
            phase: Which variant to use when the declaration is `Phased`.

        Returns:
            The aggregator, or `None` when the regime declares none (a
            terminal regime, or one taking the model-level value).

        """
        if isinstance(self.koopmans_aggregator, Phased):
            variant = (
                self.koopmans_aggregator.solve
                if phase == "solve"
                else self.koopmans_aggregator.simulate
            )
            return cast("UserFunction", variant)
        return self.koopmans_aggregator

    def get_all_functions(
        self,
        phase: Literal["solve", "simulate"] = "solve",
    ) -> MappingProxyType[str, UserFunction]:
        """Get all regime functions including utility, constraints, and transitions.

        Collect functions from four sources:
        - `self.functions` (utility and helpers)
        - `self.constraints`
        - State transitions from `self.state_transitions`
        - The regime transition (`self.transition`, keyed as `"next_regime"`)

        For `Phased` entries, the variant matching `phase` is used. A
        carried-state declaration in `states` (`Phased(solve=...,
        simulate=Grid)`) contributes its `solve` variant as a derived
        function under the state's name and its law of motion under
        `next_<name>`, mirroring how ordinary state transitions are keyed.

        Args:
            phase: Which variant to use for phase-variant entries.

        Returns:
            Read-only mapping of all regime functions.

        """

        def resolve(value: object) -> UserFunction:
            if isinstance(value, Phased):
                value = value.solve if phase == "solve" else value.simulate
            return cast("UserFunction", value)

        result: dict[str, UserFunction] = {
            name: resolve(func) for name, func in self.functions.items()
        }
        for name, spec in self.states.items():
            if isinstance(spec, Phased):
                # Carried state: the solve variant is its derived-function
                # imputation; the law of motion is its regular
                # `state_transitions` entry, collected below.
                result[name] = cast("UserFunction", spec.solve)
        result |= cast("Mapping[str, UserFunction]", self.constraints)
        if self.transition is not None:
            collected = collect_state_transitions(self.states, self.state_transitions)
            result |= {name: resolve(func) for name, func in collected.items()}
            transition = self.transition
            if isinstance(transition, Phased):
                transition = (
                    transition.solve if phase == "solve" else transition.simulate
                )
            if isinstance(transition, Mapping):
                # Per-target regime transition: one entry per declared target,
                # mirroring how per-target state laws are keyed.
                for target_regime_name, cell in transition.items():
                    result[f"next_regime__{target_regime_name}"] = cast(
                        "UserFunction", cell
                    )
            else:
                result["next_regime"] = cast("UserFunction", transition)
        return MappingProxyType(result)

    def replace(self, **kwargs: Any) -> Regime:  # noqa: ANN401
        """Replace the attributes of the regime.

        Args:
            **kwargs: Keyword arguments to replace the attributes of the regime.

        Returns:
            A new regime with the replaced attributes.

        """
        try:
            return dataclasses.replace(self, **kwargs)
        except TypeError as e:
            raise RegimeInitializationError(
                f"Failed to replace attributes of the regime. The error was: {e}"
            ) from e


outer_unchanged: FunctionName = "__outer_unchanged__"
# Sentinel declaring that an outer state is unchanged without adjustment.
# Use it as ``OuterContinuousMargin.no_adjustment`` when the no-adjustment map is
# literally the identity. Any other value is a function name and must resolve in
# the assembled regime DAG. A sentinel, rather than a generated callable, keeps
# the public declaration serialisable and avoids callable-wrapper behaviour under
# the project's beartype claw.


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class NetOfAdjustmentCost:
    """A resources node composed by pylcm as ``before_cost - cost``."""

    name_in_dag: FunctionName
    """Name assigned to the composed post-cost resources node."""

    before_cost: FunctionName
    """Name of the user-declared cost-free resources node."""

    cost: FunctionName
    """Name of the user-declared adjustment-cost node."""

    def __post_init__(self) -> None:
        self._fail_if_names_are_not_pairwise_distinct()

    def _fail_if_names_are_not_pairwise_distinct(self) -> None:
        duplicates = _duplicate_names((self.name_in_dag, self.before_cost, self.cost))
        if duplicates:
            raise RegimeInitializationError(
                "NetOfAdjustmentCost names must be pairwise distinct; repeated "
                f"names: {duplicates}."
            )


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class LiquidMargin:
    """Names defining the liquid Euler margin of an EGM-family regime."""

    state: StateName
    """The liquid continuous state."""

    action: ActionName
    """The continuous action paid from resources."""

    resources: FunctionName | NetOfAdjustmentCost
    """The resources node, bare or composed net of an adjustment cost."""

    post_decision_state: FunctionName
    """The post-decision liquid state, conventionally savings."""

    def __post_init__(self) -> None:
        self._fail_if_names_are_not_pairwise_distinct()

    @property
    def resources_name(self) -> FunctionName:
        """Return the DAG name every downstream resources reader consumes."""
        if isinstance(self.resources, NetOfAdjustmentCost):
            return self.resources.name_in_dag
        return self.resources

    def _fail_if_names_are_not_pairwise_distinct(self) -> None:
        names = [self.state, self.action, self.resources_name, self.post_decision_state]
        if isinstance(self.resources, NetOfAdjustmentCost):
            names.extend((self.resources.before_cost, self.resources.cost))
        duplicates = _duplicate_names(names)
        if duplicates:
            raise RegimeInitializationError(
                "LiquidMargin names must be pairwise distinct; repeated names: "
                f"{duplicates}."
            )


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class OuterContinuousMargin:
    """Names defining the outer continuous margin of a nested EGM regime."""

    state: StateName
    """The second continuous state."""

    action: ActionName
    """The continuous action moving the outer state."""

    post_decision_state: FunctionName
    """This period's chosen post-decision level of the outer state."""

    no_adjustment: FunctionName
    """No-adjustment map, or :data:`lcm.outer_unchanged` for identity."""

    def __post_init__(self) -> None:
        self._fail_if_names_are_not_pairwise_distinct()

    def _fail_if_names_are_not_pairwise_distinct(self) -> None:
        names = [self.state, self.action, self.post_decision_state]
        if self.no_adjustment != outer_unchanged:
            names.append(self.no_adjustment)
        duplicates = _duplicate_names(names)
        if duplicates:
            raise RegimeInitializationError(
                "OuterContinuousMargin names must be pairwise distinct; repeated "
                f"names: {duplicates}."
            )


@dataclass(frozen=True, kw_only=True)
class _EGMFamilyRegime(Regime):
    """Shared declaration and validation for one- and two-margin EGM regimes."""

    liquid: LiquidMargin

    def __post_init__(self) -> None:
        self._fail_if_local_liquid_state_is_not_continuous()
        self._fail_if_local_liquid_action_is_not_continuous()
        self._fail_if_local_liquid_function_declarations_are_invalid()
        super().__post_init__()

    def _fail_if_local_liquid_state_is_not_continuous(self) -> None:
        if self.liquid.state not in self.states:
            return
        state = self.states[self.liquid.state]
        state = state.solve if isinstance(state, Phased) else state
        if not isinstance(state, ContinuousGrid | AgeSpecializedGrid):
            raise RegimeInitializationError(
                f"LiquidMargin.state {self.liquid.state!r} is declared locally but "
                "is not a continuous non-process solve-state grid."
            )

    def _fail_if_local_liquid_action_is_not_continuous(self) -> None:
        if self.liquid.action not in self.actions:
            return
        if not isinstance(self.actions[self.liquid.action], ContinuousGrid):
            raise RegimeInitializationError(
                f"LiquidMargin.action {self.liquid.action!r} is declared locally "
                "but is not a continuous action grid."
            )

    def _fail_if_local_liquid_function_declarations_are_invalid(self) -> None:
        resources = self.liquid.resources
        if isinstance(resources, NetOfAdjustmentCost):
            required = (resources.before_cost, resources.cost)
        else:
            required = (resources,)
        required += (self.liquid.post_decision_state,)
        missing_values = [
            name
            for name in required
            if name in self.functions and self.functions[name] is None
        ]
        if missing_values:
            raise RegimeInitializationError(
                "Liquid-margin function names explicitly masked by None: "
                f"{sorted(missing_values)}."
            )

    def _liquid_finalization_errors(self) -> list[str]:
        messages: list[str] = []
        state = self.states.get(self.liquid.state)
        state = state.solve if isinstance(state, Phased) else state
        if not isinstance(state, ContinuousGrid | AgeSpecializedGrid):
            messages.append(
                f"liquid.state {self.liquid.state!r} must name a continuous "
                "non-process solve-state grid"
            )
        if not isinstance(self.actions.get(self.liquid.action), ContinuousGrid):
            messages.append(
                f"liquid.action {self.liquid.action!r} must name a continuous "
                "action grid"
            )
        if self.functions.get(self.liquid.resources_name) is None:
            messages.append(
                f"liquid.resources {self.liquid.resources_name!r} must name the "
                "assembled resources function"
            )
        if self.functions.get(self.liquid.post_decision_state) is None:
            messages.append(
                f"liquid.post_decision_state {self.liquid.post_decision_state!r} "
                "must name an assembled regime function"
            )
        return messages

    def _validate_finalized_structure(self, *, regime_name: RegimeName) -> None:
        messages = self._liquid_finalization_errors()
        if messages:
            raise RegimeInitializationError(
                f"In EGM-family regime {regime_name!r}: {'; '.join(messages)}."
            )


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class ConsumptionSavingsRegime(_EGMFamilyRegime):
    """One-liquid-margin regime for EGM, DC-EGM, or grid search."""

    solver: OneMarginSolver | GridSearch = field(default_factory=GridSearch)

    def __post_init__(self) -> None:
        self._fail_if_solver_pairing_is_invalid()
        object.__setattr__(
            self,
            "solver",
            _bind_one_margin_solver(solver=self.solver, liquid=self.liquid),
        )
        super().__post_init__()

    def _fail_if_solver_pairing_is_invalid(self) -> None:
        if not isinstance(self.solver, OneMarginSolver | GridSearch):
            raise RegimeInitializationError(
                "ConsumptionSavingsRegime.solver must be a OneMarginSolver or "
                f"GridSearch, got {type(self.solver).__module__}."
                f"{type(self.solver).__qualname__}."
            )


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class NestedConsumptionSavingsRegime(_EGMFamilyRegime):
    """Two-margin sibling of :class:`ConsumptionSavingsRegime`."""

    outer_continuous: OuterContinuousMargin
    solver: TwoMarginSolver | GridSearch = field(default_factory=GridSearch)

    def __post_init__(self) -> None:
        self._fail_if_solver_pairing_is_invalid()
        self._fail_if_liquid_and_outer_names_collide()
        self._fail_if_local_outer_state_is_not_continuous()
        self._fail_if_local_outer_action_is_not_continuous()
        self._fail_if_local_outer_function_declarations_are_invalid()
        object.__setattr__(
            self,
            "solver",
            _bind_two_margin_solver(
                solver=self.solver,
                liquid=self.liquid,
                outer=self.outer_continuous,
            ),
        )
        super().__post_init__()

    def _fail_if_solver_pairing_is_invalid(self) -> None:
        if not isinstance(self.solver, TwoMarginSolver | GridSearch):
            raise RegimeInitializationError(
                "NestedConsumptionSavingsRegime.solver must be a TwoMarginSolver "
                f"or GridSearch, got {type(self.solver).__module__}."
                f"{type(self.solver).__qualname__}."
            )

    def _fail_if_liquid_and_outer_names_collide(self) -> None:
        liquid_names = {
            self.liquid.state,
            self.liquid.action,
            self.liquid.resources_name,
            self.liquid.post_decision_state,
        }
        if isinstance(self.liquid.resources, NetOfAdjustmentCost):
            liquid_names |= {
                self.liquid.resources.before_cost,
                self.liquid.resources.cost,
            }
        outer_names = {
            self.outer_continuous.state,
            self.outer_continuous.action,
            self.outer_continuous.post_decision_state,
        }
        if self.outer_continuous.no_adjustment != outer_unchanged:
            outer_names.add(self.outer_continuous.no_adjustment)
        collisions = sorted(liquid_names & outer_names)
        if collisions:
            raise RegimeInitializationError(
                "Liquid and outer margin names must not collide; repeated names: "
                f"{collisions}."
            )

    def _fail_if_local_outer_state_is_not_continuous(self) -> None:
        name = self.outer_continuous.state
        if name not in self.states:
            return
        state = self.states[name]
        state = state.solve if isinstance(state, Phased) else state
        if not isinstance(state, ContinuousGrid | AgeSpecializedGrid):
            raise RegimeInitializationError(
                f"OuterContinuousMargin.state {name!r} is declared locally but "
                "is not a continuous non-process solve-state grid."
            )

    def _fail_if_local_outer_action_is_not_continuous(self) -> None:
        name = self.outer_continuous.action
        if name not in self.actions:
            return
        if not isinstance(self.actions[name], ContinuousGrid):
            raise RegimeInitializationError(
                f"OuterContinuousMargin.action {name!r} is declared locally but "
                "is not a continuous action grid."
            )

    def _fail_if_local_outer_function_declarations_are_invalid(self) -> None:
        required = [self.outer_continuous.post_decision_state]
        if self.outer_continuous.no_adjustment != outer_unchanged:
            required.append(self.outer_continuous.no_adjustment)
        missing_values = [
            name
            for name in required
            if name in self.functions and self.functions[name] is None
        ]
        if missing_values:
            raise RegimeInitializationError(
                "Outer-margin function names explicitly masked by None: "
                f"{sorted(missing_values)}."
            )

    def _validate_finalized_structure(self, *, regime_name: RegimeName) -> None:
        messages = self._liquid_finalization_errors()
        outer = self.outer_continuous
        state = self.states.get(outer.state)
        state = state.solve if isinstance(state, Phased) else state
        if not isinstance(state, ContinuousGrid | AgeSpecializedGrid):
            messages.append(
                f"outer_continuous.state {outer.state!r} must name a continuous "
                "non-process solve-state grid"
            )
        if not isinstance(self.actions.get(outer.action), ContinuousGrid):
            messages.append(
                f"outer_continuous.action {outer.action!r} must name a continuous "
                "action grid"
            )
        if self.functions.get(outer.post_decision_state) is None:
            messages.append(
                f"outer_continuous.post_decision_state "
                f"{outer.post_decision_state!r} must name an assembled regime function"
            )
        if (
            outer.no_adjustment != outer_unchanged
            and self.functions.get(outer.no_adjustment) is None
        ):
            messages.append(
                f"outer_continuous.no_adjustment {outer.no_adjustment!r} must "
                "name an assembled regime function"
            )
        if messages:
            raise RegimeInitializationError(
                f"In nested consumption-savings regime {regime_name!r}: "
                f"{'; '.join(messages)}."
            )


def _bind_one_margin_solver(
    *, solver: OneMarginSolver | GridSearch, liquid: LiquidMargin
) -> OneMarginSolver | GridSearch:
    if isinstance(solver, GridSearch):
        return solver
    return solver._with_liquid_margin(_bound_liquid_margin(liquid))  # noqa: SLF001


def _bind_two_margin_solver(
    *,
    solver: TwoMarginSolver | GridSearch,
    liquid: LiquidMargin,
    outer: OuterContinuousMargin,
) -> TwoMarginSolver | GridSearch:
    if isinstance(solver, GridSearch):
        return solver
    return solver._with_margins(  # noqa: SLF001
        liquid=_bound_liquid_margin(liquid),
        outer=_BoundOuterContinuousMargin(
            state=outer.state,
            action=outer.action,
            post_decision_state=outer.post_decision_state,
            no_adjustment=outer.no_adjustment,
        ),
    )


def _bound_liquid_margin(liquid: LiquidMargin) -> _BoundLiquidMargin:
    resources = liquid.resources
    if isinstance(resources, NetOfAdjustmentCost):
        return _BoundLiquidMargin(
            state=liquid.state,
            action=liquid.action,
            resources=resources.name_in_dag,
            post_decision_state=liquid.post_decision_state,
            before_cost=resources.before_cost,
            cost=resources.cost,
        )
    return _BoundLiquidMargin(
        state=liquid.state,
        action=liquid.action,
        resources=resources,
        post_decision_state=liquid.post_decision_state,
    )


def _duplicate_names(names: list[str] | tuple[str, ...]) -> list[str]:
    counts = Counter(names)
    return sorted(name for name, count in counts.items() if count > 1)


def _composition_rule_message(
    *, resources: NetOfAdjustmentCost, prefix: str = ""
) -> str:
    return (
        f"{prefix}With NetOfAdjustmentCost, functions[{resources.name_in_dag!r}] "
        f"must not exist, functions[{resources.before_cost!r}] and "
        f"functions[{resources.cost!r}] must exist, and pylcm composes "
        f"{resources.name_in_dag!r} = {resources.before_cost!r} - "
        f"{resources.cost!r}."
    )
