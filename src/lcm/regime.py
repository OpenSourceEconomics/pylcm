"""The user-facing `Regime` definition.

The validators and the identity transition live behind a leading underscore in
`_lcm.user_regime_validation` and `_lcm.regime_building.transitions`. This
module is intentionally thin: the public class definition. A non-terminal
regime that supplies no `H` gets `lcm.temporal_aggregation.H_linear` at model build.

"""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, cast

from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.grids import DiscreteGrid, Grid
from _lcm.regime_building.phases import normalize_regime_phases
from _lcm.regime_building.transitions import collect_state_transitions
from _lcm.typing import ActionName, ActiveFunction, FunctionName, RegimeName, StateName
from _lcm.user_regime_validation import (
    _validate_collective_regime,
    _validate_logical_consistency,
    _validate_mapping_contents,
)
from _lcm.utils.containers import (
    ensure_containers_are_immutable,
)
from lcm.certainty_equivalent import CertaintyEquivalent
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.solvers import GridSearch, Solver
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.transition import MarkovTransition
from lcm.typing import UserFunction


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class SamePeriodRef:
    """Declaration of a same-period cross-regime reference value (E2).

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
    CURRENT period's arrays (design doc `pylcm-extension-collective-regimes.md`
    §2 E2; EKL 2019 eq. 11 reads the singles' period-t values from inside the
    married period-t problem).
    """

    regime: RegimeName
    """Name of the reference regime whose same-period V is read.

    Must be another regime of the model, active in every period the declaring
    regime is active. No transition edge between the two regimes is required —
    same-period reference reads work across otherwise unconnected regime
    "islands" (that is the point of E2).
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

    `Phased` gives each phase its own variant (matching form required; for
    per-target dicts, identical key sets).
    """

    active: ActiveFunction = lambda _age: True
    """Callable that takes age (float) and returns True if regime is active."""

    # `None` masks a model-level entry of the same name.
    states: Mapping[StateName, Grid | Phased | None] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of state variable names to grids or phase-variant declarations.

    A plain `Grid` value is a state shared by both phases.
    `Phased(solve=callable, simulate=Grid)` declares a carried state: a
    derived function (no grid axis) in the solve phase and a seeded, evolved
    state in the simulate phase, whose law of motion is its regular
    `state_transitions` entry.
    """

    state_transitions: Mapping[
        StateName,
        UserFunction
        | MarkovTransition
        | Phased
        | None
        # `Phased` inside a per-target dict passes the type check so the
        # validator can reject it with the outermost-only explanation.
        | Mapping[RegimeName, UserFunction | MarkovTransition | Phased],
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
    - `DCEGM(...)`: endogenous grid method for one continuous state and one
      continuous action; the regime must satisfy the DC-EGM model contract,
      which is validated at `Model` construction time.
    """

    taste_shocks: ExtremeValueTasteShocks | None = None
    """EV1 taste shocks on the regime's discrete-action combinations.

    When set, the shock scale becomes the runtime param
    `{"taste_shocks": {"scale": ...}}` and the solve aggregates discrete
    actions via the smoothed expected maximum instead of the hard maximum.
    Requires at least one discrete action.
    """

    certainty_equivalent: CertaintyEquivalent | None = None
    """Nonlinear certainty equivalent over the next-period value distribution.

    When set, the solve aggregates the continuation as
    `g⁻¹(Σ_r p_r · E_w[g(V')])` instead of the linear expectation, and the
    transform parameters become runtime params under the pseudo-function
    name `certainty_equivalent`. Only non-terminal regimes solved by
    `GridSearch` support it.
    """

    description: str = ""
    """Description of the regime."""

    stakeholders: tuple[str, ...] | None = None
    """Names of the stakeholders whose individual values this regime carries.

    `None` (the default) is the singleton case: the regime has one implicit
    stakeholder, one value function, and follows today's exact code path — no
    behavior change. A non-`None` tuple declares a *collective regime*: a couple
    (or other multi-party household) that solves one household argmax but reads
    off a per-stakeholder value at that common argmax, with value-aware
    feasibility and value-gated regime routing (consent / divorce).

    This is the API surface of the "collective regimes" extension (E1-E4 +
    shared shocks). The **E1 solve** is implemented for terminal and
    non-terminal regimes: a collective regime carries a per-stakeholder utility
    `functions["utility_<s>"]` for each stakeholder `<s>` and household Pareto
    `weights`; its solve reads off each stakeholder's own value at the shared
    household argmax, and a non-terminal collective regime aggregates the
    per-stakeholder continuation `Q^s = H(u^s, E[V'^s])` (see the design doc
    `pylcm-extension-collective-regimes.md` v2.1, §2 E1). A non-terminal
    collective regime's transition targets must all be collective regimes with
    the identical `stakeholders` tuple — per-stakeholder routing to different
    regimes (value gates, divorce) is E3'. Collective-regime simulation, EV1
    taste shocks, nonlinear certainty equivalents, and non-GridSearch solvers
    on a collective regime still raise `NotImplementedError`.
    """

    weights: Mapping[str, float] | None = None
    """Household Pareto weights `λ_s` per stakeholder for a collective regime.

    Used only when `stakeholders is not None`: the collective solve maximizes the
    household scalarization `O = Σ_s λ_s Q^s` over the feasible action set. When
    omitted (the default), equal weights `1/len(stakeholders)` are used; supply
    an explicit mapping to express unequal Pareto weights (e.g. EKL's λ=0.5 on
    each partner). Ignored — and must be `None` — for a singleton regime.
    """

    value_constraints: Mapping[FunctionName, UserFunction] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Value-aware feasibility predicates for a non-terminal collective regime (E2).

    Each entry maps a constraint name to a predicate returning `True` where the
    (state, action) combination is feasible. Unlike ordinary `constraints`
    (which are evaluated before and independently of `Q`), a value constraint
    is evaluated AFTER the per-stakeholder action values and may read, as named
    arguments:

    - `Q_<s>` for each stakeholder `<s>` — that stakeholder's own action value
      `Q^s(x, a)` (felicity plus discounted continuation) at the cell;
    - each key of `same_period_refs` — the reference regime's same-period value
      interpolated at the projected state (e.g. the divorcee's single value);
    - ordinary states, actions, regime functions, and parameters via the DAG
      (a predicate parameter such as EKL's `Delta_j` surfaces in the params
      template under the constraint's name).

    The final action mask is the AND of ordinary constraints and all value
    constraints; the household argmax runs over the masked set, and a state
    cell whose mask is empty publishes the divorce flag `D = True` (returned by
    the solve alongside V — never conflated with a numeric `-inf` value, which
    can occur on-path). EKL 2019 eq. 11 is exactly
    `Q_j >= V_single_j(pi_j(x)) - Delta_j` for each stakeholder `j`.

    Only non-terminal collective regimes may declare value constraints.
    """

    same_period_refs: Mapping[str, SamePeriodRef] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Same-period cross-regime reference values read by `value_constraints` (E2).

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
        # COLLECTIVE-REGIMES (E1): the solve is implemented for terminal and
        # non-terminal collective regimes. A collective regime is validated
        # here (per-stakeholder `utility_<s>`, weights, >=1 discrete action;
        # out-of-scope features — taste shocks, certainty equivalents,
        # non-GridSearch solvers — are rejected) and then solves via the
        # collective kernels. The default `None` (singleton) path never enters
        # this branch, so today's behavior is provably untouched. See
        # `pylcm-extension-collective-regimes.md` §2.
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
                "a collective regime (E2); they read the per-stakeholder action "
                "values `Q_<s>`, which only exist when `stakeholders` is set. "
                "Use ordinary `constraints` for a singleton regime."
            )
        elif self.same_period_refs:
            raise RegimeInitializationError(
                "`same_period_refs` declares same-period reference values for a "
                "collective regime's `value_constraints` (E2); it is only "
                "meaningful together with `stakeholders`. Omit it for a "
                "singleton regime."
            )

        _validate_mapping_contents(self)
        _validate_logical_consistency(self)

        def make_immutable(name: str) -> None:
            value = ensure_containers_are_immutable(getattr(self, name))
            object.__setattr__(self, name, value)

        # Completeness (a `utility` entry, default-`H` injection, transition
        # coverage) is validated when the model finalizes its regimes
        # — model-level slots may still satisfy it after merging.
        make_immutable("functions")
        make_immutable("states")
        make_immutable("state_transitions")
        make_immutable("actions")
        make_immutable("constraints")
        make_immutable("derived_categoricals")
        make_immutable("value_constraints")
        make_immutable("same_period_refs")

        # The phase grammar (states matrix, carried laws, regime-transition
        # variants) is validated by the normalizer; the per-phase spec it
        # builds is consumed during model processing.
        normalize_regime_phases(self)

    def get_all_functions(
        self,
        phase: Literal["solve", "simulate"] = "solve",
    ) -> MappingProxyType[str, UserFunction]:
        """Get all regime functions including utility, constraints, and transitions.

        Collect functions from four sources:
        - `self.functions` (utility, helpers, H)
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
