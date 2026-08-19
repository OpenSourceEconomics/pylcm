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
from lcm.typing import UserFunction, outer_unchanged


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
        _validate_mapping_contents(self)
        _validate_logical_consistency(self)

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
