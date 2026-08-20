"""Specialized regime declarations for consumption-savings models.

The general `Regime` owns model-wide regime vocabulary. This module owns the
liquid- and outer-margin declarations and the specialized regime classes that
bind those roles into endogenous-grid solvers.
"""

from dataclasses import dataclass, field
from typing import ClassVar, cast

from beartype import beartype
from dags import get_annotations, with_signature
from dags.annotations import ensure_annotations_are_strings
from dags.signature import rename_arguments

import lcm.solvers as _solvers
from _lcm.beartype_conf import REGIME_CONF
from _lcm.grids import ContinuousGrid
from _lcm.post_decision_bound import _PostDecisionLowerBound
from _lcm.solution.contract import _BoundLiquidMargin, _BoundOuterContinuousMargin
from _lcm.typing import ActionName, FunctionName, RegimeName, StateName
from _lcm.utils.containers import find_duplicates
from _lcm.utils.error_messages import format_messages
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.regime import Regime
from lcm.transition import AgeSpecializedGrid
from lcm.typing import UserFunction

_DIRECT_RESOURCES_FUNCTION: FunctionName = "_lcm_direct_liquid_resources"

__all__ = [
    "ConsumptionSavingsRegime",
    "LiquidMargin",
    "NestedConsumptionSavingsRegime",
    "NetOfAdjustmentCost",
    "OuterContinuousMargin",
    "outer_unchanged",
    "post_decision_lower_bound",
]

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
        duplicates = _repeated_names((self.name_in_dag, self.before_cost, self.cost))
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

    resources: StateName | FunctionName | NetOfAdjustmentCost
    """The liquid state, a resources node, or resources net of adjustment cost."""

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
        names = [self.state, self.action, self.post_decision_state]
        if self.resources_name != self.state:
            names.append(self.resources_name)
        if isinstance(self.resources, NetOfAdjustmentCost):
            names.extend((self.resources.before_cost, self.resources.cost))
        duplicates = _repeated_names(names)
        if duplicates:
            raise RegimeInitializationError(
                "LiquidMargin names must be pairwise distinct; repeated names: "
                f"{duplicates}."
            )


@beartype(conf=REGIME_CONF)
def post_decision_lower_bound(*, margin: LiquidMargin, lower: float) -> UserFunction:
    """Declare a lower bound on a margin's post-decision state, checkably.

    An endogenous-grid solver enforces its borrowing limit through the savings
    grid, whose lowest node is the limit that the solve and the simulation both
    obey. Declaring the bound states that number explicitly, so a disagreement
    with the grid is refused when the model is built instead of the grid's
    value quietly taking precedence.

        liquid = LiquidMargin(
            state="wealth",
            action="consumption",
            resources="wealth",
            post_decision_state="savings",
        )
        constraints={"borrowing_limit": post_decision_lower_bound(
            margin=liquid, lower=0.0
        )}

    Taking the margin rather than the post-decision state's name is what makes
    the two impossible to disagree: there is no second spelling of the name to
    keep in step.

    The result is an ordinary constraint callable evaluating
    `post_decision_state >= lower`, so it is legal wherever a constraint is. A
    solver whose savings grid already enforces the bound proves it and drops
    it; grid search, which enforces nothing implicitly, evaluates it.

    Args:
        margin: The Euler margin whose post-decision state is bounded below.
        lower: The bound itself. Must equal the savings grid's lowest node
            exactly, where the solver has one.

    Returns:
        A constraint callable carrying the declared bound.

    """
    return _PostDecisionLowerBound(
        post_decision=margin.post_decision_state, lower_bound=lower
    )


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
    """No-adjustment map, or `lcm.outer_unchanged` for identity."""

    def __post_init__(self) -> None:
        self._fail_if_names_are_not_pairwise_distinct()

    def _fail_if_names_are_not_pairwise_distinct(self) -> None:
        names = [self.state, self.action, self.post_decision_state]
        if self.no_adjustment != outer_unchanged:
            names.append(self.no_adjustment)
        duplicates = _repeated_names(names)
        if duplicates:
            raise RegimeInitializationError(
                "OuterContinuousMargin names must be pairwise distinct; repeated "
                f"names: {duplicates}."
            )


@dataclass(frozen=True, kw_only=True)
class _EGMFamilyRegime(Regime):
    """Shared declaration and validation for one- and two-margin EGM regimes."""

    _accepts_margin_solver: ClassVar[bool] = True

    liquid: LiquidMargin

    def _augment_phase_functions(
        self, functions: dict[FunctionName, UserFunction]
    ) -> dict[FunctionName, UserFunction]:
        """Make a directly named liquid state available at the solver's DAG seam."""
        if self.liquid.resources_name != self.liquid.state:
            return functions
        if _DIRECT_RESOURCES_FUNCTION in functions:
            raise RegimeInitializationError(
                f"Function name {_DIRECT_RESOURCES_FUNCTION!r} is reserved for a "
                "LiquidMargin whose resources role is filled by its state."
            )

        state = self.liquid.state
        post_decision_name = self.liquid.post_decision_state
        post_decision = functions.get(post_decision_name)
        annotations = (
            ensure_annotations_are_strings(get_annotations(post_decision))
            if post_decision is not None
            else {}
        )
        state_annotation = annotations.get(state, "FloatND")
        if state_annotation == "no_annotation_found":
            state_annotation = "FloatND"

        @with_signature(
            args={state: state_annotation}, return_annotation=state_annotation
        )
        def direct_resources(**kwargs: object) -> object:
            return kwargs[state]

        direct_resources.__dict__["_lcm_internal_no_params"] = True

        augmented = {
            **functions,
            _DIRECT_RESOURCES_FUNCTION: cast("UserFunction", direct_resources),
        }
        if post_decision is not None and state in annotations:
            augmented[post_decision_name] = rename_arguments(
                post_decision,
                mapper={state: _DIRECT_RESOURCES_FUNCTION},
            )
        return augmented

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
                "is not a continuous solve-state grid."
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
        elif resources == self.liquid.state:
            required = ()
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
                "solve-state grid"
            )
        if not isinstance(self.actions.get(self.liquid.action), ContinuousGrid):
            messages.append(
                f"liquid.action {self.liquid.action!r} must name a continuous "
                "action grid"
            )
        if (
            self.liquid.resources_name != self.liquid.state
            and self.functions.get(self.liquid.resources_name) is None
        ):
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
                f"In EGM-family regime {regime_name!r}: {format_messages(messages)}"
            )


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class ConsumptionSavingsRegime(_EGMFamilyRegime):
    """One-liquid-margin regime for EGM, DC-EGM, or grid search."""

    solver: _solvers.OneMarginSolver | _solvers.GridSearch = field(
        default_factory=_solvers.GridSearch
    )

    def __post_init__(self) -> None:
        self._fail_if_solver_pairing_is_invalid()
        object.__setattr__(
            self,
            "solver",
            _bind_one_margin_solver(solver=self.solver, liquid=self.liquid),
        )
        super().__post_init__()

    def _fail_if_solver_pairing_is_invalid(self) -> None:
        if not isinstance(self.solver, _solvers.OneMarginSolver | _solvers.GridSearch):
            raise RegimeInitializationError(
                "ConsumptionSavingsRegime.solver must be a OneMarginSolver or "
                f"GridSearch, got {type(self.solver).__module__}."
                f"{type(self.solver).__qualname__}."
            )


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class NestedConsumptionSavingsRegime(_EGMFamilyRegime):
    """Two-margin sibling of `ConsumptionSavingsRegime`."""

    outer_continuous: OuterContinuousMargin
    solver: _solvers.TwoMarginSolver | _solvers.GridSearch = field(
        default_factory=_solvers.GridSearch
    )

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
        if not isinstance(self.solver, _solvers.TwoMarginSolver | _solvers.GridSearch):
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
                "is not a continuous solve-state grid."
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
                "solve-state grid"
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
                f"{format_messages(messages)}"
            )


def _bind_one_margin_solver(
    *, solver: _solvers.OneMarginSolver | _solvers.GridSearch, liquid: LiquidMargin
) -> _solvers.OneMarginSolver | _solvers.GridSearch:
    if isinstance(solver, _solvers.GridSearch):
        return solver
    return solver._with_liquid_margin(_bound_liquid_margin(liquid))  # noqa: SLF001


def _bind_two_margin_solver(
    *,
    solver: _solvers.TwoMarginSolver | _solvers.GridSearch,
    liquid: LiquidMargin,
    outer: OuterContinuousMargin,
) -> _solvers.TwoMarginSolver | _solvers.GridSearch:
    if isinstance(solver, _solvers.GridSearch):
        return solver
    return solver._with_margins(  # noqa: SLF001
        liquid=_bound_liquid_margin(liquid),
        outer=_BoundOuterContinuousMargin(
            state=outer.state,
            action=outer.action,
            post_decision_state=outer.post_decision_state,
            # `outer_unchanged` is a declaration, not a function name. Resolving
            # it here, at the one seam where a public margin becomes a bound
            # one, is what lets every engine consumer read the identity map as
            # the absence of a candidate function.
            no_adjustment=(
                None if outer.no_adjustment == outer_unchanged else outer.no_adjustment
            ),
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
        resources=(
            _DIRECT_RESOURCES_FUNCTION if resources == liquid.state else resources
        ),
        post_decision_state=liquid.post_decision_state,
    )


def _repeated_names(names: list[str] | tuple[str, ...]) -> list[str]:
    """Return the names occurring more than once, in a deterministic order."""
    return sorted(find_duplicates(names))


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
