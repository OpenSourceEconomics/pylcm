"""The user-facing `Problem`: an action problem several regimes may hold."""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from beartype import beartype

import lcm.solvers as _solvers
from _lcm.beartype_conf import REGIME_CONF
from _lcm.typing import ActionName, FunctionName, RegimeName, StateName
from _lcm.utils.containers import ensure_containers_are_immutable
from lcm.certainty_equivalent import CertaintyEquivalent
from lcm.collective import ValueDependentConstraint
from lcm.decoupled import Decoupled
from lcm.exceptions import RegimeInitializationError
from lcm.grids import DiscreteGrid
from lcm.phased import Phased
from lcm.regime import (
    ActionEntry,
    ConstraintEntry,
    FunctionEntry,
    Regime,
    StateEntry,
    StateTransitionEntry,
)
from lcm.shocks import ShockGroupName
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.transition import JointTransition
from lcm.typing import UserFunction


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Problem:
    """An action problem: states, actions, constraints, functions and laws of motion.

    A problem is valued with next period's regimes and with nothing from the
    current period. Several regimes may hold one problem through
    `Regime(problem=...)`. Holding the same problem means the same declaration,
    not the same value: holders may differ in their exits and in the edges the
    model declares for them, and each holder's value follows from its own. The
    parameters of a problem's functions stay per regime, keyed by the names of
    the regimes that hold it.
    """

    states: Mapping[StateName, StateEntry] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """As `Regime.states`."""

    state_transitions: Mapping[StateName, StateTransitionEntry] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """As `Regime.state_transitions`."""

    joint_transitions: Mapping[RegimeName, Mapping[str, JointTransition | Phased]] = (
        field(default_factory=lambda: MappingProxyType({}))
    )
    """As `Regime.joint_transitions`."""

    actions: Mapping[ActionName, ActionEntry] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """As `Regime.actions`."""

    functions: Mapping[FunctionName, FunctionEntry] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """As `Regime.functions`."""

    constraints: Mapping[FunctionName, ConstraintEntry] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """As `Regime.constraints`."""

    output_only_functions: tuple[FunctionName, ...] = ()
    """As `Regime.output_only_functions`."""

    derived_categoricals: Mapping[FunctionName, DiscreteGrid] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """As `Regime.derived_categoricals`."""

    solver: _solvers.Solver = field(default_factory=_solvers.GridSearch)
    """As `Regime.solver`."""

    taste_shocks: ExtremeValueTasteShocks | None = None
    """As `Regime.taste_shocks`."""

    shocks: tuple[ShockGroupName, ...] = ()
    """As `Regime.shocks`."""

    koopmans_aggregator: UserFunction | Phased | Decoupled | None = None
    """As `Regime.koopmans_aggregator`."""

    certainty_equivalent: CertaintyEquivalent | None = None
    """As `Regime.certainty_equivalent`."""

    def __post_init__(self) -> None:
        for name in (
            "functions",
            "states",
            "state_transitions",
            "joint_transitions",
            "actions",
            "constraints",
            "derived_categoricals",
        ):
            object.__setattr__(
                self, name, ensure_containers_are_immutable(getattr(self, name))
            )
        value_dependent = sorted(
            name
            for name, constraint in self.constraints.items()
            if isinstance(constraint, ValueDependentConstraint)
        )
        if value_dependent:
            raise RegimeInitializationError(
                f"Problem constraints {value_dependent} are value-dependent. A "
                "value-dependent constraint compares action values, which is a "
                "regime's rule, not part of a problem; declare it on a regime that "
                "declares its own slots."
            )


def lowered_slots(*, regime: Regime) -> tuple[str, ...]:
    """Return the slots a regime takes from the problem it holds.

    Each problem slot of a regime that holds a problem is either still the
    `Regime` default, and then takes the problem's object, or already is the
    problem's object, as after `replace` of a holder that changes only its
    `exit` or `description`. Anything else, an object carried over from
    another problem included, is a declaration of the regime's own.

    Args:
        regime: A regime, with or without a `problem`.

    Returns:
        The names of the problem's slots, or an empty tuple when the regime
        holds no problem.

    Raises:
        RegimeInitializationError: If the regime is a specialized regime, or if one
            of its problem slots is neither the default nor the problem's object.
    """
    problem = regime.problem
    if problem is None:
        return ()
    if type(regime) is not Regime:
        raise RegimeInitializationError(
            f"A {type(regime).__name__} cannot hold a Problem: its margin "
            "declarations belong to its own problem. Declare its slots itself."
        )
    slots = tuple(declaration.name for declaration in dataclasses.fields(Problem))
    defaults = {
        declaration.name: declaration.default
        if declaration.default is not dataclasses.MISSING
        else declaration.default_factory()  # ty: ignore[call-non-callable]
        for declaration in dataclasses.fields(Regime)
        if declaration.name in slots
    }
    declared = [
        name
        for name in slots
        if getattr(regime, name) is not getattr(problem, name)
        and getattr(regime, name) != defaults[name]
    ]
    if declared:
        raise RegimeInitializationError(
            "A regime that holds a problem declares none of the problem's slots, "
            f"but this one also declares {declared}: each is neither left at its "
            "default nor the problem's own object. Declare them on the Problem, or "
            "pass `problem=None` and declare the slots on the regime."
        )
    return slots
