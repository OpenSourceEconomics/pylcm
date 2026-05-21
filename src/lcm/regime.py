"""User-facing regime types: `Regime`, `MarkovTransition`, `SolveSimulateFunctionPair`.

The validators and the identity transition live behind a leading underscore in
`_lcm.regime` and `_lcm.regime_building.transitions`. This module is
intentionally thin: the public class definitions plus the private default
Bellman aggregator (`_default_H`), which `Regime` injects when a non-terminal
regime supplies no `H`.

"""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, cast

from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.grids import DiscreteGrid, Grid
from _lcm.regime.validation import (
    _validate_logical_consistency,
    _validate_mapping_contents,
)
from _lcm.regime_building.transitions import collect_state_transitions
from _lcm.typing import ActionName, ActiveFunction, FunctionName, RegimeName, StateName
from _lcm.utils.containers import (
    ensure_containers_are_immutable,
)
from lcm.exceptions import RegimeInitializationError
from lcm.transition import MarkovTransition, SolveSimulateFunctionPair
from lcm.typing import FloatND, UserFunction


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Regime:
    """User-facing regime definition.

    `Model` processes instances of this class into the canonical regime form
    (`_lcm.engine.Regime`) used internally by the solver and simulator.

    State transitions are specified via `state_transitions`, mapping state names to
    transition functions. A bare callable is deterministic; wrap in `MarkovTransition`
    for stochastic transitions. `None` marks a fixed state (identity auto-generated).
    Stochastic processes have intrinsic transitions and must not appear in
    `state_transitions`.

    The `transition` field on the regime itself is the *regime* transition function.
    A regime with `transition=None` is terminal — no separate `terminal` flag is
    needed.

    """

    transition: UserFunction | MarkovTransition | None
    """Regime transition function, or `None` for terminal regimes.

    A bare callable is deterministic. Wrap in `MarkovTransition` for stochastic
    regime transitions that return probability distributions.
    """

    active: ActiveFunction = lambda _age: True
    """Callable that takes age (float) and returns True if regime is active."""

    states: Mapping[StateName, Grid] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of state variable names to grid objects."""

    state_transitions: Mapping[
        StateName,
        UserFunction
        | MarkovTransition
        | None
        | Mapping[RegimeName, UserFunction | MarkovTransition],
    ] = field(default_factory=lambda: MappingProxyType({}))
    """Mapping of state names to transition functions, `None`, or per-target dicts.

    Every non-process state must have an entry — omitting a state raises an error.
    `None` marks a fixed state (identity auto-generated internally). Wrap in
    `MarkovTransition` for stochastic transitions. Per-target dicts map target
    regime names to transition functions — every reachable target must be listed.
    """

    actions: Mapping[ActionName, Grid] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of action variable names to grid objects."""

    functions: Mapping[FunctionName, UserFunction] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of function names to callables; must include 'utility'."""

    constraints: Mapping[FunctionName, UserFunction] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of constraint names to constraint functions."""

    derived_categoricals: Mapping[FunctionName, DiscreteGrid] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Categorical grids for DAG function outputs not in states/actions."""

    description: str = ""
    """Description of the regime."""

    @property
    def terminal(self) -> bool:
        """Whether this is a terminal regime (derived from transition being None)."""
        return self.transition is None

    @property
    def stochastic_regime_transition(self) -> bool:
        """Whether the regime transition is stochastic (MarkovTransition)."""
        return isinstance(self.transition, MarkovTransition)

    def __post_init__(self) -> None:
        _validate_mapping_contents(self)
        _validate_logical_consistency(self)

        def make_immutable(name: str) -> None:
            value = ensure_containers_are_immutable(getattr(self, name))
            object.__setattr__(self, name, value)

        # Inject default aggregation function H if not provided by user.
        # Terminal regimes don't need H since Q = U directly (no E_next_V).
        if not self.terminal and "H" not in self.functions:
            object.__setattr__(self, "functions", {**self.functions, "H": _default_H})
        make_immutable("functions")
        make_immutable("states")
        make_immutable("state_transitions")
        make_immutable("actions")
        make_immutable("constraints")
        make_immutable("derived_categoricals")

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

        For `SolveSimulateFunctionPair` entries, the variant matching `phase` is
        used.

        Args:
            phase: Which variant to use for `SolveSimulateFunctionPair` entries.

        Returns:
            Read-only mapping of all regime functions.

        """
        result: dict[str, UserFunction] = {}
        for name, func in self.functions.items():
            if isinstance(func, SolveSimulateFunctionPair):
                result[name] = cast(
                    "UserFunction",
                    func.solve if phase == "solve" else func.simulate,
                )
            else:
                result[name] = func
        result |= dict(self.constraints)
        if callable(self.transition):
            result |= collect_state_transitions(self.states, self.state_transitions)
            result["next_regime"] = self.transition
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


def _default_H(
    utility: FloatND, E_next_V: FloatND, discount_factor: FloatND
) -> FloatND:
    """Default Bellman aggregator: `U + β · E[V_next]`."""
    return utility + discount_factor * E_next_V
