import dataclasses
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAliasType, cast, overload

from lcm.exceptions import RegimeInitializationError
from lcm.grids import DiscreteGrid, Grid
from lcm.interfaces import SolveSimulateFunctionPair
from lcm.typing import (
    ActionName,
    ActiveFunction,
    ContinuousState,
    DiscreteState,
    FloatND,
    FunctionName,
    RegimeName,
    StateName,
    UserFunction,
)
from lcm.utils.containers import (
    ensure_containers_are_immutable,
)


@dataclass(frozen=True)
class MarkovTransition:
    """Wrapper marking a transition function as stochastic (Markov).

    Wrap a transition function in `MarkovTransition` to indicate that it returns
    a probability distribution over next states (for state transitions) or over
    next regimes (for regime transitions), rather than a deterministic next value.

    Use at both the state and regime level:

        # Stochastic state transition (in Regime.state_transitions)
        state_transitions={"health": MarkovTransition(health_probs)}

        # Stochastic regime transition
        Regime(transition=MarkovTransition(regime_probs), ...)

    A bare callable (without the wrapper) is deterministic at both levels.

    """

    func: Callable[..., FloatND]
    """The transition function returning a probability distribution."""

    def __post_init__(self) -> None:
        if not callable(self.func):
            raise RegimeInitializationError(
                f"MarkovTransition requires a callable, "
                f"but got {type(self.func).__name__}: {self.func!r}"
            )
        # Copy __wrapped__ and __annotations__ from the wrapped function so
        # that inspect.signature and dags see the original signature. We use
        # object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, "__wrapped__", self.func)
        object.__setattr__(
            self, "__annotations__", getattr(self.func, "__annotations__", {})
        )

    def __call__(self, *args: Any, **kwargs: Any) -> FloatND:  # noqa: ANN401
        return self.func(*args, **kwargs)


def _default_H(utility: float, E_next_V: float, discount_factor: float) -> float:
    return utility + discount_factor * E_next_V


class _IdentityTransition:
    """Identity transition function for fixed states.

    Used by `get_all_functions()` so the params template includes fixed states.
    The `_is_auto_identity` attribute lets validation distinguish auto-generated
    identities from user-provided transitions.

    """

    _is_auto_identity: bool = True

    def __init__(self, state_name: StateName, *, annotation: TypeAliasType) -> None:
        self._state_name = state_name
        self.__name__ = f"next_{state_name}"
        param = inspect.Parameter(
            state_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=annotation,
        )
        self.__signature__ = inspect.Signature(
            [param],
            return_annotation=annotation,
        )
        self.__annotations__ = {state_name: annotation, "return": annotation}

    @overload
    def __call__(self, **kwargs: DiscreteState) -> DiscreteState: ...
    @overload
    def __call__(self, **kwargs: ContinuousState) -> ContinuousState: ...
    def __call__(
        self, **kwargs: DiscreteState | ContinuousState
    ) -> DiscreteState | ContinuousState:
        return kwargs[self._state_name]


@dataclass(frozen=True, kw_only=True)
class Regime:
    """A user regime which can be processed into an internal regime.

    State transitions are specified via `state_transitions`, mapping state names to
    transition functions. A bare callable is deterministic; wrap in `MarkovTransition`
    for stochastic transitions. `None` marks a fixed state (identity auto-generated).
    ShockGrids have intrinsic transitions and must not appear in `state_transitions`.

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

    Every non-shock state must have an entry — omitting a state raises an error.
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
        from lcm.regime_building.validation import (  # noqa: PLC0415
            validate_attribute_types,
            validate_logical_consistency,
        )

        validate_attribute_types(self)
        validate_logical_consistency(self)

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
            from lcm.regime_building.validation import (  # noqa: PLC0415
                collect_state_transitions,
            )

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
