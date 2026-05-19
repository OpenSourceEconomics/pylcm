"""User-facing regime types: `Regime`, `MarkovTransition`, `SolveSimulateFunctionPair`.

The validators, the default Bellman aggregator, the identity transition, and
the `validate_transition_probs` helpers all live behind a leading underscore
in `lcm._regime` and `lcm.regime_building.transitions`. This module is
intentionally a thin layer of public class definitions plus the deprecated
`validate_transition_probs` function.

"""

import dataclasses
import inspect
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

import jax.numpy as jnp
from beartype import beartype

from lcm._beartype_conf import REGIME_CONF
from lcm._grids import DiscreteGrid, Grid
from lcm._regime._helpers import _default_H
from lcm._regime._transition_probs import (
    _build_expected_shape,
    _build_grids,
    _extract_markov_transition,
)
from lcm._regime._validation import (
    _validate_logical_consistency,
    _validate_mapping_contents,
)
from lcm.exceptions import RegimeInitializationError
from lcm.typing import (
    ActionName,
    ActiveFunction,
    FloatND,
    FunctionName,
    RegimeName,
    StateName,
    UserFunction,
)
from lcm.utils.ast_inspection import _get_func_indexing_params
from lcm.utils.containers import (
    ensure_containers_are_immutable,
)

# Genuine circular import: model.py imports from this module at module level.
# The `model` parameter of `validate_transition_probs` is annotated with the
# fully-qualified `lcm.api.model.Model` so the beartype claw resolves it by
# importing `lcm.model` at first call — long after the import cycle settles —
# rather than at module-init time. Importing `lcm.model` here keeps `lcm` a
# bound name for the type checker.
if TYPE_CHECKING:
    import lcm.api.model


class SolveSimulateFunctionPair[S, T]:
    """Container for phase-specific function variants.

    Use this to provide different implementations of a function for the solve
    and simulate phases.  For example, naive beta-delta discounting uses
    exponential discounting during backward induction (solve) but
    present-biased discounting for action selection (simulate).

    Variants may have different parameter signatures.  The params template is
    the union of both variants' parameters; each variant receives only the
    kwargs it expects.

    """

    __slots__ = ("simulate", "solve")

    def __init__(self, *, solve: S, simulate: T) -> None:
        self.solve = solve
        self.simulate = simulate


@beartype(conf=REGIME_CONF)
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
        # Copy __wrapped__ and __annotations__ from the wrapped function so
        # that inspect.signature and dags see the original signature. We use
        # object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, "__wrapped__", self.func)
        object.__setattr__(
            self, "__annotations__", getattr(self.func, "__annotations__", {})
        )

    def __call__(self, *args: Any, **kwargs: Any) -> FloatND:  # noqa: ANN401
        return self.func(*args, **kwargs)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Regime:
    """User-facing regime definition.

    `Model` processes instances of this class into the canonical regime form
    (`lcm.engine.Regime`) used internally by the solver and simulator.

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
            from lcm.regime_building.transitions import (  # noqa: PLC0415
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


def validate_transition_probs(
    *,
    probs: FloatND,
    model: lcm.api.model.Model,
    regime_name: RegimeName,
    state_name: StateName,
    target_regime_name: RegimeName | None = None,
) -> None:
    """Validate a state transition probability array.

    .. deprecated::
        State transition probabilities are now validated automatically during
        `model.solve()` and `model.simulate()` (unless `log_level="off"`) via
        `validate_state_transitions_all_periods` in
        `lcm/_transition_checks.py`. This manual helper will be removed in a
        future release. Drop the call — the model checks for you.

    Check that the array has the shape expected from the function signature,
    that all values are in [0, 1], that rows sum to 1, and that the function's
    `probs_array[…]` subscripts match the signature parameter order.

    For per-target state transitions (where `state_transitions[state_name]` is
    a dict mapping target regime names to `MarkovTransition` instances), pass
    `target_regime_name` to select the specific transition to validate.

    Regime transition probabilities are validated automatically before solve
    via `validate_regime_transitions_all_periods` in
    `lcm/_transition_checks.py`; this helper covers only state transitions.

    Args:
        probs: The transition probability array to validate.
        model: The LCM Model instance.
        regime_name: Name of the regime.
        state_name: Name of the state with a `MarkovTransition`.
        target_regime_name: Target regime name for per-target state
            transitions. Required when the state transition is a per-target
            dict.

    Raises:
        TypeError: If the transition is not a `MarkovTransition`.
        ValueError: If the shape is wrong, values are outside [0, 1], or rows
            don't sum to 1.

    """
    warnings.warn(
        "lcm.validate_transition_probs is deprecated: state transition "
        "probabilities are now validated automatically during model.solve() "
        "and model.simulate() (unless log_level='off'). Drop this call.",
        DeprecationWarning,
        stacklevel=2,
    )
    regime = model.user_regimes[regime_name]
    raw_transition = regime.state_transitions[state_name]
    markov = _extract_markov_transition(
        raw_transition=raw_transition,
        state_name=state_name,
        regime_name=regime_name,
        target_regime_name=target_regime_name,
    )
    func = markov.func
    grids = _build_grids(regime)
    n_outcomes = len(grids[state_name].categories)

    indexing_params = _get_func_indexing_params(
        func=func, array_param_name="probs_array"
    )

    sig = inspect.signature(func)
    sig_order = [
        p for p in sig.parameters if p != "probs_array" and p in indexing_params
    ]
    if indexing_params != sig_order:
        func_name = getattr(func, "__name__", "<unknown>")
        msg = (
            f"In function '{func_name}', `probs_array` is indexed as "
            f"`probs_array[{', '.join(indexing_params)}]` but the signature "
            f"order is `probs_array[{', '.join(sig_order)}]`."
        )
        raise ValueError(msg)

    expected_shape = _build_expected_shape(
        indexing_params=indexing_params,
        n_outcomes=n_outcomes,
        grids=grids,
        model=model,
    )

    if probs.shape != expected_shape:
        msg = f"Expected shape {expected_shape} but got {probs.shape}."
        raise ValueError(msg)

    if jnp.any(probs < 0) or jnp.any(probs > 1):
        msg = "All values must be in [0, 1]."
        raise ValueError(msg)

    row_sums = jnp.sum(probs, axis=-1)
    if not jnp.allclose(row_sums, 1.0, atol=1e-6):
        msg = "Rows must sum to 1 along the last axis."
        raise ValueError(msg)
