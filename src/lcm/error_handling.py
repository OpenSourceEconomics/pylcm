import inspect
from collections.abc import Callable
from types import MappingProxyType
from typing import TYPE_CHECKING, overload

import jax
import jax.numpy as jnp
import pandas as pd
from jax import Array

from lcm.ages import AgeGrid
from lcm.exceptions import (
    InvalidRegimeTransitionProbabilitiesError,
    InvalidValueFunctionError,
)
from lcm.grids import DiscreteGrid
from lcm.interfaces import InternalRegime
from lcm.regime import MarkovTransition, Regime
from lcm.typing import (
    FlatRegimeParams,
    FloatND,
    InternalParams,
    RegimeName,
    ScalarFloat,
)

if TYPE_CHECKING:
    from lcm.model import Model


def validate_value_function_array(*, V_arr: Array, age: ScalarFloat) -> None:
    """Validate the value function array for NaN values.

    This function checks the value function array for any NaN values. If any such values
    are found, we raise an `InvalidValueFunctionError`.

    Args:
        V_arr: The value function array to validate.
        age: The age for which the value function is being validated.

    Raises:
        InvalidValueFunctionError: If the value function array contains NaN values.

    """
    if jnp.any(jnp.isnan(V_arr)):
        raise InvalidValueFunctionError(
            f"The value function array at age {age} contains NaN values. This "
            "may be due to various reasons:\n"
            "- The user-defined functions returned invalid values.\n"
            "- It is impossible to reach an active regime, resulting in NaN regime\n"
            "  transition probabilities."
        )


def validate_regime_transition_probs(
    *,
    regime_transition_probs: MappingProxyType[str, Array],
    active_regimes_next_period: tuple[str, ...],
    regime_name: str,
    age: ScalarFloat,
    next_age: ScalarFloat,
    state_action_values: MappingProxyType[str, Array] | None = None,
) -> None:
    """Validate regime transition probabilities.

    Check that probabilities are finite, sum to 1 across all regimes, and that
    inactive regimes have zero probability.

    Args:
        regime_transition_probs: Immutable mapping of regime names to probability
            arrays.
        active_regimes_next_period: Tuple of regime names active in the next period.
        regime_name: Name of the source regime (for error messages).
        age: Current age (for error messages).
        next_age: Next age (for error messages).
        state_action_values: Optional immutable mapping of state/action names to arrays,
            included in error messages to help diagnose which inputs cause violations.

    Raises:
        InvalidRegimeTransitionProbabilitiesError: If probabilities are non-finite,
            outside [0, 1], don't sum to 1, or assign positive probability to inactive
            regimes.

    """
    all_probs = jnp.stack(list(regime_transition_probs.values()))

    if jnp.any(~jnp.isfinite(all_probs)):
        raise InvalidRegimeTransitionProbabilitiesError(
            f"Non-finite values in regime transition probabilities from "
            f"'{regime_name}' between ages {age} and {next_age}. Check the "
            f"'next_regime' function of the '{regime_name}' regime."
        )

    if jnp.any(all_probs < 0) or jnp.any(all_probs > 1):
        raise InvalidRegimeTransitionProbabilitiesError(
            f"Regime transition probabilities from '{regime_name}' between ages {age} "
            f"and {next_age} contain values outside [0, 1]. Check the 'next_regime' "
            f"function of the '{regime_name}' regime."
        )

    sum_all = jnp.sum(all_probs, axis=0)
    if not jnp.allclose(sum_all, 1.0):
        detail = _format_sum_violation(
            sum_all,
            state_action_values=state_action_values,
        )
        raise InvalidRegimeTransitionProbabilitiesError(
            f"Regime transition probabilities from '{regime_name}' between ages {age} "
            f"and {next_age} do not sum to 1.0. {detail}\n"
            f"Check the 'next_regime' function of the '{regime_name}' regime."
        )

    inactive = set(regime_transition_probs) - set(active_regimes_next_period)
    for r in inactive:
        if jnp.any(regime_transition_probs[r] > 0):
            raise InvalidRegimeTransitionProbabilitiesError(
                f"Regime '{r}' is inactive at age {next_age} but has positive "
                f"transition probability from '{regime_name}' between ages {age} and "
                f"{next_age}. Either make '{r}' active or ensure its probability is 0."
            )


def _format_sum_violation(
    sum_all: Array,
    *,
    state_action_values: MappingProxyType[str, Array] | None = None,
) -> str:
    """Format a human-readable description of probability sum violations.

    Args:
        sum_all: Array of probability sums (per-subject).
        state_action_values: Optional immutable mapping of state/action names to arrays,
            included in the output to show which inputs cause violations.

    Returns:
        Formatted string describing which sums violate the sum-to-1 constraint.

    """
    sum_all = jnp.atleast_1d(sum_all)
    if state_action_values is not None:
        state_action_values = MappingProxyType(
            {name: jnp.atleast_1d(arr) for name, arr in state_action_values.items()}
        )
    failing_mask = ~jnp.isclose(sum_all, 1.0)
    failing_indices = jnp.where(failing_mask)[0]
    failing_sums = sum_all[failing_mask]
    n_failing = int(failing_indices.shape[0])
    n_show = min(n_failing, 5)
    data: dict[str, list[float]] = {
        "subject": failing_indices[:n_show].tolist(),
    }
    if state_action_values is not None:
        for name, arr in state_action_values.items():
            data[name] = [float(arr[i]) for i in failing_indices[:n_show]]
    data["sum"] = failing_sums[:n_show].tolist()
    df = pd.DataFrame(data)
    return (
        f"{n_failing} of {sum_all.shape[0]} probability vectors do not sum to 1.0.\n"
        f"First failing entries:\n{df.to_string(index=False)}"
    )


def validate_regime_transitions_all_periods(
    *,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    internal_params: InternalParams,
    ages: AgeGrid,
) -> None:
    """Validate regime transition probabilities for all periods before solve.

    For each period (except the last), for each active non-terminal regime, evaluate
    the regime transition function on all grid points and check that inactive regimes
    receive zero probability.

    Args:
        internal_regimes: Immutable mapping of regime names to internal regimes.
        internal_params: Immutable mapping of regime names to flat parameter mappings.
        ages: Age grid for the model.

    Raises:
        InvalidRegimeTransitionProbabilitiesError: If any inactive regime receives
            positive transition probability.

    """
    for period in range(ages.n_periods - 1):
        active_regimes_next_period = tuple(
            name
            for name, regime in internal_regimes.items()
            if period + 1 in regime.active_periods
        )

        for name, internal_regime in internal_regimes.items():
            if period not in internal_regime.active_periods:
                continue
            if internal_regime.terminal:
                continue

            _validate_regime_transition_single(
                internal_regime=internal_regime,
                regime_params=internal_params[name],
                active_regimes_next_period=active_regimes_next_period,
                regime_name=name,
                period=period,
                ages=ages,
            )


def _validate_regime_transition_single(
    *,
    internal_regime: InternalRegime,
    regime_params: FlatRegimeParams,
    active_regimes_next_period: tuple[str, ...],
    regime_name: str,
    period: int,
    ages: AgeGrid,
) -> None:
    """Validate regime transition probabilities for a single regime and period.

    Evaluate the regime transition function on the Cartesian product of all grid
    variables it accepts, using `jax.vmap` for vectorised evaluation.

    """
    regime_transition_func = (
        internal_regime.regime_transition_probs.solve  # ty: ignore[unresolved-attribute]
    )

    state_action_space = internal_regime.state_action_space(
        regime_params=regime_params,
    )

    # Filter params to only those accepted by the transition function
    accepted_params = set(inspect.signature(regime_transition_func).parameters)
    filtered_params = {k: v for k, v in regime_params.items() if k in accepted_params}

    # Collect only grid variables the transition function accepts
    grids: dict[str, Array] = {
        k: v for k, v in state_action_space.states.items() if k in accepted_params
    } | {k: v for k, v in state_action_space.actions.items() if k in accepted_params}

    # Build flat Cartesian product and vmap over all combinations
    grid_var_names = list(grids.keys())
    grid_arrays = list(grids.values())

    if grid_arrays:
        mesh = jnp.meshgrid(*grid_arrays, indexing="ij")
        flat_arrays = [m.ravel() for m in mesh]

        def _call(
            *args: Array,
            _names: list[str] = grid_var_names,
            _params: dict = filtered_params,
            _func: object = regime_transition_func,
            _period: int = period,
            _age: ScalarFloat = ages.values[period],  # noqa: PD011
        ) -> MappingProxyType[str, Array]:
            kwargs = dict(zip(_names, args, strict=True))
            return _func(  # ty: ignore[call-non-callable]
                **kwargs, **_params, period=_period, age=_age
            )

        regime_transition_probs: MappingProxyType[str, Array] = jax.vmap(_call)(
            *flat_arrays
        )
        point = dict(zip(grid_var_names, flat_arrays, strict=True))
    else:
        regime_transition_probs: MappingProxyType[str, Array] = (  # ty: ignore[invalid-assignment]
            regime_transition_func(
                **filtered_params,
                period=period,
                age=ages.values[period],  # noqa: PD011
            )
        )
        point: dict[str, Array] = {}

    validate_regime_transition_probs(
        regime_transition_probs=regime_transition_probs,
        active_regimes_next_period=active_regimes_next_period,
        regime_name=regime_name,
        age=ages.values[period],  # noqa: PD011
        next_age=ages.values[period + 1],  # noqa: PD011
        state_action_values=MappingProxyType(point),
    )


@overload
def validate_transition_probs(
    *,
    probs: FloatND,
    model: Model,
    regime_name: str,
    state_name: str,
) -> None: ...


@overload
def validate_transition_probs(
    *,
    probs: FloatND,
    model: Model,
    regime_name: str,
) -> None: ...


def validate_transition_probs(
    *,
    probs: FloatND,
    model: Model,
    regime_name: str,
    state_name: str | None = None,
) -> None:
    """Validate a transition probability array for shape, values, and row sums.

    When ``state_name`` is provided, validate a state transition probability array.
    When omitted, validate a regime transition probability array.

    Args:
        probs: The transition probability array to validate.
        model: The LCM Model instance.
        regime_name: Name of the regime.
        state_name: Name of the state with a `MarkovTransition`. If ``None``,
            validate a regime transition instead.

    Raises:
        TypeError: If the transition is not a `MarkovTransition`.
        ValueError: If the shape is wrong, values are outside [0, 1], or rows
            don't sum to 1.

    """
    regime = model.regimes[regime_name]

    if state_name is not None:
        raw_transition = regime.state_transitions[state_name]
        if not isinstance(raw_transition, MarkovTransition):
            msg = (
                f"State '{state_name}' in regime '{regime_name}' is not a "
                f"MarkovTransition. Got {type(raw_transition).__name__}."
            )
            raise TypeError(msg)
        func = raw_transition.func
        all_grids = _build_all_grids(model, regime)
        n_outcomes = len(all_grids[state_name].categories)
    else:
        if not isinstance(regime.transition, MarkovTransition):
            msg = (
                f"Regime '{regime_name}' does not have a stochastic regime "
                f"transition. Got {type(regime.transition).__name__}."
            )
            raise TypeError(msg)
        func = regime.transition.func
        all_grids = _build_all_grids(model, regime)
        n_outcomes = len(model.regime_names_to_ids)

    indexing_params = _get_indexing_params(func)
    expected_shape = _build_expected_shape(
        indexing_params, n_outcomes, all_grids, model
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


def _build_all_grids(
    model: Model,
    regime: Regime,
) -> dict[str, DiscreteGrid]:
    """Collect all DiscreteGrid instances from model states and regime actions."""
    lookup: dict[str, DiscreteGrid] = {}
    for r in model.regimes.values():
        for state_name, grid in r.states.items():
            if isinstance(grid, DiscreteGrid) and state_name not in lookup:
                lookup[state_name] = grid
    for name, grid in regime.actions.items():
        if isinstance(grid, DiscreteGrid):
            lookup[name] = grid
    return lookup


def _get_indexing_params(func: Callable) -> list[str]:
    """Return indexing parameter names (all except ``probs_array``)."""
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())
    return [name for name in param_names if name != "probs_array"]


def _build_expected_shape(
    indexing_params: list[str],
    n_outcomes: int,
    all_grids: dict[str, DiscreteGrid],
    model: Model,
) -> tuple[int, ...]:
    """Compute expected shape for a transition probability array."""
    shape: list[int] = []
    for param_name in indexing_params:
        if param_name == "period":
            shape.append(model.n_periods)
        elif param_name in all_grids:
            shape.append(len(all_grids[param_name].categories))
        else:
            msg = (
                f"Cannot determine expected size for parameter '{param_name}'. "
                f"It is not 'period' and not a DiscreteGrid state or action."
            )
            raise ValueError(msg)
    shape.append(n_outcomes)
    return tuple(shape)
