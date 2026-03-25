import ast
import inspect
import textwrap
import warnings
from collections.abc import Callable, Mapping
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
    ScalarInt,
)

# Genuine circular import: model.py imports from this module at module level.
# Safe because Model is only used at runtime in validate_transition_probs,
# which is never called during module initialisation.
if TYPE_CHECKING:
    from lcm.model import Model


def validate_V(
    *, V_arr: Array, age: ScalarInt | ScalarFloat, regime_name: str | None = None
) -> None:
    """Validate the value function array for NaN values.

    This function checks the value function array for any NaN values. If any such values
    are found, we raise an `InvalidValueFunctionError`.

    Args:
        V_arr: The value function array to validate.
        age: The age for which the value function is being validated.
        regime_name: Name of the regime (for error messages).

    Raises:
        InvalidValueFunctionError: If the value function array contains NaN values.

    """
    if jnp.any(jnp.isnan(V_arr)):
        n_nan = int(jnp.sum(jnp.isnan(V_arr)))
        total = int(V_arr.size)
        regime_part = f" in regime '{regime_name}'" if regime_name else ""
        raise InvalidValueFunctionError(
            f"The value function array at age {age}{regime_part} contains NaN values "
            f"({n_nan} of {total} values are NaN). This may be due to various "
            "reasons:\n"
            "- The user-defined functions returned invalid values.\n"
            "- It is impossible to reach an active regime, resulting in NaN regime\n"
            "  transition probabilities."
        )


def validate_regime_transition_probs(
    *,
    regime_transition_probs: MappingProxyType[str, Array],
    active_regimes_next_period: tuple[str, ...],
    regime_name: str,
    age: ScalarInt | ScalarFloat,
    next_age: ScalarInt | ScalarFloat,
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
    last_period = ages.n_periods - 1
    non_terminal_active_at_last = [
        name
        for name, regime in internal_regimes.items()
        if not regime.terminal and last_period in regime.active_periods
    ]
    if non_terminal_active_at_last:
        raise InvalidRegimeTransitionProbabilitiesError(
            f"Non-terminal regime(s) {non_terminal_active_at_last} are active at the "
            f"last period (age {ages.exact_values[last_period]}). Non-terminal regimes "
            "must not be active at the last period because there is no next period to "
            "transition to. Adjust the 'active' function on these regimes to exclude "
            "the last age."
        )

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
        internal_regime.solve_functions.compute_regime_transition_probs
    )
    assert regime_transition_func is not None  # noqa: S101

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
            _age: ScalarInt | ScalarFloat = ages.values[period],  # noqa: PD011
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


def _get_indexing_params(func: Callable) -> list[str]:
    """Return indexing parameter names (all except `probs_array`)."""
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())
    return [name for name in param_names if name != "probs_array"]


def _validate_probs_array_indexing(func: Callable) -> None:
    """Check that `probs_array[...]` indexing order matches parameter order.

    Handles two cases reliably:

    - Single index: `probs_array[health]`
    - Tuple of bare names: `probs_array[period, health]`

    For computed indices (`probs_array[period - 1, health]`), variable aliasing,
    or multiple subscripts in different branches, a warning is emitted instead.

    Args:
        func: The transition function to inspect.

    Raises:
        ValueError: If bare-name indices don't match the expected parameter order.

    """
    sig = inspect.signature(func)
    if "probs_array" not in sig.parameters:
        return

    try:
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
    except OSError, TypeError:
        return

    expected = _get_indexing_params(func)
    func_name = getattr(func, "__name__", "<unknown>")

    subscripts = _collect_probs_array_subscripts(tree)

    if not subscripts:
        warnings.warn(
            f"Function '{func_name}' has a `probs_array` parameter but no "
            f"`probs_array[...]` subscript was found. Cannot validate indexing order.",
            UserWarning,
            stacklevel=2,
        )
        return

    if len(subscripts) > 1:
        warnings.warn(
            f"Function '{func_name}' has multiple `probs_array[...]` subscripts. "
            f"Cannot validate indexing order automatically.",
            UserWarning,
            stacklevel=2,
        )
        return

    index_names = _extract_bare_names(subscripts[0])

    if index_names is None:
        warnings.warn(
            f"Function '{func_name}' uses computed indices in "
            f"`probs_array[...]`. Cannot validate indexing order automatically.",
            UserWarning,
            stacklevel=2,
        )
        return

    if index_names != expected:
        msg = (
            f"In function '{func_name}', `probs_array` is indexed as "
            f"`probs_array[{', '.join(index_names)}]` but the expected order "
            f"(from the function signature) is "
            f"`probs_array[{', '.join(expected)}]`."
        )
        raise ValueError(msg)


def _collect_probs_array_subscripts(tree: ast.Module) -> list[ast.expr]:
    """Find all `probs_array[...]` subscript slice nodes in an AST."""
    return [
        node.slice
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "probs_array"
    ]


def _extract_bare_names(slice_node: ast.expr) -> list[str] | None:
    """Extract bare variable names from a subscript slice.

    Return ``None`` if any index element is not a bare `ast.Name` (e.g. a
    `BinOp` or `Call`).
    """
    if isinstance(slice_node, ast.Name):
        return [slice_node.id]

    if isinstance(slice_node, ast.Tuple):
        names: list[str] = []
        for elt in slice_node.elts:
            if not isinstance(elt, ast.Name):
                return None
            names.append(elt.id)
        return names

    return None


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
    state_name: str,
    target_regime_name: str,
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
    target_regime_name: str | None = None,
) -> None:
    """Validate a transition probability array for shape, values, and row sums.

    When ``state_name`` is provided, validate a state transition probability array.
    When omitted, validate a regime transition probability array.

    For per-target state transitions (where ``state_transitions[state_name]`` is a
    dict mapping target regime names to `MarkovTransition` instances), pass
    ``target_regime_name`` to select the specific transition to validate.

    Args:
        probs: The transition probability array to validate.
        model: The LCM Model instance.
        regime_name: Name of the regime.
        state_name: Name of the state with a `MarkovTransition`. If ``None``,
            validate a regime transition instead.
        target_regime_name: Target regime name for per-target state transitions.
            Required when the state transition is a per-target dict.

    Raises:
        TypeError: If the transition is not a `MarkovTransition`.
        ValueError: If the shape is wrong, values are outside [0, 1], or rows
            don't sum to 1.

    """
    regime = model.regimes[regime_name]

    if state_name is not None:
        raw_transition = regime.state_transitions[state_name]
        markov = _extract_markov_transition(
            raw_transition,
            state_name=state_name,
            regime_name=regime_name,
            target_regime_name=target_regime_name,
        )
        func = markov.func
        all_grids = _build_all_grids(regime)
        n_outcomes = len(all_grids[state_name].categories)
    else:
        if not isinstance(regime.transition, MarkovTransition):
            msg = (
                f"Regime '{regime_name}' does not have a stochastic regime "
                f"transition. Got {type(regime.transition).__name__}."
            )
            raise TypeError(msg)
        func = regime.transition.func
        all_grids = _build_all_grids(regime)
        n_outcomes = len(model.regime_names_to_ids)

    _validate_probs_array_indexing(func)

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


def _extract_markov_transition(
    raw_transition: object,
    *,
    state_name: str,
    regime_name: str,
    target_regime_name: str | None,
) -> MarkovTransition:
    """Extract a MarkovTransition from a raw transition, handling per-target dicts."""
    if isinstance(raw_transition, MarkovTransition):
        return raw_transition

    if isinstance(raw_transition, Mapping):
        if target_regime_name is None:
            targets = sorted(raw_transition.keys())
            msg = (
                f"State '{state_name}' in regime '{regime_name}' uses per-target "
                f"transitions. Pass target_regime_name to select one of: {targets}."
            )
            raise TypeError(msg)
        if target_regime_name not in raw_transition:
            msg = (
                f"Target regime '{target_regime_name}' not found in per-target "
                f"transitions for state '{state_name}' in regime '{regime_name}'. "
                f"Available targets: {sorted(raw_transition.keys())}."
            )
            raise ValueError(msg)
        entry = raw_transition[target_regime_name]  # ty: ignore[invalid-argument-type]
        if not isinstance(entry, MarkovTransition):
            msg = (
                f"Per-target transition for '{target_regime_name}' in state "
                f"'{state_name}' of regime '{regime_name}' is not a "
                f"MarkovTransition. Got {type(entry).__name__}."
            )
            raise TypeError(msg)
        return entry

    msg = (
        f"State '{state_name}' in regime '{regime_name}' is not a "
        f"MarkovTransition. Got {type(raw_transition).__name__}."
    )
    raise TypeError(msg)


def _build_all_grids(regime: Regime) -> dict[str, DiscreteGrid]:
    """Collect all DiscreteGrid instances from regime states and actions."""
    return {
        name: grid
        for name, grid in (*regime.states.items(), *regime.actions.items())
        if isinstance(grid, DiscreteGrid)
    }


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
