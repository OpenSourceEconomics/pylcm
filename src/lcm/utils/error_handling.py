import ast
import inspect
import logging
import textwrap
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, overload

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
from lcm.interfaces import InternalRegime, StateActionSpace
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
    *,
    V_arr: Array,
    age: ScalarInt | ScalarFloat,
    regime_name: str | None = None,
    partial_solution: object = None,
    compute_intermediates: Callable | None = None,
    state_action_space: StateActionSpace | None = None,
    next_regime_to_V_arr: MappingProxyType | None = None,
    internal_params: Mapping | None = None,
) -> None:
    """Validate the value function array for NaN values.

    When `compute_intermediates` is provided, NaN detection triggers a
    diagnostic run of the (already productmapped + JIT-compiled) closure to
    pinpoint which intermediate (U, F, E[V], Q) contains NaN.

    Args:
        V_arr: The value function array to validate.
        age: The age for which the value function is being validated.
        regime_name: Name of the regime (for error messages).
        partial_solution: Value function arrays for periods completed before
            the error. Attached to the exception for debug snapshots.
        compute_intermediates: Productmap + reduction closure (already
            JIT-compiled by `_build_compute_intermediates_per_period`)
            for the regime/period whose V array is being validated.
        state_action_space: StateActionSpace for the current regime/period.
        next_regime_to_V_arr: Next-period value function arrays.
        internal_params: Flat regime parameters.

    Raises:
        InvalidValueFunctionError: If the value function array contains NaN values.

    """
    if not jnp.any(jnp.isnan(V_arr)):
        return

    n_nan = int(jnp.sum(jnp.isnan(V_arr)))
    total = int(V_arr.size)
    regime_part = f" in regime '{regime_name}'" if regime_name else ""
    all_nan = n_nan == total
    fraction_hint = "all" if all_nan else f"{n_nan} of {total}"
    exc = InvalidValueFunctionError(
        f"Value function at age {age}{regime_part}: {fraction_hint} values "
        f"are NaN.\n\n"
        "NaN propagates through Q = U + beta * E[V]. Common causes:\n"
        "- A missing feasibility constraint (e.g. negative leisure passed "
        "to a fractional exponent).\n"
        "- A regime parameter is NaN.\n"
        "- The utility function returned NaN (e.g. log of a non-positive "
        "argument).\n"
        "- The regime transition function returned NaN probabilities "
        "(e.g. from a NaN survival probability or a NaN fixed param).\n"
        "- A per-target state_transitions dict omits a reachable target "
        "(non-zero transition probability to an incomplete target).\n\n"
        "To diagnose, re-solve with debug logging:\n\n"
        '  model.solve(params=params, log_level="debug", '
        'log_path="./debug/")\n\n'
        "The snapshot saved on failure contains diagnostics that pinpoint "
        "where NaN enters (U, E[V], or regime transitions). See the "
        "debugging guide:\n"
        "https://pylcm.readthedocs.io/en/latest/user_guide/debugging/"
    )
    exc.partial_solution = partial_solution

    if compute_intermediates is not None and state_action_space is not None:
        try:
            _enrich_with_diagnostics(
                exc=exc,
                compute_intermediates=compute_intermediates,
                state_action_space=state_action_space,
                next_regime_to_V_arr=next_regime_to_V_arr,
                internal_params=internal_params,
                regime_name=regime_name or "",
                age=float(age),
            )
        except Exception:  # noqa: BLE001
            logging.getLogger("lcm").warning(
                "Diagnostic enrichment failed; raising original NaN error",
                exc_info=True,
            )

    raise exc


def _enrich_with_diagnostics(
    *,
    exc: InvalidValueFunctionError,
    compute_intermediates: Callable,
    state_action_space: StateActionSpace,
    next_regime_to_V_arr: MappingProxyType | None,
    internal_params: Mapping | None,
    regime_name: str,
    age: float,
) -> None:
    """Run diagnostic intermediates and attach summary to exception.

    `compute_intermediates` is productmap-wrapped over the full state-action
    space (same structure as `max_Q_over_a`) and fused with an on-device
    reduction step in a single JIT region — so the full-shape U/F/E/Q
    arrays never materialise in host-visible memory. It returns a flat
    dict of scalars + per-dimension vectors.

    Args:
        exc: The `InvalidValueFunctionError` to enrich with a diagnostic
            note and a `diagnostics` attribute.
        compute_intermediates: Fused productmap + reduction closure for the
            regime/period whose V array contained NaN.
        state_action_space: State-action space for the regime/period; used
            to build call kwargs and label per-dimension reductions.
        next_regime_to_V_arr: Immutable mapping of next-period value
            function arrays per regime (or `None`).
        internal_params: Optional mapping of flat regime parameter values.
        regime_name: Name of the regime whose V array failed validation.
        age: Age at which the V array failed validation.

    """
    all_names = (*state_action_space.state_names, *state_action_space.action_names)
    state_action_kwargs: dict[str, Any] = {
        **state_action_space.states,
        **state_action_space.actions,
    }
    # Drop any flat regime params that collide with state/action names so
    # they don't silently overwrite the grids.
    param_kwargs = (
        {k: v for k, v in internal_params.items() if k not in state_action_kwargs}
        if internal_params
        else {}
    )
    call_kwargs: dict[str, Any] = {
        **state_action_kwargs,
        "next_regime_to_V_arr": next_regime_to_V_arr,
        **param_kwargs,
    }

    reductions = compute_intermediates(**call_kwargs)
    exc.diagnostics = _summarize_diagnostics(
        reductions=reductions,
        variable_names=all_names,
        regime_name=regime_name,
        age=age,
    )
    exc.add_note(_format_diagnostic_summary(exc.diagnostics))


def _summarize_diagnostics(
    *,
    reductions: Mapping[str, Any],
    variable_names: tuple[str, ...],
    regime_name: str,
    age: float,
) -> dict[str, Any]:
    """Restructure the flat reduction pytree into the summary dict shape.

    Pure host-side — no device computation. Consumes the output of the
    fused compute-and-reduce function built in
    `_build_compute_intermediates_per_period`.

    Args:
        reductions: Flat mapping of reduction keys (`{metric}_overall`,
            `{metric}_by_{name}`, and `regime_probs`) to device arrays.
        variable_names: Tuple of state + action names in the order that
            matches the productmap axes.
        regime_name: Name of the regime for the summary header.
        age: Age for the summary header.

    Returns:
        Dict with per-metric `"overall"` and `"by_dim"` entries plus a
        `"regime_probs"` mapping, suitable for `_format_diagnostic_summary`.

    """
    summary: dict[str, Any] = {"regime_name": regime_name, "age": age}

    for key_out, key_in in [
        ("U_nan_fraction", "U_nan"),
        ("E_nan_fraction", "E_nan"),
        ("Q_nan_fraction", "Q_nan"),
        ("F_feasible_fraction", "F_feasible"),
    ]:
        by_dim: dict[str, list[float]] = {}
        for name in variable_names:
            k = f"{key_in}_by_{name}"
            if k in reductions:
                by_dim[name] = reductions[k].tolist()
        summary[key_out] = {
            "overall": float(reductions[f"{key_in}_overall"]),
            "by_dim": by_dim,
        }

    summary["regime_probs"] = {
        k: float(v) for k, v in reductions["regime_probs"].items()
    }
    return summary


def _format_diagnostic_summary(summary: dict[str, Any]) -> str:
    """Format diagnostic summary for exception note.

    Args:
        summary: Nested summary dict as produced by `_summarize_diagnostics`.

    Returns:
        Human-readable multi-line string suitable for `Exception.add_note`.

    """
    lines = [
        f"\nDiagnostics for regime '{summary['regime_name']}' at age {summary['age']}:",
    ]

    u_frac = summary.get("U_nan_fraction", {}).get("overall", 0)
    e_frac = summary.get("E_nan_fraction", {}).get("overall", 0)
    f_feas = summary.get("F_feasible_fraction", {}).get("overall", 0)
    lines.append(f"  F: {f_feas:.4f} feasible")
    lines.append(
        f"  Among feasible state-action pairs:  "
        f"U: {u_frac:.4f} NaN  |  E[V]: {e_frac:.4f} NaN"
    )

    probs = summary.get("regime_probs", {})
    if probs:
        prob_parts = [f"{t}: {p:.4f}" for t, p in probs.items()]
        lines.append(f"  Regime probs: {' | '.join(prob_parts)}")

    for label, key in (("U", "U_nan_fraction"), ("E[V]", "E_nan_fraction")):
        info = summary.get(key, {})
        frac = info.get("overall", 0)
        by_dim = info.get("by_dim", {})
        if frac > 0 and by_dim:
            lines.append(
                f"  {label} NaN fraction by state (among feasible state-action pairs):"
            )
            for dim_name, values in by_dim.items():
                formatted = ", ".join(f"{v:.2f}" for v in values)
                lines.append(f"    {dim_name:24s} [{formatted}]")

    return "\n".join(lines)


def validate_regime_transition_probs(
    *,
    regime_transition_probs: MappingProxyType[str, Array],
    active_regimes_next_period: tuple[RegimeName, ...],
    regime_name: RegimeName,
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
            sum_all=sum_all,
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
    *,
    sum_all: Array,
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
                internal_regimes=internal_regimes,
                regime_params=internal_params[name],
                active_regimes_next_period=active_regimes_next_period,
                regime_name=name,
                period=period,
                ages=ages,
            )


def _validate_regime_transition_single(
    *,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    regime_params: FlatRegimeParams,
    active_regimes_next_period: tuple[RegimeName, ...],
    regime_name: RegimeName,
    period: int,
    ages: AgeGrid,
) -> None:
    """Validate regime transition probabilities for a single regime and period.

    Evaluate the regime transition function on the Cartesian product of all grid
    variables it accepts, using `jax.vmap` for vectorised evaluation.

    """
    internal_regime = internal_regimes[regime_name]
    # Non-None guaranteed: only called for non-terminal regimes
    regime_transition_func = (
        internal_regime.solve_functions.compute_regime_transition_probs
    )

    state_action_space = internal_regime.state_action_space(
        regime_params=regime_params,
    )

    # Filter params to only those accepted by the transition function
    accepted_params = set(inspect.signature(regime_transition_func).parameters)  # ty: ignore[invalid-argument-type]
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
            regime_transition_func(  # ty: ignore[call-non-callable]
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

    _validate_no_reachable_incomplete_targets(
        internal_regimes=internal_regimes,
        regime_transition_probs=regime_transition_probs,
        active_regimes_next_period=active_regimes_next_period,
        regime_name=regime_name,
        age=ages.values[period],  # noqa: PD011
    )


def _validate_no_reachable_incomplete_targets(
    *,
    internal_regimes: MappingProxyType[RegimeName, InternalRegime],
    regime_transition_probs: MappingProxyType[str, Array],
    active_regimes_next_period: tuple[RegimeName, ...],
    regime_name: RegimeName,
    age: ScalarInt | ScalarFloat,
) -> None:
    """Check that targets with incomplete stochastic transitions are unreachable.

    A target is "incomplete" from the source regime if the source's
    `transitions[target_regime_name]` does not cover all of the target's
    stochastic state needs. Such targets must have zero transition
    probability, otherwise the continuation value cannot be computed. This
    includes self-transitions (regime reaches itself): omitting the
    self-entry in a per-target dict is a common user error.

    """
    solve_functions = internal_regimes[regime_name].solve_functions
    transitions = solve_functions.transitions
    stochastic_names = solve_functions.stochastic_transition_names

    for target_regime_name in active_regimes_next_period:
        target_regime = internal_regimes[target_regime_name]
        target_state_names = tuple(target_regime.variable_info.query("is_state").index)
        needs = {
            f"next_{s}" for s in target_state_names if f"next_{s}" in stochastic_names
        }
        if not needs:
            continue
        if target_regime_name in transitions and needs.issubset(
            transitions[target_regime_name]
        ):
            continue
        if not jnp.any(regime_transition_probs[target_regime_name] > 0):
            continue
        missing = sorted(needs - set(transitions.get(target_regime_name, {})))
        if target_regime_name not in transitions:
            missing = sorted(f"next_{s}" for s in target_state_names)
        raise InvalidRegimeTransitionProbabilitiesError(
            f"Regime '{regime_name}' at age {age} has positive transition "
            f"probability to '{target_regime_name}', but '{regime_name}' "
            f"does not provide state transition(s) for: {missing}. Extend "
            f"`state_transitions` in '{regime_name}' to cover "
            f"'{target_regime_name}' (via a per-target dict if the "
            f"transition differs by target), or ensure "
            f"'{target_regime_name}' is unreachable."
        )


def _get_func_indexing_params(
    *,
    func: Callable,
    array_param_name: str,
) -> list[str]:
    """Return indexing parameter names by inspecting array subscripts.

    Inspect `array_param_name`'s subscripts in the function source for
    `param[x, y, ...]` patterns where all index elements are bare names
    that are also function parameters.

    Args:
        func: The function to inspect.
        array_param_name: The array parameter whose subscripts to inspect.

    Returns:
        List of indexing parameter names, or empty list if no array
        subscripts are found (scalar function).

    Raises:
        TypeError: If the function source cannot be inspected (e.g., lambda).
        ValueError: If computed indices are used instead of bare names.

    """
    func_name = getattr(func, "__name__", "<unknown>")

    if func_name == "<lambda>":
        msg = "Cannot inspect lambda functions. Define a named function instead."
        raise TypeError(msg)

    try:
        source = textwrap.dedent(inspect.getsource(func))
    except OSError, TypeError:
        msg = (
            f"Cannot inspect source of '{func_name}'. "
            f"Define a named function instead of a lambda."
        )
        raise TypeError(msg) from None

    tree = ast.parse(source)
    sig = inspect.signature(func)
    param_names = set(sig.parameters)

    subscripts = _collect_subscripts(tree=tree, param_name=array_param_name)
    if not subscripts:
        return []

    if len(subscripts) > 1:
        msg = (
            f"Function '{func_name}' has multiple `{array_param_name}[...]` "
            f"subscripts. Use exactly one subscript so the indexing order "
            f"can be determined unambiguously."
        )
        raise ValueError(msg)

    names = _extract_bare_names(subscripts[0])

    if names is not None and all(n in param_names for n in names):
        return names

    if names is not None:
        non_params = [n for n in names if n not in param_names]
        msg = (
            f"Function '{func_name}' indexes `{array_param_name}` with names "
            f"{non_params} that are not function parameters. All subscript "
            f"indices must be function parameters (not aliased variables)."
        )
        raise ValueError(msg)

    if _slice_references_params(slice_node=subscripts[0], param_names=param_names):
        msg = (
            f"Function '{func_name}' uses computed indices in "
            f"`{array_param_name}[...]`. Use bare parameter names as indices. "
            f"If you need a computed index, extract it into a separate "
            f"function in the regime (e.g., "
            f"`adjusted_period(period): return period - 1`) "
            f"and use the function output as the index."
        )
        raise ValueError(msg)

    return []


def _slice_references_params(
    *,
    slice_node: ast.expr,
    param_names: set[str],
) -> bool:
    """Check if any `ast.Name` in the slice is a function parameter.

    Args:
        slice_node: AST node for the subscript slice.
        param_names: Set of function parameter names.

    Returns:
        `True` if any bare name in the slice matches a parameter.

    """
    return any(
        isinstance(node, ast.Name) and node.id in param_names
        for node in ast.walk(slice_node)
    )


def _collect_subscripts(
    *,
    tree: ast.Module,
    param_name: str,
) -> list[ast.expr]:
    """Find all `param_name[...]` subscript slice nodes in an AST.

    Args:
        tree: Parsed AST module.
        param_name: Name of the parameter to search for subscripts.

    Returns:
        List of AST slice nodes from matching subscripts.

    """
    return [
        node.slice
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == param_name
    ]


def _extract_bare_names(slice_node: ast.expr) -> list[str] | None:
    """Extract bare variable names from a subscript slice.

    Return `None` if any index element is not a bare `ast.Name` (e.g. a
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

    When `state_name` is provided, validate a state transition probability array.
    When omitted, validate a regime transition probability array.

    For per-target state transitions (where `state_transitions[state_name]` is a
    dict mapping target regime names to `MarkovTransition` instances), pass
    `target_regime_name` to select the specific transition to validate.

    Args:
        probs: The transition probability array to validate.
        model: The LCM Model instance.
        regime_name: Name of the regime.
        state_name: Name of the state with a `MarkovTransition`. If `None`,
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
            raw_transition=raw_transition,
            state_name=state_name,
            regime_name=regime_name,
            target_regime_name=target_regime_name,
        )
        func = markov.func
        grids = _build_grids(regime)
        n_outcomes = len(grids[state_name].categories)
    else:
        if not isinstance(regime.transition, MarkovTransition):
            msg = (
                f"Regime '{regime_name}' does not have a stochastic regime "
                f"transition. Got {type(regime.transition).__name__}."
            )
            raise TypeError(msg)
        func = regime.transition.func
        grids = _build_grids(regime)
        n_outcomes = len(model.regime_names_to_ids)

    indexing_params = _get_func_indexing_params(
        func=func, array_param_name="probs_array"
    )

    # Cross-check subscript order against signature order
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


def _extract_markov_transition(
    *,
    raw_transition: object,
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


def _build_grids(regime: Regime) -> dict[str, DiscreteGrid]:
    """Collect all DiscreteGrid instances from regime states and actions."""
    return {
        name: grid
        for name, grid in (*regime.states.items(), *regime.actions.items())
        if isinstance(grid, DiscreteGrid)
    }


def _build_expected_shape(
    *,
    indexing_params: list[str],
    n_outcomes: int,
    grids: dict[str, DiscreteGrid],
    model: Model,
) -> tuple[int, ...]:
    """Compute expected shape for a transition probability array."""
    shape: list[int] = []
    for param_name in indexing_params:
        if param_name == "period":
            shape.append(model.n_periods)
        elif param_name in grids:
            shape.append(len(grids[param_name].categories))
        else:
            msg = (
                f"Cannot determine expected size for parameter '{param_name}'. "
                f"It is not 'period' and not a DiscreteGrid state or action."
            )
            raise ValueError(msg)
    shape.append(n_outcomes)
    return tuple(shape)
