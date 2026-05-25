"""Optional `additional_targets` computation for `SimulationResult.to_dataframe`."""

import inspect
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from dags import concatenate_functions

from _lcm.engine import Regime
from _lcm.typing import FlatRegimeParams, RegimeName
from _lcm.utils.dispatchers import vmap_1d
from lcm.exceptions import InvalidAdditionalTargetsError
from lcm.typing import BoolND, FloatND, IntND, UserFunction


def _resolve_targets(
    *,
    additional_targets: list[str] | Literal["all"] | None,
    available_targets: list[str],
) -> list[str] | None:
    """Resolve and validate additional targets.

    Args:
        additional_targets: User-provided targets specification.
        available_targets: List of all available target names.

    Returns:
        Resolved list of target names, or None if no targets requested.

    Raises:
        InvalidAdditionalTargetsError: If any target is not available.

    """
    if additional_targets is None:
        return None
    if additional_targets == "all":
        return available_targets

    invalid = set(additional_targets) - set(available_targets)
    if invalid:
        raise InvalidAdditionalTargetsError(
            f"Targets {invalid} not found in any regime. "
            f"Available targets: {available_targets}"
        )

    return additional_targets


def _collect_all_available_targets(
    regimes: MappingProxyType[RegimeName, Regime],
) -> set[str]:
    """Collect all available target names across all regimes."""
    all_targets: set[str] = set()
    for regime in regimes.values():
        all_targets.update(_get_available_targets_for_regime(regime))
    return all_targets


def _get_available_targets_for_regime(regime: Regime) -> set[str]:
    """Get available target names for a single regime.

    Includes:

    - DAG function names from `simulate_functions.functions` (minus `H` and
      stochastic weight functions);
    - constraint names;
    - `derived_categoricals` declared on the regime, whether backed by a
      DAG function or by a `regime_params` constant — declaration is the
      explicit opt-in to exposing the name in `available_targets`.
    """
    excluded = {"H"} | _get_stochastic_weight_function_names(regime)
    sim = regime.simulate_functions
    return (
        {name for name in sim.functions if name not in excluded}
        | sim.constraints.keys()
        | set(regime.derived_categoricals)
    )


def _get_stochastic_weight_function_names(regime: Regime) -> set[str]:
    """Get names of internal stochastic weight functions.

    These are functions named `weight_{transition_name}` that return probability arrays
    for stochastic state transitions. They should not be exposed as available targets.
    """
    stochastic_transition_names = regime.simulate_functions.stochastic_transition_names
    return {
        f"weight_{target_regime}__{transition_name}"
        for target_regime, target_transitions in (
            regime.simulate_functions.transitions.items()
        )
        for transition_name in target_transitions
        if transition_name in stochastic_transition_names
    }


def _filter_targets_for_regime(
    *,
    targets: list[str],
    regime: Regime,
) -> list[str]:
    """Filter targets to only those available in this regime."""
    available = _get_available_targets_for_regime(regime)
    return [t for t in targets if t in available]


def _compute_targets(
    *,
    data: dict[str, FloatND | IntND | BoolND | np.ndarray | Sequence[str]],
    targets: list[str],
    regime: Regime,
    regime_params: FlatRegimeParams,
) -> dict[str, FloatND | IntND | BoolND]:
    """Compute additional targets for a regime.

    Targets split into two groups:

    - DAG-backed: the name appears in the regime's functions pool. Computed
      via `concatenate_functions` and vmapped over per-row data.
    - Constant-backed: the name is a declared `derived_categorical` whose
      value is resolvable from the regime's flat params (regime-level
      `fixed_params` or runtime `regime_params`). Broadcast as a single
      value across every row of this regime.
    """
    functions_pool = _build_functions_pool(regime)
    all_params = {**regime.resolved_fixed_params, **regime_params}

    dag_targets, constant_targets = _split_targets(
        targets=targets,
        functions_pool=functions_pool,
        derived_categoricals=regime.derived_categoricals,
        all_params=all_params,
    )

    result: dict[str, FloatND | IntND | BoolND] = {}

    if dag_targets:
        target_func = _create_target_function(
            functions_pool=functions_pool, targets=dag_targets
        )
        flat_param_names = frozenset(all_params.keys())
        variables = _get_function_variables(
            func=target_func, param_names=flat_param_names
        )
        vectorized_func = vmap_1d(func=target_func, variables=variables)
        kwargs = {k: jnp.asarray(v) for k, v in data.items() if k in variables}
        dag_result = vectorized_func(**all_params, **kwargs)
        result.update({k: jnp.squeeze(v) for k, v in dag_result.items()})

    if constant_targets:
        n_rows = len(data["period"])
        for name in constant_targets:
            value = jnp.asarray(_lookup_unqualified_param(name, all_params))
            result[name] = jnp.broadcast_to(value, (n_rows,))

    return result


def _split_targets(
    *,
    targets: list[str],
    functions_pool: dict[str, UserFunction],
    derived_categoricals: Mapping[str, object],
    all_params: Mapping[str, object],
) -> tuple[list[str], list[str]]:
    """Partition targets into DAG-backed and constant-backed groups.

    A target is constant-backed when it is a declared `derived_categorical`
    that resolves from the regime's flat params (under one of the
    qualified `{func}__{name}` keys the param-template builder emits)
    rather than from a DAG function in the regime's functions pool.
    Function backing takes precedence when both are present.
    """
    dag: list[str] = []
    constant: list[str] = []
    for t in targets:
        if t in functions_pool:
            dag.append(t)
        elif t in derived_categoricals and (
            _lookup_unqualified_param(t, all_params) is not None
        ):
            constant.append(t)
        else:
            dag.append(t)
    return dag, constant


def _lookup_unqualified_param(
    name: str, all_params: Mapping[str, object]
) -> object | None:
    """Find a regime-flat-params value by its unqualified name.

    The template builder emits keys like `{function}__{param}` for every
    function that consumes `{param}`. A `derived_categorical` named `X`
    that is supplied as a regime-level constant lands under one (or
    several) such qualified keys, all carrying the same broadcast value.
    Returning the first match suffices.
    """
    if name in all_params:
        return all_params[name]
    suffix = f"__{name}"
    for key, value in all_params.items():
        if key.endswith(suffix):
            return value
    return None


def _build_functions_pool(regime: Regime) -> dict[str, UserFunction]:
    """Build pool of available functions for target computation."""
    sim = regime.simulate_functions
    pool: dict[str, UserFunction] = {
        **{k: v for k, v in sim.functions.items() if k != "H"},
        **sim.constraints,
    }
    if sim.compute_regime_transition_probs is not None:
        pool["regime_transition_probs"] = sim.compute_regime_transition_probs
    return pool


def _create_target_function(
    *,
    functions_pool: dict[str, UserFunction],
    targets: list[str],
) -> UserFunction:
    """Create combined function for computing targets."""
    return concatenate_functions(
        functions=functions_pool,
        targets=targets,
        return_type="dict",
        set_annotations=True,
    )


def _get_function_variables(
    *,
    func: Callable[..., Any],
    param_names: frozenset[str],
) -> tuple[str, ...]:
    """Get variable names from signature, excluding flat param names."""
    return tuple(p for p in inspect.signature(func).parameters if p not in param_names)
