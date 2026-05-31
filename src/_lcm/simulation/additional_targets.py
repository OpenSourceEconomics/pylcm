"""Optional `additional_targets` computation for `SimulationResult.to_dataframe`."""

import inspect
from collections.abc import Callable, Sequence
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
    """Get available target names for a single regime."""
    excluded = {"H"} | _get_stochastic_weight_function_names(regime)
    sim = regime.simulate_functions
    return {
        name for name in sim.functions if name not in excluded
    } | sim.constraints.keys()


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
    data: dict[str, np.ndarray | FloatND | IntND | BoolND | Sequence[str]],
    targets: list[str],
    regime: Regime,
    regime_params: FlatRegimeParams,
    subject_batch_size: int | None = None,
) -> dict[str, FloatND | IntND | BoolND | np.ndarray]:
    """Compute additional targets for a regime.

    The target DAG is vmapped over the regime's in-regime subject-period rows. When
    `subject_batch_size` is set, the rows are processed in chunks and each chunk's
    outputs are pulled to host before the next runs, so the fused-DAG device
    workspace is bounded by the chunk rather than the full population. Values are
    identical to the single-pass evaluation.
    """
    functions_pool = _build_functions_pool(regime)
    target_func = _create_target_function(
        functions_pool=functions_pool, targets=targets
    )
    # Merge resolved fixed params with runtime params so that the target
    # function (built from raw user functions) receives all needed arguments.
    all_params = {**regime.resolved_fixed_params, **regime_params}
    flat_param_names = frozenset(all_params.keys())
    variables = _get_function_variables(func=target_func, param_names=flat_param_names)
    vectorized_func = vmap_1d(func=target_func, variables=variables)

    inputs = {k: v for k, v in data.items() if k in variables}
    n_rows = len(data["period"])

    if subject_batch_size is None or subject_batch_size >= n_rows:
        kwargs = {k: jnp.asarray(v) for k, v in inputs.items()}
        result = vectorized_func(**all_params, **kwargs)
        return {k: jnp.squeeze(v) for k, v in result.items()}

    # Slice the (host-resident) inputs and move only one chunk to the device at a
    # time. Squeeze the *concatenated* result, never a chunk — an uneven final
    # chunk of one row would otherwise lose its row axis.
    chunk_outputs: list[dict[str, np.ndarray]] = []
    for start in range(0, n_rows, subject_batch_size):
        stop = min(start + subject_batch_size, n_rows)
        chunk_kwargs = {k: jnp.asarray(v[start:stop]) for k, v in inputs.items()}
        chunk_result = vectorized_func(**all_params, **chunk_kwargs)
        chunk_outputs.append({k: np.asarray(v) for k, v in chunk_result.items()})

    result: dict[str, FloatND | IntND | BoolND | np.ndarray] = {}
    for name in chunk_outputs[0]:
        per_chunk = [out[name] for out in chunk_outputs]
        # A target with no per-subject variable (a constant, e.g. a terminal-regime
        # `utility`) yields the same scalar from every chunk; keep one as a 0-d
        # jax.Array to match the single-pass dtype rather than concatenating scalars.
        if per_chunk[0].ndim == 0:
            result[name] = jnp.asarray(per_chunk[0])
        else:
            result[name] = np.squeeze(np.concatenate(per_chunk))
    return result


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
