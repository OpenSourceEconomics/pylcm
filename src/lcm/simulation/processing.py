from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pandas as pd
from dags import concatenate_functions
from jax import Array

from lcm.dispatchers import vmap_1d

if TYPE_CHECKING:
    from lcm.interfaces import InternalRegime, InternalSimulationPeriodResults
    from lcm.typing import InternalUserFunction, ParamsDict


def process_simulated_data(
    results: dict[int, InternalSimulationPeriodResults],
    internal_regime: InternalRegime,
    params: ParamsDict,
    additional_targets: list[str] | None = None,
) -> dict[str, Array]:
    """Process and flatten the simulation results.

    This function produces a dict of arrays for each var with dimension (n_periods *
    n_initial_states,). The arrays are flattened, so that the resulting dictionary has a
    one-dimensional array for each variable. The length of this array is the number of
    periods times the number of initial states. The order of array elements is given by
    an outer level of periods and an inner level of initial states ids.

    Args:
        results: Dict with simulation results. Each dict contains the value,
            actions, and states for one period. Actions and states are stored in a
            nested dictionary.
        internal_regime: Internal regime instance.
        params: Parameters.
        additional_targets: List of additional targets to compute.

    Returns:
        Dict with processed simulation results. The keys are the variable names and the
        values are the flattened arrays, with dimension (n_periods * n_initial_states,).
        Additionally, the period variable is added.

    """
    n_initial_states = len(results[0].value)

    nan_array = jnp.full(n_initial_states, jnp.nan, dtype=jnp.float64)

    list_of_dicts = [
        {
            "period": jnp.full_like(d.subject_ids, period),
            "subject_ids": d.subject_ids,
            "value": d.value,
            **d.actions,
            **d.states,
        }
        for period, d in results.items()
    ]
    dict_of_lists = {
        key: [d.get(key, nan_array) for d in list_of_dicts]
        for key in list(list_of_dicts[0])
    }
    out = {key: jnp.concatenate(values) for key, values in dict_of_lists.items()}
    if additional_targets is not None:
        calculated_targets = _compute_targets(
            out,
            targets=additional_targets,
            functions=internal_regime.get_all_functions(),
            params=params,
        )
        out = {**out, **calculated_targets}

    return out


def as_panel(processed: dict[str, Array], n_periods: int) -> pd.DataFrame:
    """Convert processed simulation results to panel.

    Args:
        processed: Dict with processed simulation results.
        n_periods: Number of periods.

    Returns:
        Panel with the simulation results. The index is a multi-index with the first
        level corresponding to the initial state id and the second level corresponding
        to the period. The columns correspond to the value, and the action and state
        variables, and potentially auxiliary variables.

    """
    return pd.DataFrame(processed)


def _compute_targets(
    processed_results: dict[str, Array],
    targets: list[str],
    functions: dict[str, InternalUserFunction],
    params: ParamsDict,
) -> dict[str, Array]:
    """Compute targets.

    Args:
        processed_results: Dict with processed simulation results. Values must be
            one-dimensional arrays.
        targets: List of targets to compute.
        functions: Dict with functions that are used to compute targets.
        params: Dict with parameters.

    Returns:
        Dict with computed targets.

    """
    target_func = concatenate_functions(
        functions=functions,
        targets=targets,
        return_type="dict",
        set_annotations=True,
    )

    # get list of variables over which we want to vectorize the target function
    variables = tuple(
        p for p in list(inspect.signature(target_func).parameters) if p != "params"
    )

    target_func = vmap_1d(target_func, variables=variables)

    kwargs = {k: v for k, v in processed_results.items() if k in variables}
    return target_func(params=params, **kwargs)
