from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pandas as pd
from dags import concatenate_functions
from jax import Array

from lcm.dispatchers import vmap_1d

if TYPE_CHECKING:
    from lcm.interfaces import InternalRegime, SimulationResults
    from lcm.typing import InternalUserFunction, ParamsDict, RegimeName


def process_simulated_data(
    results: dict[int, SimulationResults],
    internal_regime: InternalRegime,
    params: ParamsDict,
    additional_targets: dict[RegimeName, list[str]] | None = None,
) -> pd.DataFrame:
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
        DataFrame with processed simulation results. The columns are the variable names
        and their values are the flattened arrays, with dimension (n_periods *
        n_initial_states,). Additionally, the period variable is added.

    """
    n_initial_states = len(results[0].V_arr)

    nan_array = jnp.full(n_initial_states, jnp.nan, dtype=jnp.float64)

    list_of_dicts = [
        {
            "period": jnp.full(n_initial_states, period),
            "subject_id": jnp.arange(n_initial_states),
            "in_regime": d.in_regime,
            "value": d.V_arr,
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
    if additional_targets is not None and internal_regime.name in additional_targets:
        functions_pool = {
            **internal_regime.functions,
            **internal_regime.constraints,
            "utility": internal_regime.utility,
            "regime_transition_probs": internal_regime.regime_transition_probs.simulate,
        }

        calculated_targets = _compute_targets(
            out,
            targets=additional_targets[internal_regime.name],
            # Have to ignore the type error here because regime_transition_probs does
            # not conform to InternalUserFunction protocol, but fixing that would
            # require significant refactoring.
            functions=functions_pool,  # type: ignore[arg-type]
            params=params[internal_regime.name],
        )
        out = {**out, **calculated_targets}
    df = pd.DataFrame(out)
    return df[df["in_regime"] == 1].drop("in_regime", axis=1)


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
