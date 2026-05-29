"""DataFrame assembly for `SimulationResult.to_dataframe`."""

from collections.abc import Sequence
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pandas as pd

from _lcm.engine import PeriodRegimeSimulationData
from _lcm.simulation.result_metadata import ResultMetadata
from _lcm.typing import ActionName, RegimeName, StateName
from lcm.ages import AgeGrid
from lcm.typing import BoolND, FloatND, IntND


def _create_flat_dataframe(
    *,
    raw_results: MappingProxyType[
        RegimeName, MappingProxyType[int, PeriodRegimeSimulationData]
    ],
    metadata: ResultMetadata,
    additional_targets: list[str] | None,
    ages: AgeGrid,
) -> pd.DataFrame:
    """Create a single flat DataFrame from all regime results.

    Requested `additional_targets` are read from each period's
    pre-computed `intermediates` (evaluated at the realised actions
    during simulation), so the dataframe needs neither the compiled
    `Regime` objects nor `flat_params`.
    """
    regime_dfs = [
        _process_regime(
            regime_name=name,
            regime_results=raw_results[name],
            regime_states=metadata.regime_to_states[name],
            regime_actions=metadata.regime_to_actions[name],
            additional_targets=additional_targets,
            ages=ages,
        )
        for name in metadata.regime_names
        if raw_results[name]
    ]

    return _assemble_dataframe(
        regime_dfs=regime_dfs,
        state_names=metadata.state_names,
        action_names=metadata.action_names,
    )


def _process_regime(
    *,
    regime_name: RegimeName,
    regime_results: MappingProxyType[int, PeriodRegimeSimulationData],
    regime_states: tuple[str, ...],
    regime_actions: tuple[str, ...],
    additional_targets: list[str] | None,
    ages: AgeGrid,
) -> pd.DataFrame:
    """Process results for a single regime into a DataFrame.

    Target columns named in `additional_targets` are pulled from each
    period's `intermediates`; a regime that does not expose a requested
    target simply omits that column, and `pd.concat` fills it with NaN
    when the per-regime frames are combined.
    """
    period_dicts = [
        _extract_period_data(
            result=result,
            period=period,
            regime_states=regime_states,
            regime_actions=regime_actions,
            additional_targets=additional_targets,
        )
        for period, result in regime_results.items()
    ]

    data: dict[str, np.ndarray | FloatND | IntND | BoolND | Sequence[str]] = dict(
        _concatenate_and_filter(period_dicts)
    )

    data["age"] = ages.values[data["period"]]  # noqa: PD011
    data["regime_name"] = [regime_name] * len(data["period"])

    return pd.DataFrame(data)


def _extract_period_data(
    *,
    result: PeriodRegimeSimulationData,
    period: int,
    regime_states: tuple[str, ...],
    regime_actions: tuple[str, ...],
    additional_targets: list[str] | None,
) -> dict[str, np.ndarray | FloatND | IntND | BoolND]:
    """Extract data from a single period's simulation results."""
    data: dict[str, np.ndarray | FloatND | IntND | BoolND] = {
        "subject_id": jnp.arange(len(result.in_regime), dtype=jnp.int32),
        "period": jnp.full_like(result.in_regime, period, dtype=jnp.int32),
        "_in_regime": result.in_regime,
        "value": result.V_arr,
    }

    for name in regime_states:
        if name in result.states:
            data[name] = result.states[name]

    for name in regime_actions:
        if name in result.actions:
            data[name] = result.actions[name]

    for name in additional_targets or ():
        if name in result.intermediates:
            data[name] = result.intermediates[name]

    return data


def _concatenate_and_filter(
    period_dicts: list[dict[str, np.ndarray | FloatND | IntND | BoolND]],
) -> dict[str, np.ndarray]:
    """Concatenate period data on host and filter to in-regime subjects.

    Walks `period_dicts` one period at a time. For each leaf the
    transfer goes through `_to_host`, which falls back to `np.asarray`
    for single-device arrays and uses shard iteration for sharded ones
    (each shard transfers its local data independently, side-stepping
    the implicit XLA all-gather that a `np.asarray` on a sharded array
    would trigger). After each period's leaves are on host, that
    period's dict is cleared so the device buffers become
    GC-eligible — peak device residency is one per-period dict's
    leaves, regardless of how many periods the result spans.

    The function mutates `period_dicts` (every dict is emptied on
    completion). The caller treats the list as consumed.
    """
    keys = [k for k in period_dicts[0] if k != "_in_regime"]

    mask_chunks: list[np.ndarray] = []
    host_chunks: dict[str, list[np.ndarray]] = {key: [] for key in keys}

    for d in period_dicts:
        mask_chunks.append(_to_host(d["_in_regime"]).astype(bool))
        for key in keys:
            host_chunks[key].append(_to_host(d[key]))
        d.clear()

    mask = np.concatenate(mask_chunks)
    del mask_chunks

    result: dict[str, np.ndarray] = {}
    for key in keys:
        column = np.concatenate(host_chunks.pop(key))
        result[key] = column[mask]
        del column

    return result


def _to_host(value: np.ndarray | FloatND | IntND | BoolND) -> np.ndarray:
    """Copy a jax.Array (or numpy array) to a host-resident `np.ndarray`.

    For a value with at most one addressable shard the call collapses
    to `np.asarray`, which on a single-device jax.Array is a direct
    D2H copy. For a sharded value the loop walks
    `addressable_shards`, pulls each shard's local data to host, and
    drops it into the right slice of a host-allocated output via
    `shard.index`. This skips XLA's implicit all-gather into a
    contiguous device buffer — the contiguous reassembly happens in
    host memory, where the multi-GiB output is cheap.
    """
    shards = getattr(value, "addressable_shards", ())
    if len(shards) <= 1:
        return np.asarray(value)
    out = np.empty(value.shape, dtype=value.dtype)
    for shard in shards:
        out[shard.index] = np.asarray(shard.data)
    return out


def _assemble_dataframe(
    *,
    regime_dfs: list[pd.DataFrame],
    state_names: list[StateName],
    action_names: list[ActionName],
) -> pd.DataFrame:
    """Combine regime DataFrames, add missing columns, reorder, and sort."""
    if not regime_dfs:
        return _empty_dataframe(state_names=state_names, action_names=action_names)

    df = pd.concat(regime_dfs, ignore_index=True)
    df = _add_missing_columns(df=df, state_names=state_names, action_names=action_names)
    df = _reorder_columns(df=df, state_names=state_names, action_names=action_names)
    return df.sort_values(["subject_id", "period"]).reset_index(drop=True)


def _empty_dataframe(
    *,
    state_names: list[StateName],
    action_names: list[ActionName],
) -> pd.DataFrame:
    """Create empty DataFrame with correct columns."""
    columns = ["subject_id", "period", "regime_name", "value"]
    columns.extend(state_names)
    columns.extend(action_names)
    return pd.DataFrame(columns=pd.Index(columns))


def _add_missing_columns(
    *,
    df: pd.DataFrame,
    state_names: list[StateName],
    action_names: list[ActionName],
) -> pd.DataFrame:
    """Add NaN columns for states/actions not present in DataFrame."""
    for name in state_names:
        if name not in df.columns:
            df[name] = float("nan")
    for name in action_names:
        if name not in df.columns:
            df[name] = float("nan")
    return df


def _reorder_columns(
    *,
    df: pd.DataFrame,
    state_names: list[StateName],
    action_names: list[ActionName],
) -> pd.DataFrame:
    """Reorder columns: id, period, regime_name, value, states, actions, rest."""
    base = ["subject_id", "period", "regime_name", "value"]
    known = set(base) | set(state_names) | set(action_names)
    rest = [c for c in df.columns if c not in known]
    return df[base + state_names + action_names + rest]


def _convert_to_categorical(
    *,
    df: pd.DataFrame,
    metadata: ResultMetadata,
) -> pd.DataFrame:
    """Convert discrete columns to pandas Categorical dtype with string labels.

    Converts:
    - regime_name column: uses regime_names as categories
    - discrete state/action columns: uses categories from simulation metadata

    """
    df = df.copy()

    df["regime_name"] = pd.Categorical(
        df["regime_name"], categories=metadata.regime_names
    )

    for var_name, merged_categories in metadata.discrete_categories.items():
        if var_name not in df.columns:
            continue

        needs_remap = any(
            metadata.regime_discrete_categories.get((rn, var_name)) != merged_categories
            for rn in metadata.regime_names
            if (rn, var_name) in metadata.regime_discrete_categories
        )

        if needs_remap:
            df[var_name] = _remap_codes_per_regime(
                df=df,
                var_name=var_name,
                merged_categories=merged_categories,
                ordered=metadata.discrete_ordered[var_name],
                metadata=metadata,
            )
        else:
            df[var_name] = _codes_to_categorical(
                codes=df[var_name],
                categories=merged_categories,
                ordered=metadata.discrete_ordered[var_name],
            )

    return df


def _remap_codes_per_regime(
    *,
    df: pd.DataFrame,
    var_name: str,
    merged_categories: tuple[str, ...],
    ordered: bool,
    metadata: ResultMetadata,
) -> pd.Categorical:
    """Map per-regime integer codes to labels, then build a merged Categorical.

    When regimes define different categories for the same variable, the raw integer
    codes in the DataFrame correspond to each regime's own category ordering. This
    function converts per-regime codes to string labels, then wraps them in a
    Categorical with the merged category set.

    """
    labels = pd.Series(pd.NA, index=df.index, dtype="string")

    for regime_name in metadata.regime_names:
        regime_cats = metadata.regime_discrete_categories.get((regime_name, var_name))
        if regime_cats is None:
            continue

        mask = df["regime_name"] == regime_name
        if not mask.any():
            continue

        codes_in_regime = df.loc[mask, var_name]
        valid = codes_in_regime.notna()
        int_codes = codes_in_regime[valid].astype(int)
        mapped = int_codes.map(dict(enumerate(regime_cats))).to_numpy()
        labels[mask & valid] = mapped

    return pd.Categorical(labels, categories=list(merged_categories), ordered=ordered)


def _codes_to_categorical(
    *,
    codes: pd.Series,
    categories: tuple[str, ...],
    ordered: bool = False,
) -> pd.Categorical | pd.Series:
    """Convert integer codes to Categorical, handling NaN and out-of-range values.

    If values are outside the valid category range, returns the original series
    unchanged to avoid data loss.

    """
    codes_array = codes.to_numpy()
    has_nan = pd.isna(codes_array)
    n_categories = len(categories)

    valid_values = codes_array[~has_nan]
    if len(valid_values) > 0:
        int_values = valid_values.astype(int)
        if int_values.min() < 0 or int_values.max() >= n_categories:
            return codes

    if has_nan.any():
        int_codes = [-1 if pd.isna(c) else int(c) for c in codes_array]
        return pd.Categorical.from_codes(
            int_codes,
            categories=pd.Index(categories),
            ordered=ordered,
        )

    return pd.Categorical.from_codes(
        codes_array.astype(int),
        categories=pd.Index(categories),
        ordered=ordered,
    )
