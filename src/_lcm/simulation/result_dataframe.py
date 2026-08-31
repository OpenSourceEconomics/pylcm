"""DataFrame assembly for `SimulationResult.to_dataframe`."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pandas as pd

from _lcm.engine import PeriodRegimeSimulationData, Regime
from _lcm.simulation.additional_targets import (
    _compute_targets,
    _filter_targets_for_regime,
)
from _lcm.simulation.result_metadata import ResultMetadata
from _lcm.typing import ActionName, FlatParams, FlatRegimeParams, RegimeName, StateName
from lcm.ages import AgeGrid
from lcm.exceptions import PyLCMError
from lcm.typing import BoolND, FloatND, IntND


def _create_flat_dataframe(
    *,
    raw_results: MappingProxyType[
        RegimeName, MappingProxyType[int, PeriodRegimeSimulationData]
    ],
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    metadata: ResultMetadata,
    additional_targets: list[str] | None,
    ages: AgeGrid,
    subject_batch_size: int | None = None,
) -> pd.DataFrame:
    """Create a single flat DataFrame from all regime results.

    `regimes` may be empty (or missing entries) when `additional_targets`
    is `None` — in that case only the regime *name* is needed and the
    compiled `Regime` objects can be released ahead of the dataframe
    construction to free their XLA program workspaces. When
    `additional_targets` is set the matching regime objects must be
    present.
    """
    regime_dfs = [
        _process_regime(
            regime_name=name,
            regime=regimes.get(name),
            regime_results=raw_results[name],
            regime_states=metadata.regime_to_states[name],
            regime_actions=metadata.regime_to_actions[name],
            publishes_nested_policy=metadata.regime_to_publishes_nested_policy[name],
            regime_params=flat_params[name],
            stakeholders=metadata.regime_to_stakeholders.get(name),
            publishes_role=bool(metadata.stakeholder_names_to_ids),
            additional_targets=additional_targets,
            ages=ages,
            subject_batch_size=subject_batch_size,
        )
        for name in metadata.regime_names
        if raw_results[name]
    ]

    return _assemble_dataframe(
        regime_dfs=regime_dfs,
        state_names=metadata.state_names,
        action_names=metadata.action_names,
        stakeholder_value_names=_stakeholder_value_column_names(
            regime_names=metadata.regime_names,
            regime_to_stakeholders=metadata.regime_to_stakeholders,
        ),
    )


def _stakeholder_value_column_names(
    *,
    regime_names: list[RegimeName],
    regime_to_stakeholders: Mapping[RegimeName, tuple[str, ...] | None],
) -> list[str]:
    """Name the per-stakeholder value columns in the order the regimes declare them.

    Each collective regime contributes `value_<stakeholder>` for every entry of
    its `stakeholders` tuple, in that tuple's order — the same order that fixes
    the trailing axis of its value function. Regimes are walked in
    `regime_names` order and a name already contributed by an earlier regime
    keeps its first position, so two regimes sharing a stakeholder share one
    column.

    Args:
        regime_names: Names of all regimes in the model.
        regime_to_stakeholders: Mapping of regime names to their ordered
            stakeholder names, `None` for a singleton regime.

    Returns:
        List of column names, in publication order.

    """
    names: list[str] = []
    for regime_name in regime_names:
        for stakeholder in regime_to_stakeholders.get(regime_name) or ():
            column = f"value_{stakeholder}"
            if column not in names:
                names.append(column)
    return names


def _process_regime(
    *,
    regime_name: RegimeName,
    regime: Regime | None,
    regime_results: MappingProxyType[int, PeriodRegimeSimulationData],
    regime_states: tuple[str, ...],
    regime_actions: tuple[str, ...],
    publishes_nested_policy: bool,
    regime_params: FlatRegimeParams,
    stakeholders: tuple[str, ...] | None,
    publishes_role: bool,
    additional_targets: list[str] | None,
    ages: AgeGrid,
    subject_batch_size: int | None = None,
) -> pd.DataFrame:
    """Process results for a single regime into a DataFrame.

    `publishes_role` is a property of the MODEL, not of this regime: a model
    with any collective regime publishes `own_stakeholder` for every regime,
    since a row that leaves a household still has to say it now occupies no
    role. A model with none would carry a column that is the sentinel in every
    row, so it publishes none.

    `regime` is required only when `additional_targets` is set. With
    `additional_targets=None`, only `regime_name` is read, so callers
    may pass `regime=None` after dropping compiled `Regime` objects to
    free device workspaces. `stakeholders` is read unconditionally (from
    `ResultMetadata`, computed while `regime` was still guaranteed
    present) so a COLLECTIVE regime's `value` column can be split even
    when `regime` itself is `None`.

    Raises:
        PyLCMError: If the regime's recorded value carries a stakeholder axis
            but no stakeholder names are available to name its columns.

    """
    period_dicts = [
        _extract_period_data(
            result=result,
            period=period,
            regime_states=regime_states,
            regime_actions=regime_actions,
            publishes_nested_policy=publishes_nested_policy,
            publishes_role=publishes_role,
        )
        for period, result in regime_results.items()
    ]

    data: dict[str, np.ndarray | FloatND | IntND | BoolND | Sequence[str]] = dict(
        _concatenate_and_filter(period_dicts)
    )

    recorded_value = data["value"]
    if stakeholders is not None:
        del data["value"]
        data.update(
            _split_stakeholder_value(
                value=np.asarray(recorded_value), stakeholders=stakeholders
            )
        )
    elif np.ndim(recorded_value) > 1:
        msg = (
            f"Regime {regime_name!r} recorded a value with a trailing "
            f"stakeholder axis of length {np.shape(recorded_value)[-1]}, but the "
            "result carries no `regime_to_stakeholders` entry for it, so its "
            "per-stakeholder value columns cannot be named. The result was "
            "written without that metadata; re-simulate to produce a result "
            "that can be published as a dataframe."
        )
        raise PyLCMError(msg)

    data["age"] = ages.values[data["period"]]  # noqa: PD011
    data["regime_name"] = [regime_name] * len(data["period"])

    if additional_targets:
        if regime is None:
            msg = (
                f"additional_targets requested for regime {regime_name!r} but "
                "the Regime object is unavailable. Pass the regime when "
                "constructing the dataframe."
            )
            raise ValueError(msg)
        targets_for_regime = _filter_targets_for_regime(
            targets=additional_targets, regime=regime
        )
        if targets_for_regime:
            target_values = _compute_targets(
                data=data,
                targets=targets_for_regime,
                regime=regime,
                regime_params=regime_params,
                subject_batch_size=subject_batch_size,
            )
            n_rows = len(data["period"])
            data.update(
                {
                    # A target that reads no per-subject variable evaluates to one
                    # value for the whole regime and arrives 0-d. Give it its rows
                    # here rather than leaving pandas to broadcast it: pandas does
                    # not recognize a 0-d device array as a numeric scalar and
                    # would build an object column, costing it arithmetic,
                    # aggregation and Arrow round-tripping.
                    name: jnp.full(n_rows, value) if jnp.ndim(value) == 0 else value
                    for name, value in target_values.items()
                }
            )

    return pd.DataFrame(data)


def _split_stakeholder_value(
    *,
    value: np.ndarray,
    stakeholders: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Split a COLLECTIVE regime's 2D `value` column into per-stakeholder columns.

    A collective regime's recorded value carries a trailing stakeholder axis
    (`PeriodRegimeSimulationData.V_arr`, shape `(n_rows, n_stakeholders)`), so
    `pd.DataFrame` cannot ingest it as a single "value" column the way it does
    for a singleton regime's 1D array. Naming is deterministic:
    `value_<stakeholder>`, one 1D column per entry of `stakeholders`, in the
    same order as the trailing axis (fixed by `Regime.stakeholders` — see
    `_lcm.regime_building.Q_and_F`).
    """
    return {
        f"value_{stakeholder}": value[:, i]
        for i, stakeholder in enumerate(stakeholders)
    }


def _extract_period_data(
    *,
    result: PeriodRegimeSimulationData,
    period: int,
    regime_states: tuple[str, ...],
    regime_actions: tuple[str, ...],
    publishes_nested_policy: bool,
    publishes_role: bool,
) -> dict[str, FloatND | IntND | BoolND]:
    """Extract data from a single period's simulation results."""
    data: dict[str, FloatND | IntND | BoolND] = {
        "subject_id": jnp.arange(len(result.in_regime), dtype=jnp.int32),
        "period": jnp.full_like(result.in_regime, period, dtype=jnp.int32),
        "_in_regime": result.in_regime,
        "value": result.V_arr,
    }

    # The role each row occupies. Only a model with a collective regime has
    # roles to occupy; elsewhere the column would be the no-role sentinel in
    # every row of every user's frame.
    if publishes_role:
        data["own_stakeholder"] = result.own_stakeholder

    # Per-subject flag that a continuous-outer off-grid policy read was refused
    # and the grid-argmax pair retained. Only NNBEGM regimes can publish the flag;
    # omitting it elsewhere avoids a constant-False column in unrelated results.
    if publishes_nested_policy:
        data["nested_policy_fallback"] = result.nested_policy_fallback

    for name in regime_states:
        if name in result.states:
            data[name] = result.states[name]

    for name in regime_actions:
        if name in result.actions:
            data[name] = result.actions[name]

    return data


def _concatenate_and_filter(
    period_dicts: list[dict[str, FloatND | IntND | BoolND]],
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


def _to_host(value: FloatND | IntND | BoolND) -> np.ndarray:
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
    stakeholder_value_names: Sequence[str] = (),
) -> pd.DataFrame:
    """Combine regime DataFrames, add missing columns, reorder, and sort."""
    if not regime_dfs:
        return _empty_dataframe(state_names=state_names, action_names=action_names)

    df = pd.concat(regime_dfs, ignore_index=True)
    df = _add_missing_columns(df=df, state_names=state_names, action_names=action_names)
    df = _reorder_columns(
        df=df,
        state_names=state_names,
        action_names=action_names,
        stakeholder_value_names=stakeholder_value_names,
    )
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
    stakeholder_value_names: Sequence[str] = (),
) -> pd.DataFrame:
    """Reorder columns: id, period, regime_name, role, value, states, actions, rest.

    `subject_id`, `period` and `regime_name` identify a row and are named
    unconditionally, so a frame that lacks one raises here rather than being
    published one column narrower. Two entries are conditional. A model with
    no collective regime publishes no `own_stakeholder`, and where it is
    published it joins the identifying block: which role a row occupies says
    who the row *is*, alongside which regime it is in. `"value"` is the other:
    an all-collective result has none, because `_process_regime` replaces it
    with `value_<stakeholder>` columns.

    Those stakeholder columns are named by `stakeholder_value_names`, which
    carries each collective regime's own `stakeholders` order — the order that
    also fixes the trailing axis of its value function. Selecting them by name
    rather than by a `value_` prefix leaves an ordinary computed column such as
    `value_of_leisure` in the trailing `rest` block where it belongs.
    """
    base = ["subject_id", "period", "regime_name"]
    if "own_stakeholder" in df.columns:
        base = [*base, "own_stakeholder"]
    if "value" in df.columns:
        base = [*base, "value"]
    stakeholder_value_cols = [c for c in stakeholder_value_names if c in df.columns]
    known = (
        set(base) | set(state_names) | set(action_names) | set(stakeholder_value_cols)
    )
    rest = [c for c in df.columns if c not in known]
    return df[base + stakeholder_value_cols + state_names + action_names + rest]


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

    if "own_stakeholder" in df.columns and metadata.stakeholder_names_to_ids:
        df["own_stakeholder"] = _roles_to_categorical(
            codes=df["own_stakeholder"],
            stakeholder_names_to_ids=metadata.stakeholder_names_to_ids,
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

    return pd.Categorical(  # ty: ignore[invalid-return-type]
        labels, categories=list(merged_categories), ordered=ordered
    )


def _roles_to_categorical(
    *,
    codes: pd.Series,
    stakeholder_names_to_ids: Mapping[str, int],
) -> pd.Categorical:
    """Label the role codes each row carries.

    A row in a singleton regime occupies no role, and the honest label for that
    is no label at all: it becomes a missing entry rather than a category
    competing with the real roles in a `value_counts` or a groupby.

    Args:
        codes: The raw `own_stakeholder` column.
        stakeholder_names_to_ids: The model's role vocabulary.

    Returns:
        The column as an unordered Categorical over the declared roles.
    """
    names = list(stakeholder_names_to_ids)
    positions = {
        code: names.index(name) for name, code in stakeholder_names_to_ids.items()
    }
    raw = codes.to_numpy()
    labels = np.full(raw.shape, -1, dtype=np.int64)
    for code, position in positions.items():
        labels[raw == code] = position
    return pd.Categorical.from_codes(  # ty: ignore[invalid-return-type]
        labels, categories=pd.Index(names), ordered=False
    )


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
        return pd.Categorical.from_codes(  # ty: ignore[invalid-return-type]
            int_codes,
            categories=pd.Index(categories),
            ordered=ordered,
        )

    return pd.Categorical.from_codes(  # ty: ignore[invalid-return-type]
        codes_array.astype(int),
        categories=pd.Index(categories),
        ordered=ordered,
    )
