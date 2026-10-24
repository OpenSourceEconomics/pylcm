"""Normalize tagged current innovations and gather their original dated rows."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import jax
import numpy as np
import pandas as pd
from jaxtyping import Int

from _lcm.engine import Regime
from _lcm.shock_overrides import CurrentIIDOverride
from _lcm.simulation.initial_conditions import resolve_initial_periods
from _lcm.simulation.shock_paths import _subject_occupancy
from _lcm.simulation.subject_groups import SubjectRows
from _lcm.time import TimeAxis
from _lcm.typing import RegimeNamesToIds
from lcm.exceptions import InvalidInitialConditionsError, InvalidSimulationInputError
from lcm.shocks import BaseInnovations, PhysicalValues, ShockName, ShockOverride
from lcm.typing import (
    RegimeName,
    StateName,
    UserInitialConditions,
    UserInitialConditionsInput,
)

type SimulationInitialConditionsInput = Mapping[
    StateName,
    jax.Array | np.ndarray | pd.Series | pd.Categorical | ShockOverride,
]
type ShockPathsInput = Mapping[StateName, jax.Array | np.ndarray | ShockOverride]
type CurrentIIDOverrideSchema = tuple[
    tuple[ShockName, Literal["base", "physical"], str], ...
]


@dataclass(frozen=True, kw_only=True)
class CurrentIIDOverrides:
    """Host-owned call inputs; only gathered cells cross the native boundary."""

    entries: Mapping[ShockName, ShockOverride]
    """Entry-only vectors, with their original scientific tags."""
    paths: Mapping[ShockName, ShockOverride]
    """Common or original-row paths, superseding entry values by name."""
    starts: Int[np.ndarray, "..."]  # noqa: UP037 - jaxtyping shape string
    """Actual computational entry periods in original caller order."""

    @property
    def schema(self) -> CurrentIIDOverrideSchema:
        """Return array-free tag/dtype identity for exact native prototypes."""
        return tuple(
            (
                name,
                "base" if isinstance(value, BaseInnovations) else "physical",
                value.values.dtype.str,  # noqa: PD011 - tagged numeric record
            )
            for name, value in sorted((dict(self.entries) | dict(self.paths)).items())
        )


def split_current_iid_inputs(  # noqa: C901, PLR0912 - distinct entry/path boundaries
    *,
    initial: SimulationInitialConditionsInput | pd.DataFrame,
    paths: ShockPathsInput | None,
    names: frozenset[ShockName],
) -> tuple[
    UserInitialConditionsInput | pd.DataFrame,
    Mapping[ShockName, ShockOverride],
    Mapping[ShockName, ShockOverride],
    Mapping[StateName, jax.Array | np.ndarray],
]:
    """Extract current tags before state encoding, refusing ambiguous raw values."""
    entries = {}
    if isinstance(initial, pd.DataFrame):
        if names & set(initial.columns):
            raise InvalidInitialConditionsError(
                "Current-IID entry values require a mapping with BaseInnovations "
                "or PhysicalValues records."
            )
        ordinary = initial
    else:
        ordinary = {}
        for name, value in initial.items():
            if name in names:
                if not isinstance(value, BaseInnovations | PhysicalValues):
                    raise InvalidInitialConditionsError(
                        f"Current-IID entry {name!r} requires BaseInnovations "
                        "or PhysicalValues."
                    )
                entries[name] = value
            elif isinstance(value, BaseInnovations | PhysicalValues):
                raise InvalidInitialConditionsError(
                    f"Tagged entry {name!r} is not a named current-IID shock."
                )
            else:
                ordinary[name] = value
    current_paths, state_paths = {}, {}
    for name, value in (paths or {}).items():
        if name in names:
            if not isinstance(value, BaseInnovations | PhysicalValues):
                raise InvalidSimulationInputError(
                    f"Current-IID path {name!r} requires BaseInnovations "
                    "or PhysicalValues."
                )
            current_paths[name] = value
        elif isinstance(value, BaseInnovations | PhysicalValues):
            raise InvalidSimulationInputError(
                f"Tagged path {name!r} is not a named current-IID shock."
            )
        else:
            state_paths[name] = value
    return (
        ordinary,
        MappingProxyType(entries),
        MappingProxyType(current_paths),
        MappingProxyType(state_paths),
    )


def resolve_current_iid_overrides(  # noqa: C901, PLR0912 - shapes and demanded periods
    *,
    entries: Mapping[ShockName, ShockOverride],
    paths: Mapping[ShockName, ShockOverride],
    initial: UserInitialConditions,
    regimes: Mapping[RegimeName, Regime],
    ages: TimeAxis,
    regime_names_to_ids: RegimeNamesToIds,
) -> CurrentIIDOverrides | None:
    """Validate dated shapes and demanded coverage without uploading full banks."""
    if not entries and not paths:
        return None
    consumed = {
        name
        for regime in regimes.values()
        if (plan := regime.simulation.current_iid_draw_plan) is not None
        for name in plan.physical_names
    }
    if unused := (entries.keys() | paths.keys()) - consumed:
        raise InvalidSimulationInputError(
            f"Tagged current-IID inputs {sorted(unused)} have no simulation reader."
        )
    if "age" not in initial or "regime_id" not in initial:
        raise InvalidInitialConditionsError(
            "Tagged current-IID inputs require entry time and regime_id."
        )
    count = len(initial["regime_id"])
    starts, on_grid = resolve_initial_periods(ages=ages, initial_ages=initial["age"])
    if not np.all(on_grid):
        raise InvalidInitialConditionsError(
            "Current-IID entry times must be on the model clock."
        )
    normalized_entries, normalized_paths = {}, {}
    for source, target, dated in (
        (entries, normalized_entries, False),
        (paths, normalized_paths, True),
    ):
        for name, item in source.items():
            values = np.asarray(item.values)
            if dated:
                valid_shape = (values.ndim == 1 and values.size > 0) or (
                    values.ndim == 2  # noqa: PLR2004 - row-by-period path matrix
                    and values.shape[0] == count
                    and values.shape[-1] > 0
                )
            else:
                valid_shape = values.ndim == 1 and values.shape[0] == count
            if not valid_shape:
                shape = "(periods,) or (subjects, periods)" if dated else "(subjects,)"
                raise InvalidSimulationInputError(
                    f"Current-IID {'path' if dated else 'entry'} {name!r} "
                    f"requires shape {shape} in original row order; got {values.shape}."
                )
            target[name] = type(item)(values=values)
    occupancy = _subject_occupancy(
        initial_ages=initial["age"],
        regime_ids=np.asarray(initial["regime_id"]),
        ages=ages,
        regimes=regimes,
        regime_names_to_ids=regime_names_to_ids,
    )
    for name, item in normalized_paths.items():
        values = item.values  # noqa: PD011 - tagged numeric record
        for regime_name, regime in regimes.items():
            plan = regime.simulation.current_iid_draw_plan
            if plan is None or name not in plan.physical_names:
                continue
            for period in regime.simulation.programs.decision:
                if (
                    occupancy[regime_name][:, period].any()
                    and period >= values.shape[-1]
                ):
                    raise InvalidSimulationInputError(
                        f"Current-IID path {name!r} does not cover period {period} "
                        f"in regime {regime_name!r}."
                    )
    return CurrentIIDOverrides(
        entries=MappingProxyType(normalized_entries),
        paths=MappingProxyType(normalized_paths),
        starts=starts,
    )


def gather_current_iid_overrides(
    *,
    bank: CurrentIIDOverrides | None,
    names: tuple[ShockName, ...],
    period: int,
    subject_slice: slice | SubjectRows,
) -> Mapping[ShockName, CurrentIIDOverride]:
    """Gather original rows, repeating the final real row only for inert padding."""
    if bank is None:
        return MappingProxyType({})
    rows = (
        subject_slice.rows
        if isinstance(subject_slice, SubjectRows)
        else np.arange(subject_slice.start or 0, subject_slice.stop)
    )
    rows = np.minimum(rows, len(bank.starts) - 1)
    result = {}
    for name in names:
        if name not in bank.entries and name not in bank.paths:
            continue
        if name in bank.paths:
            item = bank.paths[name]
            payload = item.values  # noqa: PD011 - tagged numeric record
            if period < payload.shape[-1]:
                values = (
                    np.full(len(rows), payload[period], dtype=payload.dtype)
                    if payload.ndim == 1
                    else payload[rows, period]
                )
                active = bank.starts[rows] <= period
            else:
                values = np.zeros(len(rows), dtype=payload.dtype)
                active = np.zeros(len(rows), dtype=bool)
        else:
            item = bank.entries[name]
            values = item.values[rows]  # noqa: PD011 - tagged numeric record
            active = bank.starts[rows] == period
        result[name] = CurrentIIDOverride(
            kind="base" if isinstance(item, BaseInnovations) else "physical",
            values=values,
            active=active,
        )
    return MappingProxyType(result)


def current_iid_override_prototype(
    *,
    schema: CurrentIIDOverrideSchema,
    names: tuple[ShockName, ...],
    n_subjects: int,
    sharding: jax.sharding.Sharding,
) -> Mapping[ShockName, CurrentIIDOverride]:
    """Build exact array-free native operands without public input constructors."""
    return MappingProxyType(
        {
            name: CurrentIIDOverride(
                kind=kind,
                values=jax.ShapeDtypeStruct(
                    (n_subjects,), np.dtype(dtype), sharding=sharding
                ),
                active=jax.ShapeDtypeStruct((n_subjects,), np.bool_, sharding=sharding),
            )
            for name, kind, dtype in schema
            if name in names
        }
    )
