"""User-facing snapshot dataclasses, snapshot loader, and solution save/load.

I/O helpers and the snapshot writers live behind a leading underscore in
`lcm._persistence`. This module is intentionally a thin layer of public
snapshot dataclasses plus three public top-level functions
(`load_snapshot`, `save_solution`, `load_solution`).

"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from lcm._persistence._io import (
    _get_platform,
    _load_h5,
    _save_h5,
)
from lcm._persistence._snapshots import (
    _bind_forward_refs as _bind_snapshot_forward_refs,
)
from lcm.typing import (
    FloatND,
    InitialConditions,
    PeriodToRegimeToVArr,
    RegimeName,
    UserParams,
)

if TYPE_CHECKING:
    from lcm.api.model import Model
    from lcm.api.result import SimulationResult

    # Type-checker view: full precision.
    _ModelOrNone = Model | None
    _SimulationResultOrNone = SimulationResult | None
else:
    # Runtime view used by beartype's annotation evaluator. `Model` and
    # `SimulationResult` cannot be imported here (circular), so collapse
    # to `Any`. The snapshot dataclasses are serialization carriers; the
    # API surface that needs strict checking is the snapshot writers,
    # which beartype polices via their own parameters.
    _ModelOrNone = Any
    _SimulationResultOrNone = Any


def _bind_forward_refs(
    *,
    model_cls: type,
    simulation_result_cls: type,
) -> None:
    """Forward `Model` / `SimulationResult` bindings to `_persistence._snapshots`."""
    _bind_snapshot_forward_refs(
        model_cls=model_cls, simulation_result_cls=simulation_result_cls
    )


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SolveSnapshot:
    """Snapshot of a solve run for offline reconstruction."""

    model: _ModelOrNone
    """The Model instance."""

    params: UserParams | None
    """User parameters passed to solve."""

    period_to_regime_to_V_arr: PeriodToRegimeToVArr | None
    """Immutable mapping of periods to regime value function arrays."""

    platform: str
    """Platform string, e.g. `"x86_64-Linux"`."""


@dataclass(frozen=True)
class SimulateSnapshot:
    """Snapshot of a simulate run for offline reconstruction."""

    model: _ModelOrNone
    """The Model instance."""

    params: UserParams | None
    """User parameters passed to simulate."""

    initial_conditions: InitialConditions | None
    """Immutable mapping of state names and `"regime_id"` to canonical-dtype arrays."""

    period_to_regime_to_V_arr: PeriodToRegimeToVArr | None
    """Immutable mapping of periods to regime value function arrays."""

    result: _SimulationResultOrNone
    """SimulationResult object."""

    platform: str
    """Platform string, e.g. `"x86_64-Linux"`."""


def load_snapshot(
    path: Path,
    *,
    exclude: Sequence[str] = (),
) -> SolveSnapshot | SimulateSnapshot:
    """Load a debug snapshot directory from disk.

    Args:
        path: Path to the snapshot directory (e.g. `solve_snapshot_001/`).
        exclude: Field names to skip loading
            (e.g. `["period_to_regime_to_V_arr"]` to save memory).
            Excluded fields are set to `None`.

    Returns:
        A `SolveSnapshot` or `SimulateSnapshot`.

    """
    import cloudpickle  # noqa: PLC0415

    path = Path(path)

    with (path / "metadata.json").open() as fh:
        metadata = json.load(fh)

    snapshot_type = metadata["snapshot_type"]
    current_platform = _get_platform()
    saved_platform = metadata["platform"]
    if saved_platform != current_platform:
        logger.warning(
            "Snapshot created on %s but loading on %s — environment may not match",
            saved_platform,
            current_platform,
        )

    fields = metadata["fields"]

    loaded: dict[str, Any] = {"platform": saved_platform}

    # Load pickle fields
    for field_name in fields:
        if field_name in exclude:
            loaded[field_name] = None
            continue
        pkl_path = path / f"{field_name}.pkl"
        if pkl_path.exists():
            with pkl_path.open("rb") as fh:
                loaded[field_name] = cloudpickle.load(fh)

    # Load period_to_regime_to_V_arr from HDF5 if not excluded
    h5_path = path / "arrays.h5"
    if h5_path.exists() and "period_to_regime_to_V_arr" not in exclude:
        loaded["period_to_regime_to_V_arr"] = _load_h5(h5_path)
    elif "period_to_regime_to_V_arr" not in exclude:
        loaded["period_to_regime_to_V_arr"] = None
        logger.warning(
            "arrays.h5 not found in %s; period_to_regime_to_V_arr set to None",
            path,
        )

    if snapshot_type == "solve":
        return SolveSnapshot(
            model=loaded.get("model"),
            params=loaded.get("params"),
            period_to_regime_to_V_arr=loaded.get("period_to_regime_to_V_arr"),
            platform=saved_platform,
        )
    if snapshot_type == "simulate":
        return SimulateSnapshot(
            model=loaded.get("model"),
            params=loaded.get("params"),
            initial_conditions=loaded.get("initial_conditions"),
            period_to_regime_to_V_arr=loaded.get("period_to_regime_to_V_arr"),
            result=loaded.get("result"),
            platform=saved_platform,
        )
    msg = f"Unknown snapshot_type: {snapshot_type!r}"
    raise ValueError(msg)


def save_solution(
    *,
    period_to_regime_to_V_arr: MappingProxyType[
        int, MappingProxyType[RegimeName, FloatND]
    ],
    path: str | Path,
) -> Path:
    """Save value function arrays from solve() to an HDF5 file.

    Args:
        period_to_regime_to_V_arr: Immutable mapping of periods to regime
            value function arrays.
        path: File path to save the HDF5 file.

    Returns:
        The path where the object was saved.

    Raises:
        FileNotFoundError: If the parent directory does not exist.

    """
    p = Path(path)
    if not p.parent.is_dir():
        raise FileNotFoundError(f"Parent directory does not exist: {p.parent}")
    _save_h5(p, period_to_regime_to_V_arr)
    return p


def load_solution(
    *,
    path: str | Path,
) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
    """Load value function arrays from an HDF5 file.

    Args:
        path: File path to read the HDF5 file from.

    Returns:
        Immutable mapping of periods to regime value function arrays.

    """
    return _load_h5(Path(path))
