"""User-facing snapshot dataclasses, snapshot loader, and solution save/load.

Debug snapshots keep their separate reproduction format. Durable solutions use
a versioned archive of labelled metadata and independently addressable
numerical payloads.

"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeAlias

import cloudpickle

from _lcm.persistence.io import (
    _get_platform,
    _load_h5,
)
from _lcm.persistence.snapshots import (
    _bind_forward_refs as _bind_snapshot_forward_refs,
)
from _lcm.persistence.solution import load_solution_archive, save_solution_archive
from _lcm.solution.period_replay import PeriodReplay, replay_period
from _lcm.typing import InitialConditions, PeriodToRegimeToVArr
from lcm.solver_api import SolutionResult
from lcm.typing import UserParams

__all__ = [
    "PeriodReplay",
    "SimulateSnapshot",
    "SolveSnapshot",
    "load_legacy_solution",
    "load_snapshot",
    "load_solution",
    "replay_period",
    "save_solution",
]

if TYPE_CHECKING:
    from lcm.model import Model
    from lcm.result import SimulationResult

    # Type-checker view: full precision.
    _ModelOrNone = Model | None
    _SimulationResultOrNone = SimulationResult | None
    _SolutionResultBoundary: TypeAlias = SolutionResult  # noqa: UP040
else:
    # Runtime view used by beartype's annotation evaluator. `Model` and
    # `SimulationResult` cannot be imported here (circular), so collapse
    # to `Any`. The snapshot dataclasses are serialization carriers; the
    # API surface that needs strict checking is the snapshot writers,
    # which beartype polices via their own parameters.
    _ModelOrNone = Any
    _SimulationResultOrNone = Any
    _SolutionResultBoundary = object


def _bind_forward_refs(
    *,
    model_cls: type,
    simulation_result_cls: type,
) -> None:
    """Forward `Model` / `SimulationResult` bindings to `_lcm.persistence.snapshots`."""
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


# keyword-only-exempt: primary-argument=path
def load_snapshot(
    path: Path, *, exclude: Sequence[str] = ()
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
    path = Path(path)

    with (path / "metadata.json").open(encoding="utf-8") as fh:
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
    solution: _SolutionResultBoundary,
    path: Path,
) -> Path:
    """Atomically persist a complete labelled solution.

    Args:
        solution: Complete result returned by :meth:`lcm.Model.solve`.
        path: Destination archive path.

    Returns:
        The path where the object was saved.

    Raises:
        FileNotFoundError: If the parent directory does not exist.

    """
    return save_solution_archive(solution=solution, path=path)


def load_solution(
    *,
    path: Path,
    verify_checksums: bool = False,
) -> _SolutionResultBoundary:
    """Load a complete solution lazily from a versioned archive.

    Args:
        path: Archive path.
        verify_checksums: Whether to verify every payload eagerly while keeping
            entries unloaded. Individual entries are always verified when
            materialized.

    Returns:
        A complete result whose value and artifact entries load independently.

    """
    return load_solution_archive(path=path, verify_checksums=verify_checksums)


def load_legacy_solution(
    *, path: Path
) -> MappingProxyType[int, MappingProxyType[str, Any]]:
    """Load a pre-schema value-only HDF5 file for explicit migration.

    Legacy files carry no model fingerprint, artifact schemas, omissions, or
    checksums and cannot authenticate a complete replay result.

    Args:
        path: Legacy HDF5 file path.

    Returns:
        The immutable period/regime value mapping stored in the legacy file.

    """
    return _load_h5(path)
