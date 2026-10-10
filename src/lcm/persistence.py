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
from typing import TYPE_CHECKING, cast

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
from _lcm.solution.public_period_capture import load_period_capture
from _lcm.typing import InitialConditions, PeriodToRegimeToVArr
from lcm.period_capture import CapturedPeriodReplay, PeriodCapture, PeriodCaptureRecord
from lcm.solver_api import SolutionResult
from lcm.typing import UserParams

__all__ = [
    "CapturedPeriodReplay",
    "PeriodCapture",
    "PeriodCaptureRecord",
    "PeriodReplay",
    "SimulateSnapshot",
    "SolveSnapshot",
    "load_legacy_solution",
    "load_period_capture",
    "load_snapshot",
    "load_solution",
    "replay_period",
    "save_solution",
]

if TYPE_CHECKING:
    from lcm.model import Model
    from lcm.result import SimulationResult

    # Type-checker view: full precision.
    type _ModelOrNone = Model | None
    type _SimulationResultOrNone = SimulationResult | None
    type _SolutionResultBoundary = SolutionResult
    type _ModelClass = type[Model]
    type _SimulationResultClass = type[SimulationResult]
else:
    # Runtime view used by beartype's annotation evaluator. `Model` and
    # `SimulationResult` cannot be imported here (circular), so collapse
    # to `object`. The snapshot dataclasses are serialization carriers; the
    # API surface that needs strict checking is the snapshot writers,
    # which beartype polices via their own parameters.
    type _ModelOrNone = object
    type _SimulationResultOrNone = object
    type _SolutionResultBoundary = object
    type _ModelClass = type[object]
    type _SimulationResultClass = type[object]

# A field a debug snapshot pickles.
type _PickledField = (
    _ModelOrNone | UserParams | InitialConditions | _SimulationResultOrNone
)


def _bind_forward_refs(
    *,
    model_cls: _ModelClass,
    simulation_result_cls: _SimulationResultClass,
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

    # Load pickle fields; an excluded or absent field stays `None`.
    pickled: dict[str, _PickledField] = {}
    for field_name in fields:
        pkl_path = path / f"{field_name}.pkl"
        if field_name not in exclude and pkl_path.exists():
            with pkl_path.open("rb") as fh:
                pickled[field_name] = cloudpickle.load(fh)
    # Each pickle holds the value its field name declares.
    model = cast("_ModelOrNone", pickled.get("model"))
    params = cast("UserParams | None", pickled.get("params"))

    # Load period_to_regime_to_V_arr from HDF5 if not excluded
    values: PeriodToRegimeToVArr | None = None
    h5_path = path / "arrays.h5"
    if h5_path.exists() and "period_to_regime_to_V_arr" not in exclude:
        values = _load_h5(h5_path)
    elif "period_to_regime_to_V_arr" not in exclude:
        logger.warning(
            "arrays.h5 not found in %s; period_to_regime_to_V_arr set to None",
            path,
        )

    if snapshot_type == "solve":
        return SolveSnapshot(
            model=model,
            params=params,
            period_to_regime_to_V_arr=values,
            platform=saved_platform,
        )
    if snapshot_type == "simulate":
        return SimulateSnapshot(
            model=model,
            params=params,
            initial_conditions=cast(
                "InitialConditions | None", pickled.get("initial_conditions")
            ),
            period_to_regime_to_V_arr=values,
            result=cast("_SimulationResultOrNone", pickled.get("result")),
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


def load_legacy_solution(*, path: Path) -> PeriodToRegimeToVArr:
    """Load a pre-schema value-only HDF5 file for explicit migration.

    Legacy files carry no model fingerprint, artifact schemas, omissions, or
    checksums and cannot authenticate a complete replay result.

    Args:
        path: Legacy HDF5 file path.

    Returns:
        The immutable period/regime value mapping stored in the legacy file.

    """
    return _load_h5(path)
