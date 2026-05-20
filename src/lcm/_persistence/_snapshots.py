"""Snapshot writers for solve and simulate runs.

Both `Model` and `SimulationResult` cannot be imported here at module level
without creating a circular import. They are brought in via TYPE_CHECKING for
type checkers and rebound via `_bind_forward_refs` at runtime so the beartype
claw can resolve the rewritten string annotations on the snapshot writers.

"""

import copy
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from lcm._persistence._io import (
    _enforce_retention,
    _next_counter,
    _save_h5,
    _save_pkl,
    _write_environment_files,
    _write_metadata,
)
from lcm.typing import InitialConditions, PeriodToRegimeToVArr, UserParams

if TYPE_CHECKING:
    from lcm.api.model import Model
    from lcm.api.result import SimulationResult
else:
    # Runtime view used by the beartype claw's annotation evaluator until
    # `_bind_forward_refs` rebinds these names to the real classes.
    Model = Any
    SimulationResult = Any


def _bind_forward_refs(
    *,
    model_cls: type,
    simulation_result_cls: type,
) -> None:
    """Bind `Model` and `SimulationResult` into this module's globals.

    The package claw rewrites string annotations in `_save_*_snapshot` into
    runtime forward references resolved against this module's globals.
    `lcm.__init__` calls this helper once both classes are loaded so the
    refs resolve at call time without depending on an ad-hoc assignment
    from outside the module.
    """
    global Model, SimulationResult  # noqa: PLW0603
    Model = model_cls  # ty: ignore[invalid-assignment]
    SimulationResult = simulation_result_cls  # ty: ignore[invalid-assignment]


def _save_solve_snapshot(
    *,
    model: Model,
    params: UserParams,
    period_to_regime_to_V_arr: PeriodToRegimeToVArr,
    log_path: Path,
    log_keep_n_latest: int,
) -> Path:
    """Save a solve snapshot directory to disk.

    Args:
        model: The Model instance.
        params: User parameters passed to solve.
        period_to_regime_to_V_arr: Value function arrays from solve.
        log_path: Parent directory for snapshot directories.
        log_keep_n_latest: Maximum number of snapshots to retain.

    Returns:
        Path to the created snapshot directory.

    """
    log_path.mkdir(parents=True, exist_ok=True)
    counter = _next_counter(log_path, prefix="solve_snapshot")
    snap_dir = log_path / f"solve_snapshot_{counter:03d}"
    snap_dir.mkdir()

    _save_pkl(snap_dir / "model.pkl", model)
    _save_pkl(snap_dir / "params.pkl", params)
    _save_h5(snap_dir / "arrays.h5", period_to_regime_to_V_arr)
    _write_metadata(snap_dir, snapshot_type="solve", fields=["model", "params"])
    _write_environment_files(snap_dir)

    _enforce_retention(
        log_path, prefix="solve_snapshot", keep_n_latest=log_keep_n_latest
    )
    return snap_dir


def _save_simulate_snapshot(
    *,
    model: Model,
    params: UserParams,
    initial_conditions: InitialConditions,
    period_to_regime_to_V_arr: PeriodToRegimeToVArr,
    result: SimulationResult,
    log_path: Path,
    log_keep_n_latest: int,
) -> Path:
    """Save a simulate snapshot directory to disk.

    Args:
        model: The Model instance.
        params: User parameters passed to simulate.
        initial_conditions: Mapping of state names and "regime_id" to arrays.
        period_to_regime_to_V_arr: Value function arrays.
        result: SimulationResult object.
        log_path: Parent directory for snapshot directories.
        log_keep_n_latest: Maximum number of snapshots to retain.

    Returns:
        Path to the created snapshot directory.

    """
    prefix = "simulate_snapshot"
    log_path.mkdir(parents=True, exist_ok=True)
    counter = _next_counter(log_path, prefix=prefix)
    snap_dir = log_path / f"{prefix}_{counter:03d}"
    snap_dir.mkdir()

    _save_pkl(snap_dir / "model.pkl", model)
    _save_pkl(snap_dir / "params.pkl", params)
    _save_pkl(snap_dir / "initial_conditions.pkl", initial_conditions)
    _save_pkl(
        snap_dir / "result.pkl",
        _strip_V_arr_from_result(result=result, model=model),
    )
    _save_h5(snap_dir / "arrays.h5", period_to_regime_to_V_arr)
    _write_metadata(
        snap_dir,
        snapshot_type="simulate",
        fields=["model", "params", "initial_conditions", "result"],
    )
    _write_environment_files(snap_dir)

    _enforce_retention(log_path, prefix=prefix, keep_n_latest=log_keep_n_latest)
    return snap_dir


def _strip_V_arr_from_result(
    *, result: SimulationResult, model: Model
) -> SimulationResult:
    """Create a copy of result with value arrays and compiled callables stripped.

    `period_to_regime_to_V_arr` is dropped to avoid storing it both in the
    pickle and in the HDF5 file. `_regimes` is overwritten with the
    model's lazy-path `regimes`: when `Model(n_subjects=N)` is set
    the result carries the AOT-compiled regimes, whose
    `jax.stages.Compiled` callables hold a `LoadedExecutable` that cannot
    be pickled. The lazy regimes carry the same metadata and cloud-pickle
    cleanly (model.pkl uses the same set).
    """
    stripped = copy.copy(result)
    object.__setattr__(stripped, "_period_to_regime_to_V_arr", MappingProxyType({}))
    object.__setattr__(stripped, "_regimes", model._regimes)  # noqa: SLF001
    return stripped
