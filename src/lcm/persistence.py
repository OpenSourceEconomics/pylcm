"""Persistence utilities for saving and loading LCM artifacts."""

import contextlib
import json
import logging
import pickle
import platform
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import cloudpickle
import h5py
import jax.numpy as jnp
import numpy as np
from jax import Array

from lcm.typing import FloatND, RegimeName, UserParams, VArrMapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SolveSnapshot:
    """Snapshot of a solve run for offline reconstruction."""

    model: object
    """The Model instance."""

    params: UserParams
    """User parameters passed to solve."""

    V_arr_dict: VArrMapping | None
    """Immutable mapping of periods to regime value function arrays."""

    platform: str
    """Platform string, e.g. `"x86_64-Linux"`."""


@dataclass(frozen=True)
class SimulateSnapshot:
    """Snapshot of a simulate run for offline reconstruction."""

    model: object
    """The Model instance."""

    params: UserParams
    """User parameters passed to simulate."""

    initial_conditions: Mapping[str, Array] | None
    """Mapping of state names (plus `"regime_id"`) to arrays."""

    V_arr_dict: VArrMapping | None
    """Immutable mapping of periods to regime value function arrays."""

    result: object | None
    """SimulationResult object."""

    platform: str
    """Platform string, e.g. `"x86_64-Linux"`."""


@dataclass(frozen=True)
class SolveAndSimulateSnapshot:
    """Snapshot of a solve-and-simulate run for offline reconstruction."""

    model: object
    """The Model instance."""

    params: UserParams
    """User parameters passed to solve_and_simulate."""

    initial_conditions: Mapping[str, Array] | None
    """Mapping of state names (plus `"regime_id"`) to arrays."""

    V_arr_dict: VArrMapping | None
    """Immutable mapping of periods to regime value function arrays."""

    result: object | None
    """SimulationResult object."""

    platform: str
    """Platform string, e.g. `"x86_64-Linux"`."""


def load_snapshot(
    path: Path,
    *,
    exclude: Sequence[str] = (),
) -> SolveSnapshot | SimulateSnapshot | SolveAndSimulateSnapshot:
    """Load a debug snapshot directory from disk.

    Args:
        path: Path to the snapshot directory (e.g. `solve_snapshot_001/`).
        exclude: Field names to skip loading (e.g. `["V_arr_dict"]` to save memory).
            Excluded fields are set to `None`.

    Returns:
        A `SolveSnapshot`, `SimulateSnapshot`, or `SolveAndSimulateSnapshot`.

    """
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

    loaded: dict[str, object] = {"platform": saved_platform}

    # Load pickle fields
    for field_name in fields:
        if field_name in exclude:
            loaded[field_name] = None
            continue
        pkl_path = path / f"{field_name}.pkl"
        if pkl_path.exists():
            with pkl_path.open("rb") as fh:
                loaded[field_name] = cloudpickle.load(fh)

    # Load V_arr_dict from HDF5 if not excluded
    h5_path = path / "arrays.h5"
    if h5_path.exists() and "V_arr_dict" not in exclude:
        loaded["V_arr_dict"] = _load_V_arr_from_h5(h5_path)

    if snapshot_type == "solve":
        return SolveSnapshot(
            model=loaded.get("model"),
            params=loaded.get("params"),
            V_arr_dict=loaded.get("V_arr_dict"),
            platform=saved_platform,
        )
    if snapshot_type == "simulate":
        return SimulateSnapshot(
            model=loaded.get("model"),
            params=loaded.get("params"),
            initial_conditions=loaded.get("initial_conditions"),
            V_arr_dict=loaded.get("V_arr_dict"),
            result=loaded.get("result"),
            platform=saved_platform,
        )
    return SolveAndSimulateSnapshot(
        model=loaded.get("model"),
        params=loaded.get("params"),
        initial_conditions=loaded.get("initial_conditions"),
        V_arr_dict=loaded.get("V_arr_dict"),
        result=loaded.get("result"),
        platform=saved_platform,
    )


def save_solve_snapshot(
    *,
    model: object,
    params: UserParams,
    V_arr_dict: VArrMapping,
    log_path: Path,
    log_keep_n_latest: int,
) -> Path:
    """Save a solve snapshot directory to disk.

    Args:
        model: The Model instance.
        params: User parameters passed to solve.
        V_arr_dict: Value function arrays from solve.
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
    _save_V_arr_to_h5(snap_dir / "arrays.h5", V_arr_dict)
    _write_metadata(snap_dir, snapshot_type="solve", fields=["model", "params"])
    _write_environment_files(snap_dir)

    _enforce_retention(
        log_path, prefix="solve_snapshot", keep_n_latest=log_keep_n_latest
    )
    return snap_dir


def save_simulate_snapshot(
    *,
    model: object,
    params: UserParams,
    initial_conditions: Mapping[str, Array],
    V_arr_dict: VArrMapping,
    result: object,
    log_path: Path,
    log_keep_n_latest: int,
) -> Path:
    """Save a simulate snapshot directory to disk.

    Args:
        model: The Model instance.
        params: User parameters passed to simulate.
        initial_conditions: Mapping of state names (plus `"regime_id"`) to arrays.
        V_arr_dict: Value function arrays.
        result: SimulationResult object.
        log_path: Parent directory for snapshot directories.
        log_keep_n_latest: Maximum number of snapshots to retain.

    Returns:
        Path to the created snapshot directory.

    """
    log_path.mkdir(parents=True, exist_ok=True)
    counter = _next_counter(log_path, prefix="simulate_snapshot")
    snap_dir = log_path / f"simulate_snapshot_{counter:03d}"
    snap_dir.mkdir()

    _save_pkl(snap_dir / "model.pkl", model)
    _save_pkl(snap_dir / "params.pkl", params)
    _save_pkl(snap_dir / "initial_conditions.pkl", initial_conditions)
    _save_pkl(snap_dir / "result.pkl", result)
    _save_V_arr_to_h5(snap_dir / "arrays.h5", V_arr_dict)
    _write_metadata(
        snap_dir,
        snapshot_type="simulate",
        fields=["model", "params", "initial_conditions", "result"],
    )
    _write_environment_files(snap_dir)

    _enforce_retention(
        log_path, prefix="simulate_snapshot", keep_n_latest=log_keep_n_latest
    )
    return snap_dir


def save_solve_and_simulate_snapshot(
    *,
    model: object,
    params: UserParams,
    initial_conditions: Mapping[str, Array],
    V_arr_dict: VArrMapping,
    result: object,
    log_path: Path,
    log_keep_n_latest: int,
) -> Path:
    """Save a solve-and-simulate snapshot directory to disk.

    Args:
        model: The Model instance.
        params: User parameters passed to solve_and_simulate.
        initial_conditions: Mapping of state names (plus `"regime_id"`) to arrays.
        V_arr_dict: Value function arrays.
        result: SimulationResult object.
        log_path: Parent directory for snapshot directories.
        log_keep_n_latest: Maximum number of snapshots to retain.

    Returns:
        Path to the created snapshot directory.

    """
    log_path.mkdir(parents=True, exist_ok=True)
    counter = _next_counter(log_path, prefix="solve_and_simulate_snapshot")
    snap_dir = log_path / f"solve_and_simulate_snapshot_{counter:03d}"
    snap_dir.mkdir()

    _save_pkl(snap_dir / "model.pkl", model)
    _save_pkl(snap_dir / "params.pkl", params)
    _save_pkl(snap_dir / "initial_conditions.pkl", initial_conditions)
    _save_pkl(snap_dir / "result.pkl", result)
    _save_V_arr_to_h5(snap_dir / "arrays.h5", V_arr_dict)
    _write_metadata(
        snap_dir,
        snapshot_type="solve_and_simulate",
        fields=["model", "params", "initial_conditions", "result"],
    )
    _write_environment_files(snap_dir)

    _enforce_retention(
        log_path,
        prefix="solve_and_simulate_snapshot",
        keep_n_latest=log_keep_n_latest,
    )
    return snap_dir


def save_solution(
    *,
    V_arr_dict: MappingProxyType[int, MappingProxyType[RegimeName, FloatND]],
    path: str | Path,
    protocol: int = pickle.HIGHEST_PROTOCOL,
) -> Path:
    """Save value function arrays from solve() to disk.

    Args:
        V_arr_dict: Immutable mapping of periods to regime value function arrays.
        path: File path to save the pickle.
        protocol: Pickle protocol version (default HIGHEST_PROTOCOL).

    Returns:
        The path where the object was saved.

    Raises:
        FileNotFoundError: If the parent directory does not exist.

    """
    return _atomic_dump(V_arr_dict, path, protocol=protocol)


def load_solution(
    *, path: str | Path
) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
    """Load value function arrays from disk.

    Args:
        path: File path to read the pickle from.

    Returns:
        Immutable mapping of periods to regime value function arrays.

    Raises:
        TypeError: If the loaded object is not a MappingProxyType.

    """
    p = Path(path)
    with p.open("rb") as f:
        obj = cloudpickle.load(f)

    if not isinstance(obj, MappingProxyType):
        raise TypeError(
            f"Pickle at {p} is {type(obj).__name__}, expected MappingProxyType"
        )
    return obj


def _get_platform() -> str:
    """Return a platform identifier string, e.g. `"x86_64-Linux"`."""
    return f"{platform.machine()}-{platform.system()}"


def _find_project_root() -> Path | None:
    """Walk up from this file to find the directory containing `pyproject.toml`."""
    current = Path(__file__).resolve().parent
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def _save_pkl(path: Path, obj: object) -> None:
    """Save an object to a pickle file atomically."""
    _atomic_dump(obj, path, protocol=pickle.HIGHEST_PROTOCOL)


def _save_V_arr_to_h5(
    path: Path,
    V_arr_dict: VArrMapping,
) -> None:
    """Write V_arr_dict to an HDF5 file.

    Datasets are stored at `/V_arr/{period}/{regime_name}`.

    """
    with h5py.File(path, "w") as fh:
        for period, regime_dict in V_arr_dict.items():
            for regime_name, arr in regime_dict.items():
                dataset_name = f"V_arr/{period}/{regime_name}"
                fh.create_dataset(dataset_name, data=np.asarray(arr))


def _load_V_arr_from_h5(path: Path) -> VArrMapping:
    """Read V_arr_dict from an HDF5 file.

    Returns:
        Immutable mapping matching the structure written by `_save_V_arr_to_h5`.

    """
    result: dict[int, dict[str, FloatND]] = {}
    with h5py.File(path, "r") as fh:
        v_arr_group = fh["V_arr"]
        for period_key in v_arr_group:
            period = int(period_key)
            result[period] = {}
            for regime_name in v_arr_group[period_key]:
                result[period][regime_name] = jnp.asarray(
                    v_arr_group[period_key][regime_name][()]
                )

    return MappingProxyType(
        {p: MappingProxyType(regimes) for p, regimes in result.items()}
    )


def _write_metadata(
    snap_dir: Path,
    *,
    snapshot_type: str,
    fields: list[str],
) -> None:
    """Write metadata.json into a snapshot directory."""
    metadata = {
        "snapshot_type": snapshot_type,
        "platform": _get_platform(),
        "fields": fields,
    }
    with (snap_dir / "metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2)


def _write_environment_files(snap_dir: Path) -> None:
    """Copy pixi.lock, pyproject.toml, and write REPRODUCE.md into the snapshot."""
    project_root = _find_project_root()
    if project_root is not None:
        for filename in ("pixi.lock", "pyproject.toml"):
            src = project_root / filename
            if src.exists():
                shutil.copy2(src, snap_dir / filename)

    platform_str = _get_platform()
    reproduce_md = f"""\
# Reproducing this run

1. Copy `pixi.lock` and `pyproject.toml` from this directory
2. Run `pixi install --frozen` to recreate the exact environment
3. Load the snapshot:

```python
from lcm import load_snapshot
snapshot = load_snapshot("{snap_dir.name}")
# Re-run: snapshot.model.solve(snapshot.params)
```

Platform: {platform_str}
"""
    (snap_dir / "REPRODUCE.md").write_text(reproduce_md)


def _next_counter(parent_path: Path, prefix: str) -> int:
    """Compute the next monotonic counter for snapshot directories with given prefix."""
    existing = sorted(parent_path.glob(f"{prefix}_*/"))
    if not existing:
        return 1
    last = existing[-1].name  # e.g. "solve_snapshot_003"
    try:
        return int(last.rsplit("_", 1)[1]) + 1
    except IndexError, ValueError:
        return len(existing) + 1


def _enforce_retention(parent_path: Path, prefix: str, *, keep_n_latest: int) -> None:
    """Delete oldest snapshot directories so that at most keep_n_latest remain."""
    existing = sorted(parent_path.glob(f"{prefix}_*/"))
    if len(existing) > keep_n_latest:
        for snap_dir in existing[: len(existing) - keep_n_latest]:
            shutil.rmtree(snap_dir)


def _atomic_dump(obj: object, path: str | Path, *, protocol: int) -> Path:
    """Serialize `obj` to `path` in an atomic (all-or-nothing) way.

    Args:
        obj: Object to serialize.
        path: File path to save the pickle.
        protocol: Int which indicates which protocol should be used by the pickler.
            The possible values are 0, 1, 2, 3, 4, 5. See
            https://docs.python.org/3/library/pickle.html.

    Returns:
        The path where the object was saved.

    """
    p = Path(path)
    if not p.parent.is_dir():
        raise FileNotFoundError(f"Parent directory does not exist: {p.parent}")

    tmp: Path | None = None
    try:
        # Write to a uniquely-named temp file in the *same directory* as the target.
        with tempfile.NamedTemporaryFile(mode="wb", dir=p.parent, delete=False) as f:
            tmp = Path(f.name)
            cloudpickle.dump(obj, f, protocol=protocol)

        # Atomic replace: after this line, readers either see the old file or the new
        # one, never a partially-written file. (Temp file is closed already, which
        # matters on Windows.)
        tmp.replace(p)
        tmp = None
        return p
    finally:
        # If anything failed before the replace succeeded, delete the temp file. We used
        # delete=False so we can close the file before replacing (needed on Windows), so
        # the context manager will not auto-delete it for us.
        if tmp is not None:
            with contextlib.suppress(OSError):
                tmp.unlink()
