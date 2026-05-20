"""Low-level I/O helpers for snapshot and solution persistence."""

import contextlib
import json
import pickle
import platform
import shutil
import tempfile
import textwrap
from pathlib import Path
from types import MappingProxyType

import cloudpickle
import h5py
import jax.numpy as jnp
import numpy as np

from lcm.typing import FloatND, PeriodToRegimeToVArr


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


def _save_h5(
    path: Path,
    period_to_regime_to_V_arr: PeriodToRegimeToVArr,
) -> None:
    """Write value function arrays to an HDF5 file.

    Datasets are stored at `/{period}/{regime_name}/V_arr`.

    """
    with h5py.File(path, "w") as fh:
        for period, regime_dict in period_to_regime_to_V_arr.items():
            for regime_name, arr in regime_dict.items():
                dataset_name = f"{period}/{regime_name}/V_arr"
                fh.create_dataset(dataset_name, data=np.asarray(arr))


def _load_h5(path: Path) -> PeriodToRegimeToVArr:
    """Read value function arrays from an HDF5 file.

    Returns:
        Immutable mapping matching the structure written by `_save_h5`.

    """
    result: dict[int, dict[str, FloatND]] = {}
    with h5py.File(path, "r") as fh:
        for period_key in fh:
            period = int(period_key)
            result[period] = {}
            for regime_name in fh[period_key]:
                result[period][regime_name] = jnp.asarray(
                    fh[period_key][regime_name]["V_arr"][()]
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
    reproduce_md = textwrap.dedent(f"""\
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
    """)
    (snap_dir / "REPRODUCE.md").write_text(reproduce_md)


def _snapshot_counter(entry: Path, prefix: str) -> int:
    """Parse the numeric counter suffix of a snapshot directory.

    Returns `-1` for a name that does not end in an integer, so callers can
    skip foreign directories rather than mis-order them.
    """
    try:
        return int(entry.name.removeprefix(f"{prefix}_"))
    except ValueError:
        return -1


def _next_counter(parent_path: Path, prefix: str) -> int:
    """Compute the next monotonic counter for snapshot directories with given prefix."""
    counters = [
        counter
        for entry in parent_path.glob(f"{prefix}_*/")
        if (counter := _snapshot_counter(entry, prefix)) >= 0
    ]
    return max(counters, default=0) + 1


def _enforce_retention(parent_path: Path, prefix: str, *, keep_n_latest: int) -> None:
    """Delete oldest snapshot directories so that at most keep_n_latest remain.

    Directories are ordered by their parsed integer counter, not by name, so
    retention stays correct once the counter grows past the zero-padded width
    (e.g. `snapshot_1000` is newer than `snapshot_999`).
    """
    existing = sorted(
        (
            entry
            for entry in parent_path.glob(f"{prefix}_*/")
            if _snapshot_counter(entry, prefix) >= 0
        ),
        key=lambda entry: _snapshot_counter(entry, prefix),
    )
    for snap_dir in existing[: max(0, len(existing) - keep_n_latest)]:
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
        if tmp is not None:
            with contextlib.suppress(OSError):
                tmp.unlink()
