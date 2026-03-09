"""Persistence utilities for saving and loading LCM artifacts."""

import contextlib
import pickle
import tempfile
from pathlib import Path
from types import MappingProxyType

import cloudpickle

from lcm.typing import FloatND, RegimeName


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
