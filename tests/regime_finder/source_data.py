"""Materialize Python source used as parser input without importing it."""

import atexit
from functools import cache
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory


@cache
def source_fixture_project() -> Path:
    temporary = TemporaryDirectory(prefix="regime-source-fixtures-")
    atexit.register(temporary.cleanup)
    destination = Path(temporary.name)
    source = Path(__file__).parent
    for name in ("examples", "source_fixtures"):
        shutil.copytree(source / name, destination / name)
    for path in destination.rglob("*.py.txt"):
        path.rename(path.with_suffix(""))
    return destination
