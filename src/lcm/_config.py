from __future__ import annotations

from pathlib import Path

TEST_DATA = Path(__file__).parent.parent.parent.resolve().joinpath("tests", "data")

LOG_DIRECTORY = ".pylcm"


def get_log_dir() -> Path:
    """Returns the absolute path to the package's log directory.

    If the directory does not exist, it will be created.

    """
    # Find project root: assume current working dir if not configured
    project_root = Path.cwd()
    log_dir = project_root / LOG_DIRECTORY
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir
