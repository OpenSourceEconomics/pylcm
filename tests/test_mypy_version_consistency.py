"""Test mypy version consistency.

Ensures that the mypy version specified in pyproject.toml (pixi testing environment)
matches the one in .pre-commit-config.yaml. This is important to avoid
inconsistencies and potential issues when running type checks across different
environments.

"""

import tomllib
from pathlib import Path

import pytest
import yaml


@pytest.mark.skip(
    reason="ty is used instead of mypy. There is no pre-commit hook for ty yet."
)
def test_mypy_version_consistency():
    mypy_version_pyproject = _get_pixi_mypy_version()
    mypy_version_pre_commit = _get_precommit_mypy_version()
    assert mypy_version_pyproject == mypy_version_pre_commit


def _get_precommit_mypy_version() -> str:
    config = yaml.safe_load(Path(".pre-commit-config.yaml").read_text())
    mypy_config = next(
        hook
        for hook in config["repos"]
        if hook["repo"] == "https://github.com/pre-commit/mirrors-mypy"
    )
    version_str: str = mypy_config["rev"]
    return version_str.removeprefix("v")


def _get_pixi_mypy_version() -> str:
    config = tomllib.loads(Path("pyproject.toml").read_text())
    version_str: str = config["tool"]["pixi"]["feature"]["testing"]["dependencies"][
        "mypy"
    ]
    return version_str.removeprefix("==")
