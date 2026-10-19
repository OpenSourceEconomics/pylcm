"""Probe native-test setup without invoking any solve, simulation or capture."""

from pathlib import Path
from typing import Never

import pytest

from tests.solution import test_nb_income_scientific_contract as native_test

_EXTERNAL = (
    "NB_SCIENTIFIC_DECISION",
    "NB_COMPLETE_OBSERVER_SOURCE_MAP",
    "NB_COMPLETE_OBSERVER_SOURCE_ROOT",
    "NB_COMPLETE_OBSERVER_OUTPUT",
)


class _CaptureReachedError(RuntimeError):
    """A setup probe reached its single intended capture boundary."""


def _probe_capture(*, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path | None]:
    """Replace the existing capture with a typed refusal before numerical work."""
    seen: dict[str, Path | None] = {}
    owned_monkeypatch = monkeypatch

    def observe(
        *,
        root: Path,
        monkeypatch: pytest.MonkeyPatch,
        source_map: Path | None = None,
        source_root: Path | None = None,
    ) -> Never:
        assert monkeypatch is owned_monkeypatch
        seen.update(root=root, source_map=source_map, source_root=source_root)
        raise _CaptureReachedError("capture boundary reached without a solve")

    monkeypatch.setattr(native_test, "capture_original_nb_income", observe)
    for name in _EXTERNAL:
        monkeypatch.delenv(name, raising=False)
    return seen


def test_ordinary_scientific_test_needs_no_external_artifact_environment(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ordinary test reaches one temporary capture without external files."""
    seen = _probe_capture(monkeypatch=monkeypatch)
    with pytest.raises(_CaptureReachedError, match="without a solve"):
        native_test.test_original_nb_income_meets_independent_scientific_contract(
            monkeypatch=monkeypatch, tmp_path=tmp_path
        )
    assert seen == {
        "root": tmp_path / "capture",
        "source_map": None,
        "source_root": None,
    }


def test_optional_export_uses_the_same_single_capture_boundary(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An export destination changes only the same invocation's artifact root."""
    seen = _probe_capture(monkeypatch=monkeypatch)
    exported = tmp_path / "exported-capture"
    monkeypatch.setenv("NB_COMPLETE_OBSERVER_OUTPUT", str(exported))
    with pytest.raises(_CaptureReachedError, match="without a solve"):
        native_test.test_original_nb_income_meets_independent_scientific_contract(
            monkeypatch=monkeypatch, tmp_path=tmp_path
        )
    assert seen == {
        "root": exported,
        "source_map": None,
        "source_root": None,
    }


def test_supplied_decision_file_remains_authenticated(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An optional decision file with foreign contents refuses before capture."""
    seen = _probe_capture(monkeypatch=monkeypatch)
    decision = tmp_path / "wrong-decision.json"
    decision.write_text("{}\n")
    monkeypatch.setenv("NB_SCIENTIFIC_DECISION", str(decision))
    with pytest.raises(AssertionError):
        native_test.test_original_nb_income_meets_independent_scientific_contract(
            monkeypatch=monkeypatch, tmp_path=tmp_path
        )
    assert seen == {}
