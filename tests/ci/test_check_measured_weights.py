"""A push is refused while a file the last CI run measured still has no weight."""

from collections.abc import Mapping

from tests.ci import ci_workloads
from tests.ci.check_measured_weights import (
    LEG_ARTIFACTS,
    CompletedRun,
    Skip,
    check_measured_weights,
)

_RUN = CompletedRun(run_id=123, attempt=1, head_sha="abc123")
_WEIGHTS = {"fp64-solution": {"tests/solution/test_old.py": 12.0}}
_PRESENT = frozenset({"tests/solution/test_old.py", "tests/solution/test_new.py"})


def _measured(_run: CompletedRun) -> Mapping[str, frozenset[str]]:
    return {"fp64-solution": frozenset(_PRESENT)}


def _measured_old_only(_run: CompletedRun) -> Mapping[str, frozenset[str]]:
    return {"fp64-solution": frozenset({"tests/solution/test_old.py"})}


def _unreachable(_run: CompletedRun) -> Mapping[str, frozenset[str]]:
    raise AssertionError("no artifact may be read")


def test_check_measured_weights_refuses_a_measured_file_with_no_weight() -> None:
    """A file the run measured but the manifest does not weigh blocks the push."""
    verdict = check_measured_weights(
        find_run=lambda: _RUN,
        read_measured=_measured,
        leg_weights=_WEIGHTS,
        present=_PRESENT,
        override=None,
    )

    assert not verdict.allowed


def test_check_measured_weights_names_the_file_and_the_refresh_command() -> None:
    """The refusal names the file, its leg and the refresh command for that leg."""
    verdict = check_measured_weights(
        find_run=lambda: _RUN,
        read_measured=_measured,
        leg_weights=_WEIGHTS,
        present=_PRESENT,
        override=None,
    )

    assert all(
        fragment in verdict.message
        for fragment in (
            "fp64-solution: tests/solution/test_new.py",
            "gh run download 123",
            "--refresh-leg fp64-solution",
        )
    )


def test_check_measured_weights_allows_a_file_added_after_the_run() -> None:
    """A file the run never executed stays allowed at median placement."""
    verdict = check_measured_weights(
        find_run=lambda: _RUN,
        read_measured=_measured_old_only,
        leg_weights=_WEIGHTS,
        present=_PRESENT,
        override=None,
    )

    assert verdict.allowed


def test_check_measured_weights_ignores_a_measured_file_deleted_since() -> None:
    """A file the run measured that no longer exists needs no weight."""
    verdict = check_measured_weights(
        find_run=lambda: _RUN,
        read_measured=_measured,
        leg_weights=_WEIGHTS,
        present=frozenset({"tests/solution/test_old.py"}),
        override=None,
    )

    assert verdict.allowed


def test_check_measured_weights_allows_a_branch_with_no_pull_request() -> None:
    """With no open PR there is no measurement to hold the manifest to."""
    verdict = check_measured_weights(
        find_run=lambda: Skip(reason="no open pull request for this branch"),
        read_measured=_unreachable,
        leg_weights=_WEIGHTS,
        present=_PRESENT,
        override=None,
    )

    assert (verdict.allowed, "no open pull request" in verdict.message) == (
        True,
        True,
    )


def test_check_measured_weights_allows_an_override_and_prints_its_reason() -> None:
    """An override passes without any lookup and leaves its reason in the output."""

    def find_run() -> CompletedRun:
        raise AssertionError("no lookup may run under an override")

    verdict = check_measured_weights(
        find_run=find_run,
        read_measured=_unreachable,
        leg_weights=_WEIGHTS,
        present=_PRESENT,
        override="weights refresh follows in the next push",
    )

    assert (
        verdict.allowed,
        "weights refresh follows in the next push" in verdict.message,
    ) == (True, True)


def test_check_measured_weights_knows_the_artifacts_of_every_weighted_leg() -> None:
    """Every leg the manifest weighs has a JUnit artifact pattern to read."""
    assert set(LEG_ARTIFACTS) == set(ci_workloads.leg_names())
