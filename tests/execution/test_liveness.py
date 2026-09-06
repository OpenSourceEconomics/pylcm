"""Tests for conservative planned-input liveness accounting."""

from collections.abc import Callable, Hashable
from typing import cast

import pytest

from _lcm.execution.liveness import PlannedInputLiveness


def test_remaining_consumers_reach_eligibility_only_at_zero() -> None:
    ledger = PlannedInputLiveness(
        dispatch_accesses={
            "first-node": ("target-V",),
            "second-node": ("target-V",),
        },
    )

    assert ledger.remaining_consumers(artifact="target-V") == 2
    assert not ledger.is_release_eligible(artifact="target-V")

    assert not ledger.commit_successful_dispatch(dispatch="first-node")
    assert ledger.remaining_consumers(artifact="target-V") == 1
    assert not ledger.is_release_eligible(artifact="target-V")

    assert ledger.commit_successful_dispatch(dispatch="second-node") == frozenset(
        {"target-V"}
    )
    assert ledger.remaining_consumers(artifact="target-V") == 0
    assert ledger.is_release_eligible(artifact="target-V")


def test_multiple_artifacts_commit_atomically() -> None:
    ledger = PlannedInputLiveness(
        dispatch_accesses={
            "joint-node": ("first", "second"),
            "last-node": ("first",),
        },
    )

    assert ledger.commit_successful_dispatch(dispatch="joint-node") == frozenset(
        {"second"}
    )
    assert ledger.remaining_counts == {"first": 1, "second": 0}


def test_unknown_successful_dispatch_is_rejected_without_partial_decrement() -> None:
    ledger = PlannedInputLiveness(dispatch_accesses={"known-node": ("known",)})

    with pytest.raises(KeyError, match="unknown planned ID"):
        ledger.commit_successful_dispatch(dispatch="unknown-node")

    assert ledger.remaining_consumers(artifact="known") == 1
    assert ledger.pending_dispatches == frozenset({"known-node"})
    with pytest.raises(KeyError, match="Unknown planned input artifact"):
        ledger.remaining_consumers(artifact="unknown")


def test_duplicate_artifact_in_one_planned_dispatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate artifact"):
        PlannedInputLiveness(
            dispatch_accesses={"node": ("target-V", "target-V")},
        )


def test_duplicate_dispatch_cannot_mask_peer_with_identical_accesses() -> None:
    ledger = PlannedInputLiveness(
        dispatch_accesses={
            "first-node": ("target-V",),
            "second-node": ("target-V",),
        },
    )

    ledger.commit_successful_dispatch(dispatch="first-node")
    with pytest.raises(RuntimeError, match="already committed"):
        ledger.commit_successful_dispatch(dispatch="first-node")

    assert ledger.remaining_consumers(artifact="target-V") == 1
    assert ledger.pending_dispatches == frozenset({"second-node"})
    with pytest.raises(RuntimeError, match="second-node"):
        ledger.assert_solve_complete()

    ledger.commit_successful_dispatch(dispatch="second-node")
    ledger.assert_solve_complete()


def test_empty_access_dispatch_is_still_required_for_solve_closure() -> None:
    ledger = PlannedInputLiveness(dispatch_accesses={"empty-node": ()})

    with pytest.raises(RuntimeError, match="empty-node"):
        ledger.assert_solve_complete()

    assert not ledger.commit_successful_dispatch(dispatch="empty-node")
    ledger.assert_solve_complete()


def test_unplanned_consumer_pin_prevents_release_eligibility() -> None:
    ledger = PlannedInputLiveness(
        dispatch_accesses={"planned-node": ("mixed-route-V",)},
        pinned_artifacts=("mixed-route-V", "legacy-only-V"),
    )

    assert ledger.remaining_consumers(artifact="legacy-only-V") == 0
    assert not ledger.is_release_eligible(artifact="legacy-only-V")

    assert not ledger.commit_successful_dispatch(dispatch="planned-node")
    assert ledger.remaining_consumers(artifact="mixed-route-V") == 0
    assert not ledger.is_release_eligible(artifact="mixed-route-V")


def test_failed_dispatch_has_no_liveness_side_effect() -> None:
    ledger = PlannedInputLiveness(dispatch_accesses={"target-node": ("target-V",)})

    def dispatch(*, fail: bool) -> None:
        if fail:
            msg = "core failed"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="core failed"):
        dispatch(fail=True)

    # The runtime commits only after the core returns successfully.
    assert ledger.remaining_consumers(artifact="target-V") == 1
    assert ledger.pending_dispatches == frozenset({"target-node"})
    dispatch(fail=False)
    ledger.commit_successful_dispatch(dispatch="target-node")
    assert ledger.remaining_consumers(artifact="target-V") == 0


def test_successful_solve_rejects_uncommitted_planned_dispatch() -> None:
    ledger = PlannedInputLiveness(
        dispatch_accesses={
            "first-node": ("first",),
            "second-node": ("first", "second"),
        },
    )
    ledger.commit_successful_dispatch(dispatch="first-node")

    with pytest.raises(RuntimeError, match=r"Successful solve.*second-node"):
        ledger.assert_solve_complete()

    ledger.commit_successful_dispatch(dispatch="second-node")
    ledger.assert_solve_complete()


@pytest.mark.parametrize(
    "build",
    [
        lambda: PlannedInputLiveness(
            dispatch_accesses={"node": (cast("Hashable", ["not-hashable"]),)}
        ),
        lambda: PlannedInputLiveness(
            dispatch_accesses={},
            pinned_artifacts=(cast("Hashable", ["not-hashable"]),),
        ),
    ],
    ids=["planned", "pinned"],
)
def test_logical_artifact_keys_must_be_hashable(
    build: Callable[[], PlannedInputLiveness[str, object]],
) -> None:
    with pytest.raises(TypeError, match="hashable logical artifact keys"):
        build()


def test_a_retained_artifact_never_becomes_release_eligible() -> None:
    """Closing every planned consumer of a retained artifact releases nothing."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={"only-node": ("kept",)},
        retained_artifacts=("kept",),
    )

    ledger.commit_successful_dispatch(dispatch="only-node")

    assert not ledger.is_release_eligible(artifact="kept")


def test_a_commit_reports_no_retained_artifact_as_newly_eligible() -> None:
    """The commit's newly-eligible set excludes retained artifacts."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={"only-node": ("kept", "free")},
        retained_artifacts=("kept",),
    )

    assert ledger.commit_successful_dispatch(dispatch="only-node") == frozenset(
        {"free"}
    )


def test_is_retained_reads_the_declared_retention_obligation() -> None:
    """Retention is a declared fact of the artifact, not of its count."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={"node": ("kept",)}, retained_artifacts=("kept",)
    )

    assert ledger.is_retained(artifact="kept")


@pytest.mark.parametrize(
    ("committed", "eligible"),
    [
        (("first-node",), False),
        (("first-node", "second-node"), True),
    ],
)
def test_an_aliased_pair_is_eligible_only_when_both_counts_close(
    *, committed: tuple[str, ...], eligible: bool
) -> None:
    """Two keys on one buffer release together, after the later count closes."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={
            "first-node": ("later-key",),
            "second-node": ("earlier-key",),
        },
        aliases={"earlier-key": "later-key"},
    )
    for dispatch in committed:
        ledger.commit_successful_dispatch(dispatch=dispatch)

    assert ledger.is_release_eligible(artifact="later-key") is eligible


def test_alias_group_is_the_transitive_closure_of_the_alias_map() -> None:
    """A key rolled twice belongs to one group with both of its later keys."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={"node": ("t0", "t1", "t2")},
        aliases={"t0": "t1", "t1": "t2"},
    )

    assert ledger.alias_group(artifact="t0") == frozenset({"t0", "t1", "t2"})


def test_an_alias_target_outside_the_plan_is_added_with_a_zero_count() -> None:
    """A rolled buffer's home key is known to the ledger even when nobody reads it."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={"node": ("t0",)}, aliases={"t0": "t1"}
    )

    assert ledger.remaining_consumers(artifact="t1") == 0


def test_a_commit_reports_the_whole_group_when_the_last_count_closes() -> None:
    """Closing the last key of a group reports every key of the group."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={
            "first-node": ("later-key",),
            "second-node": ("earlier-key",),
        },
        aliases={"earlier-key": "later-key"},
    )
    ledger.commit_successful_dispatch(dispatch="first-node")

    assert ledger.commit_successful_dispatch(dispatch="second-node") == frozenset(
        {"earlier-key", "later-key"}
    )


@pytest.mark.parametrize(
    ("accesses", "dispatch", "expected"),
    [
        ({"a": ("x",)}, "a", True),
        ({"a": ("x",), "b": ("x",)}, "a", False),
        ({"a": ("y",)}, "a", False),
    ],
)
def test_has_sole_remaining_consumer_names_the_one_dispatch_left(
    *,
    accesses: dict[str, tuple[str, ...]],
    dispatch: str,
    expected: bool,
) -> None:
    """A dispatch is the sole remaining consumer when it is the last reader."""
    ledger = PlannedInputLiveness(dispatch_accesses=accesses)

    assert ledger.has_sole_remaining_consumer(artifact="x", dispatch=dispatch) is (
        expected
    )


def test_has_sole_remaining_consumer_is_false_after_that_dispatch_committed() -> None:
    """A committed dispatch does not count as a remaining consumer."""
    ledger = PlannedInputLiveness(dispatch_accesses={"a": ("x",)})
    ledger.commit_successful_dispatch(dispatch="a")

    assert not ledger.has_sole_remaining_consumer(artifact="x", dispatch="a")


def test_is_known_distinguishes_planned_from_foreign_artifacts() -> None:
    """The ledger answers membership without raising."""
    ledger = PlannedInputLiveness(dispatch_accesses={"a": ("x",)})

    assert (ledger.is_known(artifact="x"), ledger.is_known(artifact="y")) == (
        True,
        False,
    )


def test_a_pinned_artifact_pins_its_whole_alias_group() -> None:
    """A group with one pinned key is never eligible."""
    ledger = PlannedInputLiveness(
        dispatch_accesses={"a": ("t0",)},
        pinned_artifacts=("t1",),
        aliases={"t0": "t1"},
    )
    ledger.commit_successful_dispatch(dispatch="a")

    assert not ledger.is_release_eligible(artifact="t0")
