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
