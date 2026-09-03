"""Production-seam tests for solve-input liveness commits."""

from typing import Any

import pytest

from _lcm.execution.liveness import PlannedInputLiveness
from _lcm.execution.value_transfer import ValueArtifactAddress
from _lcm.solution import backward_induction
from _lcm.typing import RegimeName
from tests.solution.test_grid_search_streaming_production_path import (
    _build_model,
    _solve_target,
)

type _Dispatch = tuple[int, RegimeName]
type _Ledger = PlannedInputLiveness[_Dispatch, ValueArtifactAddress]


def test_real_solve_rejects_one_suppressed_dispatch_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final closure names the exact real period-regime dispatch that was skipped."""
    real_commit = PlannedInputLiveness.commit_successful_dispatch

    # keyword-only-exempt: library-callback=pytest.MonkeyPatch.setattr
    def suppress_acting_commit(
        self: _Ledger,
        *,
        dispatch: _Dispatch,
    ) -> frozenset[ValueArtifactAddress]:
        if dispatch == (0, "acting"):
            return frozenset()
        return real_commit(self, dispatch=dispatch)

    monkeypatch.setattr(
        PlannedInputLiveness,
        "commit_successful_dispatch",
        suppress_acting_commit,
    )

    with pytest.raises(
        RuntimeError,
        match=r"planned dispatches uncommitted.*\(0, 'acting'\)",
    ):
        _solve_target(
            model=_build_model(),
            work=0.0,
            consumption=1.0,
        )


def test_failed_real_period_kernel_records_no_dispatch_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A core failure occurs before the solve loop can commit that dispatch ID."""
    real_commit = PlannedInputLiveness.commit_successful_dispatch
    observed: list[_Dispatch] = []

    # keyword-only-exempt: library-callback=pytest.MonkeyPatch.setattr
    def record_commit(
        self: _Ledger,
        *,
        dispatch: _Dispatch,
    ) -> frozenset[ValueArtifactAddress]:
        observed.append(dispatch)
        return real_commit(self, dispatch=dispatch)

    def fail_before_return(**_kwargs: Any) -> object:
        msg = "injected period-kernel failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(
        PlannedInputLiveness,
        "commit_successful_dispatch",
        record_commit,
    )
    monkeypatch.setattr(
        backward_induction,
        "_run_period_kernel",
        fail_before_return,
    )

    with pytest.raises(RuntimeError, match="injected period-kernel failure"):
        _solve_target(
            model=_build_model(),
            work=0.0,
            consumption=1.0,
        )

    assert observed == []
