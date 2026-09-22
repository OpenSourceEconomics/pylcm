"""Finite host dispatch releases folded values while keeping required carry banks."""

import weakref
from typing import NamedTuple

import jax
import numpy as np
import pytest

from _lcm.solution.nbegm import _RideAlongNBEGMPeriodKernel
from _lcm.solution.nnbegm import _FiniteNNBEGMPeriodKernel
from lcm import ExecutionConfig
from lcm.solver_api import EGM_CONTINUATION, KernelOutput, ResultRetention
from tests.test_models import n_nbegm_toy as toy


class _ObservedSolve(NamedTuple):
    live_values: tuple[int, ...]
    value: np.ndarray
    carry: tuple[np.ndarray, ...]


def _observe_solve(
    *, width: int, retention: ResultRetention, monkeypatch: pytest.MonkeyPatch
) -> _ObservedSolve:
    """Count live node values at dispatch entry without retaining their arrays."""
    live_values: list[int] = []
    value_refs: list[weakref.ReferenceType] = []
    publication: list[tuple[np.ndarray, tuple[np.ndarray, ...]]] = []
    active_adjuster: object | None = None
    original_inner = _RideAlongNBEGMPeriodKernel.__call__
    original_outer = _FiniteNNBEGMPeriodKernel._solve_outer

    def record_inner(
        self: _RideAlongNBEGMPeriodKernel, **kwargs: object
    ) -> KernelOutput:
        if self is active_adjuster:
            live_values.append(sum(ref() is not None for ref in value_refs))
        result = original_inner(self, **kwargs)  # ty: ignore[invalid-argument-type]
        if self is active_adjuster:
            value_refs.append(weakref.ref(result.value))
        return result

    def record_outer(self: _FiniteNNBEGMPeriodKernel, **kwargs: object) -> KernelOutput:
        nonlocal active_adjuster
        active_adjuster = self.adjuster_kernel
        result = original_outer(self, **kwargs)  # ty: ignore[invalid-argument-type]
        publication.append(
            (
                np.array(result.value),
                tuple(
                    np.array(leaf)
                    for leaf in jax.tree_util.tree_leaves(
                        result.continuations[EGM_CONTINUATION]
                    )
                ),
            )
        )
        active_adjuster = None
        return result

    with monkeypatch.context() as patch:
        patch.setattr(_RideAlongNBEGMPeriodKernel, "__call__", record_inner)
        patch.setattr(_FiniteNNBEGMPeriodKernel, "_solve_outer", record_outer)
        toy.build_model(
            variant="n_nbegm",
            n_periods=2,
            execution_config=ExecutionConfig(axis_widths={"outer_candidate": width}),
        ).solve(
            params={"discount_factor": 0.95},
            log_level="off",
            retention=retention,
        )
    ((value, carry),) = publication
    return _ObservedSolve(tuple(live_values), value, carry)


@pytest.mark.parametrize("width", [1, 2, 4])
def test_values_only_releases_node_values_and_preserves_candidate_carries(
    *, width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the current chunk's values survive; complete carry/replay banks remain.

    Replay is a positive control: it consumes every per-node value, so the same
    weak-reference recorder must see all preceding values still alive there.
    The retention choice must leave the collapsed value and every carry leaf
    unchanged, including the nodes whose value temporaries were released.
    """
    values_only = _observe_solve(
        width=width, retention=ResultRetention.VALUES, monkeypatch=monkeypatch
    )
    with_replay = _observe_solve(
        width=width,
        retention=ResultRetention.VALUES_AND_REPLAY,
        monkeypatch=monkeypatch,
    )

    assert (values_only.live_values, with_replay.live_values) == (
        tuple(index % width for index in range(toy.N_OUTER)),
        tuple(range(toy.N_OUTER)),
    )
    np.testing.assert_array_equal(values_only.value, with_replay.value)
    assert len(values_only.carry) == len(with_replay.carry)
    for actual, expected in zip(values_only.carry, with_replay.carry, strict=True):
        np.testing.assert_array_equal(actual, expected)
