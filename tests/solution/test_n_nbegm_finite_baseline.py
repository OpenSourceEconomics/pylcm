"""Frozen-baseline equivalence for the NNBEGM candidate-bank refactor.

The candidate-bank refactor is required to be numerically neutral: the finite
collapse must reproduce the pre-refactor incremental sweep. That claim is
tested against arrays frozen from the pre-refactor HEAD (`a1d9ca7` on
`feat/nested-nbegm-ez`), not against a reimplementation of the old fold — a
test that re-types the fold would pass while the shipped code did something
else. The fixture `tests/data/n_nbegm_finite_baseline.npz` holds, per
`outer_batch_size` in {0, 1, 4} on the smooth two-asset toy (x64, 3 periods):
every alive period's `V_arr` and `EGMCarry` leaves as returned by
`_NNBEGMPeriodKernel.__call__`, plus the public `Model.solve` output.

Tolerances per the design freeze (see the continuous-outer ADR): `V` and carry
value/grid within 1e-12, the carry marginal within 1e-11.
"""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import jax
import numpy as np
import pytest
from jax import config as jax_config

import _lcm.solution.solvers as solvers_mod
from _lcm.egm.outer_candidates import OuterCandidateBank
from tests.test_models import n_nbegm_toy as toy

if TYPE_CHECKING:
    from _lcm.solution.contract import KernelResult
    from _lcm.typing import PeriodToRegimeToVArr

_PARAMS = {"discount_factor": 0.95}
_BASELINE = Path(__file__).parent.parent / "data" / "n_nbegm_finite_baseline.npz"
_N_PERIODS = 3
_ALIVE_PERIODS = (0, 1)
# Flatten order of the EGMCarry pytree (breakpoints is None and drops out).
_CARRY_LEAVES = {
    0: ("endog_grid", 1e-12, 1e-12),
    1: ("value", 1e-12, 1e-12),
    2: ("marginal_utility", 1e-11, 1e-12),
    3: ("taste_shock_scale", 1e-12, 1e-12),
}


def _solve_recording_kernel_results(
    *, outer_batch_size: int, monkeypatch: pytest.MonkeyPatch
) -> tuple[PeriodToRegimeToVArr, dict[int, KernelResult]]:
    """Solve the toy, recording each period's raw `KernelResult`."""
    recorded: dict[int, KernelResult] = {}
    original_call = solvers_mod._NNBEGMPeriodKernel.__call__

    def recording_call(
        self: solvers_mod._NNBEGMPeriodKernel,
        **kwargs: object,
    ) -> KernelResult:
        result = original_call(self, **kwargs)  # ty: ignore[invalid-argument-type]
        recorded[cast("int", kwargs["period"])] = result
        return result

    monkeypatch.setattr(
        solvers_mod._NNBEGMPeriodKernel,
        "__call__",
        recording_call,
    )
    solution = toy.build_model(
        variant="n_nbegm",
        outer_batch_size=outer_batch_size,
        n_periods=_N_PERIODS,
    ).solve(params=_PARAMS, log_level="off")
    return solution, recorded


@pytest.mark.parametrize("outer_batch_size", [0, 1, 2, 4, 7])
def test_candidate_bank_collapse_matches_frozen_prerefactor_baseline(
    outer_batch_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bank-based solve reproduces the frozen pre-refactor arrays.

    Batch sizes 0/1/4 compare against their own frozen capture; 2 and 7 (the
    §24 acceptance set's remaining sizes) compare against the batch-0 capture,
    which is valid because the fold order is the node order regardless of
    chunking — batching bounds solve dispatch, never the collapse.
    """
    if not jax_config.read("jax_enable_x64"):
        pytest.skip("baseline frozen under x64")
    baseline = np.load(_BASELINE)
    tag = f"b{outer_batch_size}" if outer_batch_size in (0, 1, 4) else "b0"

    solution, recorded = _solve_recording_kernel_results(
        outer_batch_size=outer_batch_size, monkeypatch=monkeypatch
    )

    for period in _ALIVE_PERIODS:
        result = recorded[period]
        np.testing.assert_allclose(
            np.asarray(result.V_arr),
            baseline[f"{tag}:p{period}:V_arr"],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"V_arr at period {period}, batch {outer_batch_size}",
        )
        leaves = jax.tree_util.tree_leaves(result.carry)
        assert len(leaves) == len(_CARRY_LEAVES)
        for index, leaf in enumerate(leaves):
            name, rtol, atol = _CARRY_LEAVES[index]
            np.testing.assert_allclose(
                np.asarray(leaf),
                baseline[f"{tag}:p{period}:carry[<flat index {index}>]"],
                rtol=rtol,
                atol=atol,
                err_msg=(f"carry.{name} at period {period}, batch {outer_batch_size}"),
            )

    for period, regime_to_v in solution.items():
        for regime, v_arr in regime_to_v.items():
            np.testing.assert_allclose(
                np.asarray(v_arr),
                baseline[f"{tag}:solve:p{period}:{regime}"],
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"solve V at period {period}, regime {regime}",
            )


def test_candidate_bank_holds_one_entry_per_outer_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bank stacks exactly one exact solve per outer node, keeper excluded.

    Captures the bank the period kernel builds during a real solve and checks
    the §24 structural criteria: one candidate per outer-grid node, in grid
    order, all marked valid, with the keeper nowhere in the bank (its
    state-dependent outer action cannot share the one-node-per-candidate
    layout).
    """
    banks: list[OuterCandidateBank] = []
    original_build = solvers_mod._NNBEGMPeriodKernel._build_candidate_bank

    def recording_build(
        self: solvers_mod._NNBEGMPeriodKernel,
        **kwargs: object,
    ) -> OuterCandidateBank:
        bank = original_build(self, **kwargs)  # ty: ignore[invalid-argument-type]
        banks.append(bank)
        return bank

    monkeypatch.setattr(
        solvers_mod._NNBEGMPeriodKernel,
        "_build_candidate_bank",
        recording_build,
    )
    toy.build_model(variant="n_nbegm", n_periods=2).solve(
        params=_PARAMS, log_level="off"
    )

    assert banks, "the period kernel never built a candidate bank"
    for bank in banks:
        assert bank.n_candidates == toy.N_OUTER
        np.testing.assert_array_equal(
            np.asarray(bank.outer_nodes), np.asarray(toy.OUTER_GRID.to_jax())
        )
        assert bool(np.all(np.asarray(bank.candidate_valid)))
        assert np.asarray(bank.V_arr).shape[0] == toy.N_OUTER
        # Each candidate's carry slices back out with the leading axis removed.
        first = bank.candidate_carry(0)
        stacked_leaves = jax.tree_util.tree_leaves(bank.carry)
        sliced_leaves = jax.tree_util.tree_leaves(first)
        for stacked, sliced in zip(stacked_leaves, sliced_leaves, strict=True):
            assert stacked.shape[0] == toy.N_OUTER
            assert stacked.shape[1:] == sliced.shape
