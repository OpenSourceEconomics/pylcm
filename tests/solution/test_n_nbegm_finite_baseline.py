"""Reference-output equivalence for finite NNBEGM candidate aggregation.

The finite solve must reproduce stored reference outputs rather than an
independently reimplemented aggregation. The fixture
`tests/data/n_nbegm_finite_baseline.npz` holds, per
`outer_batch_size` in {0, 1, 4} on the smooth two-asset toy (x64, 3 periods):
every alive period's collapsed `V_arr` and complete candidate-bank `EGMCarry`
leaves as returned by
`_NNBEGMPeriodKernel.__call__`, plus the public `Model.solve` output.

The reference includes the corrected lower-savings corner from `78dfa1a9` and
the explicit candidate carry bank required for exact policy replay by
`37582328`.

`V` and carry value/grid agree within 1e-12; carry marginals agree within 1e-11.
"""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import jax
import numpy as np
import pytest
from jax import config as jax_config

import _lcm.solution.nnbegm as solvers_mod
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
    ).solve(params=_PARAMS, log_level="debug")
    return solution, recorded


@pytest.mark.parametrize("outer_batch_size", [0, 1, 2, 4, 7])
def test_finite_streaming_fold_matches_frozen_corrected_baseline(
    outer_batch_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The streaming solve reproduces the frozen corrected finite-search arrays.

    Batch sizes 0/1/4 compare against their own frozen capture; 2 and 7 (the
    remaining supported chunk shapes compare against the batch-0 capture,
    which is valid because the fold order is the node order regardless of
    chunking.
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
        leaves = jax.tree_util.tree_leaves(result.continuation)
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
