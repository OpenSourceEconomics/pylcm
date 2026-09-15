"""Reference-output equivalence for finite NNBEGM candidate aggregation.

The finite solve must reproduce stored reference outputs rather than an
independently reimplemented aggregation. The fixture
`tests/data/n_nbegm_finite_baseline.npz` holds, on the smooth two-asset toy
(x64, 3 periods), every alive period's collapsed `V_arr` and complete
candidate-bank `EGMCarry` leaves as returned by `_NNBEGMPeriodKernel.__call__`,
plus the public `Model.solve` output.

The outer collapse is a host loop over the candidate nodes, so its dispatch
width reschedules the loop without changing the fold order; every width is
compared against the same capture.

`V` and carry value/grid agree within 1e-12; carry marginals agree within 1e-11.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

import jax
import numpy as np
import pytest
from jax import config as jax_config

import _lcm.solution.nnbegm as solvers_mod
from lcm import ExecutionConfig
from lcm.solver_api import EGM_CONTINUATION
from lcm.typing import FloatND
from tests.test_models import n_nbegm_toy as toy

if TYPE_CHECKING:
    from lcm.solver_api import KernelOutput

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


def _host_copy(*, result: KernelOutput) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Copy one period's value and carry leaves to the host as they are produced.

    The nested node declares the leaves it reads, so the solve releases a
    period's carry once the period that reads it has run; a recorder holding the
    device arrays would read them after that release.
    """
    carry = result.continuations[EGM_CONTINUATION]
    return (
        np.asarray(result.value),
        tuple(np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(carry)),
    )


def _solve_recording_kernel_results(
    *, width: int | None, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    Mapping[int, Mapping[str, FloatND]],
    dict[int, tuple[np.ndarray, tuple[np.ndarray, ...]]],
]:
    """Solve the toy, recording each period's `KernelOutput` arrays on the host."""
    recorded: dict[int, tuple[np.ndarray, tuple[np.ndarray, ...]]] = {}
    original_call = solvers_mod._NNBEGMPeriodKernel.__call__

    def recording_call(
        self: solvers_mod._NNBEGMPeriodKernel,
        **kwargs: object,
    ) -> KernelOutput:
        result = original_call(self, **kwargs)  # ty: ignore[invalid-argument-type]
        recorded[cast("int", kwargs["period"])] = _host_copy(result=result)
        return result

    monkeypatch.setattr(
        solvers_mod._NNBEGMPeriodKernel,
        "__call__",
        recording_call,
    )
    solution = (
        toy.build_model(
            variant="n_nbegm",
            n_periods=_N_PERIODS,
            execution_config=ExecutionConfig(
                axis_widths={} if width is None else {"outer_candidate": width}
            ),
        )
        .solve(params=_PARAMS, log_level="debug")
        .values
    )
    return solution, recorded


@pytest.mark.parametrize("width", [None, 1, 2, 4, 7])
def test_finite_streaming_fold_matches_frozen_corrected_baseline(
    *, width: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The streaming solve reproduces the frozen corrected finite-search arrays.

    Every dispatch width compares against the whole-axis capture, which is
    valid because the fold order is the node order regardless of chunking.
    """
    if not jax_config.read("jax_enable_x64"):
        pytest.skip("baseline frozen under x64")
    baseline = np.load(_BASELINE)
    tag = "b0"

    solution, recorded = _solve_recording_kernel_results(
        width=width, monkeypatch=monkeypatch
    )

    for period in _ALIVE_PERIODS:
        value, leaves = recorded[period]
        np.testing.assert_allclose(
            value,
            baseline[f"{tag}:p{period}:V_arr"],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"V_arr at period {period}, width {width}",
        )
        assert len(leaves) == len(_CARRY_LEAVES)
        for index, leaf in enumerate(leaves):
            name, rtol, atol = _CARRY_LEAVES[index]
            np.testing.assert_allclose(
                leaf,
                baseline[f"{tag}:p{period}:carry[<flat index {index}>]"],
                rtol=rtol,
                atol=atol,
                err_msg=f"carry.{name} at period {period}, width {width}",
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
