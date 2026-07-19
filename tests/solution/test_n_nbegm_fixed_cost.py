"""NNBEGM branch aggregation under a uniform observed fixed cost.

Solver-level battery: the configuration guards (continuous-only, no leftover
shock state, resolvable per-period scale), and an end-to-end toy solve whose
aggregated value lies strictly between the keeper-only branch and the
deterministic hard maximum wherever the cutoff is interior, with the
analytic adjustment probability published through the solver diagnostics.
"""

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from jax import config as jax_config

import _lcm.solution.solvers as solvers_mod
from lcm import AdaptiveOuterMesh, UniformObservedFixedCost
from lcm.exceptions import RegimeInitializationError
from tests.test_models import n_nbegm_toy as toy

if TYPE_CHECKING:
    from _lcm.egm.carry import EGMCarry
    from _lcm.solution.contract import KernelResult

_PARAMS = {"discount_factor": 0.95}
_ORIGINAL_KERNEL_CALL = solvers_mod._NNBEGMPeriodKernel.__call__
_MESH = AdaptiveOuterMesh(
    initial_grid=toy.OUTER_GRID,
    max_nodes=513,
    max_refinement_rounds=10,
    value_atol=1e-4,
    value_rtol=1e-4,
    golden_iterations=40,
)
_AGGREGATOR = UniformObservedFixedCost(
    shock_name="adjustment_cost",
    scale_function="adjustment_scale",
    lower=0.0,
    upper=1.0,
)


def test_fixed_cost_requires_the_continuous_outer_search() -> None:
    with pytest.raises(RegimeInitializationError, match="AdaptiveOuterMesh"):
        toy.build_solver(variant="n_nbegm", branch_aggregator=_AGGREGATOR)


def test_unknown_scale_function_fails_at_build() -> None:
    aggregator = UniformObservedFixedCost(
        shock_name="adjustment_cost",
        scale_function="no_such_function",
        lower=0.0,
        upper=1.0,
    )
    with pytest.raises(RegimeInitializationError, match="no_such_function"):
        toy.build_model(
            variant="n_nbegm",
            n_periods=2,
            outer_search=_MESH,
            branch_aggregator=aggregator,
        ).solve(params=_PARAMS, log_level="off")


def test_state_dependent_scale_fails_at_build() -> None:
    aggregator = UniformObservedFixedCost(
        shock_name="adjustment_cost",
        # `resources` reads states — outside the per-period scalar scope.
        scale_function="resources",
        lower=0.0,
        upper=1.0,
    )
    with pytest.raises(RegimeInitializationError, match="flat params"):
        toy.build_model(
            variant="n_nbegm",
            n_periods=2,
            outer_search=_MESH,
            branch_aggregator=aggregator,
        ).solve(params=_PARAMS, log_level="off")


def _solve_recorded(
    *, aggregator: UniformObservedFixedCost | None, monkeypatch: pytest.MonkeyPatch
) -> dict[int, KernelResult]:
    recorded: dict[int, KernelResult] = {}

    def recording_call(
        self: solvers_mod._NNBEGMPeriodKernel,
        **kwargs: object,
    ) -> KernelResult:
        result = _ORIGINAL_KERNEL_CALL(self, **kwargs)  # ty: ignore[invalid-argument-type]
        recorded[cast("int", kwargs["period"])] = result
        return result

    monkeypatch.setattr(solvers_mod._NNBEGMPeriodKernel, "__call__", recording_call)
    toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        outer_search=_MESH,
        branch_aggregator=aggregator,
    ).solve(params=_PARAMS, log_level="off")
    return recorded


def test_aggregated_solve_publishes_probabilities_and_bounded_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not jax_config.read("jax_enable_x64"):
        pytest.skip("x64 run only")
    deterministic = _solve_recorded(aggregator=None, monkeypatch=monkeypatch)
    aggregated = _solve_recorded(aggregator=_AGGREGATOR, monkeypatch=monkeypatch)

    det = deterministic[0]
    agg = aggregated[0]
    assert det.diagnostics is not None
    assert det.diagnostics.adjustment_probability is None
    assert agg.diagnostics is not None
    probability = agg.diagnostics.adjustment_probability
    assert probability is not None
    assert probability.shape == agg.V_arr.shape
    p = np.asarray(probability)
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert np.any((p > 0.0) & (p < 1.0)), "no interior cutoff cell in the toy"

    v_det = np.asarray(det.V_arr)
    v_agg = np.asarray(agg.V_arr)
    both = np.isfinite(v_det) & np.isfinite(v_agg)
    assert np.all(v_agg[both] <= v_det[both] + 1e-10)
    assert np.any(v_agg[both] < v_det[both] - 1e-8)

    # No nested simulation payload under a fixed-cost aggregation: the read
    # would replay a hard maximum the solve did not use.
    assert agg.sim_policy is None

    # The expected marginal stays finite and the carry keeps its shape.
    carry = cast("EGMCarry", agg.carry)
    assert np.all(np.isfinite(np.asarray(carry.marginal_utility)))


def test_zero_width_support_is_rejected() -> None:
    with pytest.raises(RegimeInitializationError, match="support width"):
        UniformObservedFixedCost(
            shock_name="adjustment_cost",
            scale_function="adjustment_scale",
            lower=0.5,
            upper=0.5,
        )
