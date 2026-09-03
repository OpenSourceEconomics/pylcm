"""NBEGM-family kernels use each period's concrete function pool."""

import functools
from collections.abc import Callable

import numpy as np
import pytest

from lcm import AgeSpecializedFunction
from lcm.solvers import EGM
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.solution import test_egm_solver
from tests.test_models import n_nbegm_toy, nbegm_medicaid_toy


def _utility_with_bonus(
    *, base: Callable, bonus_age: float, bonus: float
) -> AgeSpecializedFunction:
    """Add a period-specific level without changing the optimal action."""

    def build(age: float) -> Callable:
        level = bonus if age == bonus_age else 0.0

        @functools.wraps(base)
        def utility(*args, **kwargs):
            return base(*args, **kwargs) + level

        return utility

    return AgeSpecializedFunction(build=build, signature=lambda age: age)


def test_egm_uses_the_utility_of_each_period_with_the_same_target():
    """Plain EGM does not reuse one age's core across a signature change."""
    solver = EGM(savings_grid=test_egm_solver._SAVINGS_GRID)
    baseline = (
        test_egm_solver._model(solver=solver)
        .solve(params=test_egm_solver._params(), log_level="debug")
        .values
    )
    specialized = (
        test_egm_solver._model(
            solver=solver,
            utility_function=_utility_with_bonus(
                base=test_egm_solver.utility,
                bonus_age=1.0,
                bonus=10.0,
            ),
        )
        .solve(params=test_egm_solver._params(), log_level="debug")
        .values
    )

    difference = np.asarray(specialized[1]["saving"]) - np.asarray(
        baseline[1]["saving"]
    )
    np.testing.assert_allclose(difference, 10.0, rtol=0.0, atol=2e-6)
    np.testing.assert_array_equal(
        np.asarray(specialized[2]["saving"]),
        np.asarray(baseline[2]["saving"]),
    )


def test_nbegm_uses_the_utility_of_each_period_with_the_same_target():
    """A current utility level enters only the age whose function declares it."""
    baseline = (
        nbegm_medicaid_toy.build_model(
            variant="nbegm",
            n_periods=4,
            envelope_arithmetic="ordinary",
        )
        .solve(params=nbegm_medicaid_toy.build_params(), log_level="debug")
        .values
    )
    specialized = (
        nbegm_medicaid_toy.build_model(
            variant="nbegm",
            n_periods=4,
            envelope_arithmetic="ordinary",
            utility_function=_utility_with_bonus(
                base=nbegm_medicaid_toy.utility,
                bonus_age=1.0,
                bonus=10.0,
            ),
        )
        .solve(params=nbegm_medicaid_toy.build_params(), log_level="debug")
        .values
    )

    difference = np.asarray(specialized[1]["alive"]) - np.asarray(baseline[1]["alive"])
    np.testing.assert_allclose(difference, 10.0, rtol=0.0, atol=2e-6)
    np.testing.assert_array_equal(
        np.asarray(specialized[2]["alive"]),
        np.asarray(baseline[2]["alive"]),
    )


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
def test_nnbegm_resolves_functions_before_rewriting_keeper_and_adjuster_pools():
    """Both nested branches retain the current period's concrete utility."""
    baseline = (
        n_nbegm_toy.build_model(variant="n_nbegm", n_periods=4)
        .solve(params={"discount_factor": 0.95}, log_level="debug")
        .values
    )
    specialized = (
        n_nbegm_toy.build_model(
            variant="n_nbegm",
            n_periods=4,
            utility_function=_utility_with_bonus(
                base=n_nbegm_toy.utility,
                bonus_age=25.0,
                bonus=10.0,
            ),
        )
        .solve(params={"discount_factor": 0.95}, log_level="debug")
        .values
    )

    difference = np.asarray(specialized[1]["alive"]) - np.asarray(baseline[1]["alive"])
    np.testing.assert_allclose(difference, 10.0, rtol=0.0, atol=2e-6)
    np.testing.assert_array_equal(
        np.asarray(specialized[2]["alive"]),
        np.asarray(baseline[2]["alive"]),
    )
