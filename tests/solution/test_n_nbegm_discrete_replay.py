"""Complete outer-times-discrete replay checks for N-NB-EGM.

The scalar oracle enumerates keeper plus every declared outer target crossed
with every insurance code. It optimizes consumption directly and never reads
the published policy bank, NNBEGM's candidate fold, or simulation's argmax.
"""

from bisect import bisect_right
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax.typing import DTypeLike

from _lcm.egm.published_policy import NNBEGMSimPolicy
from lcm import IrregSpacedGrid, LinSpacedGrid
from lcm.exceptions import InvalidSimulationInputError
from lcm.typing import UserParams
from tests.test_models import n_nbegm_discrete_toy as toy
from tests.test_models import n_nbegm_toy as smooth
from tests.test_models.n_nbegm_toy import RegimeId

_DISCOUNT_FACTOR = 0.95
_PREMIUM = 1.0
_PARAMS: UserParams = {
    "discount_factor": _DISCOUNT_FACTOR,
    "alive": {"premium": _PREMIUM},
}
_N_INNER_POINTS = 1025


def _profile_grids() -> dict[str, LinSpacedGrid]:
    return {
        "WEALTH_GRID": LinSpacedGrid(start=0.0, stop=30.0, n_points=smooth.N_WEALTH),
        "ILLIQUID_GRID": LinSpacedGrid(
            start=0.0, stop=20.0, n_points=smooth.N_ILLIQUID
        ),
        "CONSUMPTION_GRID": LinSpacedGrid(
            start=0.1, stop=20.0, n_points=smooth.N_CONSUMPTION
        ),
        "OUTER_GRID": LinSpacedGrid(start=0.0, stop=20.0, n_points=smooth.N_OUTER),
        "SAVINGS_GRID": LinSpacedGrid(
            start=smooth.SAVINGS_FLOOR, stop=35.0, n_points=60
        ),
    }


def _bracket(value: float, grid: Sequence[float]) -> tuple[int, int, float]:
    if value <= grid[0]:
        lower, upper = 0, 1
    elif value >= grid[-1]:
        lower, upper = len(grid) - 2, len(grid) - 1
    else:
        upper = bisect_right(grid, value)
        lower = upper - 1
    weight = (value - grid[lower]) / (grid[upper] - grid[lower])
    return lower, upper, weight


def _terminal_read(
    *, wealth: float, illiquid: float, scalar: type[np.floating]
) -> np.floating:
    wealth_grid = [float(scalar(x)) for x in np.asarray(smooth.WEALTH_GRID.to_jax())]
    illiquid_grid = [
        float(scalar(x)) for x in np.asarray(smooth.ILLIQUID_GRID.to_jax())
    ]
    w0, w1, ww = _bracket(wealth, wealth_grid)
    z0, z1, wz = _bracket(illiquid, illiquid_grid)

    def terminal(i: int, k: int) -> np.floating:
        return scalar(
            -scalar(smooth.TERMINAL_SCALE) / scalar(wealth_grid[i] + 1.0)
            - scalar(smooth.TERMINAL_SCALE) / scalar(illiquid_grid[k] + 1.0)
        )

    ww_s = scalar(ww)
    wz_s = scalar(wz)
    low = scalar(terminal(w0, z0) + wz_s * scalar(terminal(w0, z1) - terminal(w0, z0)))
    high = scalar(terminal(w1, z0) + wz_s * scalar(terminal(w1, z1) - terminal(w1, z0)))
    return scalar(low + ww_s * scalar(high - low))


def _score(
    *,
    wealth: float,
    illiquid: float,
    target: float,
    consumption: float,
    buy_private: int,
    scalar: type[np.floating],
) -> float:
    premium = scalar(_PREMIUM if buy_private else 0.0)
    resources = scalar(
        scalar(wealth)
        + scalar(smooth.LABOUR_INCOME)
        - scalar(scalar(target) - scalar(illiquid))
        - premium
    )
    savings = scalar(resources - scalar(consumption))
    next_wealth = scalar(scalar(1.0 + smooth.LIQUID_RATE) * savings)
    continuation = _terminal_read(
        wealth=float(next_wealth), illiquid=float(scalar(target)), scalar=scalar
    )
    exponent = scalar(1.0 - smooth.RISK_AVERSION)
    flow = scalar(scalar(consumption) ** exponent / exponent)
    if buy_private:
        flow = scalar(flow + scalar(toy.INSURANCE_UTILITY))
    return float(scalar(flow + scalar(_DISCOUNT_FACTOR) * continuation))


def _enumerate(
    *, wealth: float, illiquid: float, scalar: type[np.floating]
) -> tuple[float, float, float, int, int]:
    lower = float(scalar(smooth.CONSUMPTION_GRID.to_jax()[0]))
    upper_support = float(scalar(smooth.CONSUMPTION_GRID.to_jax()[-1]))
    outer_targets = [
        float(scalar(illiquid)),
        *[float(scalar(x)) for x in np.asarray(smooth.OUTER_GRID.to_jax())],
    ]
    best: tuple[float, float, float, int, int] | None = None
    candidate_index = 0
    for target in outer_targets:
        for buy_private in (int(toy.BuyPrivate.no), int(toy.BuyPrivate.yes)):
            premium = _PREMIUM if buy_private else 0.0
            resources = wealth + smooth.LABOUR_INCOME - (target - illiquid) - premium
            upper = min(upper_support, resources - smooth.SAVINGS_FLOOR)
            if np.isfinite(upper) and upper >= lower:
                inner_best: tuple[float, float] | None = None
                for point in range(_N_INNER_POINTS):
                    fraction = scalar(point / (_N_INNER_POINTS - 1))
                    consumption = float(
                        scalar(scalar(lower) + fraction * scalar(upper - lower))
                    )
                    value = _score(
                        wealth=wealth,
                        illiquid=illiquid,
                        target=target,
                        consumption=consumption,
                        buy_private=buy_private,
                        scalar=scalar,
                    )
                    if inner_best is None or value > inner_best[0]:
                        inner_best = value, consumption
                assert inner_best is not None
                candidate = (
                    inner_best[0],
                    target,
                    inner_best[1],
                    buy_private,
                    candidate_index,
                )
                if best is None or candidate[0] > best[0]:
                    best = candidate
            candidate_index += 1
    assert best is not None
    return best


def _simulate(*, points: tuple[float, ...], dtype: DTypeLike):
    model = toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        illiquid_investment_grid=IrregSpacedGrid(points=points),
    )
    initial_conditions = {
        "wealth": jnp.asarray([4.0, 15.0, 24.0], dtype=dtype),
        "illiquid": jnp.asarray([0.0, 12.0, 20.0], dtype=dtype),
        "age": jnp.full(3, 20.0, dtype=dtype),
        "regime_id": jnp.full(3, RegimeId.alive, dtype=jnp.int32),
    }
    return (
        model.simulate(
            params=_PARAMS,
            initial_conditions=initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level="off",
            seed=41,
        )
        .to_dataframe()
        .query("regime_name == 'alive' and period == 0")
        .sort_index()
    )


def test_policy_bank_is_the_declared_outer_times_discrete_product() -> None:
    """The published bank retains exact product order and categorical metadata."""
    _values, policies = toy.build_model(variant="n_nbegm", n_periods=2).solve(
        params=_PARAMS, log_level="off", return_simulation_policy=True
    )
    policy = policies[0]["alive"]
    assert isinstance(policy, NNBEGMSimPolicy)
    n_outer = 1 + smooth.N_OUTER
    assert policy.candidate_inner_action.shape == (
        2 * n_outer,
        smooth.N_WEALTH,
        smooth.N_ILLIQUID,
    )
    assert policy.candidate_value.shape == policy.candidate_inner_action.shape
    assert policy.candidate_outer_target.shape == policy.candidate_inner_action.shape
    assert policy.discrete_action_names == ("buy_private",)
    np.testing.assert_array_equal(
        np.asarray(policy.candidate_discrete_actions)[:, 0],
        np.tile([toy.BuyPrivate.no, toy.BuyPrivate.yes], n_outer),
    )


def test_separate_solve_and_simulate_replays_the_exact_nnbegm_policy() -> None:
    """A returned NNBEGM policy reproduces automatic solve-and-simulate."""
    foreign_grid = IrregSpacedGrid(points=(-20.0, 0.01, 20.0))
    model = toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        illiquid_investment_grid=foreign_grid,
    )
    initial_conditions = {
        "wealth": jnp.asarray([4.0, 15.0, 24.0]),
        "illiquid": jnp.asarray([0.0, 12.0, 20.0]),
        "age": jnp.full(3, 20.0),
        "regime_id": jnp.full(3, RegimeId.alive, dtype=jnp.int32),
    }

    automatic = model.simulate(
        params=_PARAMS,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="off",
        seed=41,
    ).to_dataframe()
    values, policies = model.solve(
        params=_PARAMS,
        log_level="off",
        return_simulation_policy=True,
    )
    separate = model.simulate(
        params=_PARAMS,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=values,
        policies=policies,
        log_level="off",
        seed=41,
    ).to_dataframe()

    pd.testing.assert_frame_equal(separate, automatic)


def test_separate_nnbegm_simulation_requires_the_returned_policy() -> None:
    """NNBEGM values alone fail closed instead of selecting a foreign grid action."""
    model = toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        illiquid_investment_grid=IrregSpacedGrid(points=(-20.0, 0.01, 20.0)),
    )
    initial_conditions = {
        "wealth": jnp.asarray([4.0]),
        "illiquid": jnp.asarray([12.0]),
        "age": jnp.asarray([20.0]),
        "regime_id": jnp.asarray([RegimeId.alive], dtype=jnp.int32),
    }
    values = model.solve(params=_PARAMS, log_level="off")

    with pytest.raises(
        InvalidSimulationInputError,
        match=r"NNBEGM.*policy",
    ):
        model.simulate(
            params=_PARAMS,
            initial_conditions=initial_conditions,
            period_to_regime_to_V_arr=values,
            log_level="off",
            seed=41,
        )


@pytest.mark.parametrize("use_x64", [False, True], ids=["fp32", "fp64"])
def test_public_replay_is_foreign_grid_invariant_and_matches_scalar_oracle(
    *, use_x64: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact candidate identity survives grid mutation in both required profiles."""
    scalar = np.float64 if use_x64 else np.float32
    dtype: DTypeLike = jnp.float64 if use_x64 else jnp.float32
    equality_tolerance = 2e-11 if use_x64 else 4e-5
    score_tolerance = 3e-10 if use_x64 else 8e-5
    with jax.enable_x64(use_x64):
        for name, grid in _profile_grids().items():
            monkeypatch.setattr(smooth, name, grid)
        foreign_points = (-20.0, 0.01, 20.0)
        replay = _simulate(points=foreign_points, dtype=dtype)

        winner_indices: list[int] = []
        replayed_investments: list[float] = []
        for row in replay.itertuples():
            oracle_value, target, consumption, buy_private, winner = _enumerate(
                wealth=float(row.wealth),
                illiquid=float(row.illiquid),
                scalar=scalar,
            )
            winner_indices.append(winner)
            replayed_investments.append(float(row.illiquid_investment))
            np.testing.assert_allclose(
                float(row.illiquid + row.illiquid_investment),
                target,
                rtol=equality_tolerance,
                atol=equality_tolerance,
            )
            assert int(getattr(toy.BuyPrivate, str(row.buy_private))) == buy_private
            np.testing.assert_allclose(
                float(row.consumption), consumption, rtol=0.0, atol=0.20
            )
            direct = _score(
                wealth=float(row.wealth),
                illiquid=float(row.illiquid),
                target=target,
                consumption=float(row.consumption),
                buy_private=buy_private,
                scalar=scalar,
            )
            np.testing.assert_allclose(
                float(row.value), direct, rtol=score_tolerance, atol=score_tolerance
            )
            np.testing.assert_allclose(
                float(row.value), oracle_value, rtol=0.0, atol=0.004
            )

        # The changed simulation action grid is deliberately foreign to the
        # declared NNBEGM outer targets. At least one replayed action must lie
        # outside it, proving the grid did not define the solved candidate set.
        assert any(
            not any(
                np.isclose(action, point, rtol=0.0, atol=equality_tolerance)
                for point in foreign_points
            )
            for action in replayed_investments
        )
        assert any(index % 2 == int(toy.BuyPrivate.yes) for index in winner_indices)
        assert any(index >= 2 for index in winner_indices)
