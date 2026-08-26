"""Independent candidate-set checks for N-NB-EGM.

The direct oracle below never reads the published simulation policy, calls the
generic simulation argmax, or reuses NNBEGM's chunked candidate fold. It loops
in scalar arithmetic over keeper plus ``OUTER_GRID`` and, conditional on each
outer target, a dense feasible consumption grid. Continuation values are read
with a separate scalar implementation of the public nearest-segment multilinear
state-grid convention.
"""

from bisect import bisect_right
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

from lcm import IrregSpacedGrid, LinSpacedGrid
from tests.test_models import n_nbegm_toy as toy
from tests.test_models.n_nbegm_toy import RegimeId

_PARAMS = {"discount_factor": 0.95}
_N_INNER_POINTS = 4097


def test_the_direct_variant_chooses_the_post_decision_stock_itself() -> None:
    """`brute` acts on `new_illiquid` over the nested solver's outer grid."""
    model = toy.build_model(variant="brute", n_periods=2, illiquid_grid=toy.OUTER_GRID)
    alive = model.user_regimes["alive"]
    assert set(alive.actions) == {"consumption", "new_illiquid"}


def test_the_direct_variant_reaches_exactly_the_outer_grid() -> None:
    """Its durable candidates are the outer grid — no extras, none missing."""
    model = toy.build_model(variant="brute", n_periods=2, illiquid_grid=toy.OUTER_GRID)
    durable_action = model.user_regimes["alive"].actions["new_illiquid"]
    assert durable_action is not None
    np.testing.assert_allclose(
        np.asarray(durable_action.to_jax()), np.asarray(toy.OUTER_GRID.to_jax())
    )


def test_the_keeper_candidate_costs_the_direct_variant_nothing_extra() -> None:
    """Holding the stock is free in both arms, so the keeper is shared."""
    stock = toy.OUTER_GRID.to_jax()
    np.testing.assert_allclose(
        np.asarray(toy.credited(illiquid=stock, new_illiquid=stock)),
        np.zeros(stock.shape),
        atol=0.0,
    )


def _bracket_with_nearest_segment_extrapolation(
    value: float,
    grid: Sequence[float],
) -> tuple[int, int, float]:
    """Return scalar interpolation indices and the upper-node weight."""
    if value <= grid[0]:
        lower, upper = 0, 1
    elif value >= grid[-1]:
        lower, upper = len(grid) - 2, len(grid) - 1
    else:
        upper = bisect_right(grid, value)
        lower = upper - 1
    weight = (value - grid[lower]) / (grid[upper] - grid[lower])
    return lower, upper, weight


def _terminal_grid(
    scalar: type[np.floating],
) -> tuple[list[float], list[float], list[list[float]]]:
    """Evaluate the public terminal function at every state-grid node."""
    wealth_grid = [float(scalar(x)) for x in np.asarray(toy.WEALTH_GRID.to_jax())]
    illiquid_grid = [float(scalar(x)) for x in np.asarray(toy.ILLIQUID_GRID.to_jax())]
    values = [
        [
            float(_scalar_terminal_utility(wealth, illiquid, scalar))
            for illiquid in illiquid_grid
        ]
        for wealth in wealth_grid
    ]
    return wealth_grid, illiquid_grid, values


def _bilinear_terminal_read(
    *,
    wealth: float,
    illiquid: float,
    wealth_grid: Sequence[float],
    illiquid_grid: Sequence[float],
    values: Sequence[Sequence[float]],
    scalar: type[np.floating],
) -> float:
    """Independent scalar multilinear interpolation/extrapolation."""
    w0, w1, ww = _bracket_with_nearest_segment_extrapolation(wealth, wealth_grid)
    z0, z1, wz = _bracket_with_nearest_segment_extrapolation(illiquid, illiquid_grid)
    ww_s = scalar(ww)
    wz_s = scalar(wz)
    low = scalar(
        scalar(values[w0][z0])
        + wz_s * scalar(scalar(values[w0][z1]) - scalar(values[w0][z0]))
    )
    high = scalar(
        scalar(values[w1][z0])
        + wz_s * scalar(scalar(values[w1][z1]) - scalar(values[w1][z0]))
    )
    return float(scalar(low + ww_s * scalar(high - low)))


def _scalar_terminal_utility(
    wealth: float,
    illiquid: float,
    scalar: type[np.floating],
) -> np.floating:
    """Evaluate the toy terminal formula in the requested scalar dtype."""
    return scalar(
        -scalar(toy.TERMINAL_SCALE) / scalar(scalar(wealth) + scalar(1.0))
        - scalar(toy.TERMINAL_SCALE) / scalar(scalar(illiquid) + scalar(1.0))
    )


def _scalar_resources(
    wealth: float,
    illiquid: float,
    target: float,
    scalar: type[np.floating],
) -> np.floating:
    """Evaluate the toy liquid-resource formula in scalar arithmetic."""
    credited = scalar(scalar(target) - scalar(illiquid))
    return scalar(scalar(wealth) + scalar(toy.LABOUR_INCOME) - credited)


def _scalar_utility(
    consumption: float | np.floating,
    scalar: type[np.floating],
) -> np.floating:
    """Evaluate the toy CRRA flow in scalar arithmetic."""
    exponent = scalar(1.0 - toy.RISK_AVERSION)
    return scalar(scalar(consumption) ** exponent / exponent)


def _score_pair(
    *,
    wealth: float,
    illiquid: float,
    target: float,
    consumption: float,
    scalar: type[np.floating],
    terminal_data: tuple[list[float], list[float], list[list[float]]],
) -> float:
    """Score one public joint action without any production policy helper."""
    resources = _scalar_resources(wealth, illiquid, target, scalar)
    savings = scalar(resources - scalar(consumption))
    next_wealth = scalar(scalar(1.0 + toy.LIQUID_RATE) * savings)
    wealth_grid, illiquid_grid, terminal_values = terminal_data
    continuation = scalar(
        _bilinear_terminal_read(
            wealth=float(next_wealth),
            illiquid=float(scalar(target)),
            wealth_grid=wealth_grid,
            illiquid_grid=illiquid_grid,
            values=terminal_values,
            scalar=scalar,
        )
    )
    flow = _scalar_utility(consumption, scalar)
    return float(scalar(flow + scalar(_PARAMS["discount_factor"]) * continuation))


def _enumerate_joint_candidate(
    *,
    wealth: float,
    illiquid: float,
    scalar: type[np.floating],
    terminal_data: tuple[list[float], list[float], list[list[float]]],
) -> tuple[float, float, float, int]:
    """Rank keeper + outer nodes and a dense inner feasible grid literally."""
    outer_nodes = [float(scalar(x)) for x in np.asarray(toy.OUTER_GRID.to_jax())]
    consumption_nodes = np.asarray(toy.CONSUMPTION_GRID.to_jax())
    lower_support = float(scalar(consumption_nodes[0]))
    upper_support = float(scalar(consumption_nodes[-1]))

    best: tuple[float, float, float, int] | None = None
    for candidate_index, target in enumerate([float(scalar(illiquid)), *outer_nodes]):
        resources = float(_scalar_resources(wealth, illiquid, target, scalar))
        upper = min(upper_support, resources - toy.SAVINGS_FLOOR)
        if not np.isfinite(upper) or upper < lower_support:
            continue

        inner_best: tuple[float, float] | None = None
        span = scalar(upper - lower_support)
        for point in range(_N_INNER_POINTS):
            fraction = scalar(point / (_N_INNER_POINTS - 1))
            consumption = float(scalar(scalar(lower_support) + fraction * span))
            score = _score_pair(
                wealth=wealth,
                illiquid=illiquid,
                target=target,
                consumption=consumption,
                scalar=scalar,
                terminal_data=terminal_data,
            )
            if inner_best is None or score > inner_best[0]:
                inner_best = (score, consumption)

        assert inner_best is not None
        candidate = (inner_best[0], target, inner_best[1], candidate_index)
        if best is None or candidate[0] > best[0]:
            best = candidate

    assert best is not None
    return best


def _simulate_replay(
    *,
    points: tuple[float, ...],
    outer_batch_size: int,
    dtype: DTypeLike,
):
    """Run the public solve-and-simulate path for three off-grid subjects."""
    model = toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        outer_batch_size=outer_batch_size,
        illiquid_grid=toy.ILLIQUID_GRID,
        illiquid_investment_grid=IrregSpacedGrid(points=points),
        consumption_grid=toy.CONSUMPTION_GRID,
    )
    initial_conditions = {
        "wealth": jnp.asarray([4.0, 15.0, 24.0], dtype=dtype),
        "illiquid": jnp.asarray([0.0, 12.0, 20.0], dtype=dtype),
        "age": jnp.full(3, 20.0, dtype=dtype),
        "regime_id": jnp.full(3, RegimeId.alive, dtype=jnp.int32),
    }
    result = model.simulate(
        params=_PARAMS,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="debug",
        seed=29,
    )
    return (
        result.to_dataframe()
        .query("regime_name == 'alive' and period == 0")
        .sort_index()
    )


@pytest.mark.parametrize("use_x64", [False, True], ids=["fp32", "fp64"])
def test_solve_and_simulate_matches_the_scalar_conditional_oracle(
    *,
    use_x64: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both profiles replay the oracle pair under grid refinement and batching."""
    scalar = np.float64 if use_x64 else np.float32
    dtype: DTypeLike = jnp.float64 if use_x64 else jnp.float32
    equality_tolerance = 2e-11 if use_x64 else 3e-5
    score_tolerance = 2e-10 if use_x64 else 5e-5

    with jax.enable_x64(use_x64):
        # Grid endpoints are canonical JAX scalars fixed at construction.
        # Rebuild the toy grids inside the requested profile so the fp32
        # branch does not inherit float64 endpoint arrays created during
        # collection under another profile.
        profile_grids = {
            "WEALTH_GRID": LinSpacedGrid(start=0.0, stop=30.0, n_points=toy.N_WEALTH),
            "ILLIQUID_GRID": LinSpacedGrid(
                start=0.0, stop=20.0, n_points=toy.N_ILLIQUID
            ),
            "CONSUMPTION_GRID": LinSpacedGrid(
                start=0.1, stop=20.0, n_points=toy.N_CONSUMPTION
            ),
            "ILLIQUID_INVESTMENT_GRID": LinSpacedGrid(
                start=-20.0, stop=20.0, n_points=41
            ),
            "OUTER_GRID": LinSpacedGrid(start=0.0, stop=20.0, n_points=toy.N_OUTER),
            "SAVINGS_GRID": LinSpacedGrid(
                start=toy.SAVINGS_FLOOR, stop=35.0, n_points=60
            ),
        }
        for name, grid in profile_grids.items():
            monkeypatch.setattr(toy, name, grid)

        narrow = _simulate_replay(
            points=(-20.0, 0.01, 20.0),
            outer_batch_size=0,
            dtype=dtype,
        )
        refined = _simulate_replay(
            points=(-20.0, 0.01, 5.0, 20.0),
            outer_batch_size=4,
            dtype=dtype,
        )
        terminal_data = _terminal_grid(scalar)

        for column in ("illiquid_investment", "consumption", "value"):
            np.testing.assert_allclose(
                narrow[column].to_numpy(),
                refined[column].to_numpy(),
                rtol=equality_tolerance,
                atol=equality_tolerance,
                err_msg=f"{column} changed with simulate-grid refinement/batching",
            )

        winner_indices: list[int] = []
        for row in narrow.itertuples():
            oracle_value, oracle_target, oracle_consumption, winner_index = (
                _enumerate_joint_candidate(
                    wealth=float(row.wealth),
                    illiquid=float(row.illiquid),
                    scalar=scalar,
                    terminal_data=terminal_data,
                )
            )
            winner_indices.append(winner_index)
            replayed_target = float(row.illiquid + row.illiquid_investment)
            np.testing.assert_allclose(
                replayed_target,
                oracle_target,
                rtol=equality_tolerance,
                atol=equality_tolerance,
            )
            # Candidate identity is exact; the conditional NBEGM action is an
            # interpolated policy and therefore approximate. The independent
            # dense scalar optimizer is required to agree within 0.18 units,
            # while the objective checks below remain much tighter.
            np.testing.assert_allclose(
                float(row.consumption),
                oracle_consumption,
                rtol=0.0,
                atol=0.18,
            )
            direct_score = _score_pair(
                wealth=float(row.wealth),
                illiquid=float(row.illiquid),
                target=replayed_target,
                consumption=float(row.consumption),
                scalar=scalar,
                terminal_data=terminal_data,
            )
            np.testing.assert_allclose(
                float(row.value),
                direct_score,
                rtol=score_tolerance,
                atol=score_tolerance,
            )
            np.testing.assert_allclose(
                float(row.value),
                oracle_value,
                rtol=0.0,
                atol=0.003,
            )

        assert winner_indices[-1] == 0  # keeper-first tie at target 20
        assert any(index > 0 for index in winner_indices)  # adjuster wins too
