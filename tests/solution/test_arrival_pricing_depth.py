"""Arrival pricing reads later arrival dates from stored surfaces.

An arrival reader rebuilds the target's action problem exactly at its landing.
That action problem's own arrivals, one date later, read the stored arrival
surface of that edge by ordinary interpolation, the same way a continuation
reads a stored value function. So each period's query program prices one
arrival level, whatever the number of later arrival dates, and on grid nodes the
values are those of reading every later arrival at its landing.
"""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm import policy_queries
from lcm import ExecutionConfig, load_solution
from lcm.solver_api import SolutionResult
from lcm_examples import collective_household as household
from tests.conftest import DECIMAL_PRECISION, X64_ENABLED
from tests.solution import _discrete_household as discrete

_FIXTURES = Path(__file__).parents[1] / "data" / "arrival_pricing_depth"


def _nested_values(name: str) -> Mapping[int, Mapping[str, np.ndarray]]:
    """Values captured from pricing every later arrival at its exact landing."""
    path = _FIXTURES / f"{name}_x{int(X64_ENABLED)}.json"
    stored = json.loads(path.read_text())["values"]
    return {
        int(period): {regime: np.asarray(value) for regime, value in regimes.items()}
        for period, regimes in stored.items()
    }


def _query_flops(
    *, monkeypatch: pytest.MonkeyPatch, n_periods: int
) -> dict[tuple[str, int], float]:
    """Compiled flops of each period's query chunk loop, by regime and period."""
    flops: dict[tuple[str, int], float] = {}
    original = policy_queries._PolicyQueryKernel.__call__

    def measured(self, **kwargs):
        cost = kwargs["compiled_cores"]["_chunks"].compiled.cost_analysis()
        cost = cost[0] if isinstance(cost, list) else cost
        flops[self.root_context.regime_name, self.period] = cost["flops"]
        return original(self, **kwargs)

    monkeypatch.setattr(policy_queries._PolicyQueryKernel, "__call__", measured)
    model = household.get_model(
        n_periods=n_periods,
        wealth_n_points=3,
        consumption_n_points=2,
        execution_config=ExecutionConfig(axis_widths={"query_cell": 1}),
    )
    model.solve(params=household.get_params(), log_level="off")
    return flops


def test_query_programs_do_not_grow_with_remaining_arrival_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A period with two later arrival dates costs a small multiple of one date.

    At four periods, period 1 prices one arrival date (period 2) and period 0
    prices two (periods 1 and 2). Nesting the second date inside the first
    multiplies the query program by the target's actions times its reads and
    lottery nodes; a stored surface adds one interpolated continuation target to
    each rebuilt action problem.
    """
    flops = _query_flops(monkeypatch=monkeypatch, n_periods=4)
    for regime in ("couple", "single_f", "single_m"):
        one_date, two_dates = flops[regime, 1], flops[regime, 0]
        assert two_dates <= 3 * one_date, (regime, one_date, two_dates)


def test_query_program_cost_is_the_same_for_every_earlier_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once every target is reachable, one more later arrival date costs nothing.

    At five periods, period 1 prices two later arrival dates and period 0 three.
    Each rebuilt action problem reads the next date's stored surface, so both
    programs do the same work.
    """
    flops = _query_flops(monkeypatch=monkeypatch, n_periods=5)
    for regime in ("couple", "single_f", "single_m"):
        assert flops[regime, 0] == pytest.approx(flops[regime, 1], rel=0.05), regime


def test_grid_aligned_arrivals_price_like_reads_at_every_landing() -> None:
    """Where every landing is a node, a stored surface reads the exact value there."""
    model = discrete.get_model(n_periods=4)
    result = model.solve(params=discrete.get_params(), log_level="off")
    expected = _nested_values("discrete_np4_w4")
    assert set(result.values) == set(expected)
    for period, regimes in expected.items():
        assert set(result.values[period]) == set(regimes)
        for regime, value in regimes.items():
            aaae(
                np.asarray(result.values[period][regime]),
                value,
                decimal=DECIMAL_PRECISION,
            )


def test_periods_pricing_one_arrival_date_keep_their_values() -> None:
    """Off-grid landings change values only where a second arrival date is priced.

    With four periods only period 0 prices two arrival dates; every later
    period's value is the one an exact read at every landing gives.
    """
    model = household.get_model(n_periods=4, wealth_n_points=4, consumption_n_points=3)
    result = model.solve(params=household.get_params(), log_level="off")
    expected = _nested_values("household_np4_w4_c3")
    for period in (1, 2, 3):
        for regime, value in expected[period].items():
            aaae(
                np.asarray(result.values[period][regime]),
                value,
                decimal=DECIMAL_PRECISION,
            )


def test_a_saved_solution_replays_its_stored_arrival_surfaces(tmp_path: Path) -> None:
    """A loaded archive simulates exactly like the solution that wrote it."""
    model = household.get_model(n_periods=4, wealth_n_points=4, consumption_n_points=3)
    params = household.get_params()
    solution = model.solve(params=params, log_level="off")
    loaded = load_solution(path=solution.save(path=tmp_path / "solution.h5"))

    def simulate(solved: SolutionResult) -> pd.DataFrame:
        return model.simulate(
            params=params,
            initial_conditions=household.get_initial_conditions(
                n_subjects=16, model=model
            ),
            solution=solved,
            log_level="off",
            seed=0,
        ).to_dataframe()

    pd.testing.assert_frame_equal(simulate(loaded), simulate(solution))
