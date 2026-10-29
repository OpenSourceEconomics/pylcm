"""Plain DC-EGM simulation is policy-first.

The solve publishes each branch's conditional value and policy rows on the
endogenous resources grid. Simulation picks the discrete branch by the largest
interpolated conditional value and reads the winner's policy row at the
subject's resources, so the dense action-grid argmax (`decide()`, the
`"decision"` simulation program) is neither traced nor executed while every
subject's read is accepted. `DCEGM(replay="grid")` keeps the grid argmax and
serves as the reference route.
"""

import dataclasses
from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import pytest

import _lcm.simulation.simulate as simulation
import lcm_examples.iskhakov_et_al_2017 as iskhakov
from lcm import AgeGrid, Model
from lcm_examples.iskhakov_et_al_2017 import CONSUMPTION_GRID, dead, get_edges
from tests.conftest import invariance_tolerances
from tests.test_models.deterministic import base
from tests.test_models.deterministic.dcegm_variants import (
    DCEGM_SOLVER,
    dcegm_retirement,
    dcegm_working_life,
    get_full_params,
)

pytestmark = pytest.mark.slow

_N_PERIODS = 4
_EGM_REGIMES = ("working_life", "retirement")


def _model(*, replay: str) -> Model:
    solver = dataclasses.replace(DCEGM_SOLVER, replay=replay)
    ages = AgeGrid(start=40, inclusive_stop=40 + (_N_PERIODS - 1) * 10, step="10Y")
    return Model(
        edges=get_edges(ages=ages),
        regimes={
            "working_life": dcegm_working_life.replace(solver=solver),
            "retirement": dcegm_retirement.replace(solver=solver),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=base.RegimeId,
        initial_nodes=((40, "working_life"), (40, "retirement")),
    )


def _initial_conditions(*, wealth: np.ndarray) -> dict[str, jnp.ndarray]:
    return {
        "wealth": jnp.asarray(wealth),
        "age": jnp.full(wealth.shape, 40.0),
        "regime_id": jnp.full(wealth.shape, base.RegimeId.working_life),
    }


def _decision_program_owners(*, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the regime of every executed `"decision"` simulation program."""
    running: list[str] = []
    executed: list[str] = []
    run_regime = simulation._simulate_regime_in_period
    run_program = simulation.execute_simulation_program

    def record_regime(**kwargs):
        running.append(kwargs["regime_name"])
        try:
            return run_regime(**kwargs)
        finally:
            running.pop()

    def record_program(*, family: str, **kwargs):
        if family == "decision":
            executed.append(running[-1])
        return run_program(family=family, **kwargs)

    monkeypatch.setattr(simulation, "_simulate_regime_in_period", record_regime)
    monkeypatch.setattr(simulation, "execute_simulation_program", record_program)
    return executed


def _simulate(*, model: Model, wealth: np.ndarray):
    return model.simulate(
        params=get_full_params(n_periods=_N_PERIODS),
        initial_conditions=_initial_conditions(wealth=wealth),
        seed=0,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)


_IN_SUPPORT_WEALTH = np.linspace(20.5, 180.5, 9)


def test_default_dcegm_simulate_does_not_run_the_grid_argmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-support subjects never reach the dense decision program of an EGM regime."""
    model = _model(replay="candidate")
    executed = _decision_program_owners(monkeypatch=monkeypatch)

    _simulate(model=model, wealth=_IN_SUPPORT_WEALTH)

    assert not set(executed) & set(_EGM_REGIMES), executed


def test_grid_replay_runs_the_grid_argmax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control: the spy sees the decision program on the grid route."""
    model = _model(replay="grid")
    executed = _decision_program_owners(monkeypatch=monkeypatch)

    _simulate(model=model, wealth=_IN_SUPPORT_WEALTH)

    assert set(_EGM_REGIMES) <= set(executed), executed


def _consumption_step() -> float:
    nodes = np.asarray(CONSUMPTION_GRID.to_jax())
    return float(np.max(np.diff(nodes)))


def _per_period_means(frame) -> Mapping[int, tuple[float, float]]:
    alive = frame[frame["regime_name"].isin(_EGM_REGIMES)]
    grouped = alive.groupby("period")
    return {
        int(period): (float(group["consumption"].mean()), float(group["value"].mean()))
        for period, group in grouped
    }


def test_policy_first_tracks_the_grid_route_within_the_action_grid_step() -> None:
    """Mean consumption and value agree with the grid route up to grid error.

    The grid route picks consumption on `CONSUMPTION_GRID`, so each of its
    choices is within one grid step `h` of the policy read (half a step from
    the continuous optimum, plus the read's own interpolation error). Wealth
    carries those differences forward one-for-one at zero interest, so after
    `t` periods the mean consumption gap is bounded by `(t + 1) * h`. The
    value gap is first-order in the action gap only through `u'(c) * h`
    along the path, and log utility at the sampled consumption keeps
    `u'(c) = 1 / c <= 1`, so the same `(t + 1) * h` bound applies to the
    mean value per period.
    """
    rng = np.random.default_rng(0)
    wealth = rng.uniform(20.0, 180.0, size=200)
    policy = _per_period_means(
        _simulate(model=_model(replay="candidate"), wealth=wealth)
    )
    grid = _per_period_means(_simulate(model=_model(replay="grid"), wealth=wealth))

    step = _consumption_step()
    assert policy.keys() == grid.keys()
    for period in policy:
        band = (period + 1) * step
        policy_c, policy_v = policy[period]
        grid_c, grid_v = grid[period]
        assert abs(policy_c - grid_c) <= band, (period, policy_c, grid_c, band)
        assert abs(policy_v - grid_v) <= band, (period, policy_v, grid_v, band)


def test_lookup_policy_on_default_dcegm_returns_the_simulated_decision() -> None:
    """`lookup_policy` answers on a default DC-EGM model with `simulate`'s choice.

    At every simulated working-life state, the looked-up labor supply equals the
    simulated one, and the looked-up consumption and value agree with the
    simulated ones up to the reassociation a different row count allows. The
    last subject starts beyond the published rows, so its period-0 read is
    rejected and both take the grid argmax there.
    """
    n_periods = 4
    model = iskhakov.get_dcegm_model(n_periods)
    params = iskhakov.get_params(n_periods=n_periods)
    solution = model.solve(params=params, log_level="off")
    wealth = np.append(np.linspace(5.0, 180.0, 12), 1000.0)
    frame = model.simulate(
        params=params,
        initial_conditions={
            "wealth": jnp.asarray(wealth),
            "age": jnp.full(wealth.shape, 40.0),
            "regime_id": jnp.full(wealth.shape, iskhakov.RegimeId.working_life),
        },
        solution=solution,
        seed=0,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)

    for period in (0, 1):
        rows = frame[
            (frame["period"] == period) & (frame["regime_name"] == "working_life")
        ]
        lookup = model.lookup_policy(
            params=params,
            solution=solution,
            period=period,
            regime_name="working_life",
            states={"wealth": jnp.asarray(rows["wealth"].to_numpy())},
            use_categorical_codes=True,
        )
        assert lookup.actions is not None
        np.testing.assert_array_equal(
            np.asarray(lookup.actions["labor_supply"]),
            rows["labor_supply"].to_numpy(),
        )
        for got, expected in (
            (lookup.actions["consumption"], rows["consumption"]),
            (lookup.value, rows["value"]),
        ):
            reference = expected.to_numpy()
            rtol, atol = invariance_tolerances(reference)
            np.testing.assert_allclose(np.asarray(got), reference, rtol=rtol, atol=atol)
