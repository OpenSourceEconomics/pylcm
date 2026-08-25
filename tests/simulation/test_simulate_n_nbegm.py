"""Forward simulation of an N-NB-EGM model.

The solve publishes the no-adjustment candidate's inner policy, and the
simulate phase re-optimizes both margins by grid argmax against the solved
value arrays. These tests drive the smooth two-asset toy end to end and assert
that the durable margin is genuinely re-optimized and that the inner choice
stays inside the budget the solve enforces intrinsically.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd

from _lcm.typing import PeriodToRegimeToVArr
from lcm import IrregSpacedGrid
from tests.test_models import n_nbegm_toy as toy
from tests.test_models.n_nbegm_toy import RegimeId

_PARAMS = {"discount_factor": 0.95}


def _alive_dataframe() -> pd.DataFrame:
    """Solve and simulate the smooth two-asset toy, returning the alive rows.

    Subjects start spread across the liquid and durable grids so both the
    no-adjustment candidate and the adjuster sweep can win somewhere.
    """
    n_subjects = 4
    initial_conditions = {
        "wealth": jnp.array([2.0, 8.0, 15.0, 28.0]),
        "illiquid": jnp.array([0.0, 5.0, 12.0, 20.0]),
        "age": jnp.full(n_subjects, 20.0),
        "regime_id": jnp.full(n_subjects, RegimeId.alive, dtype=jnp.int32),
    }
    result = toy.build_model(variant="n_nbegm", n_periods=3).simulate(
        params=_PARAMS,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="debug",
        seed=7,
    )
    return result.to_dataframe().query("regime_name == 'alive'")


def test_the_durable_margin_is_re_optimized_in_simulation() -> None:
    """Some subject moves the durable stock rather than holding it.

    The solve publishes the no-adjustment candidate's policy, so a simulation
    that read it without re-optimizing the outer margin would leave every
    subject's durable investment at zero.
    """
    investment = _alive_dataframe()["illiquid_investment"].to_numpy()
    assert np.any(np.abs(investment) > 1e-9)


def test_simulated_consumption_stays_inside_the_inner_budget() -> None:
    """Consumption never exceeds liquid resources less the borrowing limit.

    The inner solve enforces the limit through the Euler inversion, so the
    simulate-phase grid argmax needs it as an explicit mask; an unmasked argmax
    would let a poor subject pick consumption from the far end of the grid.
    """
    df = _alive_dataframe()
    illiquid = df["illiquid"].to_numpy()
    resources = np.asarray(
        toy.resources(
            wealth=df["wealth"].to_numpy(),
            illiquid=illiquid,
            new_illiquid=illiquid + df["illiquid_investment"].to_numpy(),
        )
    )
    borrowing_limit = toy.SAVINGS_FLOOR
    consumption = df["consumption"].to_numpy()
    assert np.all(consumption <= resources - borrowing_limit + 1e-9)


def test_simulated_consumption_is_strictly_positive() -> None:
    """Every simulated choice lies where the CRRA flow is defined."""
    assert np.all(_alive_dataframe()["consumption"].to_numpy() > 0.0)


def _solve_and_simulate_with_outer_action_points(
    points: tuple[float, ...],
) -> tuple[PeriodToRegimeToVArr, pd.DataFrame]:
    """Return the solved policy-bearing model and its alive simulation rows."""
    model = toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        illiquid_investment_grid=IrregSpacedGrid(points=points),
    )
    values, policies = model.solve(
        params=_PARAMS,
        log_level="debug",
        return_simulation_policy=True,
    )
    assert "alive" in policies[0], (
        "NNBEGM did not publish the keeper-plus-adjuster replay payload; "
        "simulation would re-optimize a foreign action-grid candidate set."
    )
    initial_conditions = {
        "wealth": jnp.array([4.0, 8.0, 15.0, 24.0]),
        "illiquid": jnp.array([0.0, 4.0, 12.0, 20.0]),
        "age": jnp.full(4, 20.0),
        "regime_id": jnp.full(4, RegimeId.alive, dtype=jnp.int32),
    }
    result = model.simulate(
        params=_PARAMS,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="debug",
        seed=19,
    )
    return values, result.to_dataframe().query(
        "regime_name == 'alive' and period == 0"
    ).sort_index()


def test_simulation_replays_the_nnbegm_candidate_when_the_action_grid_changes() -> None:
    """A simulate-only outer-grid refinement cannot change the solved pair.

    The narrow grid contains the review witness's 0.01 point; the refined grid
    inserts the foreign 5.0 point that baseline simulation can select. Neither
    grid is NNBEGM's candidate set: the solve ranks the state-specific keeper plus
    ``OUTER_GRID``. The replay must therefore be identical under both grids, and
    every emitted post-decision durable stock must be one of those solve candidates.
    """
    narrow_values, narrow = _solve_and_simulate_with_outer_action_points(
        (-20.0, 0.01, 20.0)
    )
    wide_values, wide = _solve_and_simulate_with_outer_action_points(
        (-20.0, 0.01, 5.0, 20.0)
    )

    np.testing.assert_array_equal(
        np.asarray(narrow_values[0]["alive"]),
        np.asarray(wide_values[0]["alive"]),
    )
    for column in ("illiquid_investment", "consumption", "value"):
        np.testing.assert_allclose(
            narrow[column].to_numpy(),
            wide[column].to_numpy(),
            rtol=2e-6,
            atol=2e-6,
            err_msg=column,
        )

    outer_nodes = np.asarray(toy.OUTER_GRID.to_jax())
    illiquid = narrow["illiquid"].to_numpy(dtype=float)
    targets = illiquid + narrow["illiquid_investment"].to_numpy(dtype=float)
    for target, current_illiquid in zip(targets, illiquid, strict=True):
        is_keeper = np.isclose(target, current_illiquid, rtol=2e-6, atol=2e-6)
        is_adjuster = np.any(np.isclose(target, outer_nodes, rtol=2e-6, atol=2e-6))
        assert is_keeper or is_adjuster, (
            f"simulation emitted foreign outer target {target}; "
            f"keeper={current_illiquid}, outer_grid={outer_nodes}"
        )
