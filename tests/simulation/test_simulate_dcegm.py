"""Forward simulation of DC-EGM-solved models.

DC-EGM publishes V on the regime's exogenous grids in the same layout as the
brute-force solver, so simulation runs the standard per-subject grid argmax
over the regime's action grids against the stored next-period V. Two
properties anchor this:

- a DC-EGM regime's spec carries no borrowing constraint (the EGM solve
  enforces it intrinsically), so the simulate path must synthesize the
  intrinsic budget mask `consumption <= resources - savings_grid lower bound`
  — without it, consumption points above resources edge-clamp the
  continuation and can win the argmax;
- simulated consumption is grid-restricted (the argmax runs over the
  consumption action grid, not the exact EGM policy), so paths agree with the
  brute-force simulation of the equivalent spec up to the action-grid
  resolution wherever the two solves' V arrays agree.
"""

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("lcm.solvers", reason="DC-EGM solver not yet implemented")

from tests.test_models import dcegm_paper_twin
from tests.test_models.deterministic import dcegm_variants
from tests.test_models.deterministic.base import RegimeId as FullRegimeId
from tests.test_models.deterministic.retirement_only import RetirementOnlyRegimeId

CONSUMPTION_GRID_STEP = float(
    dcegm_variants.CONSUMPTION_GRID.to_jax()[1]
    - dcegm_variants.CONSUMPTION_GRID.to_jax()[0]
)


def test_dcegm_simulated_consumption_matches_brute_force():
    """Both solver variants simulate the same consumption path on shared grids.

    The two specs are mathematically equivalent and share the wealth and
    consumption grids, so each simulate runs the same grid argmax — only the
    stored V arrays differ (both approximate the same analytical solution).
    The argmax therefore picks the same consumption node wherever the V
    difference is below the Q-gap between neighboring nodes; where Q is flat
    near its maximum, a small V difference can shift the chosen node. One
    consumption-grid step is the natural quantum of disagreement.
    """
    n_periods = 6
    params = dcegm_variants.get_retirement_only_params(n_periods)
    n_subjects = 4
    initial_conditions = {
        "wealth": jnp.array([10.0, 50.0, 150.0, 300.0]),
        "age": jnp.full(n_subjects, 40.0),
        "regime_id": jnp.full(
            n_subjects, RetirementOnlyRegimeId.retirement, dtype=jnp.int32
        ),
    }

    consumption = {}
    for solver in ["brute_force", "dcegm"]:
        model = dcegm_variants.get_retirement_only_model(solver, n_periods)
        result = model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            period_to_regime_to_V_arr=None,
            log_level="debug",
            seed=42,
        )
        df = (
            result.to_dataframe()
            .query("regime_name == 'retirement'")
            .sort_values(["subject_id", "period"])
        )
        consumption[solver] = df["consumption"].to_numpy()

    np.testing.assert_allclose(
        consumption["dcegm"],
        consumption["brute_force"],
        atol=CONSUMPTION_GRID_STEP + 1e-9,
    )


def test_dcegm_simulate_enforces_intrinsic_budget_constraint():
    """Simulated consumption never exceeds resources minus the borrowing limit.

    Low-wealth subjects with a consumption grid reaching far above their
    wealth: an unmasked argmax would pick infeasible consumption, because
    `u(c)` grows in `c` while the below-limit savings it implies only
    edge-clamp the continuation to the lowest wealth node (for log utility,
    `log(400) + beta * V(w_min) > log(w) + beta * V(w - c)` at low `w`). The
    simulate path must mask those points exactly as a declared borrowing
    constraint would.
    """
    n_periods = 4
    model = dcegm_variants.get_retirement_only_model("dcegm", n_periods)
    params = dcegm_variants.get_retirement_only_params(n_periods)
    savings_lower_bound = float(dcegm_variants.SAVINGS_GRID.to_jax()[0])
    initial_wealth = jnp.array([5.0, 10.0, 50.0])
    # The infeasible region exists by construction: the consumption grid
    # extends far above every subject's wealth.
    assert float(dcegm_variants.CONSUMPTION_GRID.to_jax()[-1]) > float(
        jnp.max(initial_wealth)
    )

    result = model.simulate(
        params=params,
        initial_conditions={
            "wealth": initial_wealth,
            "age": jnp.full(3, 40.0),
            "regime_id": jnp.full(
                3, RetirementOnlyRegimeId.retirement, dtype=jnp.int32
            ),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
        seed=7,
    )

    df = result.to_dataframe().query("regime_name == 'retirement'")
    # Resources are wealth in this model; the savings grid's lower bound is
    # the borrowing limit.
    budget = df["wealth"].to_numpy() - savings_lower_bound
    assert (df["consumption"].to_numpy() <= budget + 1e-9).all()


def test_dcegm_simulate_with_taste_shocks_work_share_decreases_in_wealth():
    """Gumbel-max simulation of a taste-shock DC-EGM model yields sane choices.

    Wealthier workers retire more often (the income motive for working
    weakens), so the simulated period-0 work share must decrease from the
    low-wealth to the high-wealth group.
    """
    model = dcegm_paper_twin.get_model("dcegm")
    params = dcegm_paper_twin.get_params(taste_shock_scale=0.2)
    n_per_group = 512
    initial_conditions = {
        "wealth": jnp.concatenate(
            [jnp.full(n_per_group, 1.0), jnp.full(n_per_group, 35.0)]
        ),
        "age": jnp.full(2 * n_per_group, float(dcegm_paper_twin.MIN_AGE)),
        "regime_id": jnp.full(
            2 * n_per_group,
            dcegm_paper_twin.TwinRegimeId.working_life,
            dtype=jnp.int32,
        ),
    }

    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="debug",
        seed=123,
    )

    df = result.to_dataframe()
    period_0 = df.query("period == 0 and regime_name == 'working_life'")
    assert len(period_0) == 2 * n_per_group
    work_share_low = (
        period_0.query("subject_id < @n_per_group")["work_choice"] == "work"
    ).mean()
    work_share_high = (
        period_0.query("subject_id >= @n_per_group")["work_choice"] == "work"
    ).mean()
    assert work_share_low > work_share_high


def test_dcegm_full_model_simulates_end_to_end():
    """The discrete-choice IJRS model with DC-EGM regimes simulates end to end.

    Workers choose labor supply and consumption each period; the simulated
    consumption respects the intrinsic budget in every regime and the result
    converts to a DataFrame.
    """
    n_periods = 6
    model = dcegm_variants.get_full_model("dcegm", n_periods)
    params = dcegm_variants.get_full_params(n_periods)
    n_subjects = 3
    savings_lower_bound = float(dcegm_variants.SAVINGS_GRID.to_jax()[0])

    result = model.simulate(
        params=params,
        initial_conditions={
            "wealth": jnp.array([10.0, 100.0, 300.0]),
            "age": jnp.full(n_subjects, 40.0),
            "regime_id": jnp.full(
                n_subjects, FullRegimeId.working_life, dtype=jnp.int32
            ),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
        seed=99,
    )

    df = result.to_dataframe()
    alive = df.query("regime_name in ['working_life', 'retirement']")
    assert not alive.empty
    assert alive["consumption"].notna().all()
    assert set(df.query("regime_name == 'working_life'")["labor_supply"]) <= {
        "work",
        "retire",
    }
    budget = alive["wealth"].to_numpy() - savings_lower_bound
    assert (alive["consumption"].to_numpy() <= budget + 1e-9).all()


def test_dcegm_simulate_accepts_precomputed_solution():
    """`simulate` consumes V arrays from a prior `solve` of a DC-EGM model."""
    n_periods = 4
    model = dcegm_variants.get_retirement_only_model("dcegm", n_periods)
    params = dcegm_variants.get_retirement_only_params(n_periods)

    period_to_regime_to_V_arr = model.solve(params=params, log_level="off")
    result = model.simulate(
        params=params,
        initial_conditions={
            "wealth": jnp.array([20.0]),
            "age": jnp.array([40.0]),
            "regime_id": jnp.array(
                [RetirementOnlyRegimeId.retirement], dtype=jnp.int32
            ),
        },
        period_to_regime_to_V_arr=period_to_regime_to_V_arr,
        log_level="debug",
        seed=1,
    )

    df = result.to_dataframe().query("regime_name == 'retirement'")
    assert (df["consumption"].to_numpy() <= df["wealth"].to_numpy() + 1e-9).all()
