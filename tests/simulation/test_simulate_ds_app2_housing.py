"""DS-2026 App.2 housing NEGM simulation stays inside the feasible region.

The NEGM solve searches the next-housing choice on an outer grid floored at a
small positive stock, but the forward simulation re-optimises over the symmetric
`housing_investment` action grid. Without a feasibility cut, a delta that drives
`next_housing = housing + housing_investment` below the floor lands where the CES
service flow `H^{1-gamma_H}` is NaN; the budget feasibility mask does not catch
that NaN, so the simulate argmax can prefer the NaN-valued action over a genuine
interior optimum and corner consumption at the grid floor. The
`housing_stays_positive` constraint masks the below-floor deltas, mirroring the
solve's floored outer grid, so the simulated policy stays interior wherever a
feasible move exists.
"""

import jax.numpy as jnp
import numpy as np

from lcm import TauchenAR1Process
from tests.test_models import ds_app2_housing as m


def _solve_and_simulate():
    model = m.build_model(n_grid=8, n_periods=3, n_consumption=60)
    params = m.build_params(tau=0.07)
    solution = model.solve(params=params, log_level="off")

    wage = model.user_regimes["working"].states["wage"]
    assert isinstance(wage, TauchenAR1Process)
    nodes = np.asarray(
        wage.compute_gridpoints(
            rho=jnp.asarray(0.82), sigma=jnp.asarray(0.11), mu=jnp.asarray(0.0)
        )
    )
    wage0 = float(nodes[len(nodes) // 2])
    # Seed liquid wealth high enough that a feasible interior move always exists,
    # so any cornering is the NaN artifact and not a genuinely infeasible state.
    liquid = np.linspace(6.0, 20.0, 6)
    initial_conditions = {
        "liquid": jnp.asarray(liquid),
        "housing": jnp.full(6, 5.0),
        "wage": jnp.full(6, wage0),
        "age": jnp.full(6, 20.0),
        "regime_id": jnp.full(6, m.HousingRegimeId.working, dtype=jnp.int32),
    }
    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=solution,
        log_level="off",
    )
    return model, result.to_dataframe()


def test_simulated_consumption_is_interior_not_cornered():
    """The working-regime simulated consumption sits above the grid floor.

    A NaN-cornered NEGM simulation pins consumption at the `0.05`
    consumption-grid floor; the feasibility cut keeps the chosen action interior,
    so the realised consumption is well above the floor for every seeded subject.
    """
    _, df = _solve_and_simulate()
    consumption = df.query("regime_name == 'working'")["consumption"]
    assert len(consumption) > 0
    assert bool(consumption.notna().all())
    assert float(consumption.min()) > 0.1


def test_simulated_next_housing_stays_at_or_above_the_floor():
    """The chosen next house never drops below the model's housing floor.

    The feasibility cut enforces `next_housing >= housing_min` in simulation, so
    every realised housing stock the policy steps into stays at or above the
    floor used to build the housing and outer grids.
    """
    model, df = _solve_and_simulate()
    housing_min = model.user_regimes["working"].states["housing"].start
    realised_housing = df["housing"].to_numpy()
    assert float(realised_housing.min()) >= housing_min - 1e-6
