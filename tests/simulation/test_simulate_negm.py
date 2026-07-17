"""Forward simulation of an NEGM model (GPU-only).

NEGM nests the same inner 1-D DC-EGM consumption-savings solve as `DCEGM`, so
its forward simulation needs the inner borrowing-feasibility mask synthesized
as an explicit constraint: the simulate-phase grid argmax does not re-run the
inverse-Euler step that enforces it intrinsically during the solve. These
tests drive the kinked two-asset toy (`tests/test_models/negm_kinked_toy.py`)
end to end and assert simulated consumption stays inside that mask.

The whole module is skipped: solving an NEGM model OOMs the local box (DC-EGM /
NEGM solves are GPU-only, see `feedback_no_heavy_tests_local`). Run it on
gpu-01.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from tests.test_models import negm_kinked_toy
from tests.test_models.negm_kinked_toy import RegimeId

pytestmark = pytest.mark.skip(
    reason="gpu-01 only: NEGM/DC-EGM solve OOMs the local box"
)

_PARAMS = {"discount_factor": 0.95, "alive": {}}


def _alive_dataframe(*, initial_conditions: dict[str, jnp.ndarray]) -> pd.DataFrame:
    """Solve and simulate the kinked toy, returning the `alive` regime rows."""
    model = negm_kinked_toy.build_model()
    result = model.simulate(
        params=_PARAMS,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="debug",
        seed=7,
    )
    return result.to_dataframe().query("regime_name == 'alive'")


def test_negm_simulate_enforces_inner_budget_constraint():
    """Simulated consumption never exceeds inner resources minus the borrowing limit.

    Low-wealth subjects face a consumption grid reaching far above their liquid
    resources, so an unmasked argmax would pick infeasible consumption. The
    synthesized inner mask `consumption <= resources - borrowing_limit` (the
    borrowing limit being the inner savings grid's lowest node) must clip the
    simulated path exactly as a declared borrowing constraint would, with the
    outer durable move already folded into `resources`.
    """
    n_subjects = 3
    initial_conditions = {
        "wealth": jnp.array([1.0, 5.0, 50.0]),
        "illiquid": jnp.full(n_subjects, 10.0),
        "age": jnp.full(n_subjects, 20.0),
        "regime_id": jnp.full(n_subjects, RegimeId.alive, dtype=jnp.int32),
    }
    # The infeasible region exists by construction: the consumption grid
    # extends far above the low-wealth subject's liquid resources.
    assert float(negm_kinked_toy.CONSUMPTION_GRID.to_jax()[-1]) > float(
        jnp.max(initial_conditions["wealth"])
    )

    df = _alive_dataframe(initial_conditions=initial_conditions)

    illiquid = df["illiquid"].to_numpy()
    investment = df["illiquid_investment"].to_numpy()
    next_illiquid = illiquid + investment
    resources = np.asarray(
        negm_kinked_toy.resources_before_outer_cost(wealth=df["wealth"].to_numpy())
        - negm_kinked_toy.credited(illiquid=illiquid, next_illiquid=next_illiquid)
    )
    borrowing_limit = float(negm_kinked_toy.SAVINGS_GRID.to_jax()[0])
    budget = resources - borrowing_limit
    assert (df["consumption"].to_numpy() <= budget + 1e-9).all()


def test_negm_simulated_consumption_is_positive_and_finite():
    """Every simulated consumption choice is strictly positive and finite.

    The CRRA flow is defined only for positive consumption, and the masked
    argmax must never select the infeasible region; a non-positive or non-finite
    simulated consumption would signal the budget mask failed to apply.
    """
    n_subjects = 4
    initial_conditions = {
        "wealth": jnp.array([2.0, 8.0, 15.0, 40.0]),
        "illiquid": jnp.array([0.0, 5.0, 12.0, 20.0]),
        "age": jnp.full(n_subjects, 20.0),
        "regime_id": jnp.full(n_subjects, RegimeId.alive, dtype=jnp.int32),
    }
    consumption = _alive_dataframe(initial_conditions=initial_conditions)[
        "consumption"
    ].to_numpy()
    assert np.all(np.isfinite(consumption))
    assert np.all(consumption > 0.0)
