"""NEGM forward simulation reproduces the solved consumption policy on CPU.

NEGM regimes carry a within-period durable law (`next_<durable>`) that the
budget constraint and — for a service-flow durable — `utility` read. The
forward-simulation decision computes that next state from the chosen action
rather than demanding it as an external input, so the simulate argmax solves and
the realised consumption matches the consumption the backward-induction policy
prescribes for the seeded states.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_models import negm_kinked_toy, negm_serviceflow_toy

_PARAMS = {"discount_factor": 0.95, "alive": {}}

# Three subjects seeded at the same liquid/illiquid states, simulated forward
# through the toy's full (deterministic, shock-free) lifecycle.
_INITIAL_WEALTH = (5.0, 10.0, 15.0)
_INITIAL_ILLIQUID = (4.0, 6.0, 8.0)

# Period-0 `alive` consumption the solved NEGM policy prescribes for the three
# seeded subjects. The toys have no shocks, so the simulated path is
# deterministic; the values are consumption-grid nodes of the simulate argmax.
_KINKED_PERIOD0_CONSUMPTION = (10.05, 11.708333, 14.195833)
_SERVICEFLOW_PERIOD0_CONSUMPTION = (4.364286, 4.364286, 4.364286)

# Accept a one-consumption-grid-step deviation: near the optimum the discrete
# argmax is nearly flat (the top-2 candidate values differ by less than the
# solve's cross-backend variation, which enters at kink-adjacent nodes of the
# degenerate last-alive-period EGM step), so different backends and CPU
# microarchitectures legitimately settle on adjacent grid nodes.
_KINKED_CONSUMPTION_STEP = (20.0 - 0.1) / 24
_SERVICEFLOW_CONSUMPTION_STEP = (20.0 - 0.1) / 14


def _simulate_period0_alive_consumption(model, regime_id) -> np.ndarray:
    """Solve, simulate, and return the period-0 `alive` consumption per subject."""
    solution = model.solve(params=_PARAMS, log_level="off")
    n_subjects = len(_INITIAL_WEALTH)
    initial_conditions = {
        "wealth": jnp.asarray(_INITIAL_WEALTH),
        "illiquid": jnp.asarray(_INITIAL_ILLIQUID),
        "age": jnp.full(n_subjects, 20.0),
        "regime_id": jnp.full(n_subjects, regime_id, dtype=jnp.int32),
    }
    result = model.simulate(
        params=_PARAMS,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=solution,
        log_level="off",
    )
    df = result.to_dataframe()
    period0 = df.query("regime_name == 'alive' and period == 0")
    return period0.sort_values("subject_id")["consumption"].to_numpy()


@pytest.mark.parametrize(
    ("build_model", "regime_id", "expected", "consumption_step"),
    [
        (
            negm_kinked_toy.build_model,
            negm_kinked_toy.RegimeId.alive,
            _KINKED_PERIOD0_CONSUMPTION,
            _KINKED_CONSUMPTION_STEP,
        ),
        (
            negm_serviceflow_toy.build_negm_model,
            negm_serviceflow_toy.RegimeId.alive,
            _SERVICEFLOW_PERIOD0_CONSUMPTION,
            _SERVICEFLOW_CONSUMPTION_STEP,
        ),
    ],
)
def test_negm_simulate_reproduces_the_solved_consumption(
    build_model, regime_id, expected, consumption_step
):
    """The simulated period-0 consumption equals the solved NEGM policy.

    Seeding three subjects at known liquid/illiquid states and stepping the
    shock-free toy forward, the realised period-0 consumption matches the
    consumption the backward-induction policy prescribes, up to one
    consumption-grid step (the backend-dependent flat-argmax band).
    """
    consumption = _simulate_period0_alive_consumption(build_model(), regime_id)
    np.testing.assert_allclose(
        consumption, np.asarray(expected), atol=consumption_step + 1e-4
    )
