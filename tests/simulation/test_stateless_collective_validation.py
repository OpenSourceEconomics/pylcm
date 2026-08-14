"""Runtime validation leaves a stateless collective regime's simulation intact.

A collective regime that declares no states still simulates one row per
subject: its recorded value array carries the subject axis in front of the
stakeholder axis, and every subject records the household's shared optimal
action. That contract is a property of the model, not of the diagnostics, so
it holds at `log_level="debug"`, which switches the value-function validation
on, exactly as it does at every quieter level.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.engine import PeriodRegimeSimulationData
from _lcm.simulation.simulate import simulate
from _lcm.solution.backward_induction import solve
from _lcm.utils.logging import get_logger
from lcm.ages import AgeGrid
from tests.conftest import DECIMAL_PRECISION
from tests.regime_building.test_collective_regime_simulate import _solve_and_process
from tests.regime_building.test_simulate_guards import _make_stateless_collective_regime

_REGIME_NAME = "stateless_couple"

_N_SUBJECTS = 3

_N_STAKEHOLDERS = 2

# The regime's payoffs are `utility_f = 10 * work` and
# `utility_m = 5 * (1 - work)`, so the household argmax takes `work` (10 + 0
# beats 0 + 5) and each subject records `(value_f, value_m) == (10.0, 0.0)`.
_WORK = 1

_EXPECTED_V_PER_SUBJECT = (10.0, 0.0)


def _simulate_stateless_collective_at_debug() -> PeriodRegimeSimulationData:
    """Simulate the stateless collective regime and return its period-0 data."""
    ages = AgeGrid(start=0, stop=1, step="Y")
    regimes_dict = _make_stateless_collective_regime()
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict, ages=ages, regime_names=list(regimes_dict)
    )
    flat_params = MappingProxyType({_REGIME_NAME: MappingProxyType({})})
    solution = solve(
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    ).value_functions
    initial_conditions = MappingProxyType(
        {
            "age": jnp.zeros(_N_SUBJECTS),
            "regime_id": jnp.zeros(_N_SUBJECTS, dtype=jnp.int32),
        }
    )
    result = simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=regimes,
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="debug"),
        period_to_regime_to_V_arr=solution,
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    )
    return result.raw_results[_REGIME_NAME][0]


def test_stateless_collective_simulated_V_has_subject_and_stakeholder_axes() -> None:
    """The simulated value array is `(n_subjects, n_stakeholders)`."""
    period_0 = _simulate_stateless_collective_at_debug()

    assert period_0.V_arr.shape == (_N_SUBJECTS, _N_STAKEHOLDERS)


def test_stateless_collective_simulated_V_records_each_stakeholder_payoff() -> None:
    """Every subject records `(value_f, value_m) == (10.0, 0.0)`."""
    period_0 = _simulate_stateless_collective_at_debug()

    aaae(
        np.asarray(period_0.V_arr),
        np.tile(_EXPECTED_V_PER_SUBJECT, (_N_SUBJECTS, 1)),
        decimal=DECIMAL_PRECISION,
    )


def test_stateless_collective_simulated_action_is_the_household_argmax() -> None:
    """Every subject takes `work`, the action maximising the household's sum."""
    period_0 = _simulate_stateless_collective_at_debug()

    np.testing.assert_array_equal(
        np.asarray(period_0.actions["work"]), np.full(_N_SUBJECTS, _WORK)
    )
