"""Simulate-time V validation considers only subjects that are in the regime.

The forward simulation evaluates every regime's policy for all subjects and
masks out-of-regime entries afterwards; those placeholder entries can be
`-inf` (the subject's state is infeasible under the other regime's policy
problem). The NaN/Inf warning must not fire on placeholders — only on the
values of subjects actually simulated in the regime. It must fire, naming the
regime and the age, on a value the regime does own.
"""

import logging

import jax.numpy as jnp
import pytest

from _lcm.utils.logging import LogLevel
from lcm import AgeGrid, LinSpacedGrid, Model, categorical
from lcm.regime import Regime as UserRegime
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND, ScalarInt
from lcm_examples.iskhakov_et_al_2017 import get_model, get_params


def _simulate(*, log_level: LogLevel) -> None:
    model = get_model(6)
    params = get_params(
        n_periods=6,
        discount_factor=0.98,
        disutility_of_work=1.0,
        interest_rate=0.0,
        wage=20.0,
    )
    wealth = jnp.linspace(1.0, 120.0, 12)
    model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.full(wealth.size, model.ages.values[0]),
            "wealth": wealth,
            "regime_id": jnp.full(
                wealth.size, model.regime_names_to_ids["working_life"]
            ),
        },
        log_level=log_level,
    )


def test_no_nan_warning_from_out_of_regime_placeholders(caplog):
    """Simulating a healthy multi-regime model emits no NaN/Inf warnings."""
    with caplog.at_level(logging.WARNING, logger="lcm"):
        _simulate(log_level="warning")

    nan_warnings = [r for r in caplog.records if "NaN/Inf in V_arr" in r.getMessage()]
    assert nan_warnings == []


@pytest.mark.parametrize("log_level", ["warning", "debug"])
def test_out_of_regime_placeholders_pass_v_validation(log_level: LogLevel) -> None:
    """Per-regime V validation ignores out-of-regime placeholder entries.

    The simulate-time `validate_V` check must consider only the subjects
    actually simulated in the regime, so placeholder values of out-of-regime
    subjects never raise — at any log level.
    """
    _simulate(log_level=log_level)


# Age at which the off-node penalty below turns the simulated value into NaN.
NAN_AGE = 40


@categorical(ordered=False)
class OffNodeRegimeId:
    work: ScalarInt
    dead: ScalarInt


def _off_node_utility(
    *, consumption: ContinuousAction, wealth: ContinuousState, age: FloatND
) -> FloatND:
    """Log utility plus a term that is NaN off a wealth node, and only at `NAN_AGE`.

    Every node of the wealth grid is an integer, so `floor(wealth) - wealth` is
    zero wherever the solver tabulates the value function and negative at a
    subject's own wealth between two nodes. The solve therefore stays finite and
    the forward simulation produces NaN for exactly one regime at one age.
    """
    return jnp.log(consumption) + jnp.where(
        age == NAN_AGE, jnp.sqrt(jnp.floor(wealth) - wealth), 0.0
    )


def _off_node_next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    """Spend and receive a unit of income."""
    return wealth - consumption + 1.0


def _off_node_borrowing_constraint(
    *, consumption: ContinuousAction, wealth: ContinuousState
) -> BoolND:
    """Consume no more than current wealth."""
    return consumption <= wealth


def _off_node_next_regime(*, age: FloatND) -> ScalarInt:
    """Stay at work until the last age at which work is possible."""
    return jnp.where(age >= 50, OffNodeRegimeId.dead, OffNodeRegimeId.work)


def _nan_producing_model() -> Model:
    """Build a two-regime model whose simulated value is NaN at `NAN_AGE` only."""
    grid = LinSpacedGrid(start=1.0, stop=5.0, n_points=5)
    work = UserRegime(
        transition=_off_node_next_regime,
        actions={"consumption": grid},
        states={"wealth": grid},
        state_transitions={"wealth": _off_node_next_wealth},
        constraints={"borrowing_constraint": _off_node_borrowing_constraint},
        functions={"utility": _off_node_utility},
        active=lambda age: age < 60,
    )
    dead = UserRegime(transition=None, functions={"utility": lambda: 0.0})
    return Model(
        regimes={"work": work, "dead": dead},
        ages=AgeGrid(start=40, stop=60, step="10Y"),
        regime_id_class=OffNodeRegimeId,
    )


def test_a_period_with_a_non_finite_value_warns_once_per_offending_regime(caplog):
    """At `warning`, each regime with a NaN owned value produces exactly one line."""
    model = _nan_producing_model()
    with caplog.at_level(logging.WARNING, logger="lcm"):
        model.simulate(
            params={"discount_factor": 0.95},
            initial_conditions={
                "wealth": jnp.array([1.5, 2.5]),
                "age": jnp.full(2, float(NAN_AGE)),
                "regime_id": jnp.full(2, OffNodeRegimeId.work),
            },
            log_level="warning",
        )
    lines = [
        record.getMessage()
        for record in caplog.records
        if "NaN/Inf" in record.getMessage()
    ]
    assert lines == [f"NaN/Inf in V_arr for regime 'work' at age {NAN_AGE}"]
