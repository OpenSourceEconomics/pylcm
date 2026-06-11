"""Simulate-time V validation considers only subjects that are in the regime.

The forward simulation evaluates every regime's policy for all subjects and
masks out-of-regime entries afterwards; those placeholder entries can be
`-inf` (the subject's state is infeasible under the other regime's policy
problem). The NaN/Inf warning must not fire on placeholders — only on the
values of subjects actually simulated in the regime.
"""

import logging

import jax.numpy as jnp

from lcm_examples.iskhakov_et_al_2017 import get_model, get_params


def test_no_nan_warning_from_out_of_regime_placeholders(caplog):
    """Simulating a healthy multi-regime model emits no NaN/Inf warnings."""
    model = get_model(6)
    params = get_params(
        6,
        discount_factor=0.98,
        disutility_of_work=1.0,
        interest_rate=0.0,
        wage=20.0,
    )
    wealth = jnp.linspace(1.0, 120.0, 12)

    with caplog.at_level(logging.WARNING, logger="lcm"):
        model.simulate(
            params=params,
            initial_conditions={
                "age": jnp.full(wealth.size, model.ages.values[0]),
                "wealth": wealth,
                "regime_id": jnp.full(
                    wealth.size, model.regime_names_to_ids["working_life"]
                ),
            },
            period_to_regime_to_V_arr=None,
            log_level="warning",
        )

    nan_warnings = [r for r in caplog.records if "NaN/Inf in V_arr" in r.getMessage()]
    assert nan_warnings == []
