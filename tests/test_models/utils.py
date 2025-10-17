from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import jax.numpy as jnp

from tests.test_models.deterministic import (
    ISKHAKOV_ET_AL_2017,
    ISKHAKOV_ET_AL_2017_STRIPPED_DOWN,
)
from tests.test_models.discrete_deterministic import ISKHAKOV_ET_AL_2017_DISCRETE
from tests.test_models.stochastic import ISKHAKOV_ET_AL_2017_STOCHASTIC

if TYPE_CHECKING:
    from lcm.regime import Regime

TEST_REGIMES = {
    "iskhakov_et_al_2017": ISKHAKOV_ET_AL_2017,
    "iskhakov_et_al_2017_stripped_down": ISKHAKOV_ET_AL_2017_STRIPPED_DOWN,
    "iskhakov_et_al_2017_discrete": ISKHAKOV_ET_AL_2017_DISCRETE,
    "iskhakov_et_al_2017_stochastic": ISKHAKOV_ET_AL_2017_STOCHASTIC,
}


def get_model(model_name: str, n_periods: int) -> Regime:
    return get_regime(model_name, n_periods)


def get_regime(regime_name: str, n_periods: int) -> Regime:
    model = deepcopy(TEST_REGIMES[regime_name])
    return model.replace(n_periods=n_periods)


def get_params(
    beta=0.95,
    disutility_of_work=0.5,
    interest_rate=0.05,
    wage=10.0,
    health_transition=None,
    partner_transition=None,
):
    # ----------------------------------------------------------------------------------
    # Transition matrices
    # ----------------------------------------------------------------------------------

    # Health shock transition:
    # ------------------------------------------------------------------------------
    # 1st dimension: Current health state
    # 2nd dimension: Current Partner state
    # 3rd dimension: Probability distribution over next period's health state
    default_health_transition = jnp.array(
        [
            # Current health state 0
            [
                # Current Partner state 0
                [0.9, 0.1],
                # Current Partner state 1
                [0.5, 0.5],
            ],
            # Current health state 1
            [
                # Current Partner state 0
                [0.5, 0.5],
                # Current Partner state 1
                [0.1, 0.9],
            ],
        ],
    )
    health_transition = (
        default_health_transition if health_transition is None else health_transition
    )

    # Partner shock transition:
    # ------------------------------------------------------------------------------
    # 1st dimension: The period
    # 2nd dimension: Current working decision
    # 3rd dimension: Current partner state
    # 4th dimension: Probability distribution over next period's partner state
    default_partner_transition = jnp.array(
        [
            # Transition from period 0 to period 1
            [
                # Current working decision 0
                [
                    # Current partner state 0
                    [0, 1.0],
                    # Current partner state 1
                    [1.0, 0],
                ],
                # Current working decision 1
                [
                    # Current partner state 0
                    [0, 1.0],
                    # Current partner state 1
                    [0.0, 1.0],
                ],
            ],
            # Transition from period 1 to period 2
            [
                # Current working decision 0
                [
                    # Current partner state 0
                    [0, 1.0],
                    # Current partner state 1
                    [1.0, 0],
                ],
                # Current working decision 1
                [
                    # Current partner state 0
                    [0, 1.0],
                    # Current partner state 1
                    [0.0, 1.0],
                ],
            ],
        ],
    )
    partner_transition = (
        default_partner_transition if partner_transition is None else partner_transition
    )

    # ----------------------------------------------------------------------------------
    # Model parameters
    # ----------------------------------------------------------------------------------
    return {
        "beta": beta,
        "utility": {"disutility_of_work": disutility_of_work},
        "next_wealth": {"interest_rate": interest_rate},
        "next_health": {},
        "borrowing_constraint": {},
        "labor_income": {"wage": wage},
        "shocks": {
            "health": health_transition,
            "partner": partner_transition,
        },
    }
