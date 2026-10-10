"""Deterministic Iskhakov et al. (2017) retirement model, re-exported for tests.

The model lives in `lcm_examples.iskhakov_et_al_2017`; this module keeps the
historical test-suite import location stable. Its `get_model` also admits
retirement at the first age: the analytical solution reports the retired value
at every age, so the tests query that problem explicitly.
"""

import functools

from lcm import Model
from lcm_examples.iskhakov_et_al_2017 import (
    CONSUMPTION_GRID,
    RETIREMENT_LAW,
    WEALTH_GRID,
    WORKING_LIFE_LAW,
    LaborSupply,
    RegimeId,
    borrowing_constraint,
    dead,
    get_params,
    is_working,
    labor_income,
    next_regime_from_retirement,
    next_regime_from_working,
    next_wealth,
    retirement,
    utility_retirement,
    utility_working,
    working_life,
)
from lcm_examples.iskhakov_et_al_2017 import (
    get_model as get_example_model,
)

__all__ = [
    "CONSUMPTION_GRID",
    "RETIREMENT_LAW",
    "WEALTH_GRID",
    "WORKING_LIFE_LAW",
    "LaborSupply",
    "RegimeId",
    "borrowing_constraint",
    "dead",
    "get_model",
    "get_params",
    "is_working",
    "labor_income",
    "next_regime_from_retirement",
    "next_regime_from_working",
    "next_wealth",
    "retirement",
    "utility_retirement",
    "utility_working",
    "working_life",
]


@functools.cache
def get_model(n_periods: int) -> Model:
    """Return the example model with working life and retirement as first-age starts."""
    example = get_example_model(n_periods=n_periods)
    assert example.ages is not None
    return Model(
        edges=example.edges,
        regimes=example.user_regimes,
        ages=example.ages,
        regime_id_class=RegimeId,
        initial_nodes={example.ages.exact_values[0]: ("working_life", "retirement")},
    )
