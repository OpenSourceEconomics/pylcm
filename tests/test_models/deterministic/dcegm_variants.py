"""Equivalent-spec pairs for solver comparisons (brute force vs DC-EGM).

The DC-EGM contract changes the model spec, not just a flag:

- the borrowing constraint is dropped (the savings grid's lower bound enforces it),
- the wealth transition consumes the post-decision state `savings` instead of
  wealth/consumption directly,
- `resources`, `savings`, and `inverse_marginal_utility` are declared as regime
  functions.

The builders here emit mathematically equivalent specs for both solvers so tests can
compare value functions on the shared wealth grid. Importable only once `lcm.solvers`
exists.
"""

import dataclasses
import functools
from typing import Literal

from lcm import AgeGrid, DiscreteGrid, IrregSpacedGrid, Model
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM
from lcm_examples.iskhakov_et_al_2017 import (
    CONSUMPTION_GRID,
    WEALTH_GRID,
    LaborSupply,
    dead,
    inverse_marginal_utility,
    is_working,
    labor_income,
    next_wealth_from_savings,
    resources,
    savings,
    utility_retirement,
    utility_working,
)
from tests.test_models.deterministic import base, retirement_only

# Exogenous end-of-period savings grid; the lower bound is the borrowing limit
# (savings >= 0 encodes the original `consumption <= wealth` constraint).
# Nodes are cubically clustered toward the borrowing limit: the value function
# curves hardest where the constraint starts to bind, and the published V is
# interpolated from endogenous points spaced like the savings nodes — a
# uniform grid under-resolves the lowest wealth nodes by orders of magnitude.
SAVINGS_GRID = IrregSpacedGrid(points=tuple(400.0 * (i / 199) ** 3 for i in range(200)))


DCEGM_SOLVER = DCEGM(
    continuous_state="wealth",
    continuous_action="consumption",
    resources="resources",
    post_decision_function="savings",
    savings_grid=SAVINGS_GRID,
    # The final decision period consumes everything, so its carry in the
    # queried resources range consists of constrained-segment points only;
    # 64 of them keep the geometric spacing ratio (and hence the carry
    # interpolation error) small.
    n_constrained_points=64,
)


dcegm_retirement = UserRegime(
    transition=retirement_only.next_regime_from_retirement,
    actions={"consumption": CONSUMPTION_GRID},
    states={"wealth": WEALTH_GRID},
    state_transitions={"wealth": next_wealth_from_savings},
    functions={
        "utility": utility_retirement,
        "resources": resources,
        "savings": savings,
        "inverse_marginal_utility": inverse_marginal_utility,
    },
    solver=DCEGM_SOLVER,
)


dcegm_working_life = UserRegime(
    transition=base.next_regime_from_working,
    actions={
        "labor_supply": DiscreteGrid(LaborSupply),
        "consumption": CONSUMPTION_GRID,
    },
    states={"wealth": WEALTH_GRID},
    state_transitions={"wealth": next_wealth_from_savings},
    functions={
        "utility": utility_working,
        "labor_income": labor_income,
        "is_working": is_working,
        "resources": resources,
        "savings": savings,
        "inverse_marginal_utility": inverse_marginal_utility,
    },
    solver=DCEGM_SOLVER,
)


dcegm_retirement_full = UserRegime(
    transition=base.next_regime_from_retirement,
    actions={"consumption": CONSUMPTION_GRID},
    states={"wealth": WEALTH_GRID},
    state_transitions={"wealth": next_wealth_from_savings},
    functions={
        "utility": utility_retirement,
        "resources": resources,
        "savings": savings,
        "inverse_marginal_utility": inverse_marginal_utility,
    },
    solver=DCEGM_SOLVER,
)


@functools.cache
def get_retirement_only_model(
    solver: Literal["brute_force", "dcegm"], n_periods: int
) -> Model:
    """Build the two-regime retirement model for the requested solver."""
    if solver == "brute_force":
        return retirement_only.get_model(n_periods)
    ages = AgeGrid(start=40, stop=40 + (n_periods - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    return Model(
        regimes={
            "retirement": dcegm_retirement.replace(
                active=lambda age, la=last_age: age < la
            ),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=retirement_only.RetirementOnlyRegimeId,
    )


@functools.cache
def get_full_model(
    solver: Literal["brute_force", "dcegm"],
    n_periods: int,
    *,
    upper_envelope: Literal["fues", "rfc", "ltm", "mss"] = "fues",
) -> Model:
    """Build the three-regime worker/retirement/dead model for the requested solver."""
    if solver == "brute_force":
        return base.get_model(n_periods)
    ages = AgeGrid(start=40, stop=40 + (n_periods - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    dcegm_solver = dataclasses.replace(DCEGM_SOLVER, upper_envelope=upper_envelope)
    return Model(
        regimes={
            "working_life": dcegm_working_life.replace(
                active=lambda age, la=last_age: age < la, solver=dcegm_solver
            ),
            "retirement": dcegm_retirement_full.replace(
                active=lambda age, la=last_age: age < la, solver=dcegm_solver
            ),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=base.RegimeId,
    )


def get_retirement_only_params(
    n_periods: int,
    *,
    discount_factor: float = 0.98,
    interest_rate: float = 0.0,
) -> dict:
    """Params for the retirement-only pair; valid for both solver variants."""
    return retirement_only.get_params(
        n_periods,
        discount_factor=discount_factor,
        interest_rate=interest_rate,
    )


def get_full_params(
    n_periods: int,
    *,
    discount_factor: float = 0.98,
    disutility_of_work: float = 1.0,
    interest_rate: float = 0.0,
    wage: float = 20.0,
) -> dict:
    """Params for the full-model pair; valid for both solver variants."""
    return base.get_params(
        n_periods=n_periods,
        discount_factor=discount_factor,
        disutility_of_work=disutility_of_work,
        interest_rate=interest_rate,
        wage=wage,
    )
