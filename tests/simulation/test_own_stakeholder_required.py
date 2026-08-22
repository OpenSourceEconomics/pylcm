"""Routing a collective source's gated edge needs the simulated row's own role.

A collective regime declares one dissolution leg per stakeholder, and each leg
sends the dissolving row into *that* stakeholder's own continuing regime under
*that* stakeholder's own state projection. A `simulate()` call over a collective
source therefore has to say which role its population carries: a cohort routed
without one would follow a single partner's dissolution path for every row —
each divorced husband simulated as his wife, with her regime and her state.

`own_stakeholder` names that role, and for a collective source it is required.
A singleton source declares a single leg carrying no role at all, so it neither
needs nor accepts one and simulates unchanged.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, Model, categorical
from lcm.typing import ScalarInt
from tests.regime_building.test_collective_regime_simulate import (
    _BETA,
    _DISSOLUTION_PARAMS,
    DissolutionRegimeId,
    _make_consent_regimes,
    _make_dissolution_regimes,
)

# The three wage nodes the dissolution miniature is solved on. Its participation
# mask empties at wage 2 alone, so exactly the middle subject dissolves.
DISSOLUTION_WAGES = (1.0, 2.0, 3.0)

# Which subject of `DISSOLUTION_WAGES` leaves the collective regime.
DISSOLVING_ROW_MEMBERSHIP = (False, True, False)

# Nobody leaves through the leg `own_stakeholder` did not name.
UNTAKEN_LEG_MEMBERSHIP = (False, False, False)

# The two wage nodes the singleton-source consent miniature is solved on: the
# low-wage row marries, the high-wage row stays single.
CONSENT_WAGES = (1.0, 2.0)

# Which consent-model subject reaches the collective target.
CONSENT_MARRIED_MEMBERSHIP = (True, False)


@categorical(ordered=False)
class ConsentRegimeId:
    """Regime ids of the singleton-source consent model."""

    single_f: ScalarInt
    single_f_terminal: ScalarInt
    single_m_terminal: ScalarInt
    married_terminal: ScalarInt


@pytest.fixture(scope="module")
def dissolution_model_and_solution():
    """Solve the two-leg collective dissolution miniature once.

    `married` carries the stakeholders `("f", "m")` and a dissolution edge whose
    legs send the wife to `single_f` and the husband to `single_m`.
    """
    model = Model(
        regimes=_make_dissolution_regimes(),
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=DissolutionRegimeId,
    )
    solution, dissolution_flags = model.solve(
        params=_DISSOLUTION_PARAMS, log_level="off", return_dissolution_flags=True
    )
    return model, solution, dissolution_flags


@pytest.fixture(scope="module")
def consent_model_and_solution():
    """Solve the singleton-source consent miniature once.

    `single_f` is a singleton whose gated edge into the collective
    `married_terminal` declares one leg, and that leg carries no role.
    """
    model = Model(
        regimes=_make_consent_regimes(),
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=ConsentRegimeId,
    )
    solution = model.solve(params={"discount_factor": _BETA}, log_level="off")
    return model, solution


@pytest.mark.parametrize("expected_in_message", ["own_stakeholder", "married"])
def test_collective_source_without_an_own_role_is_refused(
    dissolution_model_and_solution, expected_in_message
):
    """Simulating a collective source without an own role is a caller error.

    The message names the argument that is missing and the collective regime
    that needs it, so the caller can act on it.
    """
    with pytest.raises(ValueError, match=expected_in_message):
        _simulate_dissolution_cohort(
            setup=dissolution_model_and_solution, own_stakeholder=None
        )


@pytest.mark.parametrize(
    ("fallback_regime", "expected_membership"),
    [
        ("single_m", DISSOLVING_ROW_MEMBERSHIP),
        ("single_f", UNTAKEN_LEG_MEMBERSHIP),
    ],
)
def test_named_own_role_routes_the_dissolving_row_to_its_own_leg(
    dissolution_model_and_solution, fallback_regime, expected_membership
):
    """An all-husbands cohort dissolves into `single_m`, never into `single_f`.

    The role selects the leg, and the leg selects the continuing regime, so the
    wife's leg stays empty for a cohort that declared itself the husbands.
    """
    result = _simulate_dissolution_cohort(
        setup=dissolution_model_and_solution, own_stakeholder="m"
    )

    np.testing.assert_array_equal(
        np.asarray(result.raw_results[fallback_regime][1].in_regime),
        np.asarray(expected_membership),
    )


def test_singleton_source_needs_no_own_role(consent_model_and_solution):
    """A singleton source's gated edge routes without any role being declared.

    The role decides between one stakeholder's leg and another's, and a
    singleton source has only the one leg, so requiring a role there would
    refuse a model that is perfectly well specified.
    """
    model, solution = consent_model_and_solution

    result = model.simulate(
        params={"discount_factor": _BETA},
        initial_conditions={
            "wage": jnp.asarray(CONSENT_WAGES),
            "age": jnp.zeros(len(CONSENT_WAGES)),
            "regime_id": jnp.full(
                len(CONSENT_WAGES),
                model.regime_names_to_ids["single_f"],
                dtype=jnp.int32,
            ),
        },
        period_to_regime_to_V_arr=solution,
        log_level="off",
        seed=0,
    )

    np.testing.assert_array_equal(
        np.asarray(result.raw_results["married_terminal"][1].in_regime),
        np.asarray(CONSENT_MARRIED_MEMBERSHIP),
    )


def _simulate_dissolution_cohort(*, setup, own_stakeholder):
    """Simulate the dissolution miniature's three-subject cohort at one role."""
    model, solution, dissolution_flags = setup
    return model.simulate(
        params=_DISSOLUTION_PARAMS,
        initial_conditions={
            "wage": jnp.asarray(DISSOLUTION_WAGES),
            "age": jnp.zeros(len(DISSOLUTION_WAGES)),
            "regime_id": jnp.full(
                len(DISSOLUTION_WAGES),
                model.regime_names_to_ids["married"],
                dtype=jnp.int32,
            ),
        },
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=dissolution_flags,
        log_level="off",
        seed=0,
        own_stakeholder=own_stakeholder,
    )
