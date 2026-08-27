"""The marriage-market example solves, simulates and routes both partners.

`lcm_examples.collective_household` is the worked model for the collective and
value-dependent declarations, so it is exercised the way a user would: solved
under debug validation, simulated from a cohort of singles, and read off the
published frame.
"""

import numpy as np
import pytest

from lcm_examples import collective_household as household

_N_PERIODS = 4
_N_SUBJECTS = 200


@pytest.fixture(scope="module")
def solved():
    """Solve the example once at a resolution the whole module shares."""
    model = household.get_model(n_periods=_N_PERIODS)
    params = household.get_params()
    period_to_regime_to_V_arr, dissolution_flags = model.solve(
        params=params, log_level="debug", return_dissolution_flags=True
    )
    return model, params, period_to_regime_to_V_arr, dissolution_flags


@pytest.fixture(scope="module")
def simulated(solved):
    """Simulate a cohort of half women and half men, all starting single."""
    model, params, period_to_regime_to_V_arr, dissolution_flags = solved
    result = model.simulate(
        params=params,
        initial_conditions=household.get_initial_conditions(
            n_subjects=_N_SUBJECTS, model=model
        ),
        period_to_regime_to_V_arr=period_to_regime_to_V_arr,
        period_to_regime_to_dissolution_flags=dissolution_flags,
        log_level="debug",
        seed=0,
    )
    return result.to_dataframe()


def test_the_household_publishes_one_value_per_partner(solved):
    """The couple's value function carries a trailing stakeholder axis."""
    model, _, period_to_regime_to_V_arr, _ = solved

    assert period_to_regime_to_V_arr[1]["couple"].shape[-1] == 2
    assert model.user_regimes["couple"].stakeholders == ("f", "m")


def test_the_poorest_household_is_the_one_participation_rules_out(solved):
    """Dissolution is where a partner does better alone, not everywhere.

    At the last period in which the couple is active, the household on the
    lowest wealth node cannot clear both participation constraints while every
    richer node can — so exactly one cell carries the flag.
    """
    _, _, _, dissolution_flags = solved

    flags = np.asarray(dissolution_flags[_N_PERIODS - 2]["couple"])

    assert flags[0]
    assert not flags[1:].any()


def test_singles_of_both_sexes_reach_the_household(simulated):
    """Consent is mutual, so the couple is entered from either single regime."""
    entered = simulated.loc[simulated["regime_name"] == "couple", "own_stakeholder"]

    assert set(entered.dropna().unique()) == {"f", "m"}


def test_a_row_in_a_singleton_regime_carries_no_role(simulated):
    """`own_stakeholder` is a household role, so a single row has none."""
    single_rows = simulated.loc[
        simulated["regime_name"].isin(["single_f", "single_m"]), "own_stakeholder"
    ]

    assert single_rows.isna().all()


def test_every_subject_ends_in_a_terminal_regime(simulated):
    """Nobody is lost on the way: the last period accounts for every subject."""
    last = simulated.loc[simulated["period"] == _N_PERIODS - 1]

    assert len(last) == _N_SUBJECTS
    assert last["regime_name"].str.endswith("terminal").all()
