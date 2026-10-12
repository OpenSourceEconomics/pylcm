"""The marriage-market example solves, simulates and routes both partners.

`lcm_examples.collective_household` is the worked model for the collective and
value-dependent declarations, so it is exercised the way a user would: solved
under debug validation, simulated from a cohort of singles, and read off the
published frame.
"""

import numpy as np
import pytest

from lcm.solver_api import DISSOLUTION_FLAG
from lcm_examples import collective_household as household

_N_PERIODS = 4
_N_SUBJECTS = 200


@pytest.fixture(scope="module")
def solved():
    """Solve the example once at a resolution the whole module shares."""
    model = household.get_model(n_periods=_N_PERIODS)
    params = household.get_params()
    period_to_regime_to_V_arr = model.solve(params=params, log_level="debug")
    return model, params, period_to_regime_to_V_arr


@pytest.fixture(scope="module")
def simulated(solved):
    """Simulate a cohort of half women and half men, all starting single."""
    model, params, period_to_regime_to_V_arr = solved
    result = model.simulate(
        params=params,
        initial_conditions=household.get_initial_conditions(
            n_subjects=_N_SUBJECTS, model=model
        ),
        solution=period_to_regime_to_V_arr,
        log_level="debug",
        seed=0,
    )
    return result.to_dataframe()


def test_the_household_publishes_one_value_per_partner(solved):
    """The couple's value function carries a trailing stakeholder axis."""
    model, _, period_to_regime_to_V_arr = solved

    assert period_to_regime_to_V_arr.values[1]["couple"].shape[-1] == 2
    assert model.user_regimes["couple"].stakeholders == ("f", "m")


def test_households_too_poor_to_outweigh_her_income_alone_dissolve(solved):
    """Participation rules out exactly the poorest households, and no richer one.

    Living alone she keeps her whole income, while the household splits the
    pooled income evenly, so the marriage premium has to make up the difference.
    At the last period in which the couple is active, it does not on the three
    lowest wealth nodes and does on every richer one.
    """
    _, _, solution = solved
    dissolution_flags = solution.replay_artifacts.project(DISSOLUTION_FLAG)
    flags = np.asarray(dissolution_flags[_N_PERIODS - 2]["couple"])

    assert np.flatnonzero(flags).tolist() == [0, 1, 2]


def test_some_simulated_couples_dissolve(simulated):
    """A household that participation rules out sends both partners back single."""
    by_subject = simulated.sort_values(["subject_id", "period"])
    previous = by_subject.groupby("subject_id")["regime_name"].shift()
    dissolved = (previous == "couple") & by_subject["regime_name"].isin(
        ["single_f", "single_m"]
    )

    assert dissolved.any()


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
