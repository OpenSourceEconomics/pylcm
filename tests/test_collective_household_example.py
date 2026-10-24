"""The marriage-market example solves, simulates and routes both partners.

`lcm_examples.collective_household` is the worked model for the collective and
value-dependent declarations, so it is exercised the way a user would: solved
under debug validation, simulated from a cohort of singles, and read off the
published frame.
"""

from dataclasses import replace
from typing import Unpack

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.solution import backward_induction
from lcm.exits import RuleResult
from lcm.solver_api import KernelOutput, ResultRetention
from lcm_examples import collective_household as household
from tests.solution.test_solution_result import _KernelKwargs

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


def test_households_too_poor_to_outweigh_her_income_alone_dissolve():
    """Both initial roles dissolve only on the poorest household wealth nodes.

    Two periods place the initial cohort in the final nonterminal problem:
    its age-independent utility, transition and terminal payoff are the same
    as the final nonterminal problem of the longer marriage-market model.
    """
    model = household.get_model(n_periods=2)
    params = household.get_params()
    solution = model.solve(
        params=params, log_level="off", retention=ResultRetention.VALUES
    )
    wealth = model.state_grid(params=params, regime_name="couple", state_name="wealth")
    n_wealth = len(wealth)
    frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wealth": jnp.tile(wealth, 2),
            "age": jnp.zeros(2 * n_wealth),
            "regime_id": jnp.full(
                2 * n_wealth, model.regime_names_to_ids["couple"], dtype=jnp.int32
            ),
            "own_stakeholder": jnp.repeat(
                jnp.asarray(
                    [model.stakeholder_names_to_ids[role] for role in ("f", "m")]
                ),
                n_wealth,
            ),
        },
        seed=0,
        log_level="off",
    ).to_dataframe()
    initial = frame.loc[frame["period"] == 0].sort_values("subject_id")
    for i, role in enumerate(("f", "m")):
        rows = initial.iloc[i * n_wealth : (i + 1) * n_wealth]
        leaves = rows["regime_name"].to_numpy() == f"single_{role}"
        assert np.flatnonzero(leaves).tolist() == [0, 1, 2]
        assert (rows.loc[~leaves, "regime_name"] == "couple").all()


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


def test_arrival_decisions_replay_from_values_without_dissolution_artifacts():
    """Arrival support and consent do not require retained dissolution flags."""
    model = household.get_model(n_periods=3, wealth_n_points=6, consumption_n_points=6)
    params = household.get_params()
    solution = model.solve(
        params=params, log_level="off", retention=ResultRetention.VALUES
    )
    frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions=household.get_initial_conditions(n_subjects=64, model=model),
        seed=0,
        log_level="off",
    ).to_dataframe()
    terminal = frame.loc[frame["period"] == 2]
    np.testing.assert_array_equal(np.sort(terminal["subject_id"]), np.arange(64))
    assert terminal["regime_name"].str.endswith("terminal").all()


def test_participation_rule_keeps_one_common_action_with_threshold_only_slack():
    """A zero Pareto weight keeps its veto; exact ties keep the first action."""
    model = household.get_model(n_periods=2, wealth_n_points=2, consumption_n_points=2)
    rule = model.user_regimes["couple"].exits["dissolve"].rule
    assert callable(rule)
    cases = [
        ([2.0, 2.0], [1.0, 1.0], {"f": 0.5, "m": 0.5}, 0),
        ([0.5, 2.0], [100.0, 1.0], {"f": 0.0, "m": 1.0}, 1),
        ([-np.inf, 2.0], [-np.inf, 1.0], {"f": 0.0, "m": 1.0}, 1),
        ([0.0, 0.0], [1.0, 1.0], {"f": 0.5, "m": 0.5}, 2),
    ]
    for q_f, q_m, weights, expected in cases:
        selected = rule(
            Q_f=jnp.asarray(q_f),
            Q_m=jnp.asarray(q_m),
            exit_payoffs={"dissolve": {"f": jnp.asarray(1.5), "m": jnp.asarray(1.0)}},
            alternative_ids=jnp.arange(3, dtype=jnp.int32),
            pareto_weights={
                name: jnp.asarray(value) for name, value in weights.items()
            },
            slack=jnp.asarray(0.5),
        )
        assert isinstance(selected, RuleResult)
        assert int(selected.action) == expected


def test_a_missing_declared_arrival_payload_fails_before_the_period_roll(
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dated publisher must return its payload at the actual declared date."""
    original = backward_induction._run_period_kernel

    def missing_payload(**kwargs: Unpack[_KernelKwargs]) -> KernelOutput:
        output = original(**kwargs)
        if kwargs["regime_name"] == "couple" and kwargs["period"] == 1:
            return replace(output, continuations={})
        return output

    monkeypatch.setattr(backward_induction, "_run_period_kernel", missing_payload)
    model = household.get_model(n_periods=3, wealth_n_points=2, consumption_n_points=2)
    with pytest.raises(
        RuntimeError,
        match=(
            r"Regime 'couple' in period 1.*missing required.*"
            r"pylcm\.arrival\.query_operands"
        ),
    ):
        model.solve(
            params=household.get_params(),
            retention=ResultRetention.VALUES,
            log_level="off",
        )
