"""The marriage-market example solves, simulates and routes both partners.

`lcm_examples.collective_household` is the worked model for the collective and
value-dependent declarations, so it is exercised the way a user would: solved
under debug validation, simulated from a cohort of singles and couples, and read
off the published frame.
"""

from collections import Counter
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.regime_building import Q_and_F
from _lcm.regime_building.collective import NO_ROLE
from lcm import Model
from lcm.composition_actions import PCOnActions
from lcm.solver_api import ResultRetention, SolutionResult
from lcm.typing import FloatND, RegimeName, UserParamsNode
from lcm_examples import collective_household as household

_N_PERIODS = 3
_N_AGENTS = 200

type _Solved = tuple[Model, dict[str, UserParamsNode], SolutionResult]


@pytest.fixture(scope="module")
def solved() -> _Solved:
    """Solve the example once at a resolution the whole module shares."""
    model = household.get_model(n_periods=_N_PERIODS)
    params = household.get_params()
    period_to_regime_to_V_arr = model.solve(params=params, log_level="debug")
    return model, params, period_to_regime_to_V_arr


@pytest.fixture(scope="module")
def simulated(solved: _Solved) -> pd.DataFrame:
    """Simulate half the cohort starting single and half starting married.

    Starting both kinds at age 0 puts a marriage and a dissolution into the one
    landing that three periods leave before the terminal regimes take over.
    Within each half, women and men alternate.
    """
    model, params, period_to_regime_to_V_arr = solved
    n_single = _N_AGENTS // 2
    is_single = jnp.arange(_N_AGENTS) < n_single
    is_woman = jnp.arange(_N_AGENTS) % 2 == 0
    regime_ids = model.regime_names_to_ids
    role_ids = model.member_names_to_ids
    initial_conditions = {
        "wealth": jnp.concatenate(
            [
                jnp.linspace(household.WEALTH_MIN, household.WEALTH_MAX, n_single),
                jnp.linspace(
                    2 * household.WEALTH_MIN,
                    2 * household.WEALTH_MAX,
                    _N_AGENTS - n_single,
                ),
            ]
        ),
        "age": jnp.zeros(_N_AGENTS),
        "regime_id": jnp.where(
            is_single,
            jnp.where(is_woman, regime_ids["single_f"], regime_ids["single_m"]),
            regime_ids["couple"],
        ).astype(jnp.int32),
        "own_member": jnp.where(
            is_single, NO_ROLE, jnp.where(is_woman, role_ids["f"], role_ids["m"])
        ).astype(jnp.int32),
    }
    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=period_to_regime_to_V_arr,
        log_level="debug",
        seed=0,
    )
    return result.to_dataframe()


def test_the_household_publishes_one_value_per_partner(solved: _Solved) -> None:
    """The couple's value function carries a trailing member axis."""
    model, _, period_to_regime_to_V_arr = solved

    assert period_to_regime_to_V_arr.values[1]["couple"].shape[-1] == 2
    assert model.user_regimes["couple"].members == ("f", "m")


def test_households_too_poor_to_outweigh_her_income_alone_dissolve() -> None:
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
            "own_member": jnp.repeat(
                jnp.asarray([model.member_names_to_ids[role] for role in ("f", "m")]),
                n_wealth,
            ),
        },
        seed=0,
        log_level="off",
    ).to_dataframe()
    initial = frame.loc[frame["period"] == 0].sort_values("agent_id")
    for i, role in enumerate(("f", "m")):
        rows = initial.iloc[i * n_wealth : (i + 1) * n_wealth]
        leaves = rows["regime_name"].to_numpy() == f"single_{role}"
        assert np.flatnonzero(leaves).tolist() == [0, 1, 2]
        assert (rows.loc[~leaves, "regime_name"] == "couple").all()


def test_some_simulated_couples_dissolve(simulated: pd.DataFrame) -> None:
    """A couple that separates sends both members back to their single regimes."""
    by_agent = simulated.sort_values(["agent_id", "period"])
    previous = by_agent.groupby("agent_id")["regime_name"].shift()
    dissolved = by_agent.loc[
        (previous == "couple") & by_agent["regime_name"].isin(["single_f", "single_m"])
    ]

    assert set(zip(dissolved["period"], dissolved["regime_name"], strict=True)) == {
        (1, "single_f"),
        (1, "single_m"),
    }


def test_singles_of_both_sexes_reach_the_household(
    simulated: pd.DataFrame,
) -> None:
    """Consent is mutual, so the couple is entered from either single regime."""
    starts_single = simulated.loc[
        (simulated["period"] == 0)
        & simulated["regime_name"].isin(["single_f", "single_m"]),
        "agent_id",
    ]
    entered = simulated.loc[
        simulated["agent_id"].isin(starts_single)
        & (simulated["regime_name"] == "couple"),
        "own_member",
    ]

    assert set(entered.dropna().unique()) == {"f", "m"}


def test_a_row_in_a_singleton_regime_carries_no_role(
    simulated: pd.DataFrame,
) -> None:
    """`own_member` is a household role, so a single row has none."""
    single_rows = simulated.loc[
        simulated["regime_name"].isin(["single_f", "single_m"]), "own_member"
    ]

    assert single_rows.isna().all()


def test_every_agent_ends_in_a_terminal_regime(
    simulated: pd.DataFrame,
) -> None:
    """Nobody is lost on the way: the last period accounts for every agent."""
    last = simulated.loc[simulated["period"] == _N_PERIODS - 1]

    assert len(last) == _N_AGENTS
    assert last["regime_name"].str.endswith("terminal").all()


def test_composition_actions_replay_from_values_without_dissolution_artifacts() -> None:
    """Consent and separation replay without retained dissolution flags."""
    model = household.get_model(n_periods=3, wealth_n_points=6, consumption_n_points=6)
    params = household.get_params()
    solution = model.solve(
        params=params, log_level="off", retention=ResultRetention.VALUES
    )
    frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions=household.get_initial_conditions(n_agents=64, model=model),
        seed=0,
        log_level="off",
    ).to_dataframe()
    terminal = frame.loc[frame["period"] == 2]
    np.testing.assert_array_equal(np.sort(terminal["agent_id"]), np.arange(64))
    assert terminal["regime_name"].str.endswith("terminal").all()


def test_couple_dissolves_under_the_eckstein_keane_lifshitz_rule() -> None:
    """Either partner can leave, and a partner exactly indifferent leaves."""
    model = household.get_model(n_periods=2, wealth_n_points=2, consumption_n_points=2)
    declaration = model.user_regimes["couple"].composition_action
    assert getattr(declaration, "rule", None) == PCOnActions(tie="leave")


def test_each_query_action_problem_is_traced_once_per_solve(
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A query read inside another query's continuation reuses the target's trace.

    A composition action reads the destination's problem, and a consent counterparty its
    regime's problem, so the reads nest. Re-tracing a problem at every read
    multiplies the traces by the fan-out of every read.
    """
    traces: Counter[int] = Counter()
    original = Q_and_F._CollectiveQAndF.evaluated

    def counted(
        self: Q_and_F._CollectiveQAndF,
        *args: MappingProxyType[RegimeName, FloatND],
        **kwargs: Q_and_F.QAndFArg,
    ) -> Q_and_F.EvaluatedQAndF:
        traces[id(self)] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Q_and_F._CollectiveQAndF, "evaluated", counted)
    model = household.get_model(n_periods=3, wealth_n_points=2, consumption_n_points=2)
    model.solve(params=household.get_params(), log_level="off")
    assert traces
    assert max(traces.values()) == 1
