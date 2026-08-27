"""Routing a collective source's gated edge needs the simulated row's own role.

A collective regime declares one dissolution leg per stakeholder, and each leg
sends the dissolving row into *that* stakeholder's own continuing regime under
*that* stakeholder's own state projection. A `simulate()` call over a collective
source therefore has to say which role its population carries: a cohort routed
without one would follow a single partner's dissolution path for every row —
each divorced husband simulated as his wife, with her regime and her state.

`initial_conditions["own_stakeholder"]` names that role, one entry per subject,
and a subject starting in a collective regime must carry one wherever a route
it can still reach differs by role. A row keeps its role across an ordinary
regime transition, so that question is asked over the start's forward closure:
a two-leg edge the cohort runs into later demands a seed, and one in a regime
the cohort can never arrive at demands nothing. A singleton source declares a
single leg carrying no role at all, so a cohort starting there needs none
either, and both unseeded cases simulate unchanged.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import InvalidInitialConditionsError
from lcm.typing import ScalarInt
from tests.collective_fixtures import (
    make_couple_initial_conditions,
    make_two_stakeholder_model,
)
from tests.regime_building.test_collective_regime_simulate import (
    _BETA,
    _DISSOLUTION_PARAMS,
    _WAGE_3,
    DissolutionRegimeId,
    Work,
    _make_consent_regimes,
    _make_dissolution_regimes,
    _prob_one,
    _u_zero_collective,
)

# The three wage nodes the dissolution miniature is solved on. Its participation
# mask empties at wage 2 alone, so exactly the middle subject dissolves.
DISSOLUTION_WAGES = (1.0, 2.0, 3.0)

# Which subject of `DISSOLUTION_WAGES` leaves the collective regime.
DISSOLVING_ROW_MEMBERSHIP = (False, True, False)

# Nobody leaves through the leg the cohort's role did not name.
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
    """Starting a cohort in a collective regime with no role is a caller error.

    The message names the entry that is missing and the collective regime that
    needs it, so the caller can act on it.
    """
    with pytest.raises(InvalidInitialConditionsError, match=expected_in_message):
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
    """A cohort starting single routes through its gated edge with no role seeded.

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


def test_collective_source_with_no_role_dependent_route_needs_no_own_role():
    """A collective cohort whose model declares no multi-leg edge simulates unseeded.

    Nothing in such a model reads the role: there is no gated edge whose legs
    differ by stakeholder, so no route, regime or state projection can turn on
    it. Demanding a seed there would refuse a well specified model over a
    column that decides nothing.
    """
    model, params = make_two_stakeholder_model()

    result = model.simulate(
        params=params,
        initial_conditions=make_couple_initial_conditions(n_subjects=2),
        period_to_regime_to_V_arr=None,
        log_level="off",
        seed=0,
    )

    np.testing.assert_array_equal(
        np.asarray(result.raw_results["couple_terminal"][1].in_regime),
        np.asarray([True, True]),
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
            **(
                {}
                if own_stakeholder is None
                else {
                    "own_stakeholder": jnp.full(
                        len(DISSOLUTION_WAGES),
                        model.stakeholder_names_to_ids[own_stakeholder],
                        dtype=jnp.int32,
                    )
                }
            ),
        },
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=dissolution_flags,
        log_level="off",
        seed=0,
    )


@categorical(ordered=False)
class UnreachableRoleRoutingRegimeId:
    """Regime ids of the model whose role-routing regime no start can reach."""

    alone: ScalarInt
    alone_terminal: ScalarInt
    married: ScalarInt
    married_ir: ScalarInt
    married_terminal: ScalarInt
    single_f: ScalarInt
    single_f_terminal: ScalarInt
    single_m: ScalarInt
    single_m_terminal: ScalarInt


def _make_unreachable_role_routing_regimes():
    """Add a collective start that cannot reach the two-leg dissolution edge.

    `alone` carries the same stakeholders as `married` and runs out into its own
    terminal regime. Its per-target transition names `alone_terminal` alone, so
    no path leads from it to `married` and none of `married`'s legs can ever
    select on a role a subject started in `alone` with.
    """
    alone = Regime(
        transition={"alone_terminal": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_zero_collective, "utility_m": _u_zero_collective},
    )
    alone_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_3},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_zero_collective, "utility_m": _u_zero_collective},
    )
    return {
        "alone": alone,
        "alone_terminal": alone_terminal,
        **_make_dissolution_regimes(),
    }


def test_a_start_that_cannot_reach_a_role_dependent_route_needs_no_own_role():
    """A collective cohort is seeded for its own routes, not the model's.

    The role a subject starts with selects among the legs of a gated edge it
    can actually arrive at. A collective regime elsewhere in the model whose
    edge does distinguish roles decides nothing for a cohort that can never
    reach it, so demanding a seed on its account refuses a well specified
    model over a column that would go unread.
    """
    model = Model(
        regimes=_make_unreachable_role_routing_regimes(),
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=UnreachableRoleRoutingRegimeId,
    )
    solution, dissolution_flags = model.solve(
        params=_DISSOLUTION_PARAMS, log_level="off", return_dissolution_flags=True
    )

    result = model.simulate(
        params=_DISSOLUTION_PARAMS,
        initial_conditions={
            "wage": jnp.asarray(DISSOLUTION_WAGES),
            "age": jnp.zeros(len(DISSOLUTION_WAGES)),
            "regime_id": jnp.full(
                len(DISSOLUTION_WAGES),
                model.regime_names_to_ids["alone"],
                dtype=jnp.int32,
            ),
        },
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=dissolution_flags,
        log_level="off",
        seed=0,
    )

    np.testing.assert_array_equal(
        np.asarray(result.raw_results["alone_terminal"][1].in_regime),
        np.asarray([True, True, True]),
    )


@categorical(ordered=False)
class ReachableRoleRoutingRegimeId:
    """Regime ids of the model whose start runs into the role-routing regime."""

    prelude: ScalarInt
    married: ScalarInt
    married_ir: ScalarInt
    married_terminal: ScalarInt
    single_f: ScalarInt
    single_f_terminal: ScalarInt
    single_m: ScalarInt
    single_m_terminal: ScalarInt


def _shift_dissolution_one_age() -> dict:
    """Move the dissolution miniature one age later, freeing age 0 for a start."""
    regimes = _make_dissolution_regimes()
    windows = {
        "married": lambda age: (age >= 1) & (age < 2),
        "married_ir": lambda age: (age >= 2) & (age < 3),
        "married_terminal": lambda age: age >= 3,
        "single_f": lambda age: (age >= 2) & (age < 3),
        "single_f_terminal": lambda age: age >= 3,
        "single_m": lambda age: (age >= 2) & (age < 3),
        "single_m_terminal": lambda age: age >= 3,
    }
    return {
        name: regime.replace(active=windows[name]) for name, regime in regimes.items()
    }


def test_a_start_that_runs_into_a_role_dependent_route_still_needs_an_own_role():
    """A role is demanded wherever a subject can still arrive at a two-leg edge.

    A row keeps the role it started with across an ordinary regime transition,
    so a collective start that declares no role-dependent route of its own but
    runs into a regime that does is exactly the case the seed answers. Reading
    only the start regime's own edges would let such a cohort simulate with
    every row following one partner's dissolution path.
    """
    prelude = Regime(
        transition={"married": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_zero_collective, "utility_m": _u_zero_collective},
    )
    model = Model(
        regimes={"prelude": prelude, **_shift_dissolution_one_age()},
        ages=AgeGrid(start=0, stop=4, step="Y"),
        regime_id_class=ReachableRoleRoutingRegimeId,
    )
    solution, dissolution_flags = model.solve(
        params=_DISSOLUTION_PARAMS, log_level="off", return_dissolution_flags=True
    )

    with pytest.raises(InvalidInitialConditionsError, match="prelude"):
        model.simulate(
            params=_DISSOLUTION_PARAMS,
            initial_conditions={
                "wage": jnp.asarray(DISSOLUTION_WAGES),
                "age": jnp.zeros(len(DISSOLUTION_WAGES)),
                "regime_id": jnp.full(
                    len(DISSOLUTION_WAGES),
                    model.regime_names_to_ids["prelude"],
                    dtype=jnp.int32,
                ),
            },
            period_to_regime_to_V_arr=solution,
            period_to_regime_to_dissolution_flags=dissolution_flags,
            log_level="off",
            seed=0,
        )
