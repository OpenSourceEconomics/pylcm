"""Routing a collective regime's exit needs the simulated row's own role.

A collective regime declares one exit target per member, and each target
sends the leaving row into *that* member's own continuing regime under
*that* member's own state projection. A `simulate()` call over a collective
source therefore has to say which role its population carries: a cohort routed
without one would follow a single partner's dissolution path for every row —
each divorced husband simulated as his wife, with her regime and her state.

`initial_conditions["own_member"]` names that role, one entry per subject,
and a subject starting in a collective regime must carry one wherever a route
it can still reach differs by role. A row keeps its role across an ordinary
regime transition, so that question is asked over the start's forward closure:
a two-target exit the cohort runs into later demands a seed, and one in a regime
the cohort can never arrive at demands nothing. A singleton source's consented
exit gives its row the role it enters with, so a cohort starting there needs none
either, and both unseeded cases simulate unchanged.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    CollectiveUtility,
    Consent,
    Counterparty,
    DiscreteGrid,
    InitialNodes,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import InvalidInitialConditionsError
from lcm.exits import Exit, ExitTarget
from lcm.result import SimulationResult
from lcm.solver_api import ResultRetention, SolutionResult
from lcm.typing import (
    ContinuousState,
    DiscreteAction,
    FloatND,
    RegimeName,
    ScalarInt,
)
from tests.collective_fixtures import (
    make_couple_initial_conditions,
    make_two_member_model,
)
from tests.conftest import assert_agrees_to_ulp
from tests.regime_building.test_collective_regime_simulate import (
    _DISSOLUTION_PARAMS,
    _WAGE_3,
    DissolutionRegimeId,
    Work,
    _identity_wage,
    _make_dissolution_regimes,
    _u_zero_collective,
)

# The three wage nodes the dissolution miniature is solved on. The exit of
# `married_ir` leaves no action both members accept at wage 2 alone, so exactly
# the middle subject dissolves.
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

# The edges of the dissolution miniature: the couple meets its own exit in
# `married_ir`, whose targets send each member to that member's single regime.
DISSOLUTION_EDGES = {
    "married": {"married_ir": 0},
    "married_ir": {"married_terminal": 1},
    "single_f": {"single_f_terminal": 1},
    "single_m": {"single_m_terminal": 1},
}

# Grid of the two consent-model wages.
_WAGE_2 = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)


def _u_single_f(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return wage * work


def _u_single_m(wage: ContinuousState) -> FloatND:
    return jnp.where(wage < 1.5, 0.5, 5.0)


def _u_married_f(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 2.0 * wage + 0.0 * work


def _u_married_m(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return wage + 0.0 * work


def _make_consent_regimes() -> dict[RegimeName, Regime]:
    """Return a singleton `single_f` whose consented exit enters `married`.

    She marries where both partners' married values strictly exceed their single
    ones: at wage 1 (`2 > 1` and `1 > 0.5`), not at wage 2 (`2 < 5` for him).
    """
    return {
        "single_f": Regime(
            states={"wage": _WAGE_2},
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={"utility": _u_single_f},
            exit=Exit(
                targets={
                    "self": ExitTarget(
                        regime="married",
                        role="f",
                        projection={"wage": _identity_wage},
                    )
                },
                rule=Consent(
                    counterparty=Counterparty(
                        regime="single_m",
                        role="m",
                        projection={"wage": _identity_wage},
                    )
                ),
            ),
        ),
        "single_m": Regime(
            states={"wage": _WAGE_2}, functions={"utility": _u_single_m}
        ),
        "married": Regime(
            states={"wage": _WAGE_2},
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={
                "utility": CollectiveUtility(
                    utilities={"f": _u_married_f, "m": _u_married_m}
                )
            },
        ),
    }


@categorical(ordered=False)
class ConsentRegimeId:
    """Regime ids of the singleton-source consent model."""

    single_f: ScalarInt
    single_m: ScalarInt
    married: ScalarInt


@pytest.fixture(scope="module")
def dissolution_model_and_solution() -> tuple[Model, SolutionResult]:
    """Solve the two-leg collective dissolution miniature once.

    `married` carries the members `("f", "m")` into `married_ir`, whose exit
    sends the wife to `single_f` and the husband to `single_m`.
    """
    model = Model(
        regimes=_make_dissolution_regimes(),
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=DissolutionRegimeId,
        initial_nodes={0: "married"},
        edges=DISSOLUTION_EDGES,
    )
    solution = model.solve(params=_DISSOLUTION_PARAMS, log_level="off")
    return model, solution


@pytest.fixture(scope="module")
def consent_model_and_solution() -> tuple[Model, SolutionResult]:
    """Solve the singleton-source consent miniature once.

    `single_f` is a singleton whose consented exit enters the collective
    `married` as its member `f`; the exit itself names the role.
    """
    model = Model(
        regimes=_make_consent_regimes(),
        n_periods=1,
        regime_id_class=ConsentRegimeId,
        initial_nodes=InitialNodes(by_period={0: "single_f"}),
        edges={},
    )
    solution = model.solve(params={}, log_level="off")
    return model, solution


@pytest.mark.parametrize("expected_in_message", ["own_member", "married"])
def test_collective_source_without_an_own_role_is_refused(
    *,
    dissolution_model_and_solution: tuple[Model, SolutionResult],
    expected_in_message: str,
) -> None:
    """Starting a cohort in a collective regime with no role is a caller error.

    The message names the entry that is missing and the collective regime that
    needs it, so the caller can act on it.
    """
    with pytest.raises(InvalidInitialConditionsError, match=expected_in_message):
        _simulate_dissolution_cohort(
            setup=dissolution_model_and_solution, own_member=None
        )


@pytest.mark.parametrize(
    ("fallback_regime", "expected_membership"),
    [
        ("single_m", DISSOLVING_ROW_MEMBERSHIP),
        ("single_f", UNTAKEN_LEG_MEMBERSHIP),
    ],
)
def test_named_own_role_routes_the_dissolving_row_to_its_own_leg(
    *,
    dissolution_model_and_solution: tuple[Model, SolutionResult],
    fallback_regime: RegimeName,
    expected_membership: tuple[bool, ...],
) -> None:
    """An all-husbands cohort dissolves into `single_m`, never into `single_f`.

    The role selects the exit target, and the target selects the continuing
    regime, so the wife's stays empty for a cohort that declared itself the husbands.
    """
    result = _simulate_dissolution_cohort(
        setup=dissolution_model_and_solution, own_member="m"
    )

    np.testing.assert_array_equal(
        np.asarray(result.raw_results[fallback_regime][1].in_regime),
        np.asarray(expected_membership),
    )


def test_singleton_source_needs_no_own_role(
    consent_model_and_solution: tuple[Model, SolutionResult],
) -> None:
    """A cohort starting single enters the couple through its exit with no role seeded.

    The role decides between one member's exit target and another's, and a
    singleton source's consented exit names the role it enters with, so requiring
    a role there would refuse a model that is perfectly well specified.
    """
    model, solution = consent_model_and_solution

    result = model.simulate(
        params={},
        initial_conditions={
            "wage": jnp.asarray(CONSENT_WAGES),
            "period": jnp.zeros(len(CONSENT_WAGES), dtype=jnp.int32),
            "regime_id": jnp.full(
                len(CONSENT_WAGES),
                model.regime_names_to_ids["single_f"],
                dtype=jnp.int32,
            ),
        },
        solution=solution,
        log_level="off",
        seed=0,
    )

    np.testing.assert_array_equal(
        np.asarray(result.raw_results["married"][0].in_regime),
        np.asarray(CONSENT_MARRIED_MEMBERSHIP),
    )


def test_collective_source_with_no_role_dependent_route_needs_no_own_role() -> None:
    """A collective cohort whose model declares no multi-leg edge simulates unseeded.

    Nothing in such a model reads the role: there is no exit or gated edge whose
    targets differ by member, so no route, regime or state projection can turn on
    it. Demanding a seed there would refuse a well specified model over a
    column that decides nothing.
    """
    model, params = make_two_member_model()

    result = model.simulate(
        params=params,
        initial_conditions=make_couple_initial_conditions(n_subjects=2),
        log_level="off",
        seed=0,
    )

    np.testing.assert_array_equal(
        np.asarray(result.raw_results["couple_terminal"][1].in_regime),
        np.asarray([True, True]),
    )


def _simulate_dissolution_cohort(
    *, setup: tuple[Model, SolutionResult], own_member: str | None
) -> SimulationResult:
    """Simulate the dissolution miniature's three-subject cohort at one role."""
    model, solution = setup
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
                if own_member is None
                else {
                    "own_member": jnp.full(
                        len(DISSOLUTION_WAGES),
                        model.member_names_to_ids[own_member],
                        dtype=jnp.int32,
                    )
                }
            ),
        },
        solution=solution,
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


def _make_unreachable_role_routing_regimes() -> dict[RegimeName, Regime]:
    """Add a collective start that cannot reach the two-target dissolution exit.

    `alone` carries the same members as `married` and runs out into its own
    terminal regime. Its only outgoing edge leads to `alone_terminal`, so no
    path leads from it to `married_ir` and none of its exit targets can ever
    select on a role a subject started in `alone` with.
    """
    alone = Regime(
        states={"wage": _WAGE_3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_zero_collective, "m": _u_zero_collective}
            )
        },
    )
    alone_terminal = Regime(
        states={"wage": _WAGE_3},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_zero_collective, "m": _u_zero_collective}
            )
        },
    )
    return {
        "alone": alone,
        "alone_terminal": alone_terminal,
        **_make_dissolution_regimes(),
    }


def test_a_start_that_cannot_reach_a_role_dependent_route_needs_no_own_role() -> None:
    """A collective cohort is seeded for its own routes, not the model's.

    The role a subject starts with selects among the targets of an exit it can
    actually arrive at. A collective regime elsewhere in the model whose
    exit does distinguish roles decides nothing for a cohort that can never
    reach it, so demanding a seed on its account refuses a well specified
    model over a column that would go unread.
    """
    model = Model(
        regimes=_make_unreachable_role_routing_regimes(),
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=UnreachableRoleRoutingRegimeId,
        initial_nodes={0: ("alone", "married")},
        edges={"alone": {"alone_terminal": 0}, **DISSOLUTION_EDGES},
    )
    solution = model.solve(params=_DISSOLUTION_PARAMS, log_level="off")
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
        solution=solution,
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


def test_a_start_that_runs_into_a_role_dependent_route_still_needs_an_own_role() -> (
    None
):
    """A role is demanded wherever a subject can still arrive at a two-target exit.

    A row keeps the role it started with across an ordinary regime transition,
    so a collective start that declares no role-dependent route of its own but
    runs into a regime that does is exactly the case the seed answers. Reading
    only the start regime's own exit would let such a cohort simulate with
    every row following one partner's dissolution path.
    """
    prelude = Regime(
        states={"wage": _WAGE_3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_zero_collective, "m": _u_zero_collective}
            )
        },
    )
    model = Model(
        regimes={"prelude": prelude, **_make_dissolution_regimes()},
        ages=AgeGrid(start=0, inclusive_stop=4, step="Y"),
        regime_id_class=ReachableRoleRoutingRegimeId,
        initial_nodes={0: "prelude"},
        edges={
            "prelude": {"married": 0},
            "married": {"married_ir": 1},
            "married_ir": {"married_terminal": 2},
            "single_f": {"single_f_terminal": 2},
            "single_m": {"single_m_terminal": 2},
        },
    )
    solution = model.solve(params=_DISSOLUTION_PARAMS, log_level="off")
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
            solution=solution,
            log_level="off",
            seed=0,
        )


@pytest.fixture(scope="module")
def current_exit_model_and_solution() -> tuple[Model, SolutionResult]:
    """Start in the actual current-exit actor."""
    model = Model(
        regimes=_make_dissolution_regimes(),
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=DissolutionRegimeId,
        initial_nodes={1: "married_ir"},
        edges=DISSOLUTION_EDGES,
    )
    solution = model.solve(
        params=_DISSOLUTION_PARAMS,
        retention=ResultRetention.VALUES,
        log_level="off",
    )
    return model, solution


@pytest.mark.parametrize("role", [None, -1, 7], ids=["missing", "no-role", "unknown"])
def test_current_exit_requires_an_actual_source_member_role(
    *, current_exit_model_and_solution: tuple[Model, SolutionResult], role: int | None
) -> None:
    """A role-dependent current exit refuses missing, no-role and unknown codes."""
    model, solution = current_exit_model_and_solution
    initial_conditions = {
        "wage": jnp.asarray([2.0]),
        "age": jnp.asarray([1.0]),
        "regime_id": jnp.asarray([model.regime_names_to_ids["married_ir"]]),
    }
    if role is not None:
        initial_conditions["own_member"] = jnp.asarray([role], dtype=jnp.int32)
    with pytest.raises(InvalidInitialConditionsError, match="own_member"):
        model.simulate(
            params=_DISSOLUTION_PARAMS,
            initial_conditions=initial_conditions,
            solution=solution,
            log_level="off",
            seed=0,
        )


@pytest.mark.parametrize(
    ("role", "destination", "payoff"),
    [("f", "single_f", 5.0), ("m", "single_m", 0.8)],
    ids=["wife", "husband"],
)
def test_current_exit_settles_the_named_original_source_member(
    *,
    current_exit_model_and_solution: tuple[Model, SolutionResult],
    role: str,
    destination: RegimeName,
    payoff: float,
) -> None:
    """Current-exit occupants receive their member's unpaid outside value."""
    model, solution = current_exit_model_and_solution
    frame = model.simulate(
        params=_DISSOLUTION_PARAMS,
        initial_conditions={
            "wage": jnp.asarray([2.0]),
            "age": jnp.asarray([1.0]),
            "regime_id": jnp.asarray([model.regime_names_to_ids["married_ir"]]),
            "own_member": jnp.asarray([model.member_names_to_ids[role]]),
        },
        solution=solution,
        log_level="off",
        seed=0,
    ).to_dataframe()
    acting = frame.loc[frame["period"] == 1]
    np.testing.assert_array_equal(acting["subject_id"], [0])
    np.testing.assert_array_equal(acting["regime_name"].to_numpy(), [destination])
    np.testing.assert_array_equal(acting["wage"], [2.0])
    # The payoff is an outside value less a dissolution cost, each rounded in the
    # working format, so it lies within one ULP of the real number.
    value = acting["value"].to_numpy()
    assert_agrees_to_ulp(
        got=value, expected=np.asarray([payoff], dtype=value.dtype), n_ulp=1
    )
