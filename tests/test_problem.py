"""A `Problem` is an action problem that several regimes may hold.

A regime that holds a problem declares none of the problem's slots and takes the
problem's own objects for them, adding only its exit and description. Everything
the regime derives afterwards — stakeholders, value constraints, the joint-normal
lowering, validation — sees what a regime declaring the same slots by hand gives.
"""

import dataclasses
from collections.abc import Callable
from types import MappingProxyType

import cloudpickle
import jax.numpy as jnp
import pytest

from _lcm.regime_law import bind_regime_law
from _lcm.solution.fingerprint import (
    _project_user_regime_declaration,
    _semantic_fingerprint,
)
from lcm import (
    CollectiveUtility,
    DiscreteGrid,
    Exit,
    ExitTarget,
    ExtremeValueTasteShocks,
    InitialNodes,
    JointNormalIIDProcess,
    LinearAggregator,
    LinearExpectation,
    LinSpacedGrid,
    Model,
    MutualConsent,
    Problem,
    ProjectedRegimeValue,
    Regime,
    Unilateral,
    ValueDependentConstraint,
    categorical,
    fixed_transition,
)
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.regime import RegimeReplacement
from lcm.solvers import DCEGM, GridSearch
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt

_WEALTH = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)
_WORK = DiscreteGrid(values=jnp.asarray([0, 1]))

_PROBLEM_SLOTS = (
    "states",
    "state_transitions",
    "joint_transitions",
    "actions",
    "functions",
    "constraints",
    "output_only_functions",
    "derived_categoricals",
    "solver",
    "taste_shocks",
    "shocks",
    "koopmans_aggregator",
    "certainty_equivalent",
)


@categorical(ordered=False)
class _RegimeId:
    alone: ScalarInt


def _utility(*, wealth: ContinuousState, work: DiscreteAction) -> FloatND:
    return wealth + work


def _next_wealth(*, wealth: ContinuousState, work: DiscreteAction) -> FloatND:
    return wealth + work


def _feasible(*, wealth: ContinuousState, work: DiscreteAction) -> BoolND:
    return work <= wealth


def _wife(wealth: ContinuousState) -> FloatND:
    return 3.0 * wealth


def _husband(wealth: ContinuousState) -> FloatND:
    return wealth


def _other_utility(*, wealth: ContinuousState, work: DiscreteAction) -> FloatND:
    return 2.0 * wealth - work


def _shock_sum(*, x: ContinuousState, y: ContinuousState) -> FloatND:
    return x + y


def _always(Q_f: FloatND) -> BoolND:
    return Q_f > -jnp.inf


def _problem() -> Problem:
    return Problem(
        states={"wealth": _WEALTH},
        state_transitions={"wealth": _next_wealth},
        actions={"work": _WORK},
        functions={"utility": _utility},
        constraints={"feasible": _feasible},
    )


def _collective_problem() -> Problem:
    return Problem(
        states={"wealth": _WEALTH},
        functions={"utility": CollectiveUtility(utilities={"f": _wife, "m": _husband})},
    )


def _dissolution(rule: Unilateral | MutualConsent) -> Exit:
    return Exit(
        rule=rule,
        targets={
            member: ExitTarget(
                regime=f"single_{member}",
                projection={"wealth": fixed_transition("wealth")},
            )
            for member in ("f", "m")
        },
    )


def test_problem_slots_are_the_thirteen_slots_of_an_action_problem() -> None:
    """The problem's fields are the regime slots that declare an action problem."""
    assert tuple(field.name for field in dataclasses.fields(Problem)) == (
        _PROBLEM_SLOTS
    )


@pytest.mark.parametrize("slot", _PROBLEM_SLOTS)
def test_regime_holding_a_problem_takes_the_problem_s_own_objects(slot: str) -> None:
    """Each slot of a holder is the very object the problem holds."""
    problem = _problem()
    regime = Regime(problem=problem)
    assert getattr(regime, slot) is getattr(problem, slot)


def test_regime_holding_a_problem_keeps_the_problem() -> None:
    problem = _problem()
    assert Regime(problem=problem).problem is problem


def test_regime_holding_a_problem_keeps_its_own_exit_and_description() -> None:
    exit_ = _dissolution(Unilateral())
    regime = Regime(
        problem=_collective_problem(), exit=exit_, description="a married couple"
    )
    assert (regime.exit, regime.description) == (exit_, "a married couple")


@pytest.mark.parametrize("slot", _PROBLEM_SLOTS)
def test_two_regimes_holding_one_problem_share_its_objects(slot: str) -> None:
    problem = _problem()
    first = Regime(problem=problem)
    second = Regime(problem=problem, description="the same problem, held again")
    assert getattr(first, slot) is getattr(second, slot)


_DECLARED_SLOTS: tuple[RegimeReplacement, ...] = (
    {"states": {"wealth": _WEALTH}},
    {"state_transitions": {"wealth": _next_wealth}},
    {"joint_transitions": {"elsewhere": MappingProxyType({})}},
    {"actions": {"work": _WORK}},
    {"functions": {"utility": _utility}},
    {"constraints": {"feasible": _feasible}},
    {"output_only_functions": ("helper",)},
    {"derived_categoricals": {"bracket": _WORK}},
    {"solver": GridSearch(lottery_contraction=True)},
    {"taste_shocks": ExtremeValueTasteShocks()},
    {"shocks": ("match",)},
    {"koopmans_aggregator": LinearAggregator()},
    {"certainty_equivalent": LinearExpectation()},
)


def _hand_twin(problem: Problem) -> Regime:
    """A regime declaring the problem's slots by hand."""
    return Regime(
        states=problem.states,
        state_transitions=problem.state_transitions,
        joint_transitions=problem.joint_transitions,
        actions=problem.actions,
        functions=problem.functions,
        constraints=problem.constraints,
        output_only_functions=problem.output_only_functions,
        derived_categoricals=problem.derived_categoricals,
        solver=problem.solver,
        taste_shocks=problem.taste_shocks,
        shocks=problem.shocks,
        koopmans_aggregator=problem.koopmans_aggregator,
        certainty_equivalent=problem.certainty_equivalent,
    )


def test_every_slot_has_a_declaration_that_differs_from_its_default() -> None:
    """The refusal below is exercised once per problem slot."""
    assert tuple(next(iter(slot)) for slot in _DECLARED_SLOTS) == _PROBLEM_SLOTS


@pytest.mark.parametrize("borrows_solver", [False, True])
@pytest.mark.parametrize(
    "declared", _DECLARED_SLOTS, ids=[next(iter(slot)) for slot in _DECLARED_SLOTS]
)
def test_a_problem_slot_declared_beside_the_problem_is_refused(
    *, declared: RegimeReplacement, borrows_solver: bool
) -> None:
    """Handing the regime its problem's own solver changes nothing: the slot it
    declares is still not the problem's."""
    problem = _problem()
    holder: RegimeReplacement = {"problem": problem}
    if borrows_solver:
        holder["solver"] = problem.solver
    holder.update(declared)
    with pytest.raises(
        RegimeInitializationError,
        match=rf"holds a problem declares none of the problem's slots, but this "
        rf"one also declares \['{next(iter(declared))}'\]",
    ):
        Regime(**holder)


def test_replacing_the_problem_of_a_holder_is_refused() -> None:
    """The holder's slots are its first problem's, which it would now declare."""
    regime = Regime(problem=_problem())
    with pytest.raises(
        RegimeInitializationError, match="declares none of the problem's slots"
    ):
        regime.replace(problem=_problem())


def test_a_holder_handed_its_problem_s_solver_still_takes_the_other_slots() -> None:
    """The solver's identity alone does not mark a regime as already lowered."""
    problem = _problem()
    assert Regime(problem=problem, solver=problem.solver).functions is problem.functions


_REPLACE_APIS = {
    "dataclasses.replace": dataclasses.replace,
    "Regime.replace": Regime.replace,
}
_DECLARATIONS = {
    name: value for slot in _DECLARED_SLOTS for name, value in slot.items()
}


def _variant(*, first: Problem, slot: str) -> Problem:
    """The problem with one slot declared anew; every other slot, the solver
    included, is the very object the problem holds."""
    second = dataclasses.replace(first, **{slot: _DECLARATIONS[slot]})
    assert second.solver is first.solver
    return second


@pytest.mark.parametrize("replace", _REPLACE_APIS.values(), ids=_REPLACE_APIS)
@pytest.mark.parametrize(
    "slot",
    # The joint transition declared here names no node, which a holder
    # refuses on its own.
    [name for name in _PROBLEM_SLOTS if name not in {"solver", "joint_transitions"}],
)
def test_a_holder_given_a_problem_that_declares_a_slot_anew_is_refused(
    *, slot: str, replace: Callable[..., Regime]
) -> None:
    """The holder carries the first problem's declaration of the slot, which is
    not the second problem's."""
    first = _problem()
    holder = Regime(problem=first)
    with pytest.raises(
        RegimeInitializationError, match=rf"but this one also declares \['{slot}'\]"
    ):
        replace(holder, problem=_variant(first=first, slot=slot))


@pytest.mark.parametrize(
    "declared", _DECLARED_SLOTS, ids=[next(iter(slot)) for slot in _DECLARED_SLOTS]
)
def test_a_holder_declares_a_problem_slot_only_after_letting_go_of_its_problem(
    declared: RegimeReplacement,
) -> None:
    holder = Regime(problem=_problem())
    with pytest.raises(
        RegimeInitializationError,
        match=rf"but this one also declares \['{next(iter(declared))}'\]",
    ):
        holder.replace(**declared)


def test_a_holder_that_lets_go_of_its_problem_declares_its_own_slot() -> None:
    problem = _problem()
    own = Regime(problem=problem).replace(
        problem=None, functions={"utility": _other_utility}
    )
    assert (own.problem, own.functions["utility"], own.states) == (
        None,
        _other_utility,
        problem.states,
    )


@pytest.mark.parametrize(
    "change",
    [{}, {"exit": _dissolution(MutualConsent())}, {"description": "renamed"}],
    ids=["nothing", "exit", "description"],
)
@pytest.mark.parametrize("replace", _REPLACE_APIS.values(), ids=_REPLACE_APIS)
def test_a_holder_whose_own_fields_change_keeps_its_problem_and_its_objects(
    *, change: RegimeReplacement, replace: Callable[..., Regime]
) -> None:
    problem = _collective_problem()
    holder = Regime(problem=problem, exit=_dissolution(Unilateral()))
    replaced = replace(holder, **change)
    for name in [*_PROBLEM_SLOTS, "problem"]:
        assert getattr(replaced, name) is getattr(holder, name)


def test_a_detached_holder_keeps_the_objects_and_may_hold_the_problem_again() -> None:
    problem = _problem()
    detached = Regime(problem=problem).replace(problem=None)
    reattached = detached.replace(problem=problem)
    assert (detached.problem, reattached.problem) == (None, problem)
    for name in _PROBLEM_SLOTS:
        assert getattr(detached, name) is getattr(problem, name)
        assert getattr(reattached, name) is getattr(problem, name)


@pytest.mark.parametrize(
    "default",
    [{"solver": GridSearch()}, {"functions": {}}, {"states": {}}],
    ids=["solver", "functions", "states"],
)
def test_a_default_or_reset_slot_passed_beside_a_problem_is_refused(
    default: RegimeReplacement,
) -> None:
    """An explicit value is a declaration even when it equals the `Regime` default."""
    holder: RegimeReplacement = {"problem": _problem()}
    holder.update(default)
    with pytest.raises(
        RegimeInitializationError,
        match=rf"but this one also declares \['{next(iter(default))}'\]",
    ):
        Regime(**holder)


@pytest.mark.parametrize("replace", _REPLACE_APIS.values(), ids=_REPLACE_APIS)
def test_an_unpickled_holder_keeps_its_problem_and_refuses_another(
    replace: Callable[..., Regime],
) -> None:
    restored = cloudpickle.loads(cloudpickle.dumps(Regime(problem=_problem())))
    same = replace(restored, description="restored")
    assert same.problem is restored.problem
    for name in _PROBLEM_SLOTS:
        assert getattr(same, name) is getattr(restored.problem, name)
    second = dataclasses.replace(
        restored.problem, functions={"utility": _other_utility}
    )
    with pytest.raises(
        RegimeInitializationError, match=r"but this one also declares \['functions'\]"
    ):
        replace(restored, problem=second)


@pytest.mark.parametrize("slot", [*_PROBLEM_SLOTS, "problem"])
def test_replacing_the_exit_of_a_holder_keeps_the_problem_and_its_slots(
    slot: str,
) -> None:
    problem = _collective_problem()
    regime = Regime(problem=problem, exit=_dissolution(Unilateral()))
    replaced = regime.replace(exit=_dissolution(MutualConsent()))
    assert getattr(replaced, slot) is getattr(regime, slot)


def test_problem_refuses_a_value_dependent_constraint() -> None:
    with pytest.raises(
        RegimeInitializationError,
        match=r"Problem constraints \['participation'\] are value-dependent",
    ):
        Problem(
            functions={
                "utility": CollectiveUtility(utilities={"f": _wife, "m": _husband})
            },
            constraints={
                "participation": ValueDependentConstraint(
                    predicate=_always,
                    references={
                        "V_single": ProjectedRegimeValue(regime="single", projection={})
                    },
                )
            },
        )


def test_a_consumption_savings_regime_cannot_hold_a_problem() -> None:
    with pytest.raises(
        RegimeInitializationError,
        match="A ConsumptionSavingsRegime cannot hold a Problem",
    ):
        ConsumptionSavingsRegime(
            problem=_problem(),
            liquid=LiquidMargin(
                state="assets",
                action="consumption",
                resources="resources",
                post_decision_state="savings",
            ),
        )


def test_a_consumption_savings_regime_with_its_problem_s_objects_is_refused() -> None:
    solver = GridSearch()
    problem = dataclasses.replace(_problem(), solver=solver)
    with pytest.raises(
        RegimeInitializationError,
        match="A ConsumptionSavingsRegime cannot hold a Problem",
    ):
        ConsumptionSavingsRegime(
            problem=problem,
            solver=solver,
            actions=problem.actions,
            functions=problem.functions,
            liquid=LiquidMargin(
                state="assets",
                action="consumption",
                resources="resources",
                post_decision_state="savings",
            ),
        )


def test_a_plain_regime_refuses_the_endogenous_grid_solver_of_its_problem() -> None:
    """Lowering hands the solver to the regime's own margin check."""
    problem = Problem(
        functions={"utility": _utility}, solver=DCEGM(savings_grid=_WEALTH)
    )
    with pytest.raises(
        RegimeInitializationError,
        match="EGM-family solvers require regime-owned margin declarations",
    ):
        Regime(problem=problem)


def test_a_problem_is_not_a_regime_of_a_model() -> None:
    with pytest.raises(ModelInitializationError):
        Model(
            regimes={"alone": _problem()},  # ty: ignore[invalid-argument-type]
            n_periods=1,
            edges={},
            regime_id_class=_RegimeId,
            initial_nodes=InitialNodes(by_period={0: "alone"}),
        )


def test_a_collective_problem_gives_its_holders_their_stakeholders() -> None:
    """An exit naming the household's roles validates on a holder."""
    regime = Regime(problem=_collective_problem(), exit=_dissolution(Unilateral()))
    assert regime.stakeholders == ("f", "m")


@pytest.mark.parametrize("slot", ["states", "functions"])
def test_a_joint_normal_problem_lowers_like_its_hand_declared_twin(slot: str) -> None:
    joint = JointNormalIIDProcess(
        states=("x", "y"),
        n_points=3,
        gauss_hermite=True,
        mean=(0.0, 0.0),
        cov=((1.0, 0.0), (0.0, 1.0)),
    )
    held = Regime(
        problem=Problem(states={"joint": joint}, functions={"utility": _shock_sum})
    ).lower_joint_normal_processes()
    twin = Regime(
        states={"joint": joint}, functions={"utility": _shock_sum}
    ).lower_joint_normal_processes()
    assert tuple(getattr(held, slot)) == tuple(getattr(twin, slot))


def test_the_joint_normal_lowering_of_a_holder_holds_no_problem() -> None:
    """The lowered regime carries generated states and functions, not the
    problem's objects."""
    joint = JointNormalIIDProcess(
        states=("x", "y"),
        n_points=3,
        gauss_hermite=True,
        mean=(0.0, 0.0),
        cov=((1.0, 0.0), (0.0, 1.0)),
    )
    held = Regime(
        problem=Problem(states={"joint": joint}, functions={"utility": _shock_sum})
    )
    assert held.lower_joint_normal_processes().problem is None


def test_a_dict_passed_to_a_problem_does_not_reach_it_afterwards() -> None:
    functions = {"utility": _utility}
    problem = Problem(functions=functions)
    functions["helper"] = _utility
    assert tuple(problem.functions) == ("utility",)


@pytest.mark.parametrize("holds_a_problem", [False, True])
def test_regime_projection_has_no_problem_field(*, holds_a_problem: bool) -> None:
    regime = Regime(problem=_problem()) if holds_a_problem else _hand_twin(_problem())
    projection = _project_user_regime_declaration(regime, law=bind_regime_law(None))
    assert "problem" not in projection["fields"]


def test_holder_projects_like_its_hand_declared_twin() -> None:
    problem = _problem()
    held = Regime(problem=problem, description="held")
    twin = _hand_twin(problem)
    law = bind_regime_law(None)
    assert _semantic_fingerprint(
        _project_user_regime_declaration(held, law=law)
    ) == _semantic_fingerprint(_project_user_regime_declaration(twin, law=law))
