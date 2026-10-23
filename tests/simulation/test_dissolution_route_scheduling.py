"""A household rerouted in a period still acts exactly once in that period.

A household without a feasible action in a gated target takes its gate's
dissolution route and acts in the fallback regime in the same period. Two
schedules have to hold for that:

- A fallback runs after every target it serves, whatever order the regimes are
  declared in, so any acyclic arrangement of targets and fallbacks simulates.
- A gate may name its own target as the fallback, at a projected state. The
  household then decides again in the same regime at the projected state, and
  its period's row records that decision.
"""

from itertools import permutations

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    CollectiveUtility,
    DiscreteGrid,
    Gate,
    IrregSpacedGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    categorical,
)
from lcm.exceptions import NoFeasibleActionError
from lcm.solver_api import DISSOLUTION_FLAG
from lcm.transition import StochasticTransition
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt
from tests.simulation.test_dissolution_route_provenance import (
    Work,
    build_chain_model,
    reference_action,
    simulate_chain,
)


@pytest.mark.parametrize("order", list(permutations(("source", "a", "b", "c"))))
def test_every_declaration_order_simulates_the_same_panel(*, order: tuple[str, ...]):
    """`a` falls back to `b` and `b` to `c`; every landing is feasible."""
    model = build_chain_model(order=order, close_a=False, constrain_b=False)
    wages = np.array([3.0, 1.0, 2.0])
    expected = [
        (
            "b" if wage > 2.0 else "a",
            wage,
            reference_action(wage=wage, constrained=False),
        )
        for wage in wages
    ]
    result = simulate_chain(model=model, wages=wages, runtime_checks=True)
    panel = result.to_dataframe(use_categorical_codes=True)
    rows = panel.loc[panel["period"] == 1].sort_values("subject_id")
    assert (
        list(zip(rows["regime_name"], rows["wage"], rows["work"], strict=True))
        == expected
    )


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    "order", [("c", "b", "source", "a"), ("source", "b", "a", "c")]
)
def test_a_fallback_declared_first_still_receives_its_rerouted_rows(
    *, order: tuple[str, ...], enable_jit: bool
):
    """`b` is declared before `a`; an empty landing in `b` acts in `c`."""
    model = build_chain_model(
        order=order, close_a=False, constrain_b=True, enable_jit=enable_jit
    )
    wages = np.array([1.0, 2.25, 3.0])
    result = simulate_chain(model=model, wages=wages, runtime_checks=True)
    panel = result.to_dataframe(use_categorical_codes=True)
    rows = panel.loc[panel["period"] == 1].sort_values("subject_id")
    assert list(zip(rows["regime_name"], rows["wage"], strict=True)) == [
        ("a", 1.0),
        ("c", 2.25),
        ("b", 3.0),
    ]


def test_targets_and_fallbacks_in_a_cycle_simulate_when_no_household_is_rerouted():
    """`a` falls back to `b` and `b` to `a`; every landing is feasible."""
    model = build_chain_model(
        order=("source", "a", "b", "c"),
        close_a=False,
        constrain_b=False,
        b_fallback="a",
    )
    result = simulate_chain(
        model=model, wages=np.array([1.0, 3.0]), runtime_checks=False
    )
    panel = result.to_dataframe(use_categorical_codes=True)
    rows = panel.loc[panel["period"] == 1].sort_values("subject_id")
    assert list(zip(rows["regime_name"], rows["wage"], strict=True)) == [
        ("a", 1.0),
        ("b", 3.0),
    ]


def test_a_reroute_into_a_regime_that_already_ran_this_period_raises():
    """`a` runs before `b` in the cycle; an empty landing in `b` cannot act in `a`."""
    model = build_chain_model(
        order=("source", "a", "b", "c"),
        close_a=False,
        constrain_b=True,
        b_fallback="a",
    )
    with pytest.raises(
        NoFeasibleActionError, match=r"'b', period 1.*'a', which already ran"
    ):
        simulate_chain(model=model, wages=np.array([2.25]), runtime_checks=False)


@categorical(ordered=False)
class SelfRouteId:
    source: ScalarInt
    a: ScalarInt


def _source_utility(*, wage: ContinuousState) -> FloatND:
    return wage


def _utility(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return wage + work


def _identity(*, wage: ContinuousState) -> ContinuousState:
    return wage


def _project_to_three(*, wage: ContinuousState) -> ContinuousState:
    return 3.0 + 0.0 * wage


def _feasible(*, wage: ContinuousState) -> BoolND:
    return (wage - 2.0) ** 2 > 0.5


def _feasible_below_the_middle(*, wage: ContinuousState) -> BoolND:
    return wage < 1.5


def _no_dissolution(*, D_target: BoolND) -> BoolND:
    return ~D_target


def _certain(*, wage: ContinuousState) -> FloatND:
    return 1.0 + 0.0 * wage


def _self_route_model(*, enable_jit: bool, feasible=_feasible) -> Model:
    """The gate into `a` falls back to `a` itself, projecting the wage to 3."""
    grid = IrregSpacedGrid(points=np.array([1.0, 2.0, 3.0]))
    return Model(
        regimes={
            "source": Regime(
                states={"wage": grid},
                state_transitions={"wage": _identity},
                functions={"utility": _source_utility},
            ),
            "a": Regime(
                states={"wage": grid},
                actions={"work": DiscreteGrid(category_class=Work)},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _utility, "m": _utility}
                    )
                },
                constraints={"feasible": feasible},
            ),
        },
        edges={
            "source": Transition(
                law=ByAge(cases={0: {"a": StochasticTransition(func=_certain)}}),
                gates={
                    "a": Gate(
                        predicate=_no_dissolution,
                        routes={
                            "only": StakeholderRoute(
                                target_stakeholder="f",
                                fallback=ProjectedRegimeValue(
                                    regime="a",
                                    stakeholder="f",
                                    projection={"wage": _project_to_three},
                                ),
                            )
                        },
                    )
                },
            )
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        initial_nodes={0: "source"},
        regime_id_class=SelfRouteId,
        enable_jit=enable_jit,
    )


_SELF_PARAMS = {"source": {"koopmans_aggregator": {"discount_factor": 1.0}}, "a": {}}


def _simulate_self_route(*, model: Model, runtime_checks: bool):
    solution = model.solve(params=_SELF_PARAMS, log_level="off")
    result = model.simulate(
        params=_SELF_PARAMS,
        solution=solution,
        initial_conditions={
            "wage": jnp.asarray([1.0, 2.0, 3.0]),
            "age": jnp.zeros(3),
            "regime_id": jnp.full(3, model.regime_names_to_ids["source"], jnp.int32),
        },
        seed=0,
        log_level="off",
        runtime_checks=runtime_checks,
    )
    return solution, result


def test_the_self_route_gate_dissolves_only_at_the_middle_node():
    model = _self_route_model(enable_jit=True)
    solution = model.solve(params=_SELF_PARAMS, log_level="off")
    flags = np.asarray(solution.replay_artifacts.project(DISSOLUTION_FLAG)[1]["a"])
    assert flags.tolist() == [False, True, False]


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("runtime_checks", [False, True])
def test_a_self_route_decides_again_at_its_projected_state(
    *, enable_jit: bool, runtime_checks: bool
):
    """Wage 2 is empty in `a`; projected to 3 it works, as wages 1 and 3 do."""
    _, result = _simulate_self_route(
        model=_self_route_model(enable_jit=enable_jit), runtime_checks=runtime_checks
    )
    expected = [
        ("a", wage, reference_action(wage=wage, constrained=True))
        for wage in (1.0, 3.0, 3.0)
    ]
    panel = result.to_dataframe(use_categorical_codes=True)
    rows = panel.loc[panel["period"] == 1].sort_values("subject_id")
    assert rows["subject_id"].tolist() == [0, 1, 2]
    assert (
        list(zip(rows["regime_name"], rows["wage"], rows["work"], strict=True))
        == expected
    )


def test_a_self_route_is_recorded_once_at_the_state_it_left():
    _, result = _simulate_self_route(
        model=_self_route_model(enable_jit=True), runtime_checks=True
    )
    assert [
        (
            record.subject_id,
            record.period,
            record.regime,
            dict(record.state),
            record.source,
            record.fallback,
        )
        for record in result.dissolution_fallbacks
    ] == [(1, 1, "a", {"wage": 2.0}, "source", "a")]


def test_a_self_route_that_is_empty_again_raises():
    """The route is taken once: an empty projected state has nowhere to go."""
    assert reference_action(wage=3.0, constrained=False) is not None
    model = _self_route_model(enable_jit=True, feasible=_feasible_below_the_middle)
    with pytest.raises(NoFeasibleActionError, match=r"'a', period 1"):
        _simulate_self_route(model=model, runtime_checks=False)
