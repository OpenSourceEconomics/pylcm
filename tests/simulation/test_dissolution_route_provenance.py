"""Only the gate a household actually passed open hands it a dissolution route.

A source `S` has two gated edges: into `a` with fallback `b`, and into `b` with
fallback `c`. A household whose draw selects `a` and whose `a` gate closes lands
in `b` through the closed branch of `a`'s gate. If `b` has no feasible action at
that state, the household has already used its only route: the gate into `b`
was never evaluated for it, so it must not inherit that gate's fallback `c`.
Only a household that entered a regime through the open branch of a gate into
that regime may take that gate's route, and only once.

The reference below enumerates both actions of the two-action regimes directly.
"""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ByAge,
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
from lcm.transition import StochasticTransition
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt


@categorical(ordered=False)
class Work:
    off: ScalarInt
    on: ScalarInt


@categorical(ordered=False)
class RouteId:
    source: ScalarInt
    a: ScalarInt
    b: ScalarInt
    c: ScalarInt


def _source_utility(*, wage: ContinuousState) -> FloatND:
    return wage


def _utility(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return wage + work


def _identity(*, wage: ContinuousState) -> ContinuousState:
    return wage


def _feasible(*, wage: ContinuousState) -> BoolND:
    return (wage - 2.0) ** 2 > 0.5


def _gate_open(*, wage: ContinuousState) -> BoolND:
    return wage >= 0.0


def _gate_closed(*, wage: ContinuousState) -> BoolND:
    return wage < 0.0


def _to_a(*, wage: ContinuousState) -> FloatND:
    return jnp.where(wage <= 2.0, 1.0, 0.0)


def _to_b(*, wage: ContinuousState) -> FloatND:
    return jnp.where(wage > 2.0, 1.0, 0.0)


def reference_action(*, wage: float, constrained: bool) -> int | None:
    """The optimal `work` among both actions, or `None` when neither is feasible."""
    feasible = not constrained or (wage - 2.0) ** 2 > 0.5
    candidates = [(wage + work, work) for work in (0, 1) if feasible]
    return max(candidates)[1] if candidates else None


def _route(*, fallback: str, predicate: Callable[..., BoolND] = _gate_open) -> Gate:
    return Gate(
        predicate=predicate,
        routes={
            "only": StakeholderRoute(
                fallback=ProjectedRegimeValue(
                    regime=fallback, projection={"wage": _identity}
                )
            )
        },
    )


def _regime(
    *, grid: IrregSpacedGrid, source: bool = False, constrained: bool = False
) -> Regime:
    if source:
        return Regime(
            states={"wage": grid},
            state_transitions={"wage": _identity},
            functions={"utility": _source_utility},
        )
    return Regime(
        states={"wage": grid},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _utility},
        constraints={"feasible": _feasible} if constrained else {},
    )


def build_chain_model(
    *,
    order: tuple[str, ...],
    close_a: bool,
    constrain_b: bool,
    enable_jit: bool = True,
    gate_order: tuple[str, ...] = ("a", "b"),
    b_fallback: str = "c",
) -> Model:
    """`source` gates into `a` (fallback `b`) and into `b` (fallback `c`)."""
    grid = IrregSpacedGrid(points=np.array([1.0, 3.0]))
    declarations = {
        "source": _regime(grid=grid, source=True),
        "a": _regime(grid=grid),
        "b": _regime(grid=grid, constrained=constrain_b),
        "c": _regime(grid=grid),
    }
    gates = {
        "a": _route(fallback="b", predicate=_gate_closed if close_a else _gate_open),
        "b": _route(fallback=b_fallback),
    }
    return Model(
        regimes={name: declarations[name] for name in order},
        edges={
            "source": Transition(
                law=ByAge(
                    cases={
                        0: {
                            "a": StochasticTransition(func=_to_a),
                            "b": StochasticTransition(func=_to_b),
                        }
                    }
                ),
                gates={name: gates[name] for name in gate_order},
            )
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        initial_nodes={0: "source"},
        regime_id_class=RouteId,
        enable_jit=enable_jit,
    )


def simulate_chain(*, model: Model, wages: np.ndarray, runtime_checks: bool):
    params = {
        name: {"koopmans_aggregator": {"discount_factor": 1.0}}
        if name == "source"
        else {}
        for name in model.regime_names_to_ids
    }
    solution = model.solve(params=params, log_level="off")
    return model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wage": jnp.asarray(wages),
            "age": jnp.zeros(len(wages)),
            "regime_id": jnp.full(
                len(wages), model.regime_names_to_ids["source"], jnp.int32
            ),
        },
        seed=0,
        log_level="off",
        runtime_checks=runtime_checks,
    )


_EMPTY_WAGES = np.linspace(1.5, 2.0, 9)


def test_reference_finds_no_action_in_b_across_the_empty_interval():
    assert all(
        reference_action(wage=float(wage), constrained=True) is None
        for wage in _EMPTY_WAGES
    )


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("runtime_checks", [False, True])
@pytest.mark.parametrize("wage", [1.5, 1.75, 2.0])
@pytest.mark.parametrize("gate_order", [("a", "b"), ("b", "a")])
def test_a_closed_branch_does_not_borrow_another_gates_route(
    *,
    enable_jit: bool,
    runtime_checks: bool,
    wage: float,
    gate_order: tuple[str, ...],
):
    model = build_chain_model(
        order=("source", "a", "b", "c"),
        close_a=True,
        constrain_b=True,
        enable_jit=enable_jit,
        gate_order=gate_order,
    )
    with pytest.raises(NoFeasibleActionError, match=r"'b', period 1"):
        simulate_chain(
            model=model, wages=np.array([wage]), runtime_checks=runtime_checks
        )


@pytest.mark.parametrize("gate_order", [("a", "b"), ("b", "a")])
@pytest.mark.parametrize("reverse_rows", [False, True])
def test_a_mixed_cohort_does_not_borrow_another_gates_route(
    *, gate_order: tuple[str, ...], reverse_rows: bool
):
    wages = np.concatenate(([1.0, 3.0], _EMPTY_WAGES))
    if reverse_rows:
        wages = wages[::-1].copy()
    model = build_chain_model(
        order=("source", "a", "b", "c"),
        close_a=True,
        constrain_b=True,
        gate_order=gate_order,
    )
    with pytest.raises(NoFeasibleActionError, match=r"'b', period 1"):
        simulate_chain(model=model, wages=wages, runtime_checks=False)


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_an_open_branch_into_b_still_takes_its_own_route(*, enable_jit: bool):
    """The gate into `b` opens; the empty landing takes `b`'s fallback `c`."""
    model = build_chain_model(
        order=("source", "a", "b", "c"),
        close_a=False,
        constrain_b=True,
        enable_jit=enable_jit,
    )
    wages = np.array([2.25, 3.0])
    expected = [
        ("c" if reference_action(wage=w, constrained=True) is None else "b", w, 1)
        for w in wages
    ]
    result = simulate_chain(model=model, wages=wages, runtime_checks=True)
    panel = result.to_dataframe(use_categorical_codes=True)
    rows = panel.loc[panel["period"] == 1].sort_values("subject_id")
    assert (
        list(zip(rows["regime_name"], rows["wage"], rows["work"], strict=True))
        == expected
    )
