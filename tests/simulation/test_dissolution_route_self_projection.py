"""Every leg of a gate projects from the household's original realised state.

A collective source gates its target `a` with two legs: `f` falls back to `a`
itself at a projected coordinate, `m` falls back to the singleton `b` at its own
coordinate. Rows with no feasible action in `a` take their own leg. The `m` leg
must read the state the household arrived with, not the state the `f` leg just
wrote back into `a`, so the outcome does not depend on the order the legs or the
regimes are declared in.

The reference is a scalar enumeration over the discrete coordinates and the two
work choices. It imports no project, JAX or NumPy code.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    CollectiveUtility,
    DiscreteGrid,
    Gate,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    categorical,
)
from lcm.solver_api import DISSOLUTION_FLAG
from lcm.transition import StochasticTransition
from lcm.typing import BoolND, DiscreteAction, DiscreteState, FloatND, ScalarInt

_COORDINATES = (0, 1, 2, 0, 1, 2)
_ROLES = ("f", "f", "f", "m", "m", "m")


@categorical(ordered=False)
class Coordinate:
    low: ScalarInt
    middle: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class Work:
    rest: ScalarInt
    work: ScalarInt


@categorical(ordered=False)
class SelfProjectionId:
    source: ScalarInt
    a: ScalarInt
    b: ScalarInt


def _source_utility(*, x: DiscreteState) -> FloatND:
    return 1.0 * x


def _utility(*, x: DiscreteState, work: DiscreteAction) -> FloatND:
    return 1.0 * x + work


def _identity(*, x: DiscreteState) -> DiscreteState:
    return x


def _to_high(*, x: DiscreteState) -> DiscreteState:
    return 2 + 0 * x


def _to_low(*, x: DiscreteState) -> DiscreteState:
    return 0 * x


def _feasible(*, x: DiscreteState) -> BoolND:
    return x != 1


def _no_dissolution(*, D_target: BoolND) -> BoolND:
    return ~D_target


def _certain(*, x: DiscreteState) -> FloatND:
    return 1.0 + 0.0 * x


_PARAMS = {
    "source": {"koopmans_aggregator": {"discount_factor": 1.0}},
    "a": {},
    "b": {},
}


def _build_model(
    *,
    stakeholder_order: tuple[str, ...],
    regime_order: tuple[str, ...],
    self_destination: int,
    enable_jit: bool,
) -> Model:
    grid = DiscreteGrid(category_class=Coordinate)
    work = DiscreteGrid(category_class=Work)
    regimes = {
        "source": Regime(
            states={"x": grid},
            state_transitions={"x": _identity},
            functions={
                "utility": CollectiveUtility(
                    utilities=dict.fromkeys(stakeholder_order, _source_utility)
                )
            },
        ),
        "a": Regime(
            states={"x": grid},
            actions={"work": work},
            functions={
                "utility": CollectiveUtility(utilities={"f": _utility, "m": _utility})
            },
            constraints={"feasible": _feasible},
        ),
        "b": Regime(
            states={"x": grid}, actions={"work": work}, functions={"utility": _utility}
        ),
    }
    return Model(
        regimes={name: regimes[name] for name in regime_order},
        edges={
            "source": Transition(
                law=ByAge(cases={0: {"a": StochasticTransition(func=_certain)}}),
                gates={
                    "a": Gate(
                        predicate=_no_dissolution,
                        routes={
                            "f": StakeholderRoute(
                                target_stakeholder="f",
                                fallback=ProjectedRegimeValue(
                                    regime="a",
                                    stakeholder="f",
                                    projection={
                                        "x": _to_high
                                        if self_destination == 2
                                        else _to_low
                                    },
                                ),
                            ),
                            "m": StakeholderRoute(
                                target_stakeholder="m",
                                fallback=ProjectedRegimeValue(
                                    regime="b", projection={"x": _identity}
                                ),
                            ),
                        },
                    )
                },
            )
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        initial_nodes={0: "source"},
        regime_id_class=SelfProjectionId,
        enable_jit=enable_jit,
    )


def _reference_panel(
    *, coordinates: tuple[int, ...], roles: tuple[str, ...], self_destination: int
) -> list[tuple[int, str, int, int]]:
    """Enumerate the work choice once per household, projecting the original state."""
    rows = []
    for subject_id, (original, role) in enumerate(zip(coordinates, roles, strict=True)):
        if original != 1:
            regime, coordinate = "a", original
        elif role == "f":
            regime, coordinate = "a", self_destination
        else:
            regime, coordinate = "b", original
        actions = [action for action in (0, 1) if regime != "a" or coordinate != 1]
        work = max(actions, key=lambda action: coordinate + action)
        rows.append((subject_id, regime, coordinate, work))
    return rows


def _reference_records(
    *, coordinates: tuple[int, ...], roles: tuple[str, ...]
) -> list[tuple[int, int, str, dict[str, str], str, str]]:
    """Each record names the state the household left, before any projection."""
    return [
        (subject_id, 1, "a", {"x": "middle"}, "source", "a" if role == "f" else "b")
        for subject_id, (coordinate, role) in enumerate(
            zip(coordinates, roles, strict=True)
        )
        if coordinate == 1
    ]


def _simulate(
    *,
    model: Model,
    coordinates: tuple[int, ...],
    roles: tuple[str, ...],
    runtime_checks: bool,
):
    solution = model.solve(params=_PARAMS, log_level="off")
    role_ids = model.stakeholder_names_to_ids
    return model.simulate(
        params=_PARAMS,
        solution=solution,
        initial_conditions={
            "x": jnp.asarray(coordinates, dtype=jnp.int32),
            "age": jnp.zeros(len(coordinates)),
            "regime_id": jnp.full(
                len(coordinates), model.regime_names_to_ids["source"], jnp.int32
            ),
            "own_stakeholder": jnp.asarray(
                [role_ids[role] for role in roles], dtype=jnp.int32
            ),
        },
        seed=0,
        log_level="off",
        runtime_checks=runtime_checks,
    )


def _period_one_panel(*, result) -> list[tuple[int, str, int, int]]:
    panel = result.to_dataframe(use_categorical_codes=True)
    rows = panel.loc[panel["period"] == 1].sort_values("subject_id")
    return list(
        zip(
            rows["subject_id"],
            rows["regime_name"],
            rows["x"],
            rows["work"],
            strict=True,
        )
    )


def test_the_gate_into_a_dissolves_only_at_the_middle_coordinate():
    model = _build_model(
        stakeholder_order=("f", "m"),
        regime_order=("source", "a", "b"),
        self_destination=2,
        enable_jit=True,
    )
    solution = model.solve(params=_PARAMS, log_level="off")
    flags = np.asarray(solution.replay_artifacts.project(DISSOLUTION_FLAG)[1]["a"])
    assert flags.tolist() == [False, True, False]


@pytest.mark.parametrize("stakeholder_order", [("f", "m"), ("m", "f")])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("runtime_checks", [False, True], ids=["unchecked", "checked"])
def test_each_leg_projects_from_the_original_target_state(
    *, stakeholder_order: tuple[str, ...], enable_jit: bool, runtime_checks: bool
):
    model = _build_model(
        stakeholder_order=stakeholder_order,
        regime_order=("source", "a", "b"),
        self_destination=2,
        enable_jit=enable_jit,
    )
    result = _simulate(
        model=model,
        coordinates=_COORDINATES,
        roles=_ROLES,
        runtime_checks=runtime_checks,
    )
    assert _period_one_panel(result=result) == _reference_panel(
        coordinates=_COORDINATES, roles=_ROLES, self_destination=2
    )


@pytest.mark.parametrize(
    ("stakeholder_order", "regime_order", "self_destination", "reverse_rows"),
    [
        (("f", "m"), ("source", "a", "b"), 2, False),
        (("m", "f"), ("source", "a", "b"), 2, False),
        (("f", "m"), ("b", "a", "source"), 0, True),
        (("m", "f"), ("a", "b", "source"), 0, True),
    ],
)
def test_declaration_and_row_orders_leave_the_panel_unchanged(
    *,
    stakeholder_order: tuple[str, ...],
    regime_order: tuple[str, ...],
    self_destination: int,
    reverse_rows: bool,
):
    coordinates = _COORDINATES[::-1] if reverse_rows else _COORDINATES
    roles = _ROLES[::-1] if reverse_rows else _ROLES
    model = _build_model(
        stakeholder_order=stakeholder_order,
        regime_order=regime_order,
        self_destination=self_destination,
        enable_jit=True,
    )
    result = _simulate(
        model=model, coordinates=coordinates, roles=roles, runtime_checks=True
    )
    assert _period_one_panel(result=result) == _reference_panel(
        coordinates=coordinates, roles=roles, self_destination=self_destination
    )


def test_each_record_names_the_state_the_household_left():
    model = _build_model(
        stakeholder_order=("f", "m"),
        regime_order=("source", "a", "b"),
        self_destination=2,
        enable_jit=True,
    )
    result = _simulate(
        model=model, coordinates=_COORDINATES, roles=_ROLES, runtime_checks=True
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
    ] == _reference_records(coordinates=_COORDINATES, roles=_ROLES)
