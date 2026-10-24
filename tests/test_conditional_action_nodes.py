"""`ConditionalAction` keeps exactly its inactive node, on grids with adjacent nodes.

Grids can place two representable floats next to each other: a piecewise grid puts the
node after a breakpoint one representable step from the breakpoint. On such grids the
synthesized pin must still admit exactly the inactive node `k` and no neighbour, so the
feasible set is `active(d) or j == k` cell by cell. Expected cells are enumerated from
node indices, independently of the pin's expression.

Every model has one decision: utility `1 - labor`, so the optimum works (`labor = no`)
with hours at the inactive node. Deleting that node would make `labor = yes` optimal.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import (
    AgeGrid,
    ConditionalAction,
    DiscreteGrid,
    GridBreakpoint,
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    Model,
    PiecewiseLinSpacedGrid,
    PiecewiseLogSpacedGrid,
    categorical,
)
from lcm.exceptions import InvalidSimulationInputError, ModelInitializationError
from lcm.grids import ContinuousGrid
from lcm.persistence import load_solution
from lcm.regime import Regime
from lcm.typing import (
    ActionName,
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    FunctionName,
    IntND,
    ScalarInt,
)
from tests.conftest import assert_agrees_to_ulp


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class Labor:
    no: ScalarInt
    yes: ScalarInt


PARAMS = {"discount_factor": 0.95, "working": {}, "dead": {}}
WEALTH = jnp.asarray([1.0, 1.5, 2.0])


def _utility(
    *, wealth: ContinuousState, hours: ContinuousAction, labor: DiscreteAction
) -> FloatND:
    return 1.0 - labor + 0.0 * (wealth + hours)


def _utility_with_shift(
    *,
    wealth: ContinuousState,
    hours: ContinuousAction,
    labor: DiscreteAction,
    shift: DiscreteAction,
) -> FloatND:
    return 1.0 - labor + 0.0 * (wealth + hours + shift)


def _next_wealth(*, wealth: ContinuousState) -> ContinuousState:
    return wealth


def _bequest(*, wealth: ContinuousState) -> FloatND:
    return 0.0 * wealth


def _hand_pin(inactive_node: float) -> Callable[..., BoolND]:
    def hours_only_when_working(
        *, labor: DiscreteAction, hours: ContinuousAction
    ) -> BoolND:
        return (labor == Labor.yes) | (hours == inactive_node)

    return hours_only_when_working


def _working_dtype() -> type[np.floating]:
    return np.float64 if jax.config.read("jax_enable_x64") else np.float32


type _Owner = Literal["left", "right"]


def _boundary_grid(*, kind: str, owner: _Owner) -> ContinuousGrid:
    """Return a grid whose node 1.0 has a representable neighbour as the next node."""
    if kind == "irregular":
        dtype = _working_dtype()
        direction = dtype(np.inf if owner == "left" else -np.inf)
        neighbour = float(np.nextafter(dtype(1.0), direction))
        return IrregSpacedGrid(
            points=np.array(tuple(sorted((0.0, 1.0, neighbour, 2.0))))
        )
    grid_class = {
        "piecewise_lin": PiecewiseLinSpacedGrid,
        "piecewise_log": PiecewiseLogSpacedGrid,
    }[kind]
    return grid_class(
        start=0.5 if kind == "piecewise_log" else 0.0,
        stop=2.0,
        breakpoints=(GridBreakpoint(value=1.0, owner=owner),),
        points_per_segment=(3, 3),
    )


def _model_with_hours(
    *,
    hours: ContinuousGrid,
    extra_constraints: dict[FunctionName, Callable[..., BoolND]] | None = None,
    with_shift: bool = False,
) -> Model:
    ages = AgeGrid(start=0, inclusive_stop=1, step="Y")
    actions: dict[ActionName, DiscreteGrid | ContinuousGrid] = {
        "labor": DiscreteGrid(category_class=Labor),
        "hours": hours,
    }
    if with_shift:
        actions["shift"] = DiscreteGrid(values=np.array([0, 8, 12]))
    working = Regime(
        actions=actions,
        state_transitions={"wealth": _next_wealth},
        functions={"utility": _utility_with_shift if with_shift else _utility},
        constraints=extra_constraints or {},
    )
    dead = Regime(functions={"utility": _bequest})
    return Model(
        regimes={"working": working, "dead": dead},
        states={"wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=3)},
        ages=ages,
        regime_id_class=RegimeId,
        initial_nodes={ages.exact_values[0]: "working"},
        edges={"working": {"dead": 0}},
    )


def _model(
    *, grid: ContinuousGrid, conditional: bool, inactive_value: float = 1.0
) -> Model:
    if conditional:
        return _model_with_hours(
            hours=ConditionalAction(
                grid=grid,
                active_when={"labor": Labor.yes},
                inactive_value=inactive_value,
            )
        )
    return _model_with_hours(
        hours=grid, extra_constraints={"hours_only": _hand_pin(inactive_value)}
    )


def _initial_conditions() -> dict[str, FloatND | IntND]:
    return {
        "wealth": WEALTH,
        "age": jnp.zeros(3),
        "regime_id": jnp.full(3, RegimeId.working, dtype=jnp.int32),
    }


def _installed_pin(model: Model) -> Callable[..., BoolND]:
    pin = model.user_regimes["working"].constraints["conditional_action_hours"]
    assert callable(pin)
    return cast("Callable[..., BoolND]", pin)


def _assert_pin_is_node_identity(
    *, model: Model, nodes: FloatND, inactive_index: int, with_shift: bool = False
) -> None:
    """Compare the installed pin with `active(d) or j == k`, enumerated by index.

    The pin is evaluated eagerly, under `jit`, cell by cell under `vmap`, and on
    transposed arrays.
    """
    pin = _installed_pin(model)
    labor = jnp.asarray([0, 1], dtype=jnp.int32)
    if with_shift:
        # The finalized regime's constraints read discrete actions as codes and
        # decode them to the values user functions see: shift codes 0, 1, 2 are the
        # values 0, 8, 12.
        shift = jnp.asarray([0, 1, 2], dtype=jnp.int32)
        labor_cells, shift_cells, hours_cells = jnp.meshgrid(
            labor, shift, nodes, indexing="ij"
        )
        kwargs = {"labor": labor_cells, "shift": shift_cells, "hours": hours_cells}
        expected = np.array(
            [
                [
                    [
                        (d == 1 and s in (8, 12)) or j == inactive_index
                        for j in range(len(nodes))
                    ]
                    for s in (0, 8, 12)
                ]
                for d in (0, 1)
            ]
        )
    else:
        labor_cells, hours_cells = jnp.meshgrid(labor, nodes, indexing="ij")
        kwargs = {"labor": labor_cells, "hours": hours_cells}
        expected = np.array(
            [[d == 1 or j == inactive_index for j in range(len(nodes))] for d in (0, 1)]
        )
    names = tuple(kwargs)

    def evaluate_cell(*values: FloatND | IntND) -> BoolND:
        return pin(**dict(zip(names, values, strict=True)))

    flat = tuple(kwargs[name].reshape(-1) for name in names)
    np.testing.assert_array_equal(np.asarray(pin(**kwargs)), expected)
    np.testing.assert_array_equal(np.asarray(jax.jit(pin)(**kwargs)), expected)
    np.testing.assert_array_equal(
        np.asarray(jax.vmap(evaluate_cell)(*flat)), expected.reshape(-1)
    )
    np.testing.assert_array_equal(
        np.asarray(jax.jit(pin)(**{name: value.T for name, value in kwargs.items()})),
        expected.T,
    )


@pytest.mark.parametrize("kind", ["irregular", "piecewise_lin", "piecewise_log"])
@pytest.mark.parametrize("owner", ["left", "right"])
def test_boundary_node_solves_looks_up_and_simulates_like_the_hand_pin(
    *, kind: str, owner: _Owner
) -> None:
    """At a node with a representable neighbour, the optimum stays `labor = no`
    with hours at the inactive node, as with the hand-written pin."""
    grid = _boundary_grid(kind=kind, owner=owner)
    models = {c: _model(grid=grid, conditional=c) for c in (False, True)}
    solutions = {c: m.solve(params=PARAMS, log_level="off") for c, m in models.items()}
    expected = solutions[False].values[0]["working"]
    np.testing.assert_array_equal(np.asarray(expected), np.ones_like(expected))
    assert_agrees_to_ulp(
        got=solutions[True].values[0]["working"], expected=expected, n_ulp=4
    )
    lookups = {
        c: m.lookup_policy(
            params=PARAMS,
            solution=solutions[c],
            period=0,
            regime_name="working",
            states={"wealth": WEALTH},
        )
        for c, m in models.items()
    }
    assert lookups[True].actions is not None
    assert lookups[False].actions is not None
    for name in ("labor", "hours"):
        np.testing.assert_array_equal(
            np.asarray(lookups[True].actions[name]),
            np.asarray(lookups[False].actions[name]),
        )
    assert np.all(np.asarray(lookups[True].actions["labor"]) == int(Labor.no))
    assert np.all(np.asarray(lookups[True].actions["hours"]) == 1.0)
    frames = {}
    for conditional, model in models.items():
        frame = model.simulate(
            params=PARAMS,
            solution=solutions[conditional],
            initial_conditions=_initial_conditions(),
            log_level="off",
        ).to_dataframe()
        frames[conditional] = frame.loc[
            frame["regime_name"] == "working",
            ["subject_id", "period", "labor", "hours"],
        ].reset_index(drop=True)
    assert len(frames[False]) == len(WEALTH)
    pd.testing.assert_frame_equal(frames[True], frames[False])


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("exponent", [-10, 0, 10])
@pytest.mark.parametrize("odd_significand", [False, True])
def test_pin_is_node_identity_on_three_adjacent_nodes(
    *, sign: int, exponent: int, odd_significand: bool
) -> None:
    """On three consecutive representable floats, each inactive node is admitted
    alone wherever the action is inactive."""
    dtype = _working_dtype()
    center = dtype(np.ldexp(float(sign), exponent))
    if odd_significand:
        center = np.nextafter(center, dtype(np.inf))
    points = (
        float(np.nextafter(center, dtype(-np.inf))),
        float(center),
        float(np.nextafter(center, dtype(np.inf))),
    )
    grid = IrregSpacedGrid(points=np.array(points))
    nodes = grid.to_jax()
    assert bool(np.all(np.diff(np.asarray(nodes)) > 0))
    for k, inactive_value in enumerate(points):
        declaration = ConditionalAction(
            grid=grid, active_when={"labor": Labor.yes}, inactive_value=inactive_value
        )
        _assert_pin_is_node_identity(
            model=_model_with_hours(hours=declaration), nodes=nodes, inactive_index=k
        )


@pytest.mark.parametrize("owner", ["left", "right"])
@pytest.mark.parametrize("reverse_conditions", [False, True])
def test_pin_is_node_identity_under_a_conjunction_with_a_value_set(
    *, owner: _Owner, reverse_conditions: bool
) -> None:
    """Two conditions, one with two activating values, in either declaration order."""
    grid = _boundary_grid(kind="piecewise_lin", owner=owner)
    pairs: list[tuple[ActionName, tuple[int, ...]]] = [
        ("labor", (int(Labor.yes),)),
        ("shift", (8, 12)),
    ]
    if reverse_conditions:
        pairs.reverse()
    declaration = ConditionalAction(
        grid=grid, active_when=dict(pairs), inactive_value=1.0
    )
    _assert_pin_is_node_identity(
        model=_model_with_hours(hours=declaration, with_shift=True),
        nodes=grid.to_jax(),
        inactive_index=declaration.inactive_index,
        with_shift=True,
    )


@pytest.mark.parametrize("kind", ["rounded_linear", "piecewise_lin"])
def test_pin_uses_the_node_generated_at_the_build_precision(*, kind: str) -> None:
    """A declaration made at float32 pins the node the grid generates at float64,
    the precision the model is built and solved at.

    The opposite direction is not a supported path for any pylcm grid: a grid
    declared at float64 cannot generate float32 nodes.
    """
    old = jax.config.read("jax_enable_x64")
    try:
        jax.config.update("jax_enable_x64", val=False)
        grid = (
            LinSpacedGrid(start=0.0, stop=1.0, n_points=11)
            if kind == "rounded_linear"
            else _boundary_grid(kind=kind, owner="left")
        )
        declaration = ConditionalAction(
            grid=grid,
            active_when={"labor": Labor.yes},
            inactive_value=0.9 if kind == "rounded_linear" else 1.0,
        )
        jax.config.update("jax_enable_x64", val=True)
        assert bool(np.all(np.diff(np.asarray(grid.to_jax())) > 0))
        model = _model_with_hours(hours=declaration)
        _assert_pin_is_node_identity(
            model=model,
            nodes=grid.to_jax(),
            inactive_index=declaration.inactive_index,
        )
    finally:
        jax.config.update("jax_enable_x64", old)


def test_pin_refuses_actions_at_another_precision_than_the_model_was_built_at() -> None:
    """Evaluating the pin on float64 actions of a model built at float32 raises."""
    old = jax.config.read("jax_enable_x64")
    try:
        jax.config.update("jax_enable_x64", val=False)
        grid = _boundary_grid(kind="piecewise_lin", owner="left")
        model = _model(grid=grid, conditional=True)
        pin = _installed_pin(model)
        jax.config.update("jax_enable_x64", val=True)
        with pytest.raises(TypeError, match="float32"):
            pin(
                labor=jnp.asarray([0], dtype=jnp.int32),
                hours=jnp.asarray([1.0], dtype=jnp.float64),
            )
    finally:
        jax.config.update("jax_enable_x64", old)


def test_saved_solution_on_a_boundary_grid_binds_the_inactive_node(
    tmp_path: Path,
) -> None:
    """A saved solution simulates with the same declaration and is refused once the
    inactive node changes."""
    grid = _boundary_grid(kind="piecewise_lin", owner="left")
    solution = _model(grid=grid, conditional=True).solve(params=PARAMS, log_level="off")
    loaded = load_solution(path=solution.save(path=tmp_path / "boundary.lcm"))
    frame = (
        _model(grid=grid, conditional=True)
        .simulate(
            params=PARAMS,
            solution=loaded,
            initial_conditions=_initial_conditions(),
            log_level="off",
        )
        .to_dataframe()
    )
    working = frame.loc[frame["regime_name"] == "working"]
    assert len(working) == 3
    assert set(working["labor"]) == {"no"}
    assert np.all(working["hours"].to_numpy() == 1.0)
    changed = _model(grid=grid, conditional=True, inactive_value=2.0)
    with pytest.raises(InvalidSimulationInputError):
        changed.simulate(
            params=PARAMS,
            solution=loaded,
            initial_conditions=_initial_conditions(),
            log_level="off",
        )


def test_model_refuses_an_inactive_node_whose_value_repeats() -> None:
    """At float32 the first nodes of a very narrow log grid round to one value; the
    inactive node is then not identified by its value, and the model refuses it."""
    grid = LogSpacedGrid(start=1.0, stop=1.0000002, n_points=50)
    nodes = np.asarray(grid.to_jax())
    repeats = int((nodes == nodes[0]).sum()) > 1
    declaration = ConditionalAction(
        grid=grid, active_when={"labor": Labor.yes}, inactive_value=1.0
    )
    if repeats:
        with pytest.raises(ModelInitializationError, match="occurs 13 times"):
            _model_with_hours(hours=declaration)
    else:
        _model_with_hours(hours=declaration)
