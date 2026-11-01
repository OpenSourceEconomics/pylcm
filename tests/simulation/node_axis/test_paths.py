"""Apply tagged current-shock paths only where a subject consumes their cells."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import AgeGrid, ExecutionConfig, InitialNodes, IrregSpacedGrid, Model, Regime
from lcm.exceptions import InvalidSimulationInputError
from lcm.shocks import BaseInnovations, PhysicalValues, ShockGroup, Uniform, UserNodes
from lcm.typing import ContinuousState, FloatND
from tests.simulation._node_axis_models import (
    _child_regimes,
    _heterogeneous_entry_model,
    _income,
    _MixedIds,
    _path_model,
    _replay,
    _transition_model,
)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_unread_adapter_path_cells_do_not_condition_the_draw() -> None:
    """Only column one is consumed; both outside-support columns are unread."""
    model = _path_model()
    solution = model.solve(params={"discount_factor": 1.0}, log_level="off")
    path = np.asarray([[2.0, 0.5, 2.0]], dtype=jnp.asarray(0.0).dtype)
    simulated = _replay(model=model, solution=solution, path=path, seed=29)
    child = simulated.raw_results["child"][1]
    np.testing.assert_array_equal(child.shocks["eps"], [0.5])
    np.testing.assert_allclose(
        child.actions["consumption"], [0.75], rtol=2e-4, atol=2e-4
    )
    np.testing.assert_allclose(
        simulated.raw_results["done"][2].states["wealth"], [0.75], rtol=2e-4, atol=2e-4
    )


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize(
    ("column", "common", "width", "budgeted", "seed"),
    [
        (0, False, 1, False, 1),
        (2, False, 2, True, 29),
        (0, True, 2, False, 71),
        (2, True, 1, True, 91),
    ],
)
def test_unread_cell_mutations_leave_replay_unchanged(
    *, column: int, common: bool, width: int, budgeted: bool, seed: int
) -> None:
    model = _path_model(width=width, budgeted=budgeted)
    solution = model.solve(params={"discount_factor": 1.0}, log_level="off")
    dtype = jnp.asarray(0.0).dtype
    good = np.asarray([0.5, 0.5, 0.5], dtype=dtype)
    if not common:
        good = np.tile(good, (3, 1))
        good[:, 1] = [0.0, 0.5, 1.0]  # consumed endpoints remain supported
    changed = good.copy()
    if common:
        changed[column] = 2.0
    else:
        changed[:, column] = 2.0
    reference = _replay(model=model, solution=solution, path=good, seed=seed)
    actual = _replay(model=model, solution=solution, path=changed, seed=seed)
    for regime, period in (("child", 1), ("done", 2)):
        left, right = (
            reference.raw_results[regime][period],
            actual.raw_results[regime][period],
        )
        for name, expected in left.states.items():
            np.testing.assert_array_equal(right.states[name], expected)
        for name, expected in left.actions.items():
            np.testing.assert_array_equal(right.actions[name], expected)
        for name, expected in left.shocks.items():
            np.testing.assert_array_equal(right.shocks[name], expected)
        np.testing.assert_array_equal(right.V_arr, left.V_arr)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("invalid", [-1.0, 2.0])
def test_consumed_adapter_path_cell_still_requires_support(*, invalid: float) -> None:
    model = _path_model()
    solution = model.solve(params={"discount_factor": 1.0}, log_level="off")
    path = np.asarray([[0.5, invalid, 0.5]], dtype=jnp.asarray(0.0).dtype)
    with pytest.raises(InvalidSimulationInputError):
        _replay(model=model, solution=solution, path=path, seed=29)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize(("base", "reverse"), [(False, False), (True, True)])
def test_heterogeneous_nonreader_entry_ignores_all_unconsumed_cells(
    *, base: bool, reverse: bool
) -> None:
    """One subject never reads eps; another reads it only in column one.

    Uniform(0,1) makes the valid base and physical coordinates equal. The value
    two is outside both supports, but is placed only in unconsumed cells. A row
    permutation checks that consumption belongs to original subjects.
    """
    model = _heterogeneous_entry_model()
    params = {"discount_factor": 1.0}
    solution = model.solve(params=params, log_level="off", max_compilation_workers=1)
    dtype = jnp.asarray(0.0).dtype
    order = np.asarray([1, 0] if reverse else [0, 1])
    initial = {
        "wealth": np.ones(2, dtype=dtype),
        "age": np.asarray([2, 1], dtype=np.int32)[order],
        "regime_id": np.asarray(
            [model.regime_names_to_ids["done"], model.regime_names_to_ids["child"]],
            dtype=np.int32,
        )[order],
    }
    reference = np.full((2, 3), 0.5, dtype=dtype)
    changed = np.asarray([[2.0, 2.0, 2.0], [2.0, 0.5, 2.0]], dtype=dtype)[order]
    tag = BaseInnovations if base else PhysicalValues
    frames = []
    for values in (reference, changed):
        simulated = model.simulate(
            params=params,
            solution=solution,
            initial_conditions=initial,
            shock_paths={"eps": tag(values=values)},
            seed=29,
            log_level="off",
        )
        frames.append(simulated.to_dataframe())
    pd.testing.assert_frame_equal(frames[1], frames[0], check_exact=True)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("budgeted", [False, True])
@pytest.mark.parametrize("base", [False, True])
@pytest.mark.parametrize("dependent", [False, True])
def test_only_realized_destination_reads_its_supplied_shock(
    *, budgeted: bool, base: bool, dependent: bool
) -> None:
    """A terminal destination ignores the same-date reader's invalid path cell."""

    model = _transition_model(budgeted=budgeted, dependent=dependent)
    dtype = jnp.asarray(0.0).dtype
    values = np.asarray([[2.0, 0.5, 2.0], [2.0, 2.0, 2.0]], dtype=dtype)
    tag = BaseInnovations if base else PhysicalValues
    result = model.simulate(
        params={"discount_factor": 1.0},
        initial_conditions={
            "wealth": np.asarray([1.0, 2.0], dtype=dtype),
            "age": np.zeros(2, dtype=np.int32),
            "regime_id": np.full(
                2, model.regime_names_to_ids["parent"], dtype=np.int32
            ),
        },
        shock_paths={"eps": tag(values=values)},
        seed=29,
        log_level="off",
    )
    frame = result.to_dataframe(use_categorical_codes=True).reset_index()
    actual = frame.query("period == 1 and regime_name == 'child'")["consumption"]
    np.testing.assert_allclose(actual, [0.75], rtol=2e-4, atol=2e-4)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("budgeted", [False, True])
def test_other_current_group_keeps_its_tagged_path_and_entry_precedence(
    *, budgeted: bool
) -> None:
    """A node-axis group's unread cells do not intercept another group's tags."""

    def utility(*, eta: FloatND) -> FloatND:
        return eta

    regimes = _child_regimes()
    regimes["probe"] = Regime(functions={"utility": utility}, shocks=("other",))
    model = Model(
        regimes=regimes,
        ages=AgeGrid(exact_values=(0, 1, 2)),
        initial_nodes=InitialNodes(by_age={0: "probe", 1: "child"}),
        edges={"child": {"done": (1,)}},
        regime_id_class=_MixedIds,
        shocks={
            "income": _income(nodes=(0.0, 1.0), weights=(0.5, 0.5)),
            "other": ShockGroup(
                shocks={"eta": Uniform(start=0.0, stop=1.0)},
                nodes=UserNodes(
                    nodes=np.asarray([[0.0], [1.0]]), weights=np.asarray([0.5, 0.5])
                ),
            ),
        },
        execution_config=ExecutionConfig(
            axis_widths={"subject": 1},
            device_memory_bytes=2**27 if budgeted else None,
        ),
    )
    dtype = jnp.asarray(0.0).dtype
    result = model.simulate(
        params={"discount_factor": 1.0},
        initial_conditions={
            "wealth": np.ones(1, dtype=dtype),
            "age": np.zeros(1, dtype=np.int32),
            "regime_id": np.full(1, model.regime_names_to_ids["probe"], dtype=np.int32),
            "eta": PhysicalValues(values=np.asarray([0.75], dtype=dtype)),
        },
        shock_paths={
            "eps": PhysicalValues(values=np.full((1, 3), 2.0, dtype=dtype)),
            "eta": PhysicalValues(values=np.asarray([0.25, 0.25, 0.25], dtype=dtype)),
        },
        seed=29,
        log_level="off",
    )
    np.testing.assert_array_equal(result.raw_results["probe"][0].shocks["eta"], [0.25])


def _add_destination_shock(
    *, wealth: ContinuousState, next_eps: ContinuousState
) -> ContinuousState:
    return wealth + next_eps


def _read_current_shock(*, eps: FloatND) -> FloatND:
    return eps


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("budgeted", [False, True])
def test_transition_local_shock_is_consumed_before_the_terminal_drops_it(
    *, budgeted: bool
) -> None:
    """A terminal wealth law reads its destination-date supplied innovation."""
    regimes = _child_regimes()
    regimes["probe"] = Regime(
        states={"wealth": IrregSpacedGrid(points=np.asarray([0.375, 1.0, 2.0]))},
        functions={"utility": _read_current_shock},
        state_transitions={"wealth": {"done": _add_destination_shock}},
        shocks=("income",),
    )
    model = Model(
        regimes=regimes,
        ages=AgeGrid(exact_values=(0, 1, 2)),
        initial_nodes=InitialNodes(by_age={0: "probe", 1: "child"}),
        edges={"probe": {"done": (0,)}, "child": {"done": (1,)}},
        regime_id_class=_MixedIds,
        shocks={"income": _income(nodes=(0.0, 1.0), weights=(0.5, 0.5))},
        execution_config=ExecutionConfig(
            axis_widths={"subject": 1},
            device_memory_bytes=2**27 if budgeted else None,
        ),
    )
    dtype = jnp.asarray(0.0).dtype
    result = model.simulate(
        params={"discount_factor": 1.0},
        initial_conditions={
            "wealth": np.ones(1, dtype=dtype),
            "age": np.zeros(1, dtype=np.int32),
            "regime_id": np.full(1, model.regime_names_to_ids["probe"], dtype=np.int32),
        },
        shock_paths={
            "eps": PhysicalValues(values=np.asarray([0.5, 0.25, 0.5], dtype=dtype))
        },
        seed=29,
        log_level="off",
    )
    np.testing.assert_array_equal(
        result.raw_results["done"][1].states["wealth"], [1.25]
    )
