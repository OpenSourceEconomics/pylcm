"""A supplied process value read as `<process>_code` must stand on a node.

The position of a node exists only at that node, so a simulation refuses an
initial condition or a shock path that puts an off-node value where a regime
reads the position. A regime that reads the process only as a value keeps the
process's whole continuous support.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    Model,
    NormalIIDProcess,
    Regime,
    categorical,
)
from lcm.exceptions import InvalidInitialConditionsError, InvalidSimulationInputError
from lcm.typing import (
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
)


@categorical(ordered=False)
class Work:
    off: ScalarInt
    on: ScalarInt


@categorical(ordered=False)
class RegimeId:
    worker: ScalarInt


_NODES = (-2.0, -1.0, 0.0, 1.0, 2.0)

# Both sides of every interior node, half-cell ties, and values beyond either
# end; every one is exactly representable.
_OFF_NODES = (
    -32.0,
    -3.0,
    -1.875,
    -1.5,
    -1.125,
    -0.875,
    -0.5,
    -0.25,
    -0.125,
    0.125,
    0.25,
    0.5,
    0.875,
    1.125,
    1.5,
    1.875,
    3.0,
    32.0,
)


def _utility_by_code(*, z_code: DiscreteState, work: DiscreteAction) -> FloatND:
    return jnp.asarray(_NODES)[z_code] + 0.125 * work


def _utility_by_value(*, z: ContinuousState, work: DiscreteAction) -> FloatND:
    return z + 0.125 * work


def _model(*, code: bool, enable_jit: bool = False) -> Model:
    shock = NormalIIDProcess(
        n_points=5,
        mu=0.0,
        sigma=1.0,
        n_std=2.0,
        gauss_hermite=False,
        simulated_draws="nodes",
    )
    worker = Regime(
        states={"z": shock},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _utility_by_code if code else _utility_by_value},
    )
    return Model(
        regimes={"worker": worker},
        edges={},
        ages=AgeGrid(exact_values=(0,)),
        initial_nodes={0: "worker"},
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
    )


def _simulate(*, model: Model, value: float, entry: str) -> pd.DataFrame:
    params = {"worker": {}}
    initial = {
        "z": jnp.array([0.0 if entry == "shock_path" else value]),
        "age": jnp.array([0.0]),
        "regime_id": jnp.array([model.regime_names_to_ids["worker"]], dtype=jnp.int32),
    }
    return model.simulate(
        params=params,
        solution=model.solve(params=params, log_level="off"),
        initial_conditions=initial,
        shock_paths={"z": np.array([value])} if entry == "shock_path" else None,
        seed=0,
        log_level="off",
    ).to_dataframe()


@pytest.mark.parametrize("value", _OFF_NODES)
def test_off_node_initial_value_is_refused_where_its_code_is_read(value: float) -> None:
    with pytest.raises(InvalidInitialConditionsError, match="off the process's nodes"):
        _simulate(model=_model(code=True), value=value, entry="initial")


@pytest.mark.parametrize("value", _OFF_NODES)
def test_off_node_shock_path_is_refused_where_its_code_is_read(value: float) -> None:
    with pytest.raises(
        InvalidSimulationInputError, match=r"off the process's nodes|inadmissible"
    ):
        _simulate(model=_model(code=True), value=value, entry="shock_path")


@pytest.mark.parametrize("entry", ["initial", "shock_path"])
def test_off_node_value_is_refused_under_jit(entry: str) -> None:
    with pytest.raises(
        (InvalidInitialConditionsError, InvalidSimulationInputError), match="node"
    ):
        _simulate(model=_model(code=True, enable_jit=True), value=0.25, entry=entry)


@pytest.mark.parametrize("entry", ["initial", "shock_path"])
@pytest.mark.parametrize("value", _NODES)
def test_node_value_is_held_where_its_code_is_read(*, entry: str, value: float) -> None:
    panel = _simulate(model=_model(code=True), value=value, entry=entry)
    np.testing.assert_array_equal(panel["z"].to_numpy(), np.array([value]))


@pytest.mark.parametrize("entry", ["initial", "shock_path"])
@pytest.mark.parametrize("value", [-0.25, 0.25])
def test_off_node_value_read_by_value_is_held(*, entry: str, value: float) -> None:
    panel = _simulate(model=_model(code=False), value=value, entry=entry)
    np.testing.assert_array_equal(panel["z"].to_numpy(), np.array([value]))
