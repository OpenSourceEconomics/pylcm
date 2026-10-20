"""A carried state declared with `values` publishes those values.

The state `hours` is carried: its solve side is imputed as `20`, its simulate side
is `DiscreteGrid(values=nodes)`. No regime declares it as a plain grid, so the only
place its values appear is `Phased.simulate`. Every subject keeps its starting
node, so in period 0 the published column must equal the starting values
literally, with the full ordered node set as its categories, including nodes no
subject occupies.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import (
    DiscreteGrid,
    InitialNodes,
    LinSpacedGrid,
    Model,
    Periods,
    Phased,
    categorical,
    fixed_transition,
)
from lcm.regime import Regime
from lcm.result import SimulationResult
from lcm.typing import ContinuousState, DiscreteState, FloatND, ScalarInt


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


def _dead_utility() -> FloatND:
    return jnp.asarray(0.0)


def _wealth_utility(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _hours_as_wealth(*, hours: DiscreteState) -> FloatND:
    return jnp.asarray(hours, dtype=float)


def _carried_hours(nodes: np.ndarray) -> Phased:
    """Carry `hours`, imputing its middle node in the backward problem."""
    middle = nodes[1]

    def impute_hours() -> DiscreteState:
        return jnp.asarray(middle)

    return Phased(solve=impute_hours, simulate=DiscreteGrid(values=nodes))


def _regime(*, hours: Phased | DiscreteGrid) -> Regime:
    return Regime(
        states={
            "wealth": LinSpacedGrid(start=-50, stop=50, n_points=4),
            "hours": hours,
        },
        state_transitions={
            "wealth": _hours_as_wealth,
            "hours": fixed_transition("hours"),
        },
        functions={"utility": _wealth_utility},
    )


def _model(
    *, nodes: np.ndarray, retired_declares_directly: bool, enable_jit: bool
) -> Model:
    retired_hours = (
        DiscreteGrid(values=nodes)
        if retired_declares_directly
        else _carried_hours(nodes)
    )
    return Model(
        n_periods=3,
        regimes={
            "alive": _regime(hours=_carried_hours(nodes)),
            "retired": _regime(hours=retired_hours),
            "dead": Regime(functions={"utility": _dead_utility}),
        },
        regime_id_class=_RegimeId,
        initial_nodes=InitialNodes(by_period={0: ("alive", "retired")}),
        edges={
            "alive": {"alive": Periods(values=(0,)), "dead": Periods(values=(1,))},
            "retired": {
                "retired": Periods(values=(0,)),
                "dead": Periods(values=(1,)),
            },
        },
        enable_jit=enable_jit,
    )


def _simulate(
    *,
    nodes: np.ndarray,
    start: np.ndarray,
    retired_declares_directly: bool = False,
    enable_jit: bool = True,
) -> SimulationResult:
    n_subjects = len(start)
    return _model(
        nodes=nodes,
        retired_declares_directly=retired_declares_directly,
        enable_jit=enable_jit,
    ).simulate(
        params={"discount_factor": 1.0},
        initial_conditions={
            "wealth": np.zeros(n_subjects),
            "hours": np.asarray(start),
            "period": np.zeros(n_subjects, dtype=np.int32),
            # Alternate the subjects between the two regimes that carry `hours`.
            "regime_id": np.array(
                [_RegimeId.alive, _RegimeId.retired] * n_subjects, dtype=np.int32
            )[:n_subjects],
        },
        seed=0,
        log_level="off",
    )


def _first_period(frame: pd.DataFrame) -> pd.Series:
    rows = frame.loc[frame["period"] == 0].sort_values("subject_id")
    return rows["hours"].reset_index(drop=True)


# Integer and float nodes, shifted and scaled; the starts leave one node unoccupied.
_CASES = {
    "int": (np.array([10, 20, 30], dtype=np.int32), [20, 10, 20, 10]),
    "float": (np.array([-1.5, 0.25, 4.0], dtype=np.float32), [4.0, -1.5, 4.0, 4.0]),
}
_VARIANTS = [
    pytest.param(case, directly, enable_jit, id=f"{case}-{label}-{jit_label}")
    for case in _CASES
    for directly, label in [(False, "carried-everywhere"), (True, "declared-once")]
    for enable_jit, jit_label in [(False, "eager"), (True, "jit")]
]


def _value_frame(*, case: str, directly: bool, enable_jit: bool) -> pd.DataFrame:
    nodes, start = _CASES[case]
    return _simulate(
        nodes=nodes,
        start=np.asarray(start, dtype=nodes.dtype),
        retired_declares_directly=directly,
        enable_jit=enable_jit,
    ).to_dataframe()


@pytest.mark.parametrize(("case", "directly", "enable_jit"), _VARIANTS)
def test_carried_value_grid_column_holds_its_values(
    *, case: str, directly: bool, enable_jit: bool
) -> None:
    frame = _value_frame(case=case, directly=directly, enable_jit=enable_jit)
    assert _first_period(frame).tolist() == _CASES[case][1]


@pytest.mark.parametrize(("case", "directly", "enable_jit"), _VARIANTS)
def test_carried_value_grid_categories_are_all_nodes_in_order(
    *, case: str, directly: bool, enable_jit: bool
) -> None:
    dtype = _value_frame(case=case, directly=directly, enable_jit=enable_jit)[
        "hours"
    ].dtype
    nodes = _CASES[case][0].tolist()
    assert dtype == pd.CategoricalDtype(categories=nodes, ordered=True)


@pytest.mark.parametrize(("case", "directly", "enable_jit"), _VARIANTS)
def test_carried_value_grid_column_mean_is_the_mean_of_its_values(
    *, case: str, directly: bool, enable_jit: bool
) -> None:
    frame = _value_frame(case=case, directly=directly, enable_jit=enable_jit)
    assert _first_period(frame).astype(float).mean() == pytest.approx(
        np.mean(_CASES[case][1])
    )


@pytest.mark.parametrize("case", list(_CASES))
def test_carried_value_grid_with_categorical_codes_gives_codes(case: str) -> None:
    nodes, start = _CASES[case]
    frame = _simulate(nodes=nodes, start=np.asarray(start, dtype=nodes.dtype))
    codes = [nodes.tolist().index(value) for value in start]
    assert (
        _first_period(frame.to_dataframe(use_categorical_codes=True)).tolist() == codes
    )


def test_carried_value_grid_feather_holds_its_values(tmp_path) -> None:
    nodes, start = _CASES["int"]
    _simulate(nodes=nodes, start=np.asarray(start, dtype=nodes.dtype)).save(
        directory=tmp_path
    )
    saved = pd.read_feather(tmp_path / "simulated_data.arrow")
    assert _first_period(saved).tolist() == start


def test_carried_value_grid_load_holds_its_values(tmp_path) -> None:
    nodes, start = _CASES["int"]
    _simulate(nodes=nodes, start=np.asarray(start, dtype=nodes.dtype)).save(
        directory=tmp_path
    )
    restored = SimulationResult.load(directory=tmp_path).to_dataframe()
    assert _first_period(restored).tolist() == start
