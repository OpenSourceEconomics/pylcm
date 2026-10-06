"""Coordinate contract for `Model.state_grid` on age-specialized state axes.

An age-specialized state's value arrays are tabulated on per-period nodes, so its
coordinates are identified only together with a period index. Omitting the period
for such a state raises; age-invariant states may still omit it.
"""

import numpy as np
import pytest

from lcm import (
    AgeGrid,
    AgeSpecializedGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import InvalidSimulationInputError
from lcm.phased import Phased
from lcm.typing import ContinuousState, FloatND, ScalarInt


@categorical(ordered=False)
class EndId:
    end: ScalarInt


def _identity(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _two_state_value(*, wealth: ContinuousState, income: ContinuousState) -> FloatND:
    return wealth + 10.0 * income


def _model(
    *,
    scale: float = 2.0,
    shift: float = 0.0,
    n_points: int = 2,
    enable_jit: bool = False,
    roots: tuple[int, ...] = (0, 1),
    two_states: bool = False,
    specialized: bool = True,
) -> Model:
    def make_grid(age: float) -> LinSpacedGrid:
        factor = 1.0 if age == 0.0 else scale
        return LinSpacedGrid(
            start=shift + factor,
            stop=shift + 2.0 * factor,
            n_points=n_points,
        )

    states = {
        "wealth": (
            AgeSpecializedGrid(build=make_grid, signature=lambda age: age)
            if specialized
            else make_grid(0.0)
        )
    }
    if two_states:
        states["income"] = LinSpacedGrid(start=0.0, stop=1.0, n_points=3)
    return Model(
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=EndId,
        initial_nodes={roots: "end"},
        edges={},
        regimes={
            "end": Regime(
                regime_transitions=None,
                states=states,
                functions={"utility": _two_state_value if two_states else _identity},
            )
        },
        enable_jit=enable_jit,
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def test_omitted_period_never_silently_labels_late_values_with_early_coordinates():
    """Late-period values tabulated on [2, 4] never get the early [1, 2] nodes."""
    model = _model()
    values = model.solve(params={}, log_level="off").values
    np.testing.assert_array_equal(np.asarray(values[0]["end"]), [1.0, 2.0])
    np.testing.assert_array_equal(np.asarray(values[1]["end"]), [2.0, 4.0])
    with pytest.raises(InvalidSimulationInputError, match="period"):
        model.state_grid(params={}, regime_name="end", state_name="wealth")


def test_unvalued_period_is_rejected_after_changing_the_representative():
    """A period at which the regime is not valued is not a coordinate request."""
    model = _model(roots=(1,))
    got = model.state_grid(params={}, regime_name="end", state_name="wealth", period=1)
    np.testing.assert_array_equal(np.asarray(got), [2.0, 4.0])
    with pytest.raises(InvalidSimulationInputError, match="period"):
        model.state_grid(params={}, regime_name="end", state_name="wealth", period=0)


@pytest.mark.parametrize("period", [-1, 2, True, 0.0])
def test_invalid_period_is_not_a_value_array_coordinate_request(*, period):
    """Out-of-range, boolean and float periods raise."""
    with pytest.raises(InvalidSimulationInputError, match="period"):
        _model().state_grid(
            params={}, regime_name="end", state_name="wealth", period=period
        )


def test_invariant_axis_can_still_omit_period():
    """An age-invariant axis is the same with and without a period."""
    model = _model(specialized=False)
    for period in (None, 0, 1):
        kwargs = {} if period is None else {"period": period}
        got = model.state_grid(
            params={}, regime_name="end", state_name="wealth", **kwargs
        )
        np.testing.assert_array_equal(np.asarray(got), [1.0, 2.0])


def test_two_state_axis_order_and_only_one_axis_age_specialized():
    """Per-period axes reproduce the value array of a two-state regime."""
    model = _model(n_points=2, two_states=True)
    result = model.solve(params={}, log_level="off")
    names = model.state_names(regime_name="end")
    assert set(names) == {"wealth", "income"}
    np.testing.assert_array_equal(
        np.asarray(model.state_grid(params={}, regime_name="end", state_name="income")),
        [0.0, 0.5, 1.0],
    )
    for period in (1, 0):
        axes = {
            name: np.asarray(
                model.state_grid(
                    params={}, regime_name="end", state_name=name, period=period
                )
            )
            for name in names
        }
        coords = dict(
            zip(
                names,
                np.meshgrid(*(axes[name] for name in names), indexing="ij"),
                strict=True,
            )
        )
        expected = coords["wealth"] + 10.0 * coords["income"]
        np.testing.assert_array_equal(
            np.asarray(result.values[period]["end"]), expected
        )


@categorical(ordered=False)
class RouteId:
    source: ScalarInt
    end: ScalarInt
    other: ScalarInt


def test_a_value_only_node_is_a_valid_coordinate_request():
    """A valued but never visited period still has coordinates."""
    moving = AgeSpecializedGrid(
        build=lambda age: LinSpacedGrid(
            start=1.0 + age, stop=2.0 * (1.0 + age), n_points=2
        ),
        signature=lambda age: age,
    )
    common = LinSpacedGrid(start=2.0, stop=4.0, n_points=2)
    model = Model(
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=RouteId,
        initial_nodes={0: "source"},
        edges=Phased(solve={"source": {"end": 0}}, simulate={"source": {"other": 0}}),
        enable_jit=False,
        execution_config=ExecutionConfig(device_memory_bytes=None),
        regimes={
            "source": Regime(
                regime_transitions=Phased(solve="end", simulate="other"),
                states={"wealth": common},
                state_transitions={"wealth": fixed_transition("wealth")},
                functions={"utility": _identity},
            ),
            "end": Regime(
                regime_transitions=None,
                states={"wealth": moving},
                functions={"utility": _identity},
            ),
            "other": Regime(
                regime_transitions=None,
                states={"wealth": common},
                functions={"utility": _identity},
            ),
        },
    )
    assert (1, "end") in model.reachability.nodes
    assert (1, "end") not in model.reachability.visited_nodes
    got = model.state_grid(
        params={"discount_factor": 0.0},
        regime_name="end",
        state_name="wealth",
        period=1,
    )
    np.testing.assert_array_equal(np.asarray(got), [2.0, 4.0])


def test_period_is_an_index_not_the_calendar_age():
    """`period` indexes model periods; a calendar age is rejected."""
    model = Model(
        ages=AgeGrid(start=10, inclusive_stop=11, step="Y"),
        regime_id_class=EndId,
        initial_nodes={(10, 11): "end"},
        edges={},
        enable_jit=False,
        execution_config=ExecutionConfig(device_memory_bytes=None),
        regimes={
            "end": Regime(
                regime_transitions=None,
                functions={"utility": _identity},
                states={
                    "wealth": AgeSpecializedGrid(
                        build=lambda age: LinSpacedGrid(
                            start=age - 9.0,
                            stop=2.0 * (age - 9.0),
                            n_points=2,
                        ),
                        signature=lambda age: age,
                    )
                },
            )
        },
    )
    got = model.state_grid(params={}, regime_name="end", state_name="wealth", period=1)
    np.testing.assert_array_equal(np.asarray(got), [2.0, 4.0])
    with pytest.raises(InvalidSimulationInputError, match="period"):
        model.state_grid(params={}, regime_name="end", state_name="wealth", period=11)
