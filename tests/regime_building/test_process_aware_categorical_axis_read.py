"""A categorical axis of a process-aware V read is an exact node read.

`get_V_interpolator(interpolate_process_axes=True)` builds the reader used
wherever a projection produces a genuine VALUE for every axis of a reference
regime's V — a `ProjectedRegimeValue` projection, a gated-edge gate, a gated-edge
fallback. Such a regime may carry three kinds of axis at once, and each has its
own reading rule:

- a genuine `DiscreteGrid` axis names a category, so the incoming value is one
  of the grid's integer codes and the read at it is that node's value exactly;
- a non-folded stochastic-process axis carries a value in the process's own
  units, read by linear interpolation clamped to the node range;
- a continuous axis is read by linear interpolation, extrapolated linearly
  beyond the node range.

These tests pin all three against hand-written references, plus the two ways a
value can name no category (fractional, out of range) and the derivative of the
read.
"""

from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.ndimage import map_coordinates
from _lcm.regime_building.V import VInterpolationInfo, get_V_interpolator
from lcm import DiscreteGrid, LinSpacedGrid, NormalIIDProcess, categorical
from lcm.typing import ScalarInt
from tests.conftest import DECIMAL_PRECISION


@categorical(ordered=False)
class Three:
    a: ScalarInt
    b: ScalarInt
    c: ScalarInt


@categorical(ordered=False)
class Two:
    yes: ScalarInt
    no: ScalarInt


# Nodes: linspace(mu - n_std*sigma, mu + n_std*sigma, 5) = [-2, -1, 0, 1, 2].
_SHOCK = NormalIIDProcess(n_points=5, gauss_hermite=False, mu=0.0, sigma=1.0, n_std=2.0)
_WEALTH = LinSpacedGrid(start=0.0, stop=4.0, n_points=5)  # nodes 0, 1, 2, 3, 4


def _reader(*, state_names, discrete_states, continuous_states):
    info = VInterpolationInfo(
        state_names=tuple(state_names),
        discrete_states=MappingProxyType(discrete_states),
        continuous_states=MappingProxyType(continuous_states),
    )
    return get_V_interpolator(
        v_interpolation_info=info,
        state_prefix="next_",
        V_arr_name="V_arr",
        interpolate_process_axes=True,
    )


def _three_categorical_reader():
    """Reader over (3, 2, 3, 5, 5): three categorical axes, a process, a continuous."""
    return _reader(
        state_names=("cat0", "cat1", "cat2", "shock", "wealth"),
        discrete_states={
            "cat0": DiscreteGrid(category_class=Three),
            "cat1": DiscreteGrid(category_class=Two),
            "cat2": DiscreteGrid(category_class=Three),
            "shock": _SHOCK,
        },
        continuous_states={"wealth": _WEALTH},
    )


def _V_arr(shape):
    """Deterministic, all-distinct values so a wrong node cannot pass by luck."""
    return jnp.arange(float(np.prod(shape))).reshape(shape) * 0.5 - 3.0


@pytest.mark.parametrize(
    "codes", [(0, 0, 0), (2, 1, 1), (1, 0, 2), (2, 1, 2), (0, 1, 0)]
)
def test_read_at_categorical_codes_and_exact_nodes_is_the_node_value(codes):
    """On-grid everywhere, the read is that node's value with no arithmetic on it."""
    reader = _three_categorical_reader()
    V_arr = _V_arr((3, 2, 3, 5, 5))
    result = reader(
        next_cat0=jnp.asarray(float(codes[0])),
        next_cat1=jnp.asarray(float(codes[1])),
        next_cat2=jnp.asarray(float(codes[2])),
        next_shock=jnp.asarray(1.0),  # process node index 3
        next_wealth=jnp.asarray(2.0),  # continuous node index 2
        V_arr=V_arr,
    )
    # An index read is exact: equality, not a tolerance.
    assert float(result) == float(V_arr[codes[0], codes[1], codes[2], 3, 2])


def test_read_off_grid_equals_interpolating_the_selected_categorical_slice():
    """Only the process and continuous axes are interpolated; a categorical selects."""
    reader = _three_categorical_reader()
    V_arr = _V_arr((3, 2, 3, 5, 5))
    codes = (2, 0, 1)
    result = reader(
        next_cat0=jnp.asarray(float(codes[0])),
        next_cat1=jnp.asarray(float(codes[1])),
        next_cat2=jnp.asarray(float(codes[2])),
        next_shock=jnp.asarray(-0.7),  # strictly between nodes -1 and 0
        next_wealth=jnp.asarray(2.25),  # strictly between nodes 2 and 3
        V_arr=V_arr,
    )
    # Independent reference: slice the categorical axes by hand, then interpolate
    # the two remaining axes with the same kernel every reader shares.
    expected = map_coordinates(
        input=V_arr[codes[0], codes[1], codes[2]],
        coordinates=[
            jnp.asarray(_SHOCK.get_coordinate(jnp.asarray(-0.7))),
            jnp.asarray(_WEALTH.get_coordinate(jnp.asarray(2.25))),
        ],
    )
    assert float(result) == float(expected)


def test_minus_inf_in_a_neighbouring_category_does_not_reach_the_read():
    """A category the read did not select contributes nothing, `-inf` included."""
    reader = _three_categorical_reader()
    V_arr = _V_arr((3, 2, 3, 5, 5)).at[1].set(-jnp.inf)
    result = reader(
        next_cat0=jnp.asarray(0.0),  # neighbour category 1 is all -inf
        next_cat1=jnp.asarray(1.0),
        next_cat2=jnp.asarray(2.0),
        next_shock=jnp.asarray(0.5),
        next_wealth=jnp.asarray(1.5),
        V_arr=V_arr,
    )
    assert bool(jnp.isfinite(result))


def test_read_at_a_minus_inf_category_stays_minus_inf():
    """Selecting a category that holds `-inf` reads `-inf`, not a blend."""
    reader = _three_categorical_reader()
    V_arr = _V_arr((3, 2, 3, 5, 5)).at[1].set(-jnp.inf)
    result = reader(
        next_cat0=jnp.asarray(1.0),
        next_cat1=jnp.asarray(1.0),
        next_cat2=jnp.asarray(2.0),
        next_shock=jnp.asarray(0.5),
        next_wealth=jnp.asarray(1.5),
        V_arr=V_arr,
    )
    assert float(result) == -jnp.inf


def test_one_categorical_axis_reads_the_named_category():
    """A single categorical axis alongside a process and a continuous axis."""
    reader = _reader(
        state_names=("cat0", "shock", "wealth"),
        discrete_states={"cat0": DiscreteGrid(category_class=Three), "shock": _SHOCK},
        continuous_states={"wealth": _WEALTH},
    )
    V_arr = _V_arr((3, 5, 5))
    result = reader(
        next_cat0=jnp.asarray(2.0),
        next_shock=jnp.asarray(-2.0),  # node index 0
        next_wealth=jnp.asarray(4.0),  # node index 4
        V_arr=V_arr,
    )
    assert float(result) == float(V_arr[2, 0, 4])


def test_no_categorical_axis_leaves_the_process_and_continuous_read_intact():
    """With no categorical axis the read is the plain multilinear interpolation."""
    reader = _reader(
        state_names=("shock", "wealth"),
        discrete_states={"shock": _SHOCK},
        continuous_states={"wealth": _WEALTH},
    )
    V_arr = _V_arr((5, 5))
    result = reader(
        next_shock=jnp.asarray(-0.25),
        next_wealth=jnp.asarray(3.75),
        V_arr=V_arr,
    )
    expected = map_coordinates(
        input=V_arr,
        coordinates=[
            jnp.asarray(_SHOCK.get_coordinate(jnp.asarray(-0.25))),
            jnp.asarray(_WEALTH.get_coordinate(jnp.asarray(3.75))),
        ],
    )
    assert float(result) == float(expected)


def test_pure_continuous_axes_extrapolate_linearly_beyond_the_node_range():
    """A continuous axis keeps its linear extrapolation outside the grid."""
    reader = _reader(
        state_names=("wealth",),
        discrete_states={},
        continuous_states={"wealth": _WEALTH},
    )
    V_arr = jnp.array([0.0, 2.0, 4.0, 6.0, 8.0])
    result = reader(next_wealth=jnp.asarray(6.0), V_arr=V_arr)
    # Slope 2 per node, so the value two nodes past the last one is 8 + 2*2.
    np.testing.assert_almost_equal(float(result), 12.0, decimal=DECIMAL_PRECISION)


def test_process_axis_read_beyond_the_node_range_clamps_to_the_edge_node():
    """A process value outside the discretized support reads the nearest node."""
    reader = _reader(
        state_names=("cat0", "shock"),
        discrete_states={"cat0": DiscreteGrid(category_class=Three), "shock": _SHOCK},
        continuous_states={},
    )
    V_arr = _V_arr((3, 5))
    result = reader(
        next_cat0=jnp.asarray(1.0),
        next_shock=jnp.asarray(50.0),  # far above the top node (2.0)
        V_arr=V_arr,
    )
    assert float(result) == float(V_arr[1, 4])


def test_fractional_categorical_value_is_poisoned_under_a_trace():
    """A value between two categories names no state, so the read is NaN."""
    reader = _three_categorical_reader()
    V_arr = _V_arr((3, 2, 3, 5, 5))

    def read(cat0):
        return reader(
            next_cat0=cat0,
            next_cat1=jnp.asarray(1.0),
            next_cat2=jnp.asarray(2.0),
            next_shock=jnp.asarray(0.0),
            next_wealth=jnp.asarray(2.0),
            V_arr=V_arr,
        )

    assert bool(jnp.isnan(jax.jit(read)(jnp.asarray(1.5))))
    # Positive control: the identical traced read at a genuine code is not NaN.
    assert not bool(jnp.isnan(jax.jit(read)(jnp.asarray(1.0))))


def test_fractional_categorical_value_raises_when_evaluated_concretely():
    """Outside a trace the same value raises and names the axis."""
    reader = _three_categorical_reader()
    V_arr = _V_arr((3, 2, 3, 5, 5))
    with pytest.raises(ValueError, match="next_cat0"):
        reader(
            next_cat0=jnp.asarray(1.5),
            next_cat1=jnp.asarray(1.0),
            next_cat2=jnp.asarray(2.0),
            next_shock=jnp.asarray(0.0),
            next_wealth=jnp.asarray(2.0),
            V_arr=V_arr,
        )


@pytest.mark.parametrize("code", [-1.0, 3.0, 7.0])
def test_categorical_code_outside_the_grid_is_poisoned_under_a_trace(code):
    """An integer outside the grid's codes names no category either."""
    reader = _three_categorical_reader()
    V_arr = _V_arr((3, 2, 3, 5, 5))

    def read(cat0):
        return reader(
            next_cat0=cat0,
            next_cat1=jnp.asarray(1.0),
            next_cat2=jnp.asarray(2.0),
            next_shock=jnp.asarray(0.0),
            next_wealth=jnp.asarray(2.0),
            V_arr=V_arr,
        )

    assert bool(jnp.isnan(jax.jit(read)(jnp.asarray(code))))
    assert not bool(jnp.isnan(jax.jit(read)(jnp.asarray(2.0))))


def test_categorical_code_outside_the_grid_raises_when_evaluated_concretely():
    """Outside a trace an out-of-range code raises and names the axis."""
    reader = _three_categorical_reader()
    V_arr = _V_arr((3, 2, 3, 5, 5))
    with pytest.raises(ValueError, match="next_cat0"):
        reader(
            next_cat0=jnp.asarray(3.0),
            next_cat1=jnp.asarray(1.0),
            next_cat2=jnp.asarray(2.0),
            next_shock=jnp.asarray(0.0),
            next_wealth=jnp.asarray(2.0),
            V_arr=V_arr,
        )


@pytest.mark.parametrize("codes", [(0, 0, 0), (2, 1, 2), (1, 0, 1)])
def test_pinning_an_axis_at_an_integral_coordinate_leaves_the_kernel_read_unchanged(
    codes,
):
    """Kernel level: a pinned axis drops only corners of weight exactly zero."""
    V_arr = _V_arr((3, 2, 3, 5, 5)).at[0, 1].set(-jnp.inf)
    coordinates = [
        jnp.asarray(float(codes[0])),
        jnp.asarray(float(codes[1])),
        jnp.asarray(float(codes[2])),
        jnp.asarray(1.4),
        jnp.asarray(3.2),
    ]
    unpinned = map_coordinates(input=V_arr, coordinates=coordinates)
    pinned = map_coordinates(
        input=V_arr, coordinates=coordinates, pinned_axes=(0, 1, 2)
    )
    assert float(pinned) == float(unpinned)


def test_pinning_no_axis_leaves_the_kernel_read_unchanged():
    """The default keeps every axis interpolated, so nothing moves."""
    V_arr = _V_arr((4, 6))
    coordinates = [jnp.asarray(1.25), jnp.asarray(4.75)]
    assert float(
        map_coordinates(input=V_arr, coordinates=coordinates, pinned_axes=())
    ) == float(map_coordinates(input=V_arr, coordinates=coordinates))


def test_pinned_kernel_gradient_matches_the_unpinned_gradient():
    """A pinned axis carries no derivative of its own, and none is lost elsewhere."""
    V_arr = _V_arr((3, 5))

    def read(*, wealth, pinned_axes):
        return map_coordinates(
            input=V_arr,
            coordinates=[jnp.asarray(1.0), wealth],
            pinned_axes=pinned_axes,
        )

    query = jnp.asarray(2.4)
    unpinned = jax.grad(lambda w: read(wealth=w, pinned_axes=()))(query)
    pinned = jax.grad(lambda w: read(wealth=w, pinned_axes=(0,)))(query)
    np.testing.assert_almost_equal(
        float(pinned), float(unpinned), decimal=DECIMAL_PRECISION
    )


def test_gradient_with_respect_to_V_arr_is_carried_by_the_read_nodes_only():
    """Differentiating an on-grid read puts unit weight on exactly that node."""
    reader = _three_categorical_reader()
    V_arr = _V_arr((3, 2, 3, 5, 5))
    codes = (2, 1, 0)
    grad = jax.grad(
        lambda arr: reader(
            next_cat0=jnp.asarray(float(codes[0])),
            next_cat1=jnp.asarray(float(codes[1])),
            next_cat2=jnp.asarray(float(codes[2])),
            next_shock=jnp.asarray(1.0),  # node index 3
            next_wealth=jnp.asarray(2.0),  # node index 2
            V_arr=arr,
        )
    )(V_arr)
    expected = jnp.zeros_like(V_arr).at[codes[0], codes[1], codes[2], 3, 2].set(1.0)
    np.testing.assert_array_equal(np.asarray(grad), np.asarray(expected))


def test_gradient_with_respect_to_a_continuous_state_is_the_local_slope():
    """The continuous axis stays differentiable through the categorical selection."""
    reader = _three_categorical_reader()
    V_arr = _V_arr((3, 2, 3, 5, 5))
    codes = (1, 0, 2)
    slope = jax.grad(
        lambda wealth: reader(
            next_cat0=jnp.asarray(float(codes[0])),
            next_cat1=jnp.asarray(float(codes[1])),
            next_cat2=jnp.asarray(float(codes[2])),
            next_shock=jnp.asarray(0.0),  # node index 2
            next_wealth=wealth,
            V_arr=V_arr,
        )
    )(jnp.asarray(2.25))
    selected = V_arr[codes[0], codes[1], codes[2], 2]
    expected = float(selected[3] - selected[2]) / 1.0  # one grid step per node
    np.testing.assert_almost_equal(float(slope), expected, decimal=DECIMAL_PRECISION)
