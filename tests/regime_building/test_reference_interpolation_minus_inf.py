"""F2: an exact finite node next to an on-path `-inf` must interpolate finitely.

The linear-interpolation kernel shared by every same-period-reference / gate /
fallback V reader (`_lcm.regime_building.V.get_V_interpolator`, both the
ordinary integer-lookup-plus-continuous-tail mode and the
`interpolate_process_axes=True` mode added for a non-folded process axis)
delegates to `_lcm.regime_building.ndimage.map_coordinates`. That kernel forms
`weight_product * corner_value` for EVERY corner of the interpolation box,
including a corner whose weight is exactly `0.0` — which happens not only at
an out-of-range extrapolation but at the ordinary case of querying EXACTLY AT
a grid node (one corner gets weight `1.0`, its neighbor `0.0`). If that
zero-weight neighbor holds an admissible on-path `-inf` (e.g. an infeasible
or zero-consumption reference cell), `0.0 * -inf = nan` used to leak into the
sum and turn an exact, finite, on-grid lookup into `nan`.

Both `get_V_interpolator` modes share the identical `map_coordinates` kernel,
so fixing it once covers every reader built on top of it — the same-period
reference reader (`Q_and_F.py:_build_same_period_ref_reader`), the gated-edge
gate reader, and the gated-edge fallback reader
(`processing.py:_attach_gated_edge_folds`) alike; this test exercises the
kernel through the shared `get_V_interpolator` entry point in both modes
rather than duplicating the full multi-regime plumbing of each reader.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.ndimage import map_coordinates
from _lcm.regime_building.V import VInterpolationInfo, get_V_interpolator
from lcm import LinSpacedGrid, NormalIIDProcess

# V_arr node values: an admissible on-path -inf sits immediately next to an
# exact finite node (index 1 = 5.0); index 2 (-inf) is the corner that must be
# read with an EXACT zero weight when querying exactly at index 1.
_V_ARR = jnp.array([0.0, 5.0, -jnp.inf, 15.0, 20.0])


def test_map_coordinates_exact_node_next_to_minus_inf_is_finite():
    """Kernel-level: querying exactly at an on-grid finite node stays finite."""
    # Coordinate 1.0 is exactly node index 1 (value 5.0); its zero-weight
    # neighbor (index 2) is -inf.
    result = map_coordinates(input=_V_ARR, coordinates=[jnp.array(1.0)])
    assert bool(jnp.isfinite(result))
    np.testing.assert_allclose(float(result), 5.0)


def test_map_coordinates_pure_minus_inf_node_stays_minus_inf():
    # Querying exactly at the -inf node itself (weight 1.0 there, 0.0 on its
    # finite neighbor) must still read -inf, not annihilate it: the guard
    # only zeroes a ZERO-weight term, never a genuine positive-weight -inf.
    result = map_coordinates(input=_V_ARR, coordinates=[jnp.array(2.0)])
    assert float(result) == -jnp.inf


def test_ordinary_interpolator_mode_finite_node_next_to_minus_inf_is_finite():
    """Same scenario through the production `get_V_interpolator` entry point.

    This is the reader used whenever a same-period reference / fallback V has
    no non-folded process axis (the ordinary continuation-value path).
    """
    grid = LinSpacedGrid(start=0.0, stop=4.0, n_points=5)  # nodes 0,1,2,3,4
    info = VInterpolationInfo(
        state_names=("x",),
        discrete_states=MappingProxyType({}),
        continuous_states=MappingProxyType({"x": grid}),
    )
    interpolator = get_V_interpolator(
        v_interpolation_info=info,
        state_prefix="next_",
        V_arr_name="V_arr",
    )
    # Query exactly at node value 1.0 (index 1, the finite node next to -inf).
    result = interpolator(next_x=jnp.array(1.0), V_arr=_V_ARR)
    assert bool(jnp.isfinite(result))
    np.testing.assert_allclose(float(result), 5.0)

    # The IR (interim-ratification-style) comparison stays feasible: an
    # own-stakeholder Q of 6 dominating the reference of 5 must compare
    # `True`, not `nan >= 5 == False`.
    Q_own = jnp.array(6.0)
    assert bool(Q_own >= result) is True


def test_process_aware_interpolator_mode_finite_node_next_to_minus_inf_is_finite():
    """Same scenario for a non-folded process axis (`interpolate_process_axes=True`).

    This is the mode a same-period reference / gate / fallback reader
    auto-selects when the reference regime carries a `_ContinuousStochasticProcess`
    state (e.g. a persisting wage shock) — see commits `1b33e86` (reference /
    fallback reader) and `311861b` (gate reader), both of which route through
    the identical `map_coordinates` kernel fixed here.
    """
    # Nodes: linspace(mu - n_std*sigma, mu + n_std*sigma, n_points) = [-2,-1,0,1,2].
    process = NormalIIDProcess(
        n_points=5, gauss_hermite=False, mu=0.0, sigma=1.0, n_std=2.0
    )
    info = VInterpolationInfo(
        state_names=("shock",),
        discrete_states=MappingProxyType({"shock": process}),
        continuous_states=MappingProxyType({}),
    )
    interpolator = get_V_interpolator(
        v_interpolation_info=info,
        state_prefix="next_",
        V_arr_name="V_arr",
        interpolate_process_axes=True,
    )
    # Node index 1 has value -1.0; its zero-weight neighbor (index 2, node
    # value 0.0) is -inf in `_V_ARR`.
    result = interpolator(next_shock=jnp.array(-1.0), V_arr=_V_ARR)
    assert bool(jnp.isfinite(result))
    np.testing.assert_allclose(float(result), 5.0)

    Q_own = jnp.array(6.0)
    assert bool(Q_own >= result) is True


@pytest.mark.parametrize("coordinate", [1.5, 2.5])
def test_genuine_fractional_neighbor_of_minus_inf_still_propagates_minus_inf(
    coordinate: float,
):
    """Sanity: a GENUINE (non-zero-weight) mixture with `-inf` still is `-inf`.

    The zero-safe guard must not silently mask an actual positive-mass
    `-inf` contribution — only an exact-zero-weight one. Querying strictly
    between the finite node (index 1) and the `-inf` node (index 2) gives
    both corners a nonzero weight, so the result is genuinely `-inf`.
    """
    result = map_coordinates(input=_V_ARR, coordinates=[jnp.array(coordinate)])
    assert float(result) == -jnp.inf
