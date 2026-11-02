"""Reading a collective continuation value at every stakeholder at once.

A collective target's value array carries a trailing stakeholder axis. The
continuation reads it at the landing coordinates, which do not depend on the
stakeholder, so the read locates the landing once and takes every stakeholder's
value from the same cells:

- the traced read holds as many gathers for any number of stakeholders as for
  one;
- each stakeholder's value is the singleton read of its own slice: the
  infeasible (`-inf`) and off-grid (NaN) reads exactly, the finite ones to within
  the few ULP that reading the same slice at another batch width already moves
  them, since the compiler vectorizes the read differently per width.
"""

from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend.core import ClosedJaxpr, Jaxpr

from _lcm.regime_building.Q_and_F import _get_stakeholder_sliced_interpolator
from _lcm.regime_building.V import VInterpolationInfo, get_V_interpolator
from lcm import DiscreteGrid, LinSpacedGrid, categorical
from lcm.typing import FloatND, ScalarInt
from tests.conftest import assert_agrees_to_ulp


@categorical(ordered=False)
class Work:
    no: ScalarInt
    yes: ScalarInt


WEALTH_GRID = LinSpacedGrid(start=0, stop=4, n_points=5)
HEALTH_GRID = LinSpacedGrid(start=-1, stop=1, n_points=4)

V_INTERPOLATION_INFO = VInterpolationInfo(
    state_names=("work", "wealth", "health"),
    discrete_states=MappingProxyType({"work": DiscreteGrid(Work)}),
    continuous_states=MappingProxyType({"wealth": WEALTH_GRID, "health": HEALTH_GRID}),
)

# Landings inside the grids, on a node, and outside them (extrapolated), with the
# last one at a discrete code no category carries.
NEXT_WORK = jnp.array([0, 1, 1, 0, 1, 2], dtype=jnp.int32)
NEXT_WEALTH = jnp.array([0.3, 2.0, 3.7, -0.5, 4.6, 1.0])
NEXT_HEALTH = jnp.array([-0.2, 1.0, 0.9, 0.1, -1.3, 0.0])


def _base_interpolator():
    return get_V_interpolator(
        v_interpolation_info=V_INTERPOLATION_INFO,
        state_prefix="next_",
        V_arr_name="next_V_arr",
    )


def _stakeholder_values(n_stakeholders: int) -> FloatND:
    """Distinct values per stakeholder, with one infeasible node each."""
    shape = (2, WEALTH_GRID.n_points, HEALTH_GRID.n_points, n_stakeholders)
    values = np.random.default_rng(seed=607).normal(size=shape)
    # Stakeholder `s` finds the state infeasible at a different wealth node.
    for s in range(n_stakeholders):
        values[1, 3 - s % 2, 2, s] = -np.inf
    return jnp.asarray(values, dtype=jnp.result_type(float))


def _read(*, interpolator, next_V_arr: FloatND) -> FloatND:
    """Read at every landing, one landing per call as the continuation does."""
    return jax.vmap(
        lambda work, wealth, health: interpolator(
            next_work=work,
            next_wealth=wealth,
            next_health=health,
            next_V_arr=next_V_arr,
        )
    )(
        NEXT_WORK,
        jnp.asarray(NEXT_WEALTH, dtype=next_V_arr.dtype),
        jnp.asarray(NEXT_HEALTH, dtype=next_V_arr.dtype),
    )


def _primitive_count(*, graph: object, name: str) -> int:  # noqa: PAN001 - Foreign IR parameters include arbitrary non-graph metadata.
    """Count one primitive across nested call bodies."""
    if isinstance(graph, ClosedJaxpr):
        graph = graph.jaxpr
    if not isinstance(graph, Jaxpr):
        return 0
    total = 0
    for equation in graph.eqns:
        total += equation.primitive.name == name
        for parameter in equation.params.values():
            children = (
                parameter if isinstance(parameter, tuple | list) else (parameter,)
            )
            total += sum(_primitive_count(graph=child, name=name) for child in children)
    return total


def _traced_read(n_stakeholders: int) -> ClosedJaxpr:
    interpolator = _get_stakeholder_sliced_interpolator(
        base_interpolator=_base_interpolator(),
        V_arr_name="next_V_arr",
        n_stakeholders=n_stakeholders,
    )
    return jax.make_jaxpr(
        lambda next_V_arr: _read(interpolator=interpolator, next_V_arr=next_V_arr)
    )(_stakeholder_values(n_stakeholders))


def test_collective_read_gathers_as_often_for_three_stakeholders_as_for_one():
    """The landing is located once; every stakeholder is read from its cells."""
    one = _traced_read(1)
    three = _traced_read(3)
    gathers_for_one = _primitive_count(graph=one, name="gather")
    assert gathers_for_one > 0
    assert _primitive_count(graph=three, name="gather") == gathers_for_one


@pytest.mark.parametrize("n_stakeholders", [1, 2, 3])
def test_each_stakeholder_reads_its_slice_as_a_singleton(
    n_stakeholders: int,
):
    """Stakeholder `s`'s value is the singleton read of `next_V_arr[..., s]`."""
    next_V_arr = _stakeholder_values(n_stakeholders)
    base = _base_interpolator()
    interpolator = _get_stakeholder_sliced_interpolator(
        base_interpolator=base,
        V_arr_name="next_V_arr",
        n_stakeholders=n_stakeholders,
    )

    got = np.asarray(_read(interpolator=interpolator, next_V_arr=next_V_arr))

    assert got.shape == (NEXT_WEALTH.shape[0], n_stakeholders)
    for s in range(n_stakeholders):
        expected = np.asarray(_read(interpolator=base, next_V_arr=next_V_arr[..., s]))
        assert_agrees_to_ulp(
            got=got[..., s],
            expected=expected,
            n_ulp=4,
            operand_magnitude=float(
                np.max(np.abs(next_V_arr[np.isfinite(next_V_arr)]))
            ),
            err_msg=f"stakeholder {s}",
        )
    # The witness covers what the read must carry through: an infeasible corner
    # under a nonzero weight, and a code off the discrete axis.
    assert np.isneginf(got).any()
    assert np.isnan(got[-1]).all()
