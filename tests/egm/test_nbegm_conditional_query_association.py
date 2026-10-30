"""Original saving follows each NB candidate family and streamed held corner."""

import inspect
import math
from collections.abc import Callable
from typing import Literal, cast

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm import nbegm_step
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from lcm.typing import FloatND, IntND
from tests.conftest import (
    DECIMAL_PRECISION,
    EXACT_KERNEL_SKIP_REASON,
    assert_agrees_to_ulp,
)
from tests.egm.test_nbegm_conditional_query_producer import _OwnedQueryStep, _QueryStep
from tests.solution._crra_preferences import crra_preferences
from tests.solution.test_nbegm_streamed_interval_envelope import (
    _PARTITION_ULP,
    _continuation,
    _geometry,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

type _OldOwned = tuple[FloatND, FloatND, FloatND, IntND]


def _same_owned_channels(*, result: _OwnedQueryStep, legacy: _OldOwned) -> EGMQueryData:
    """The retained tuple publishes the same ordinary channels and native owner."""
    assert isinstance(result, tuple)
    assert len(result) == 5
    for actual, expected in zip(
        (result.value, result.marginal, result.policy, result.owner),
        legacy,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert isinstance(result.query_data, EGMQueryData)
    return result.query_data


@pytest.mark.parametrize("route", ["empty", "interval"])
def test_original_saving_at_an_interior_euler_node(
    *, route: Literal["empty", "interval"]
) -> None:
    """At resource 3 the Euler candidate retains saving 1 and consumption 2."""
    liquid = jnp.asarray([0.75, 1.5, 3.0])
    savings = jnp.asarray([0.5, 1.0, 1.5])
    function = (
        nbegm_step.nbegm_multi_interval_step_savings
        if route == "empty"
        else nbegm_step.nbegm_per_interval_continuation_step_savings
    )
    continuation = 0.5 * savings
    marginal = jnp.full_like(savings, 0.5)
    arguments = {
        "liquid_grid": liquid,
        "savings_grid": savings,
        "discount_factor": jnp.asarray(1.0),
        "preferences": crra_preferences(crra=1.0),
        "cont_value": continuation if route == "empty" else continuation[None, :],
        "cont_marginal": marginal if route == "empty" else marginal[None, :],
        "coh_slopes": jnp.asarray([1.0]),
        "coh_intercepts": jnp.asarray([0.0]),
        "breakpoints": jnp.asarray([], dtype=liquid.dtype),
        "arithmetic": "certified",
    }
    legacy = cast("Callable[..., tuple[FloatND, FloatND, FloatND]]", function)(
        **arguments
    )
    for actual, expected in zip(
        legacy,
        (
            [math.log(0.25) + 0.25, 0.25, math.log(2.0) + 0.5],
            [4.0, 1.0, 0.5],
            [0.25, 1.0, 2.0],
        ),
        strict=True,
    ):
        np.testing.assert_array_almost_equal(
            actual, expected, decimal=DECIMAL_PRECISION
        )
    if route == "interval":
        owned = cast("Callable[..., _OldOwned]", function)(
            **arguments, return_owner=True
        )
        np.testing.assert_array_equal(owned[3][-1], 1)
    assert "retain_query" in inspect.signature(function).parameters
    result = cast("Callable[..., _QueryStep]", function)(**arguments, retain_query=True)
    for actual, expected in zip(
        (result.value, result.marginal, result.policy), legacy, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(result.query_data.savings, [0.5, 0.5, 1.0])


def test_upper_corner_keeps_the_original_maximum_saving() -> None:
    """The upper-savings candidate carries saving 40 and residual consumption 12."""
    liquid = jnp.linspace(0.5, 50.0, 40)
    savings = jnp.linspace(0.0, 40.0, 60)
    continuation = jnp.linspace(0.0, 30.0, 60)[None, :]
    arguments = {
        "liquid_grid": liquid,
        "savings_grid": savings,
        "cont_value": continuation,
        "cont_marginal": jnp.linspace(0.5, 0.02, 60)[None, :],
        "discount_factor": jnp.asarray(0.95),
        "preferences": crra_preferences(crra=2.0),
        "coh_slopes": jnp.asarray([1.0]),
        "coh_intercepts": jnp.asarray([2.0]),
        "breakpoints": jnp.asarray([], dtype=liquid.dtype),
        "arithmetic": "certified",
    }
    function = nbegm_step.nbegm_per_interval_continuation_step_savings
    legacy = cast("Callable[..., _OldOwned]", function)(**arguments, return_owner=True)
    assert np.isfinite(np.asarray(legacy[0])).all()
    consumption = 52.0 - np.asarray(savings)
    node_value = -1.0 / consumption + 0.95 * np.asarray(continuation[0])
    np.testing.assert_allclose(legacy[0][-1], node_value.max(), atol=1e-2, rtol=1e-2)
    np.testing.assert_array_almost_equal(legacy[2][-1], 12.0, decimal=DECIMAL_PRECISION)
    np.testing.assert_array_almost_equal(
        legacy[1][-1], 1.0 / 144.0, decimal=DECIMAL_PRECISION
    )
    assert 60 + 2 * 40 <= int(legacy[3][-1]) < 60 + 4 * 40
    assert "retain_query" in inspect.signature(function).parameters
    result = cast("Callable[..., _OwnedQueryStep]", function)(
        **arguments, retain_query=True, return_owner=True
    )
    data = _same_owned_channels(result=result, legacy=legacy)
    np.testing.assert_array_equal(data.savings[-1], savings[-1])


def test_flat_lower_corner_keeps_its_selected_nonzero_saving_node() -> None:
    """The flat interval's lower-corner record carries the dense optimum at node 19."""
    liquid = jnp.linspace(-5.0, 20.0, 60)
    savings = jnp.linspace(0.0, 18.0, 120)
    continuation = jnp.linspace(1.0, 5.0, 120)[None, :]
    arguments = {
        "liquid_grid": liquid,
        "savings_grid": savings,
        "cont_value": continuation,
        "cont_marginal": jnp.linspace(1.0, 0.2, 120)[None, :],
        "discount_factor": jnp.asarray(0.95),
        "preferences": crra_preferences(crra=2.0),
        "coh_slopes": jnp.asarray([0.0]),
        "coh_intercepts": jnp.asarray([5.0]),
        "breakpoints": jnp.asarray([], dtype=liquid.dtype),
        "arithmetic": "certified",
    }
    consumption = 5.0 - np.asarray(savings)
    feasible = consumption > 0.0
    node_value = np.where(
        feasible,
        -1.0 / np.where(feasible, consumption, 1.0)
        + 0.95 * np.asarray(continuation[0]),
        -np.inf,
    )
    assert int(np.argmax(node_value)) == 19
    function = nbegm_step.nbegm_per_interval_continuation_step_savings
    legacy = cast("Callable[..., _OldOwned]", function)(**arguments, return_owner=True)
    assert np.isfinite(np.asarray(legacy[0])).all()
    np.testing.assert_allclose(legacy[0], node_value.max(), atol=1e-4, rtol=1e-4)
    np.testing.assert_array_equal(legacy[1], 0.0)
    np.testing.assert_array_almost_equal(
        legacy[2], consumption[19], decimal=DECIMAL_PRECISION
    )
    assert bool(
        np.all((np.asarray(legacy[3]) >= 120) & (np.asarray(legacy[3]) < 120 + 2 * 60))
    )
    assert "retain_query" in inspect.signature(function).parameters
    result = cast("Callable[..., _OwnedQueryStep]", function)(
        **arguments, retain_query=True, return_owner=True
    )
    data = _same_owned_channels(result=result, legacy=legacy)
    np.testing.assert_array_equal(data.savings, savings[19])
    np.testing.assert_array_equal(data.corner_continuation, continuation[:, 0])


@pytest.mark.parametrize("streamed", [False, True])
def test_extra_target_keeps_its_own_interval_saving(*, streamed: bool) -> None:
    """A winning extra target carries saving 7/8 from the query's interval.

    The extra target sits 0.1 above the smooth continuation (a cliff jump), so it
    beats the interior Euler saving 5/6 that the smooth continuation alone yields.
    """
    liquid = jnp.asarray([0.75, 1.5, 3.0])
    savings = jnp.asarray([0.5, 1.0, 1.5])
    extra = jnp.asarray([[0.75], [0.875]])
    value = jnp.broadcast_to(jnp.log(3.0 * savings + 4.0), (2, 3))
    marginal = jnp.broadcast_to(3.0 / (3.0 * savings + 4.0), (2, 3))
    extra_value = jnp.log(3.0 * extra + 4.0) + 0.1

    def read(indices: IntND) -> tuple[FloatND, FloatND, FloatND]:
        return (
            jnp.concatenate((value[indices], extra_value[indices]), axis=-1),
            jnp.concatenate(
                (marginal[indices], 3.0 / (3.0 * extra[indices] + 4.0)), axis=-1
            ),
            extra[indices],
        )

    arguments = {
        "liquid_grid": liquid,
        "savings_grid": savings,
        "cont_value": None if streamed else value,
        "cont_marginal": None if streamed else marginal,
        "discount_factor": jnp.asarray(1.0),
        "preferences": crra_preferences(crra=1.0),
        "coh_slopes": jnp.asarray([1.0, 1.0]),
        "coh_intercepts": jnp.asarray([0.0, 0.0]),
        "breakpoints": jnp.asarray([1.0]),
        "extra_savings": None if streamed else extra,
        "extra_cont_value": None if streamed else extra_value,
        "interval_block_reader": read if streamed else None,
        "interval_width": 1 if streamed else 0,
        "arithmetic": "certified",
    }
    function = nbegm_step.nbegm_per_interval_continuation_step_savings
    legacy = cast("Callable[..., _OldOwned]", function)(**arguments, return_owner=True)
    for actual, expected in zip(
        (legacy[0][-1], legacy[1][-1], legacy[2][-1]),
        (math.log(901.0 / 64.0) + 0.1, 8.0 / 17.0, 17.0 / 8.0),
        strict=True,
    ):
        np.testing.assert_array_almost_equal(
            actual, expected, decimal=DECIMAL_PRECISION
        )
    assert "retain_query" in inspect.signature(function).parameters
    result = cast("Callable[..., _OwnedQueryStep]", function)(
        **arguments, retain_query=True, return_owner=True
    )
    data = _same_owned_channels(result=result, legacy=legacy)
    np.testing.assert_array_equal(data.savings[-1], extra[1, 0])
    np.testing.assert_array_equal(data.owner_id[-1], 1)


def test_partial_interval_block_keeps_every_original_corner_slot() -> None:
    """A width-two stream retains all seven held floor values without padding writes."""
    value, marginal = _continuation()
    function = nbegm_step.nbegm_per_interval_continuation_step_savings
    reference = cast("Callable[..., _OldOwned]", function)(
        **_geometry(), cont_value=value, cont_marginal=marginal, return_owner=True
    )

    def read(indices: IntND) -> tuple[FloatND, FloatND]:
        return value[indices], marginal[indices]

    arguments = {
        **_geometry(),
        "cont_value": None,
        "cont_marginal": None,
        "interval_block_reader": read,
        "interval_width": 2,
        "arithmetic": "certified",
    }
    legacy = cast("Callable[..., _OldOwned]", function)(**arguments, return_owner=True)
    for actual, expected in zip(legacy[:3], reference[:3], strict=True):
        assert_agrees_to_ulp(
            got=np.asarray(actual), expected=np.asarray(expected), n_ulp=_PARTITION_ULP
        )
    assert "retain_query" in inspect.signature(function).parameters
    result = cast("Callable[..., _OwnedQueryStep]", function)(
        **arguments, retain_query=True, return_owner=True
    )
    data = _same_owned_channels(result=result, legacy=legacy)
    assert data.corner_continuation is not None
    assert data.corner_continuation.shape == (7,)
    np.testing.assert_array_equal(data.corner_continuation, value[:, 0])
    np.testing.assert_array_equal(data.corner_kind, [QueryNodeKind.FINITE] * 7)
    np.testing.assert_array_equal(data.corner_interval_id, jnp.arange(7))
    np.testing.assert_array_equal(data.corner_side_id, [0] * 7)
