"""NB query rows retain original saving, native liquid marginal and held corners."""

import inspect
import math
from collections.abc import Callable
from typing import Literal, Protocol, cast

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.nbegm_step import (
    nbegm_multi_interval_step_savings,
    nbegm_per_interval_continuation_step_savings,
)
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from lcm.typing import FloatND, IntND
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.solution._crra_preferences import crra_preferences

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


class _QueryStep(Protocol):
    value: FloatND
    marginal: FloatND
    policy: FloatND
    query_data: EGMQueryData


class _OwnedQueryStep(_QueryStep, Protocol):
    owner: IntND


@pytest.mark.parametrize("route", ["empty", "interval", "streamed_interval"])
def test_nbegm_producer_keeps_owned_saving_and_interval_floor_continuation(
    *, route: Literal["empty", "interval", "streamed_interval"]
) -> None:
    """One floor belongs to every conditional tuple and its held interval corner."""
    liquid = jnp.asarray([0.75, 1.5, 3.0])
    savings = jnp.asarray([0.5, 1.0, 1.5])
    preferences = crra_preferences(crra=1.0)
    parameters = {
        "liquid_grid": liquid,
        "savings_grid": savings,
        "discount_factor": jnp.asarray(1.0),
        "preferences": preferences,
        "arithmetic": "certified",
    }
    if route == "empty":
        function = nbegm_multi_interval_step_savings
        continuation = jnp.log(3.0 * savings + 4.0)
        continuation_marginal = 3.0 / (3.0 * savings + 4.0)
        arguments = {
            **parameters,
            "cont_value": continuation,
            "cont_marginal": continuation_marginal,
            "coh_slopes": jnp.asarray([1.0]),
            "coh_intercepts": jnp.asarray([0.0]),
            "breakpoints": jnp.asarray([], dtype=liquid.dtype),
        }
        corner_values = [math.log(5.5)]
    else:
        function = nbegm_per_interval_continuation_step_savings
        intercepts = jnp.asarray([4.0, 7.0])[:, None]
        continuation = jnp.log(3.0 * savings[None, :] + intercepts)
        continuation_marginal = 3.0 / (3.0 * savings[None, :] + intercepts)

        def read(interval_indices: IntND) -> tuple[FloatND, FloatND]:
            return continuation[interval_indices], continuation_marginal[
                interval_indices
            ]

        arguments = {
            **parameters,
            "cont_value": None if route == "streamed_interval" else continuation,
            "cont_marginal": None
            if route == "streamed_interval"
            else continuation_marginal,
            "coh_slopes": jnp.asarray([1.0, 1.0]),
            "coh_intercepts": jnp.asarray([0.0, 0.0]),
            "breakpoints": jnp.asarray([1.0]),
            "coh_grid": liquid,
            "interval_block_reader": read if route == "streamed_interval" else None,
            "interval_width": 1 if route == "streamed_interval" else 0,
        }
        corner_values = [math.log(5.5), math.log(8.5)]
    # The strict derivative at the first actual saving node selects its own corner.
    assert -1.0 / (0.75 - 0.5) + 3.0 / 5.5 < 0.0
    legacy = cast("Callable[..., tuple[FloatND, FloatND, FloatND]]", function)(
        **arguments
    )
    assert len(legacy) == 3
    owned_legacy: tuple[FloatND, FloatND, FloatND, IntND] | None = None
    for actual, expected in zip(
        (legacy[0][0], legacy[1][0], legacy[2][0]),
        (math.log(11.0 / 8.0), 4.0, 0.25),
        strict=True,
    ):
        np.testing.assert_array_almost_equal(
            actual, expected, decimal=DECIMAL_PRECISION
        )
    if route != "empty":
        assert -1.0 / (1.5 - 0.5) + 3.0 / 8.5 < 0.0
        for actual, expected in zip(
            (legacy[0][1], legacy[1][1], legacy[2][1]),
            (math.log(17.0 / 2.0), 1.0, 1.0),
            strict=True,
        ):
            np.testing.assert_array_almost_equal(
                actual, expected, decimal=DECIMAL_PRECISION
            )
        read_owned = cast(
            "Callable[..., tuple[FloatND, FloatND, FloatND, IntND]]", function
        )
        owned_legacy = read_owned(**arguments, return_owner=True)
        assert len(owned_legacy) == 4
        for actual, expected in zip(owned_legacy[:3], legacy, strict=True):
            np.testing.assert_array_equal(actual, expected)
    assert "retain_query" in inspect.signature(function).parameters, (
        "The actual NB step does not retain its original conditional query tuple"
    )
    produce = cast("Callable[..., _QueryStep]", function)
    result = produce(**arguments, retain_query=True)
    assert isinstance(result, tuple)
    assert len(result) == 4
    for actual, expected in zip(
        (result.value, result.marginal, result.policy), legacy, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    data = result.query_data
    _assert_query_data(data=data, liquid=liquid, corner_values=corner_values)
    if route != "empty":
        np.testing.assert_array_equal(data.savings[1], 0.5)
        np.testing.assert_array_equal(data.owner_id[1], 1)
        np.testing.assert_array_equal(data.side_id[1], 0)
        np.testing.assert_array_equal(data.link_valid, [False, True])
        produce_owned = cast("Callable[..., _OwnedQueryStep]", function)
        owned = produce_owned(**arguments, retain_query=True, return_owner=True)
        _assert_owned_result(
            owned=owned, legacy=legacy, owned_legacy=owned_legacy, data=data
        )


def _assert_query_data(
    *, data: EGMQueryData, liquid: FloatND, corner_values: list[float]
) -> None:
    """The retained conditional tuple carries its original saving and held corners."""
    assert isinstance(data, EGMQueryData)
    np.testing.assert_array_equal(data.floor_used, 0.5)
    assert data.savings.shape == liquid.shape
    np.testing.assert_array_equal(data.savings[0], 0.5)
    np.testing.assert_array_equal(data.node_kind[0], QueryNodeKind.FINITE)
    np.testing.assert_array_equal(data.owner_id[0], 0)
    np.testing.assert_array_equal(data.side_id[0], 0)
    assert data.corner_continuation is not None
    assert data.corner_kind is not None
    assert data.corner_interval_id is not None
    assert data.corner_side_id is not None
    np.testing.assert_array_almost_equal(
        data.corner_continuation, corner_values, decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(
        data.corner_kind, [QueryNodeKind.FINITE] * len(corner_values)
    )
    np.testing.assert_array_equal(
        data.corner_interval_id, list(range(len(corner_values)))
    )
    np.testing.assert_array_equal(data.corner_side_id, [0] * len(corner_values))


def _assert_owned_result(
    *,
    owned: _OwnedQueryStep,
    legacy: tuple[FloatND, FloatND, FloatND],
    owned_legacy: tuple[FloatND, FloatND, FloatND, IntND] | None,
    data: EGMQueryData,
) -> None:
    """The owned tuple shares every ordinary channel and retained query leaf."""
    assert isinstance(owned, tuple)
    assert len(owned) == 5
    for actual, expected in zip(
        (owned.value, owned.marginal, owned.policy), legacy, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    assert owned_legacy is not None
    np.testing.assert_array_equal(owned.owner, owned_legacy[3])
    owned_leaves = owned.query_data.leaves()
    for path, expected in data.leaves().items():
        np.testing.assert_array_equal(owned_leaves[path], expected)
