"""Keep complete represented tuples when rounded Euler coordinates coincide."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.euler import invert_euler
from _lcm.egm.one_asset_egm_step import RetiredEGMResult, _solve_viable_feature_row
from _lcm.egm.preferences import Preferences
from lcm.typing import Float1D, FloatND
from tests.conftest import X64_ENABLED

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


def _reciprocal(value: FloatND) -> FloatND:
    return 1.0 / value


def _candidate_consumption(*, marginals: Float1D, compiled: bool) -> Float1D:
    def consume() -> Float1D:
        return invert_euler(
            expected_marginal_continuation=marginals,
            discount_factor=1.0,
            inverse_marginal_utility=_reciprocal,
        )

    return jax.jit(consume)() if compiled else consume()


def _read_row(
    *,
    savings: Float1D,
    values: Float1D,
    marginals: Float1D,
    cash: Float1D,
    compiled: bool,
) -> RetiredEGMResult:
    """Read the real feature-row consumer with original log preferences."""

    def read() -> RetiredEGMResult:
        return _solve_viable_feature_row(
            value_next=values,
            marginal_next=marginals,
            liquid_grid=cash,
            savings_grid=savings,
            discount_factor=1.0,
            preferences=Preferences(
                utility=jnp.log,
                marginal_utility=_reciprocal,
                inverse_marginal_utility=_reciprocal,
            ),
            marginal_return=jnp.ones_like(savings),
            next_liquid=savings,
        )

    return jax.jit(read)() if compiled else read()


def _assert_supported_tuple(
    *,
    result: RetiredEGMResult,
    values: Float1D,
    consumption: Float1D,
    savings: Float1D,
    marginals: Float1D,
    rescore: tuple[bool, ...],
    corner: float,
) -> None:
    assert result.coverage_missing is not None
    np.testing.assert_array_equal(result.coverage_missing, [False] * len(values))
    unsigned = np.uint64 if X64_ENABLED else np.uint32
    for actual, expected in (
        (result.value, values),
        (result.consumption, consumption),
        (result.savings, savings),
        (result.marginal, marginals),
    ):
        assert actual is not None
        np.testing.assert_array_equal(
            np.asarray(actual).view(unsigned), np.asarray(expected).view(unsigned)
        )
    np.testing.assert_array_equal(result.rescore_interior, rescore)
    np.testing.assert_array_equal(result.corner_continuation, corner)


@pytest.mark.parametrize("compiled", [False, True])
def test_adjacent_savings_with_rounded_equal_support_keep_the_right_tuple(
    *, compiled: bool
) -> None:
    """Read the rightmost equal-coordinate tuple and its next original segment."""
    dtype = np.float64 if X64_ENABLED else np.float32
    unsigned = np.uint64 if X64_ENABLED else np.uint32
    one = dtype(1.0)
    declared = np.asarray(
        [np.nextafter(one, dtype(-np.inf)), one, np.nextafter(one, dtype(np.inf)), 3.0],
        dtype=dtype,
    )
    savings = jnp.asarray(declared)
    np.testing.assert_array_equal(
        np.asarray(savings).view(unsigned), declared.view(unsigned)
    )
    np.testing.assert_array_equal(jnp.diff(savings) > 0.0, [True, True, True])
    consumption = _candidate_consumption(
        marginals=jnp.ones_like(savings), compiled=compiled
    )
    np.testing.assert_array_equal(consumption, [1.0, 1.0, 1.0, 1.0])
    endog = savings + consumption
    np.testing.assert_array_equal(
        np.asarray(endog).view(unsigned),
        np.asarray([2.0, 2.0, 2.0, 4.0], dtype=dtype).view(unsigned),
    )
    result = _read_row(
        savings=savings,
        values=savings,
        marginals=jnp.ones_like(savings),
        cash=jnp.asarray([2.0, 3.0]),
        compiled=compiled,
    )
    np.testing.assert_array_equal(result.consumption, [1.0, 1.0])
    _assert_supported_tuple(
        result=result,
        values=jnp.asarray([declared[2], 2.0]),
        consumption=jnp.asarray([1.0, 1.0]),
        savings=jnp.asarray([declared[2], 2.0]),
        marginals=jnp.asarray([1.0, 1.0]),
        rescore=(False, True),
        corner=float(declared[0]),
    )


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("trailing", [False, True], ids=["all-equal", "trailing"])
def test_exact_and_above_duplicate_endpoint_keep_one_complete_tuple(
    *, compiled: bool, trailing: bool
) -> None:
    """Exact and upper-clamped reads use the same final original endpoint."""
    dtype = np.float64 if X64_ENABLED else np.float32
    unsigned = np.uint64 if X64_ENABLED else np.uint32
    last = np.nextafter(dtype(1.5), dtype(np.inf))
    declared = np.asarray(([1.0] if trailing else []) + [1.5, last], dtype=dtype)
    savings = jnp.asarray(declared)
    np.testing.assert_array_equal(
        np.asarray(savings).view(unsigned), declared.view(unsigned)
    )
    np.testing.assert_array_equal(jnp.diff(savings) > 0.0, [True] * (len(declared) - 1))
    # W(A)=2A-(A-A_last)^2 is increasing and concave on this entire interval.
    values = 2.0 * savings - (savings - last) ** 2
    marginals = 2.0 - 2.0 * (savings - last)
    np.testing.assert_array_equal(
        marginals[-2:], [np.nextafter(dtype(2.0), dtype(np.inf)), dtype(2.0)]
    )
    consumption = _candidate_consumption(marginals=marginals, compiled=compiled)
    expected_consumption = np.asarray(
        [0.5 - np.finfo(dtype).eps / 2.0, 0.5], dtype=dtype
    )
    np.testing.assert_array_equal(
        np.asarray(consumption[-2:]).view(unsigned), expected_consumption.view(unsigned)
    )
    endog = savings + consumption
    expected_size = 3 if trailing else 2
    assert (
        savings.shape
        == values.shape
        == marginals.shape
        == consumption.shape
        == endog.shape
        == (expected_size,)
    )
    np.testing.assert_array_equal(jnp.isfinite(values), [True] * expected_size)
    np.testing.assert_array_equal(jnp.isfinite(endog), [True] * expected_size)
    candidate_values = jnp.log(consumption) + values
    for suffix in (savings[-2:], consumption[-2:], candidate_values[-2:]):
        np.testing.assert_array_equal(jnp.isfinite(suffix), [True, True])
        suffix_bits = np.asarray(suffix).view(unsigned)
        assert suffix_bits[0] != suffix_bits[1]
    np.testing.assert_array_equal(
        np.asarray(endog[-2:]).view(unsigned),
        np.asarray([2.0, 2.0], dtype=dtype).view(unsigned),
    )
    if trailing:
        assert bool(endog[0] < endog[1])
    result = _read_row(
        savings=savings,
        values=values,
        marginals=marginals,
        cash=jnp.asarray([2.0, 3.0]),
        compiled=compiled,
    )
    _assert_supported_tuple(
        result=result,
        values=jnp.repeat(jnp.log(jnp.asarray(0.5)) + values[-1], 2),
        consumption=jnp.asarray([0.5, 0.5]),
        savings=jnp.asarray([last, last]),
        marginals=jnp.asarray([2.0, 2.0]),
        rescore=(False, False),
        corner=float(values[0]),
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_true_endogenous_descent_remains_missing_coverage(*, compiled: bool) -> None:
    """A descending Euler map cannot certify an interior represented row."""
    savings = jnp.asarray([0.0, 1.0, 2.0])
    marginals = jnp.asarray([0.25, 1.0, 1.0])
    np.testing.assert_array_equal(
        savings + _candidate_consumption(marginals=marginals, compiled=compiled),
        [4.0, 2.0, 3.0],
    )
    result = _read_row(
        savings=savings,
        values=jnp.asarray([0.0, 1.0, 2.0]),
        marginals=marginals,
        cash=jnp.asarray([4.0]),
        compiled=compiled,
    )
    np.testing.assert_array_equal(result.coverage_missing, [True])
    np.testing.assert_array_equal(result.value, [-np.inf])
    np.testing.assert_array_equal(result.marginal, [0.0])
    np.testing.assert_array_equal(result.rescore_interior, [False])


@pytest.mark.parametrize("compiled", [False, True])
def test_dead_original_node_still_prevents_bridging_its_gap(*, compiled: bool) -> None:
    """Finite endpoints separated by an impossible original node form no segment."""
    result = _read_row(
        savings=jnp.asarray([0.0, 1.0, 2.0]),
        values=jnp.asarray([0.0, -np.inf, 2.0]),
        marginals=jnp.asarray([1.0, 0.0, 1.0]),
        cash=jnp.asarray([2.0]),
        compiled=compiled,
    )
    np.testing.assert_array_equal(result.coverage_missing, [True])
    np.testing.assert_array_equal(result.value, [-np.inf])
    np.testing.assert_array_equal(result.marginal, [0.0])
    np.testing.assert_array_equal(result.rescore_interior, [False])


@pytest.mark.parametrize("compiled", [False, True])
def test_isolated_exact_node_and_upper_clamp_keep_the_original_tuple(
    *, compiled: bool
) -> None:
    """An isolated viable endpoint supports its exact coordinate and upper clamp."""
    result = _read_row(
        savings=jnp.asarray([0.0, 1.0, 2.0]),
        values=jnp.asarray([-np.inf, 2.0, -np.inf]),
        marginals=jnp.asarray([0.0, 1.0, 0.0]),
        cash=jnp.asarray([2.0, 3.0]),
        compiled=compiled,
    )
    _assert_supported_tuple(
        result=result,
        values=jnp.asarray([2.0, 2.0]),
        consumption=jnp.asarray([1.0, 1.0]),
        savings=jnp.asarray([1.0, 1.0]),
        marginals=jnp.asarray([1.0, 1.0]),
        rescore=(False, False),
        corner=-np.inf,
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_active_nan_continuation_still_invalidates_a_finite_floor_corner(
    *, compiled: bool
) -> None:
    """A finite supported corner cannot hide invalid active continuation values."""
    result = _read_row(
        savings=jnp.asarray([0.0, 1.0, 2.0]),
        values=jnp.asarray([0.0, np.nan, 2.0]),
        marginals=jnp.asarray([1.0, 1.0, 1.0]),
        cash=jnp.asarray([0.5]),
        compiled=compiled,
    )
    np.testing.assert_array_equal(jnp.isnan(result.value), [True])
    np.testing.assert_array_equal(jnp.isnan(result.marginal), [True])
    np.testing.assert_array_equal(result.consumption, [0.5])
    np.testing.assert_array_equal(result.savings, [0.0])
    np.testing.assert_array_equal(result.coverage_missing, [False])
    np.testing.assert_array_equal(result.rescore_interior, [False])
