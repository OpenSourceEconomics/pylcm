"""Augmented EGM rows preserve activity, exact tuples and original support gaps."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.one_asset_egm_step import (
    RetiredEGMResult,
    _EGMSavingsSupport,
    _prepare_augmented_savings_support,
    egm_one_asset_step,
)
from tests.conftest import DECIMAL_PRECISION
from tests.solution._crra_preferences import crra_preferences

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


def _step(
    *,
    savings: tuple[float, ...],
    values: tuple[float, ...],
    marginals: tuple[float, ...],
    cash: tuple[float, ...],
    support: _EGMSavingsSupport | None,
    compiled: bool,
    feature_floor: bool = True,
) -> RetiredEGMResult:
    """Run the numeric row with prepared identity-law continuation readings."""

    def solve() -> RetiredEGMResult:
        return egm_one_asset_step(
            next_value=jnp.asarray(values),
            next_marginal=jnp.asarray(marginals),
            liquid_grid=jnp.asarray(cash),
            next_liquid_grid=jnp.asarray(savings),
            savings_grid=jnp.asarray(savings),
            discount_factor=1.0,
            preferences=crra_preferences(crra=1.0),
            next_liquid=jnp.asarray(savings),
            marginal_return=jnp.ones(len(savings)),
            continuation_value=jnp.asarray(values),
            continuation_marginal=jnp.asarray(marginals),
            feature_floor=feature_floor,
            savings_support=support,
        )

    return jax.jit(solve)() if compiled else solve()


def _assert_channels(
    *, result: RetiredEGMResult, expected: tuple[tuple[float, ...], ...]
) -> None:
    """Check value, consumption, original saving and native marginal together."""
    assert result.savings is not None
    np.testing.assert_array_almost_equal(
        np.stack((result.value, result.consumption, result.savings, result.marginal)),
        expected,
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_inactive_corrupt_padding_preserves_the_finite_candidate_tuple(
    *, compiled: bool
) -> None:
    """Absent tail slots neither poison a finite row nor split its true links."""
    support = _EGMSavingsSupport(
        floor=jnp.asarray(0.0),
        active=jnp.asarray([True, True, True, False, False]),
        candidate_id=jnp.asarray([0, 1, 2, -1, -2], dtype=jnp.int32),
    )
    result = _step(
        savings=(0.0, 1.0, 2.0, 0.0, 0.0),
        values=(math.log(4.0), math.log(5.0), math.log(6.0), np.nan, np.inf),
        marginals=(0.25, 0.2, 1.0 / 6.0, np.nan, -1.0),
        cash=(2.0, 5.0, 7.0),
        support=support,
        compiled=compiled,
    )
    _assert_channels(
        result=result,
        expected=(
            (math.log(8.0), math.log(20.0), math.log(30.0)),
            (2.0, 4.5, 5.5),
            (0.0, 0.5, 1.5),
            (0.5, 2.0 / 9.0, 2.0 / 11.0),
        ),
    )
    np.testing.assert_array_equal(result.coverage_missing, [False, False, False])
    np.testing.assert_array_equal(result.rescore_interior, [False, True, True])


@pytest.mark.parametrize("compiled", [False, True])
def test_exact_original_boundary_duplicate_preserves_adjacent_interiors(
    *, compiled: bool
) -> None:
    """Identical canonical reads at one saving leave both neighboring links live."""
    savings = (0.0, 1.0, 1.0, 2.0)
    values = (math.log(4.0), math.log(5.0), math.log(5.0), math.log(6.0))
    marginals = (0.25, 0.2, 0.2, 1.0 / 6.0)
    support = _EGMSavingsSupport(
        floor=jnp.asarray(0.0),
        active=jnp.asarray([True, True, True, True]),
        candidate_id=jnp.asarray([0, 1, -1, 2], dtype=jnp.int32),
    )
    keep, rank = _prepare_augmented_savings_support(
        savings_grid=jnp.asarray(savings),
        next_liquid=jnp.asarray(savings),
        marginal_return=jnp.ones(4),
        value_next=jnp.asarray(values),
        marginal_next=jnp.asarray(marginals),
        savings_support=support,
    )
    np.testing.assert_array_equal(keep, [True, True, False, True])
    np.testing.assert_array_equal(rank, [0, 1, 1, 2])
    result = _step(
        savings=savings,
        values=values,
        marginals=marginals,
        cash=(2.0, 5.0, 7.0),
        support=support,
        compiled=compiled,
    )
    _assert_channels(
        result=result,
        expected=(
            (math.log(8.0), math.log(20.0), math.log(30.0)),
            (2.0, 4.5, 5.5),
            (0.0, 0.5, 1.5),
            (0.5, 2.0 / 9.0, 2.0 / 11.0),
        ),
    )
    np.testing.assert_array_equal(result.coverage_missing, [False, False, False])
    np.testing.assert_array_equal(result.rescore_interior, [False, True, True])


@pytest.mark.parametrize("changed", ["saving", "landing", "return", "value", "M"])
def test_one_ulp_tuple_difference_is_not_an_exact_duplicate(*, changed: str) -> None:
    """Equality folding requires every represented numeric field to agree exactly."""
    arrays = {
        "saving": jnp.asarray([0.0, 1.0, 1.0, 2.0]),
        "landing": jnp.asarray([0.0, 1.0, 1.0, 2.0]),
        "return": jnp.asarray([1.0, 1.0, 1.0, 1.0]),
        "value": jnp.log(jnp.asarray([4.0, 5.0, 5.0, 6.0])),
        "M": jnp.asarray([0.25, 0.2, 0.2, 1.0 / 6.0]),
    }
    array = arrays[changed]
    arrays[changed] = array.at[2].set(jnp.nextafter(array[2], jnp.inf))
    assert bool(arrays[changed][1] != arrays[changed][2])
    keep, rank = _prepare_augmented_savings_support(
        savings_grid=arrays["saving"],
        next_liquid=arrays["landing"],
        marginal_return=arrays["return"],
        value_next=arrays["value"],
        marginal_next=arrays["M"],
        savings_support=_EGMSavingsSupport(
            floor=jnp.asarray(0.0),
            active=jnp.asarray([True, True, True, True]),
            candidate_id=jnp.asarray([0, 1, -1, 2], dtype=jnp.int32),
        ),
    )
    np.testing.assert_array_equal(keep, [True, True, True, True])
    np.testing.assert_array_equal(rank, [0, 1, 2, 3])


@pytest.mark.parametrize("compiled", [False, True])
def test_economic_original_gap_survives_duplicate_folding_and_inactive_padding(
    *, compiled: bool
) -> None:
    """A missing original bracket stays missing while its intact neighbor survives."""
    result = _step(
        savings=(0.0, 1.0, 2.0, 2.0, 3.0, 0.0, 0.0),
        values=(
            0.0,
            -np.inf,
            math.log(3.0),
            math.log(3.0),
            math.log(4.0),
            np.nan,
            np.inf,
        ),
        marginals=(1.0, 0.0, 1.0, 1.0, 0.5, np.nan, -1.0),
        cash=(0.5, 2.0, 3.0, 4.0),
        support=_EGMSavingsSupport(
            floor=jnp.asarray(0.0),
            active=jnp.asarray([True, True, True, True, True, False, False]),
            candidate_id=jnp.asarray([0, 1, 2, -1, 3, -2, -3], dtype=jnp.int32),
        ),
        compiled=compiled,
    )
    np.testing.assert_array_almost_equal(
        result.value,
        [math.log(0.5), -np.inf, math.log(3.0), 0.5 * math.log(24.0)],
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        result.marginal, [2.0, 0.0, 1.0, 2.0 / 3.0], decimal=DECIMAL_PRECISION
    )
    assert result.savings is not None
    np.testing.assert_array_almost_equal(
        result.consumption[jnp.asarray([0, 2, 3])],
        [0.5, 1.0, 1.5],
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_almost_equal(
        result.savings[jnp.asarray([0, 2, 3])],
        [0.0, 2.0, 2.5],
        decimal=DECIMAL_PRECISION,
    )
    np.testing.assert_array_equal(result.coverage_missing, [False, True, False, False])
    np.testing.assert_array_equal(result.rescore_interior, [False, False, False, True])


@pytest.mark.parametrize("compiled", [False, True])
def test_distinct_savings_with_equal_cash_do_not_authorize_an_interval(
    *, compiled: bool
) -> None:
    """Distinct tuples sharing endogenous cash retain strict-order refusal."""
    savings = (0.0, 2.0, 4.0)
    marginals = (0.25, 0.5, 0.125)
    np.testing.assert_array_equal(
        1.0 / jnp.asarray(marginals) + jnp.asarray(savings), [4.0, 4.0, 12.0]
    )
    result = _step(
        savings=savings,
        values=(math.log(4.0), math.log(6.0), math.log(8.0)),
        marginals=marginals,
        cash=(2.0, 5.0),
        support=_EGMSavingsSupport(
            floor=jnp.asarray(0.0),
            active=jnp.asarray([True, True, True]),
            candidate_id=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        ),
        compiled=compiled,
    )
    np.testing.assert_array_almost_equal(
        result.value, [math.log(8.0), -np.inf], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_equal(result.marginal, [0.5, 0.0])
    np.testing.assert_array_equal(result.coverage_missing, [False, True])
    np.testing.assert_array_equal(result.rescore_interior, [False, False])


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("channel", ["value", "marginal"])
def test_active_corruption_survives_an_exact_finite_corner(
    *, compiled: bool, channel: str
) -> None:
    """A corrupt active original cannot disappear behind a healthy floor corner."""
    result = _step(
        savings=(0.0, 1.0, 2.0),
        values=(
            math.log(4.0),
            np.nan if channel == "value" else math.log(5.0),
            math.log(6.0),
        ),
        marginals=(0.25, np.nan if channel == "marginal" else 0.2, 1.0 / 6.0),
        cash=(2.0,),
        support=_EGMSavingsSupport(
            floor=jnp.asarray(0.0),
            active=jnp.asarray([True, True, True]),
            candidate_id=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        ),
        compiled=compiled,
    )
    np.testing.assert_array_equal(np.isnan(result.value), [True])
    np.testing.assert_array_equal(np.isnan(result.marginal), [True])
    np.testing.assert_array_equal(result.consumption, [2.0])
    np.testing.assert_array_equal(result.savings, [0.0])
    np.testing.assert_array_equal(result.coverage_missing, [False])
    np.testing.assert_array_equal(result.rescore_interior, [False])


@pytest.mark.parametrize("compiled", [False, True])
def test_absent_support_metadata_preserves_the_ordinary_numeric_row(
    *, compiled: bool
) -> None:
    """The ordinary row retains exact represented channels and absent extra leaves."""
    result = _step(
        savings=(0.0, 1.0, 2.0),
        values=(math.log(4.0), math.log(5.0), math.log(6.0)),
        marginals=(0.25, 0.2, 1.0 / 6.0),
        cash=(2.0, 5.0, 7.0),
        support=None,
        compiled=compiled,
        feature_floor=False,
    )
    _assert_channels(
        result=result,
        expected=(
            (math.log(8.0), math.log(20.0), math.log(30.0)),
            (2.0, 4.5, 5.5),
            (0.0, 0.5, 1.5),
            (0.5, 2.0 / 9.0, 2.0 / 11.0),
        ),
    )
    assert result.corner_continuation is None
    assert result.coverage_missing is None
    np.testing.assert_array_equal(result.rescore_interior, [False, True, True])
