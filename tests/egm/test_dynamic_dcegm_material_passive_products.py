"""Retain a finite one-eighth contribution across two normal passive factors."""

import math
from fractions import Fraction
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from lcm.typing import ScalarFloat
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm import test_dynamic_dcegm_passive_products as query_cases

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _material_numbers() -> tuple[ScalarFloat, ScalarFloat, ScalarFloat]:
    """Keep normal factors and a normal final value around a subnormal joint weight."""
    dtype = np.dtype(jnp.asarray(1.0).dtype)
    assert dtype in (np.dtype(np.float32), np.dtype(np.float64))
    exponent, scale_exponent, minimum_exponent, subnormal_exponent = (
        (64, 126, 126, 149)
        if dtype == np.dtype(np.float32)
        else (512, 1022, 1022, 1074)
    )
    factor = Fraction(1, 2**exponent)
    scale_exact = Fraction(2**scale_exponent)
    joint = factor**2
    assert Fraction(1, 2**subnormal_exponent) <= joint
    assert joint < Fraction(1, 2**minimum_exponent)
    exact_value = joint * scale_exact / 2
    assert exact_value == Fraction(1, 8)
    epsilon = jnp.asarray(math.ldexp(1.0, -exponent), dtype=dtype)
    scale = jnp.asarray(math.ldexp(1.0, scale_exponent), dtype=dtype)
    expected = jnp.asarray(float(exact_value), dtype=dtype)
    for number in (epsilon, scale, 2 * scale, expected):
        assert number.shape == ()
        assert np.dtype(number.dtype) == dtype
        assert np.isfinite(number)
        assert number >= np.finfo(dtype).tiny
    np.testing.assert_array_equal(expected, 0.125)
    np.testing.assert_array_equal(
        epsilon * scale, math.ldexp(1.0, scale_exponent - exponent)
    )
    return epsilon, scale, expected


def test_canonical_dc_query_keeps_one_eighth_from_normal_passive_inputs(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read one eighth from the canonical query's retained finite corner."""
    monkeypatch.setattr(query_cases, "_case_numbers", _material_numbers)
    query_cases.test_canonical_dc_query_retains_normal_corner_terms_from_tiny_joint_weights()


def test_actual_dc_factory_keeps_one_eighth_from_computed_corner(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain one eighth through the actual producer and factory callbacks."""
    monkeypatch.setattr(query_cases, "_case_numbers", _material_numbers)
    query_cases.test_actual_dc_factory_retains_computed_tiny_passive_corner(
        tmp_path=tmp_path, monkeypatch=monkeypatch
    )
