"""Genuine native critical signs over original represented Hermite operands."""

from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hatch_build
from _lcm.egm.interp import _hermite_coefficients
from _lcm.egm.upper_envelope._exact_affine.ffi import UNRESOLVED_STATUS
from _lcm.egm.upper_envelope._exact_affine.hermite_critical import (
    HERMITE_CRITICAL_F32_LIMBS,
    HERMITE_CRITICAL_F64_LIMBS,
    hermite_critical_bernstein_signs,
)
from lcm.typing import FloatND, ReferenceName

_NAMES = ("x0", "h", "v0", "v1", "a", "b", "L", "A", "lower", "upper")
_BASE = (0.0, 1.0, 0.0, 4.0, 0.0, 0.0, -1.0, 0.125, 0.0, 1.0)
_NATIVE_COVERAGE = pytest.mark.coverage(
    backends=("cpu", "gpu-small", "gpu-large"), precisions="both"
)


def _arguments(*, values: tuple[float, ...]) -> dict[ReferenceName, FloatND]:
    return dict(zip(_NAMES, (jnp.asarray(value) for value in values), strict=True))


def _reference(*, arguments: Mapping[ReferenceName, FloatND]) -> tuple[int, ...]:
    """Evaluate the derivative identity independently as exact rational algebra."""
    x0, width, v0, v1, a, b, floor, scale, lower, upper = (
        Fraction(float(np.asarray(arguments[name]))) for name in _NAMES
    )
    c0 = v1 - v0 + a
    c1 = 2 * b - 4 * a
    c2 = 3 * (a - b)

    def polynomial(point: Fraction) -> Fraction:
        position = (point - x0) / width
        derivative = (c0 + c1 * position + c2 * position**2) / width
        return width**3 * ((point - floor) * derivative - scale)

    def derivative(point: Fraction) -> Fraction:
        position = point - x0
        return (
            c0 * width**2
            + c1 * (position + point - floor) * width
            + c2 * (position**2 + 2 * (point - floor) * position)
        )

    span = upper - lower
    controls = (
        3 * polynomial(lower),
        3 * polynomial(lower) + span * derivative(lower),
        3 * polynomial(upper) - span * derivative(upper),
        3 * polynomial(upper),
    )
    return tuple((value > 0) - (value < 0) for value in controls)


@_NATIVE_COVERAGE
@pytest.mark.parametrize(
    "values",
    [
        _BASE,
        (0.0, 1.0, 0.0, 1.5, 0.5, 0.5, -1.0, 2.25, 0.0, 1.0),
        (-1.0, 2.0, 3.0, -4.0, 7.0, -5.0, -2.0, 0.25, -0.75, 0.75),
        (0.0, 1.0, -4.0, 3.0, -5.0, 7.0, -2.0, 3.0, 0.125, 0.875),
    ],
)
def test_native_controls_match_independent_exact_algebra(
    *, values: tuple[float, ...]
) -> None:
    """Every integer coefficient magnitude participates in real native signs."""
    arguments = _arguments(values=values)
    signs, status = jax.jit(hermite_critical_bernstein_signs)(**arguments)
    assert int(status) == 0
    np.testing.assert_array_equal(np.asarray(signs), _reference(arguments=arguments))


@_NATIVE_COVERAGE
def test_tangent_without_endpoint_reversal_stays_unresolved_as_an_interval() -> None:
    """A repeated critical root is not excluded by equal endpoint signs."""
    arguments = _arguments(values=(0.0, 1.0, 0.0, 1.5, 0.5, 0.5, -1.0, 2.25, 0.0, 1.0))
    signs, status = hermite_critical_bernstein_signs(**arguments)
    assert int(status) == 0
    np.testing.assert_array_equal(np.asarray(signs), [-1, 1, 1, -1])


@_NATIVE_COVERAGE
def test_original_hermite_materialization_supplies_the_native_operands() -> None:
    """The actual limiter outputs define the cubic tested by the native handler."""
    arguments = _arguments(values=_BASE)
    lower, upper = jnp.asarray(2.0), jnp.asarray(-3.0)
    a, b, applicable = _hermite_coefficients(
        bracket_width=arguments["h"],
        safe_width=arguments["h"],
        fp_lower=arguments["v0"],
        fp_upper=arguments["v1"],
        slope_lower=lower,
        slope_upper=upper,
    )
    assert bool(applicable)
    assert np.isfinite(np.asarray(a))
    assert np.isfinite(np.asarray(b))
    arguments |= {"a": a, "b": b}
    signs, status = hermite_critical_bernstein_signs(**arguments)
    assert int(status) == 0
    np.testing.assert_array_equal(np.asarray(signs), _reference(arguments=arguments))


@_NATIVE_COVERAGE
def test_changed_dynamic_parameters_use_the_same_compiled_native_program() -> None:
    """Runtime scale, bound, width and correction changes are actual operands."""
    arguments = _arguments(values=_BASE)
    compiled = jax.jit(hermite_critical_bernstein_signs).lower(**arguments).compile()
    for changes in (
        {},
        {"A": jnp.asarray(16.0)},
        {"L": jnp.asarray(0.5)},
        {"h": jnp.asarray(2.0), "a": jnp.asarray(-3.0), "b": jnp.asarray(5.0)},
    ):
        actual = arguments | changes
        signs, status = compiled(**actual)
        assert int(status) == 0
        np.testing.assert_array_equal(np.asarray(signs), _reference(arguments=actual))


@_NATIVE_COVERAGE
@pytest.mark.parametrize("extreme", ["subnormal", "maximum", "cancellation"])
def test_extreme_finite_operands_retain_their_bits_and_exact_sign(
    *, extreme: str
) -> None:
    """Dyadic products neither flush subnormal bits nor overflow a float product."""
    dtype = np.asarray(jnp.asarray(0.0)).dtype
    tiny = np.nextafter(dtype.type(0.0), dtype.type(1.0))
    maximum = np.finfo(dtype).max
    if extreme == "subnormal":
        values = (0.0, tiny, -tiny, tiny, -tiny, tiny, -tiny, tiny, 0.0, tiny)
    elif extreme == "maximum":
        values = (
            -maximum,
            maximum,
            maximum,
            -maximum,
            maximum,
            -maximum,
            -maximum,
            maximum,
            -maximum,
            0.0,
        )
    else:
        values = (0.0, 1.0, maximum, maximum, 0.0, 0.0, 0.0, tiny, 0.25, 0.75)
    arguments = _arguments(values=tuple(float(value) for value in values))
    word = np.uint64 if dtype.itemsize == 8 else np.uint32
    for name, value in zip(_NAMES, values, strict=True):
        expected = np.asarray(value, dtype=dtype)
        actual = np.asarray(arguments[name])
        assert np.isfinite(actual)
        np.testing.assert_array_equal(actual.view(word), expected.view(word))
    signs, status = jax.jit(hermite_critical_bernstein_signs)(**arguments)
    assert int(status) == 0
    np.testing.assert_array_equal(np.asarray(signs), _reference(arguments=arguments))


@_NATIVE_COVERAGE
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("A", 0.0),
        ("A", -1.0),
        ("A", float("nan")),
        ("a", float("inf")),
        ("h", -0.0),
        ("lower", -1.0),
        ("upper", 2.0),
        ("upper", 0.0),
    ],
)
def test_native_invalid_input_has_mandatory_unresolved_status(
    *, name: ReferenceName, value: float
) -> None:
    """Invalid geometry or poison cannot publish apparently positive controls."""
    arguments = _arguments(values=_BASE) | {name: jnp.asarray(value)}
    signs, status = hermite_critical_bernstein_signs(**arguments)
    assert int(status) == UNRESOLVED_STATUS
    np.testing.assert_array_equal(np.asarray(signs), np.zeros(4, dtype=np.int32))


@_NATIVE_COVERAGE
def test_broadcasting_declares_the_actual_native_accumulator_workspace() -> None:
    """The lowered custom call includes its fixed accumulator result allocation."""
    arguments = _arguments(values=_BASE)
    arguments |= {
        "A": jnp.asarray([[0.125], [16.0]]),
        "upper": jnp.asarray([0.25, 0.5, 1.0]),
    }
    function = jax.jit(hermite_critical_bernstein_signs)
    lowered = function.lower(**arguments)
    signs, status = function(**arguments)
    assert signs.shape == (2, 3, 4)
    assert status.shape == (2, 3)
    assert np.all(np.asarray(status) == 0)
    expected = np.asarray([[[1] * 4] * 3, [[-1] * 4] * 3], dtype=np.int32)
    np.testing.assert_array_equal(np.asarray(signs), expected)
    limbs = (
        HERMITE_CRITICAL_F64_LIMBS
        if arguments["h"].dtype == jnp.float64
        else HERMITE_CRITICAL_F32_LIMBS
    )
    assert f"tensor<2x3x2x{limbs}xui32>" in str(lowered.compiler_ir())


def test_critical_wrapper_refuses_nonworking_float_dtype() -> None:
    """The native ABI accepts only its explicitly implemented float formats."""
    arguments = _arguments(values=_BASE)
    arguments = {name: value.astype(jnp.float16) for name, value in arguments.items()}
    with pytest.raises(TypeError, match="float32 or float64"):
        hermite_critical_bernstein_signs(**arguments)


def test_native_fingerprint_changes_on_a_header_only_edit(tmp_path: Path) -> None:
    """The compiled predicate cannot survive an unrecorded header-only change."""
    directory = tmp_path / hatch_build.PACKAGE_DIR
    directory.mkdir(parents=True)
    (tmp_path / "hatch_build.py").write_text("hook")
    (directory / "handler_symbols.py").write_text("symbols")
    header = directory / "exact_hermite_critical_core.h"
    header.write_text("first")
    before = hatch_build.native_source_fingerprint(root=tmp_path)
    header.write_text("second")
    assert hatch_build.native_source_fingerprint(root=tmp_path) != before


@_NATIVE_COVERAGE
def test_exact_endpoint_critical_roots_keep_zero_controls() -> None:
    """Endpoint contacts retain exact zeros beside a strictly positive interior."""
    arguments = _arguments(values=(0.0, 1.0, -0.0, 1.5, 0.5, 0.5, -1.0, 2.0, -0.0, 1.0))
    signs, status = hermite_critical_bernstein_signs(**arguments)
    assert int(status) == 0
    np.testing.assert_array_equal(np.asarray(signs), [0, 1, 1, 0])
    np.testing.assert_array_equal(np.asarray(signs), _reference(arguments=arguments))


def test_critical_wrapper_refuses_mixed_operand_dtypes() -> None:
    """Broadcasting never silently promotes a mismatched native operand."""
    arguments = _arguments(values=_BASE)
    arguments["a"] = arguments["a"].astype(jnp.float16)
    with pytest.raises(TypeError, match="share one dtype"):
        hermite_critical_bernstein_signs(**arguments)
