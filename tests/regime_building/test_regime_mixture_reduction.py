"""Class-level contract tests for the no-target-axis regime mixture reduction."""

import contextlib
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.Q_and_F import (
    _bitonic_value_order_network,
    _compare_swap_values,
    _normalized_regime_mixture,
    _sum_regime_mixture,
)
from _lcm.zero_safe import zero_safe_weighted_term
from lcm.typing import FloatND

_K_SWEEP = (1, 2, 3, 5, 8, 16)
_EXPECTED_NETWORK_COMPARISONS = {1: 0, 2: 1, 3: 3, 5: 9, 8: 24, 16: 80}


@contextlib.contextmanager
def _x64(*, enabled: bool):
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _bits(values: object) -> np.ndarray:
    arr = np.asarray(values)
    return arr.view(np.uint32 if arr.dtype == np.float32 else np.uint64)


def _apply_python_network(values: list[int]) -> list[int]:
    out = list(values)
    for left, right, ascending in _bitonic_value_order_network(len(out)):
        should_swap = out[left] > out[right] if ascending else out[left] < out[right]
        if should_swap:
            out[left], out[right] = out[right], out[left]
    return out


@pytest.mark.parametrize("k", _K_SWEEP)
def test_bitonic_network_sweep_is_correct_and_subquadratic(k: int):
    """The static trace uses the advertised bounded comparison network."""
    network = _bitonic_value_order_network(k)
    assert len(network) == _EXPECTED_NETWORK_COMPARISONS[k]
    assert _apply_python_network(list(reversed(range(k)))) == list(range(k))

    bubble_comparisons = k * (k - 1) // 2
    if k >= 5:
        assert len(network) < bubble_comparisons


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_compare_swap_network_matches_jax_total_order(precision: str):
    """NaNs, signed zeros, infinities, and duplicates retain JAX sort order."""
    use_x64 = precision == "fp64"
    with _x64(enabled=use_x64):
        dtype = jnp.float64 if use_x64 else jnp.float32
        values = jnp.asarray(
            [jnp.nan, 1.0, -0.0, jnp.inf, -jnp.inf, 0.0, 1.0, -2.0],
            dtype=dtype,
        )
        ordered = [values[i] for i in range(values.size)]
        for left, right, ascending in _bitonic_value_order_network(len(ordered)):
            ordered[left], ordered[right] = _compare_swap_values(
                ordered[left], ordered[right], ascending=ascending
            )

        got = jnp.stack(ordered)
        expected = jnp.sort(values)
        np.testing.assert_array_equal(_bits(got), _bits(expected))


def test_compare_swap_preserves_one_nan_and_one_finite_operand():
    """Unlike min/max, the production comparator remains a permutation."""
    nan = jnp.asarray(jnp.nan, dtype=jnp.float32)
    finite = jnp.asarray(1.0, dtype=jnp.float32)

    minimum = jnp.minimum(nan, finite)
    maximum = jnp.maximum(nan, finite)
    assert bool(jnp.isnan(minimum) & jnp.isnan(maximum))

    lower, upper = _compare_swap_values(nan, finite, ascending=True)
    assert float(lower) == 1.0
    assert bool(jnp.isnan(upper))


def test_compare_swap_preserves_two_distinct_nan_payloads():
    """The invalid NaN class may reorder, but no payload is duplicated or lost."""
    payload_bits = jnp.asarray([0x7FC00001, 0x7FC00002], dtype=jnp.uint32)
    payloads = jax.lax.bitcast_convert_type(payload_bits, jnp.float32)
    lower, upper = _compare_swap_values(payloads[0], payloads[1], ascending=True)
    got = np.sort(_bits(jnp.stack([lower, upper])))
    np.testing.assert_array_equal(got, np.sort(np.asarray(payload_bits)))


@pytest.mark.parametrize("k", _K_SWEEP)
def test_no_stack_reduction_is_permutation_invariant_and_within_roundoff(k: int):
    """Target order changes no bits and the old reducer differs only by roundoff."""
    rng = np.random.default_rng(20260823 + k)
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    numpy_dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    n_cells = 257

    raw_probabilities = rng.uniform(0.05, 1.0, size=(k, n_cells))
    probabilities = raw_probabilities / raw_probabilities.sum(axis=0, keepdims=True)
    exponents = rng.integers(
        -100 if numpy_dtype == np.float32 else -900,
        100 if numpy_dtype == np.float32 else 900,
        size=(k, n_cells),
    )
    values = np.ldexp(rng.uniform(-1.0, 1.0, size=(k, n_cells)), exponents)
    probabilities = probabilities.astype(numpy_dtype)
    values = values.astype(numpy_dtype)

    # Exercise zero-mass extended-real safety in one cell without changing mass.
    if k > 1:
        probabilities[0, 0] = 0.0
        probabilities[1:, 0] /= probabilities[1:, 0].sum()
        values[0, 0] = -np.inf

    p = jnp.asarray(probabilities, dtype=dtype)
    v = jnp.asarray(values, dtype=dtype)

    def production(probability: FloatND, value: FloatND) -> FloatND:
        terms = [(f"r{i}", probability[i], value[i]) for i in range(k)]
        return _sum_regime_mixture(terms, like=value[0])

    def oracle(probability: FloatND, value: FloatND) -> FloatND:
        # Literal pre-repair spelling: stack the unmultiplied operands, form one
        # vectorized zero-safe product, then reduce the target axis in value order.
        contributions = zero_safe_weighted_term(
            weight=probability,
            value=value,
            subnormal_is_accounted_for=False,
        )
        if k <= 2:
            return jnp.sum(contributions, axis=0)
        return jnp.sum(jnp.sort(contributions, axis=0), axis=0)

    got = jax.jit(production)(p, v)
    expected = jax.jit(oracle)(p, v)

    # The no-stack reduction has a source-level order; the old target-axis
    # `reduce` delegates its association and product fusion to backend codegen.
    # Their bits are therefore not a portable identity (JAX 0.11.1 differs from
    # 0.9.0.1 at K=2 and K=16), but both must remain within the ordinary
    # summand-scale floating-point bound.
    with np.errstate(invalid="ignore"):
        finite_values = np.where(
            (probabilities == 0) & ~np.isfinite(values),
            0,
            probabilities * values,
        )
    summand_scale = np.sum(np.abs(finite_values), axis=0, dtype=numpy_dtype)
    eps = np.finfo(numpy_dtype).eps
    smallest_normal = np.finfo(numpy_dtype).tiny
    tolerance = 4 * k * eps * summand_scale + smallest_normal
    assert np.all(np.abs(np.asarray(got) - np.asarray(expected)) <= tolerance)

    # Target declaration order and regime alpha-renaming are economically inert.
    # Exercise both a reversal and a rotation for every supported K class.
    for order in (
        np.arange(k - 1, -1, -1),
        np.roll(np.arange(k), 1),
    ):
        permuted = jax.jit(production)(p[order], v[order])
        np.testing.assert_array_equal(_bits(permuted), _bits(got))


def test_two_target_reduction_preserves_stored_product_bits():
    """The no-stack spelling must not contract a product into the final add."""
    probabilities = jnp.asarray(
        [0.42095494270324707, 0.5790450572967529], dtype=jnp.float32
    )
    values = jnp.asarray(
        [4.4386952338486274e-14, 7.219655679768875e-15], dtype=jnp.float32
    )
    got = _sum_regime_mixture(
        [("r0", probabilities[0], values[0]), ("r1", probabilities[1], values[1])],
        like=values[0],
    )
    expected = jnp.sum(
        zero_safe_weighted_term(
            weight=probabilities,
            value=values,
            subnormal_is_accounted_for=False,
        ),
        axis=0,
    )
    np.testing.assert_array_equal(_bits(got), _bits(expected))
    assert int(_bits(got)) == 684585997


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_two_target_value_order_is_invariant_across_separate_compilations(
    precision: str,
):
    """Swapping two valid targets cannot change mixture bits or the argmax."""
    use_x64 = precision == "fp64"
    with _x64(enabled=use_x64):
        dtype = jnp.float64 if use_x64 else jnp.float32
        if use_x64:
            p0 = 0.39074607530255345
            p1 = 0.6092539246974465
            v0 = 711.1235208253597
            v1 = -455.481208676875
            competing = 0.36501080552022324
        else:
            p0 = 0.41636961698532104
            p1 = 0.583630383014679
            v0 = -89.87742614746094
            v1 = 63.76618576049805
            competing = -0.20634535

        probabilities = jnp.asarray([p0, p1], dtype=dtype)
        values = jnp.asarray([v0, v1], dtype=dtype)

        @jax.jit
        def forward(probability: FloatND, value: FloatND) -> FloatND:
            return _sum_regime_mixture(
                [("r0", probability[0], value[0]), ("r1", probability[1], value[1])],
                like=value[0],
            )

        @jax.jit
        def reverse(probability: FloatND, value: FloatND) -> FloatND:
            return _sum_regime_mixture(
                [("r1", probability[1], value[1]), ("r0", probability[0], value[0])],
                like=value[0],
            )

        @jax.jit
        def oracle(probability: FloatND, value: FloatND) -> FloatND:
            contributions = zero_safe_weighted_term(
                weight=probability,
                value=value,
                subnormal_is_accounted_for=False,
            )
            return jnp.sum(jnp.sort(contributions), axis=0)

        got_forward = forward(probabilities, values)
        got_reverse = reverse(probabilities, values)
        expected = oracle(probabilities, values)
        np.testing.assert_array_equal(_bits(got_forward), _bits(got_reverse))
        np.testing.assert_array_equal(_bits(got_forward), _bits(expected))

        competitor = jnp.asarray(competing, dtype=dtype)
        assert bool((got_forward > competitor) == (got_reverse > competitor))


def test_two_target_compare_swap_preserves_duplicate_contribution_gradients():
    """The one-comparison K=2 path preserves the derivative of the full sum."""
    weights = jnp.asarray([0.5, 0.25])
    values = jnp.asarray([2.0, 4.0])

    def mixture(w: FloatND, v: FloatND) -> FloatND:
        return _sum_regime_mixture([("r0", w[0], v[0]), ("r1", w[1], v[1])], like=v[0])

    grad_weights, grad_values = jax.jit(jax.grad(mixture, argnums=(0, 1)))(
        weights, values
    )
    np.testing.assert_array_equal(np.asarray(grad_weights), np.asarray(values))
    np.testing.assert_array_equal(np.asarray(grad_values), np.asarray(weights))


def test_value_ordering_preserves_gradients_at_duplicate_contributions():
    """The compare keys are nondifferentiable; the selected values are not."""
    weights = jnp.asarray([0.5, 0.25, 0.25])
    values = jnp.asarray([2.0, 4.0, 4.0])  # every contribution equals one

    def mixture(w: FloatND, v: FloatND) -> FloatND:
        return _sum_regime_mixture([(f"r{i}", w[i], v[i]) for i in range(3)], like=v[0])

    grad_weights, grad_values = jax.jit(jax.grad(mixture, argnums=(0, 1)))(
        weights, values
    )
    np.testing.assert_array_equal(np.asarray(grad_weights), np.asarray(values))
    np.testing.assert_array_equal(np.asarray(grad_values), np.asarray(weights))


def test_invalid_nan_and_negative_probability_fail_visibly():
    live_nan = _sum_regime_mixture(
        [
            ("finite", jnp.asarray(0.5), jnp.asarray(2.0)),
            ("nan", jnp.asarray(0.5), jnp.asarray(jnp.nan)),
            ("other", jnp.asarray(0.0), jnp.asarray(-jnp.inf)),
        ],
        like=jnp.asarray(0.0),
    )
    assert bool(jnp.isnan(live_nan))

    negative_mass = _normalized_regime_mixture(
        mixture=jnp.asarray(1.0),
        probability_mass=jnp.asarray(1.0),
        has_negative_probability=jnp.ones((), dtype=bool),
    )
    assert bool(jnp.isnan(negative_mass))


def test_subnormal_regime_weight_keeps_its_finite_contribution():
    """Separate calls retain the exponent-moving zero-safe multiplication."""
    with _x64(enabled=False):
        weight = jnp.asarray(2.0**-128, dtype=jnp.float32)
        value = jnp.asarray(2.0**126, dtype=jnp.float32)
        got = _sum_regime_mixture(
            [("rare", weight, value)], like=jnp.asarray(0.0, dtype=jnp.float32)
        )
        assert float(got) == pytest.approx(0.25)


def test_optimized_hlo_has_no_materialized_target_axis():
    """Bounded CI proxy: inspect optimized HLO, not a mutable ASV timeout."""
    k = 3
    cell_shape = (64, 2, 32)
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    dtype_name = "f64" if jax.config.jax_enable_x64 else "f32"
    abstract = jax.ShapeDtypeStruct(cell_shape, dtype)

    def kernel(*args: FloatND) -> FloatND:
        probabilities = args[:k]
        values = args[k:]
        terms = [(f"r{i}", probabilities[i], values[i]) for i in range(k)]
        return _sum_regime_mixture(terms, like=values[0])

    compiled = jax.jit(kernel).lower(*([abstract] * (2 * k))).compile()
    optimized_hlo = compiled.as_text()
    assert optimized_hlo is not None
    full_axis_dimensions = (k, *cell_shape)
    forbidden_shapes = {
        f"{dtype_name}[{','.join(map(str, permutation))}]"
        for permutation in itertools.permutations(full_axis_dimensions)
    }

    assert not any(shape in optimized_hlo for shape in forbidden_shapes)
    assert " sort(" not in optimized_hlo

    memory_analysis = compiled.memory_analysis()
    assert memory_analysis is not None
    temp_bytes = memory_analysis.temp_size_in_bytes
    cell_bytes = (
        int(np.prod(cell_shape))
        * np.dtype(np.float64 if jax.config.jax_enable_x64 else np.float32).itemsize
    )
    assert temp_bytes <= 5 * cell_bytes


def test_full_shape_mixture_stays_within_profile_temporary_memory_budget():
    """The application-sized compile stays below the declared temporary budget."""
    k = 3
    cell_shape = (100_000, 2, 500)
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    abstract = jax.ShapeDtypeStruct(cell_shape, dtype)

    def kernel(*args: FloatND) -> FloatND:
        probabilities = args[:k]
        values = args[k:]
        terms = [(f"r{i}", probabilities[i], values[i]) for i in range(k)]
        return _sum_regime_mixture(terms, like=values[0])

    compiled = jax.jit(kernel).lower(*([abstract] * (2 * k))).compile()
    memory_analysis = compiled.memory_analysis()
    assert memory_analysis is not None
    budget_mib = 20 if jax.config.jax_enable_x64 else 10
    assert memory_analysis.temp_size_in_bytes <= budget_mib * 1024**2
