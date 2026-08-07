"""The vectorized golden-section primitive — bracket-local, safeguarded.

The PR-3 gate battery from the continuous-outer plan: interior maxima to
golden-section accuracy, boundary maxima found *exactly* (the endpoint
safeguard — golden section alone can only approach a boundary), deterministic
smaller-abscissa tie-breaking on flat objectives, degenerate brackets,
heterogeneous vectorized brackets, invalid-mask propagation, NaN-probe
sanitization, JIT/vmap composition, and x64 determinism.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.optimization.golden_section import (
    _INV_PHI,
    GoldenSectionResult,
    maximize_golden_section,
)
from lcm.typing import FloatND

_ITERATIONS = 40


def test_interior_quadratic_maximum_to_golden_accuracy() -> None:
    """An interior parabola vertex is located to the bracket-shrink accuracy."""
    result = maximize_golden_section(
        lambda x: -((x - 0.3) ** 2),
        lower=jnp.array([0.0]),
        upper=jnp.array([1.0]),
        iterations=_ITERATIONS,
    )
    tol = float(_INV_PHI**_ITERATIONS)  # final bracket width for unit domain
    np.testing.assert_allclose(np.asarray(result.x), [0.3], atol=2 * tol)
    np.testing.assert_allclose(np.asarray(result.value), [0.0], atol=4 * tol**2)
    assert bool(result.valid[0])
    assert bool(result.converged[0])


@pytest.mark.parametrize(("slope", "argmax"), [(1.0, 1.0), (-1.0, 0.0)])
def test_boundary_maximum_is_found_exactly(slope: float, argmax: float) -> None:
    """A monotone objective's boundary maximum is returned EXACTLY.

    Golden section converges toward, never onto, a boundary; only the explicit
    endpoint evaluation makes the boundary abscissa itself a candidate.
    """
    result = maximize_golden_section(
        lambda x: slope * x,
        lower=jnp.array([0.0]),
        upper=jnp.array([1.0]),
        iterations=_ITERATIONS,
    )
    assert float(result.x[0]) == argmax
    assert float(result.value[0]) == slope * argmax


def test_flat_objective_ties_break_to_the_smaller_abscissa() -> None:
    """On a constant objective every candidate ties; the LOWER edge wins.

    The canonical outer tie rule is "choose the smaller outer action"; the
    selection folds candidates in increasing-abscissa order with a strict
    `>`, so the tie is deterministic, not float-noise-dependent.
    """
    result = maximize_golden_section(
        jnp.zeros_like,
        lower=jnp.array([2.0]),
        upper=jnp.array([5.0]),
        iterations=17,
    )
    assert float(result.x[0]) == 2.0
    assert float(result.value[0]) == 0.0


def test_degenerate_bracket_is_accepted() -> None:
    """`lower == upper` is a legal bracket: the point itself is returned."""
    result = maximize_golden_section(
        lambda x: -(x**2),
        lower=jnp.array([1.5]),
        upper=jnp.array([1.5]),
        iterations=8,
    )
    assert float(result.x[0]) == 1.5
    np.testing.assert_allclose(float(result.value[0]), -2.25)
    assert bool(result.valid[0])


def test_vectorized_heterogeneous_brackets() -> None:
    """Cells with different brackets and different maxima resolve per cell."""
    centers = jnp.array([0.25, 0.5, 2.0, -3.0])
    lower = jnp.array([0.0, 0.0, -1.0, -10.0])
    upper = jnp.array([1.0, 1.0, 5.0, 10.0])

    def objective(x: FloatND) -> FloatND:
        return -((x - centers) ** 2)

    result = maximize_golden_section(
        objective, lower=lower, upper=upper, iterations=_ITERATIONS
    )
    np.testing.assert_allclose(np.asarray(result.x), np.asarray(centers), atol=1e-6)


def test_invalid_mask_propagates_and_reports_neg_inf() -> None:
    """Masked cells are not searched: `x = lower`, `value = -inf`, invalid."""
    result = maximize_golden_section(
        lambda x: x,
        lower=jnp.array([0.0, 0.0]),
        upper=jnp.array([1.0, 1.0]),
        iterations=10,
        valid=jnp.array([True, False]),
    )
    assert bool(result.valid[0])
    assert not bool(result.valid[1])
    assert float(result.x[1]) == 0.0
    assert float(result.value[1]) == -jnp.inf
    assert float(result.x[0]) == 1.0  # the valid cell still optimizes


def test_inverted_bracket_is_invalid_not_an_error() -> None:
    """`upper < lower` marks the cell invalid rather than raising in-trace."""
    result = maximize_golden_section(
        lambda x: x,
        lower=jnp.array([1.0]),
        upper=jnp.array([0.0]),
        iterations=5,
    )
    assert not bool(result.valid[0])
    assert float(result.value[0]) == -jnp.inf


def test_nan_probes_are_treated_as_neg_inf_not_poison() -> None:
    """A NaN objective region must not poison the bracket (fmax semantics)."""

    def objective(x: FloatND) -> FloatND:
        # NaN left half, quadratic right half with the max at 0.75.
        return jnp.where(x < 0.5, jnp.nan, -((x - 0.75) ** 2))

    result = maximize_golden_section(
        objective,
        lower=jnp.array([0.0]),
        upper=jnp.array([1.0]),
        iterations=_ITERATIONS,
    )
    assert np.isfinite(float(result.value[0]))
    np.testing.assert_allclose(float(result.x[0]), 0.75, atol=1e-5)


def test_jit_and_vmap_compose() -> None:
    """The primitive JITs and vmaps (static iteration count, no Python
    branching on traced values)."""

    def run(center: FloatND) -> FloatND:
        result = maximize_golden_section(
            lambda x: -((x - center) ** 2),
            lower=jnp.zeros(()),
            upper=jnp.ones(()),
            iterations=30,
        )
        return result.x

    centers = jnp.array([0.2, 0.4, 0.6])
    jitted = jax.jit(jax.vmap(run))
    np.testing.assert_allclose(np.asarray(jitted(centers)), centers, atol=1e-5)


def test_x64_determinism_same_inputs_same_bits() -> None:
    """Two identical x64 runs agree bit for bit (no hidden randomness)."""
    if not jax.config.read("jax_enable_x64"):
        pytest.skip("x64 run only")

    def solve() -> GoldenSectionResult:
        return maximize_golden_section(
            lambda x: jnp.sin(3.0 * x) - 0.1 * x,
            lower=jnp.linspace(0.0, 0.5, 7),
            upper=jnp.linspace(1.5, 4.0, 7),
            iterations=51,
        )

    first, second = solve(), solve()
    np.testing.assert_array_equal(np.asarray(first.x), np.asarray(second.x))
    np.testing.assert_array_equal(np.asarray(first.value), np.asarray(second.value))


def test_width_tolerance_gates_converged() -> None:
    """`converged` reflects the requested final bracket width."""
    tight = maximize_golden_section(
        lambda x: -(x**2),
        lower=jnp.array([0.0]),
        upper=jnp.array([1.0]),
        iterations=40,
        width_tolerance=1e-6,
    )
    loose = maximize_golden_section(
        lambda x: -(x**2),
        lower=jnp.array([0.0]),
        upper=jnp.array([1.0]),
        iterations=3,
        width_tolerance=1e-6,
    )
    assert bool(tight.converged[0])
    assert not bool(loose.converged[0])


def test_negative_iterations_raise() -> None:
    """A negative static budget is a config error, caught eagerly."""
    with pytest.raises(ValueError, match="iterations"):
        maximize_golden_section(
            lambda x: x, lower=jnp.zeros(1), upper=jnp.ones(1), iterations=-1
        )
