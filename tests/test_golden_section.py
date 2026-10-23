"""Behavioral tests for the bracket-local golden-section primitive.

The battery covers interior maxima to
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
        objective=lambda x: -((x - 0.3) ** 2),
        lower=jnp.array([0.0]),
        upper=jnp.array([1.0]),
        iterations=_ITERATIONS,
    )
    tol = float(_INV_PHI**_ITERATIONS)  # final bracket width for unit domain
    # ...but a bracket cannot shrink below the format's own resolution on the domain,
    # so under `--precision=32` (eps ~1.2e-7) the forty iterations stop resolving long
    # before the golden factor says they do, and `4 * tol**2` then asks for a value
    # accuracy no float32 search could deliver. Flooring `tol` at one epsilon of the
    # unit domain states the achievable accuracy at either precision and leaves float64
    # untouched: there 2.2e-16 is seven decades below the golden width.
    tol = max(tol, float(np.finfo(jnp.zeros(()).dtype).eps))
    np.testing.assert_allclose(np.asarray(result.x), [0.3], atol=2 * tol)
    np.testing.assert_allclose(np.asarray(result.value), [0.0], atol=4 * tol**2)
    assert bool(result.valid[0])
    assert bool(result.converged[0])


@pytest.mark.parametrize(("slope", "argmax"), [(1.0, 1.0), (-1.0, 0.0)])
def test_boundary_maximum_is_found_exactly(*, slope: float, argmax: float) -> None:
    """A monotone objective's boundary maximum is returned EXACTLY.

    Golden section converges toward, never onto, a boundary; only the explicit
    endpoint evaluation makes the boundary abscissa itself a candidate.
    """
    result = maximize_golden_section(
        objective=lambda x: slope * x,
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
        objective=jnp.zeros_like,
        lower=jnp.array([2.0]),
        upper=jnp.array([5.0]),
        iterations=17,
    )
    assert float(result.x[0]) == 2.0
    assert float(result.value[0]) == 0.0


def test_degenerate_bracket_is_accepted() -> None:
    """`lower == upper` is a legal bracket: the point itself is returned."""
    result = maximize_golden_section(
        objective=lambda x: -(x**2),
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
        objective=objective, lower=lower, upper=upper, iterations=_ITERATIONS
    )
    np.testing.assert_allclose(np.asarray(result.x), np.asarray(centers), atol=1e-6)


def test_invalid_mask_propagates_and_reports_neg_inf() -> None:
    """Masked cells are not searched: `x = lower`, `value = -inf`, invalid."""
    result = maximize_golden_section(
        objective=lambda x: x,
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
        objective=lambda x: x,
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
        objective=objective,
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
            objective=lambda x: -((x - center) ** 2),
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
            objective=lambda x: jnp.sin(3.0 * x) - 0.1 * x,
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
        objective=lambda x: -(x**2),
        lower=jnp.array([0.0]),
        upper=jnp.array([1.0]),
        iterations=40,
        width_tolerance=1e-6,
    )
    loose = maximize_golden_section(
        objective=lambda x: -(x**2),
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
            objective=lambda x: x, lower=jnp.zeros(1), upper=jnp.ones(1), iterations=-1
        )


def _search(
    *, path: str, objective, lower: FloatND, upper: FloatND, iterations: int
) -> GoldenSectionResult:
    """Run the plain search, or the status search with the abscissa as companion."""
    if path == "plain":
        return maximize_golden_section(
            objective=objective, lower=lower, upper=upper, iterations=iterations
        )
    return maximize_golden_section(
        objective=objective,
        lower=lower,
        upper=upper,
        iterations=iterations,
        objective_with_status=lambda x: (
            objective(x),
            (x,),
            jnp.zeros_like(x, dtype=bool),
        ),
    )


def _interior_plateau(x: FloatND) -> FloatND:
    """One on the open bracket (2, 5), zero at both endpoints."""
    return jnp.where((x > 2.0) & (x < 5.0), 1.0, 0.0)


def _equal_endpoints(x: FloatND) -> FloatND:
    """A parabola with exactly equal values at the endpoints -1 and 1."""
    return x * x


def _upper_shelf(x: FloatND) -> FloatND:
    """Zero below 0.5 and one from 0.5 up to and including the upper endpoint."""
    return jnp.where(x >= 0.5, 1.0, 0.0)


_TIE_CASES = {
    "interior": (_interior_plateau, 2.0, 5.0),
    "endpoints": (_equal_endpoints, -1.0, 1.0),
    "interior-and-upper": (_upper_shelf, 0.0, 1.0),
}


@pytest.mark.parametrize("path", ["plain", "status"])
def test_exact_interior_ties_keep_the_left_sub_bracket(*, path: str) -> None:
    """With `fa == fb` at every step the lower edge never moves."""
    result = _search(
        path=path,
        objective=_interior_plateau,
        lower=jnp.array([2.0]),
        upper=jnp.array([5.0]),
        iterations=17,
    )
    assert float(result.lower[0]) == 2.0
    assert float(result.upper[0]) < 2.01
    assert 2.0 < float(result.x[0]) <= float(result.upper[0])
    assert float(result.value[0]) == 1.0


@pytest.mark.parametrize("path", ["plain", "status"])
def test_exactly_equal_endpoints_select_the_lower_endpoint(*, path: str) -> None:
    """The strict `>` in the final selection keeps the lower of two equal ends."""
    result = _search(
        path=path,
        objective=_equal_endpoints,
        lower=jnp.array([-1.0]),
        upper=jnp.array([1.0]),
        iterations=12,
    )
    assert (float(result.x[0]), float(result.value[0])) == (-1.0, 1.0)


@pytest.mark.parametrize("path", ["plain", "status"])
def test_an_upper_endpoint_tied_with_an_interior_point_loses(*, path: str) -> None:
    """An interior probe equal to the upper endpoint's value is the smaller abscissa."""
    result = _search(
        path=path,
        objective=_upper_shelf,
        lower=jnp.array([0.0]),
        upper=jnp.array([1.0]),
        iterations=12,
    )
    assert 0.5 <= float(result.x[0]) < 1.0
    assert float(result.value[0]) == 1.0


@pytest.mark.parametrize("case", tuple(_TIE_CASES))
def test_plain_and_status_searches_agree_bitwise_on_ties(*, case: str) -> None:
    """Both entry points select, bracket and report the same bits on exact ties."""
    objective, lower, upper = _TIE_CASES[case]
    plain, status = (
        _search(
            path=path,
            objective=objective,
            lower=jnp.array([lower]),
            upper=jnp.array([upper]),
            iterations=12,
        )
        for path in ("plain", "status")
    )
    assert status.companions is not None
    actual = [status.x, status.value, status.lower, status.upper, status.companions[0]]
    expected = [plain.x, plain.value, plain.lower, plain.upper, plain.x]
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
