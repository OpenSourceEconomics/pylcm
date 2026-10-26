"""Dimensionless accuracy and explicit admission for normal-density sensitivities."""

import math
from fractions import Fraction
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.processes import density_derivative_admission as admission
from _lcm.processes.ar1 import _TauchenOnNodesProcess
from lcm import TauchenAR1Process
from lcm.typing import FloatND, ScalarFloat
from tests.conftest import DECIMAL_PRECISION, X64_ENABLED
from tests.test_models.pr626_density_sensitivity_oracle import (
    density_pairwise_derivative,
)

type _Parameter = Literal["mu", "rho", "source", "sigma"]
type _Method = Literal["jvp", "jacfwd", "jacrev"]
type _Family = Literal["ordinary", "central", "higher_moment"]


class _DensityInputs(NamedTuple):
    nodes: tuple[float, ...]
    source: float
    rho: float
    mu: float
    sigma: float


def _inputs(
    *, family: _Family, physical_scale: float, nonzero_source: bool = False
) -> _DensityInputs:
    if family == "ordinary":
        nodes, source, rho, mu, sigma = (-1.4, -0.1, 0.7, 1.2, 2.0), 0.25, 0.3, 0.2, 0.8
    else:
        a = 2.0 ** (-1000 if X64_ENABLED else -100)
        sigma = 2.0 ** (-950 if X64_ENABLED else -80)
        multipliers = (
            (-1, 0, 1)
            if family == "central"
            else tuple(
                sorted((0, 3, 5, 6, 9, 10, 12, 15, -1, -2, -4, -7, -8, -11, -13, -14))
            )
        )
        nodes = tuple(value * a for value in multipliers)
        source, rho, mu = (
            (1.0 if nonzero_source else 0.0),
            0.0,
            (a / 8 if family == "central" else 0.0),
        )
    dtype = np.float64 if X64_ENABLED else np.float32
    return _DensityInputs(
        tuple(
            float(dtype(physical_scale * value))
            for value in sorted(nodes, reverse=physical_scale < 0)
        ),
        float(dtype(physical_scale * source)),
        float(dtype(rho)),
        float(dtype(physical_scale * mu)),
        float(dtype(abs(physical_scale) * sigma)),
    )


def _derivative(
    *, case: _DensityInputs, parameter: _Parameter, method: _Method, compiled: bool
) -> FloatND:
    dtype = jnp.float64 if X64_ENABLED else jnp.float32
    process = TauchenAR1Process.on_nodes(
        nodes=np.asarray(case.nodes), transition_rule="density"
    )
    assert isinstance(process, _TauchenOnNodesProcess)
    point = jnp.asarray(getattr(case, parameter), dtype=dtype)

    def evaluate(value: ScalarFloat) -> FloatND:
        return process.compute_transition_row(
            value if parameter == "source" else jnp.asarray(case.source, dtype=dtype),
            rho=value if parameter == "rho" else jnp.asarray(case.rho, dtype=dtype),
            mu=value if parameter == "mu" else jnp.asarray(case.mu, dtype=dtype),
            sigma=value
            if parameter == "sigma"
            else jnp.asarray(case.sigma, dtype=dtype),
        )

    if method == "jvp":

        def differentiate(value: ScalarFloat) -> FloatND:
            return jax.jvp(evaluate, (value,), (jnp.ones_like(value),))[1]
    else:
        differentiate = (
            jax.jacfwd(evaluate) if method == "jacfwd" else jax.jacrev(evaluate)
        )
    operation = jax.jit(differentiate) if compiled else differentiate
    actual = operation(point)
    jax.block_until_ready(actual)
    jax.effects_barrier()
    return actual


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("method", ["jvp", "jacfwd", "jacrev"])
@pytest.mark.parametrize(
    "physical_scale", [1.0, 16.0, -1.0], ids=["original_units", "rescaled", "reflected"]
)
@pytest.mark.parametrize(
    ("family", "parameter"),
    [
        ("ordinary", "mu"),
        ("ordinary", "rho"),
        ("ordinary", "source"),
        ("ordinary", "sigma"),
        ("central", "mu"),
        ("central", "sigma"),
        ("higher_moment", "mu"),
    ],
)
def test_density_sensitivities_meet_the_dimensionless_error_budget(
    *,
    family: _Family,
    parameter: _Parameter,
    physical_scale: float,
    method: _Method,
    compiled: bool,
) -> None:
    case = _inputs(family=family, physical_scale=physical_scale)
    expected = np.asarray(
        density_pairwise_derivative(
            nodes=case.nodes,
            source=case.source,
            rho=case.rho,
            mu=case.mu,
            sigma=case.sigma,
            parameter=parameter,
        )
    )
    actual = np.asarray(
        _derivative(case=case, parameter=parameter, method=method, compiled=compiled)
    )
    scale = 1.0 if parameter == "rho" else case.sigma
    tau = 10.0**-DECIMAL_PRECISION
    assert np.isfinite(actual).all()
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(actual * scale, expected * scale, atol=tau, rtol=tau)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("method", ["jvp", "jacfwd", "jacrev"])
@pytest.mark.parametrize(
    "physical_scale", [1.0, 16.0], ids=["original_units", "rescaled"]
)
def test_density_rejects_an_uncertified_dimensionless_rho_sensitivity(
    *,
    physical_scale: float,
    method: _Method,
    compiled: bool,
) -> None:
    case = _inputs(family="central", physical_scale=physical_scale, nonzero_source=True)
    expected = np.asarray(
        density_pairwise_derivative(
            nodes=case.nodes,
            source=case.source,
            rho=case.rho,
            mu=case.mu,
            sigma=case.sigma,
            parameter="rho",
        )
    )
    tau = 10.0**-DECIMAL_PRECISION
    assert np.isfinite(expected).all()
    assert abs(expected[1]) > tau + tau * abs(expected[1])
    with pytest.raises((ArithmeticError, RuntimeError), match=r"(?i)(accuracy|certif)"):
        _derivative(case=case, parameter="rho", method=method, compiled=compiled)


def _word_fraction(word: tuple[float, int]) -> Fraction:
    return Fraction(word[0]) * Fraction(2) ** word[1]


@pytest.mark.parametrize("operation", ["add", "multiply", "divide"])
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((0.75, 2000), (0.625, -2000)),
        ((-0.5000000000000001, -2000), (0.9999999999999999, 2000)),
        ((0.9999999999999999, 1), (0.5000000000000001, 1)),
        ((-0.75, -1074), (0.5, -1074)),
        ((0.0, -100000), (0.75, 1000)),
    ],
)
def test_density_error_enclosures_contain_independent_rational_operations(
    *,
    operation: Literal["add", "multiply", "divide"],
    left: tuple[float, int],
    right: tuple[float, int],
) -> None:
    left_radius = (0.5, left[1] - 54) if left[0] else (0.0, -100000)
    right_radius = (0.5, right[1] - 54)
    a, b = _word_fraction(left), _word_fraction(right)
    ra, rb = _word_fraction(left_radius), _word_fraction(right_radius)
    if operation == "add":
        result = admission._add((left, left_radius), (right, right_radius))
        endpoints = (a - ra + b - rb, a + ra + b + rb)
    elif operation == "multiply":
        result = admission._multiply(
            left=(left, left_radius), right=(right, right_radius)
        )
        endpoints = tuple(x * y for x in (a - ra, a + ra) for y in (b - rb, b + rb))
    else:
        result = admission._divide(
            left=(left, left_radius), right=(right, right_radius)
        )
        endpoints = tuple(x / y for x in (a - ra, a + ra) for y in (b - rb, b + rb))
    center, radius = (_word_fraction(value) for value in result)
    assert center - radius <= min(endpoints) <= max(endpoints) <= center + radius


def test_density_logarithm_constant_encloses_the_exact_series() -> None:
    z = Fraction(1, 3)
    lower = 2 * sum((z ** (2 * k + 1) / (2 * k + 1) for k in range(64)), Fraction(0))
    remainder = 2 * z**129 / (129 * (1 - z * z))
    high = Fraction(0.6931471805599453)
    radius = Fraction(1, 2**55)
    assert high - radius <= lower <= lower + remainder <= high + radius


@pytest.mark.parametrize("family", ["ordinary", "central", "higher_moment"])
def test_density_assembled_bounds_cover_the_independent_law(
    *,
    family: _Family,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _inputs(family=family, physical_scale=1.0)
    recorded: list[tuple[tuple[float, int], tuple[float, int]]] = []
    original = admission._accuracy_bound

    def capture(
        *,
        candidate: tuple[float, int],
        enclosure: tuple[tuple[float, int], tuple[float, int]],
        scale: tuple[float, int],
        epsilon: float,
        min_normal_exponent: int,
    ) -> tuple[tuple[float, int], tuple[float, int]]:
        result = original(
            candidate=candidate,
            enclosure=enclosure,
            scale=scale,
            epsilon=epsilon,
            min_normal_exponent=min_normal_exponent,
        )
        recorded.append(result)
        return result

    monkeypatch.setattr(admission, "_accuracy_bound", capture)
    actual = np.asarray(
        _derivative(case=case, parameter="mu", method="jvp", compiled=False)
    )
    assert len(recorded) == 4 * len(case.nodes)
    for column, parameter in enumerate(("mu", "rho", "source", "sigma")):
        expected = density_pairwise_derivative(
            nodes=case.nodes,
            source=case.source,
            rho=case.rho,
            mu=case.mu,
            sigma=case.sigma,
            parameter=parameter,
        )
        scale = Fraction(1.0 if parameter == "rho" else case.sigma)
        for index, reference in enumerate(expected):
            error_word, center_word = recorded[column * len(case.nodes) + index]
            error, center = _word_fraction(error_word), _word_fraction(center_word)
            target = Fraction(reference) * scale
            # The oracle returns a binary64 rounding of its high-precision result.
            oracle_rounding = (
                Fraction(math.ulp(reference)) * abs(scale) if reference else Fraction(0)
            )
            assert abs(center - target) <= error + oracle_rounding
            if parameter == "mu":
                assert (
                    abs(Fraction(float(actual[index])) * scale - target)
                    <= error + oracle_rounding
                )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("method", ["jvp", "jacfwd", "jacrev"])
def test_density_mapped_ambiguity_rejects_zero_source_sensitivities(
    *,
    compiled: bool,
    method: _Method,
) -> None:
    case = _inputs(family="central", physical_scale=1.0)
    process = TauchenAR1Process.on_nodes(
        nodes=np.asarray(case.nodes), transition_rule="density"
    )
    assert isinstance(process, _TauchenOnNodesProcess)

    def evaluate(value: ScalarFloat) -> FloatND:
        return process.compute_transition_row(
            value,
            rho=jnp.asarray(0.0),
            mu=jnp.asarray(case.mu),
            sigma=jnp.asarray(case.sigma),
        )

    if method == "jvp":

        def differentiate(value: ScalarFloat) -> FloatND:
            return jax.jvp(evaluate, (value,), (jnp.zeros_like(value),))[1]
    else:
        differentiate = (
            jax.jacfwd(evaluate) if method == "jacfwd" else jax.jacrev(evaluate)
        )
    operation = jax.vmap(differentiate)
    if compiled:
        operation = jax.jit(operation)

    def run_mapped() -> None:
        actual = operation(jnp.asarray([0.0, 1.0]))
        jax.block_until_ready(actual)
        jax.effects_barrier()

    with pytest.raises((ArithmeticError, RuntimeError), match=r"(?i)(accuracy|certif)"):
        run_mapped()


@pytest.mark.parametrize("field", ["powers", "coefficient_exponents"])
def test_density_certificate_rejects_signed_integer_extrema(*, field: str) -> None:
    dtype = np.float64 if X64_ENABLED else np.float32
    scalar_zero = np.asarray(0.0, dtype=dtype)
    powers = np.zeros(2, dtype=np.int64)
    exponents = np.zeros((4, 2), dtype=np.int64)
    if field == "powers":
        powers[0] = np.iinfo(np.int64).min
    else:
        exponents[0, 0] = np.iinfo(np.int64).min
    with pytest.raises(ArithmeticError, match=r"(?i)(power|exponent)"):
        admission._checked_density_coefficients(
            np.asarray([-1.0, 1.0], dtype=dtype),
            scalar_zero,
            scalar_zero,
            scalar_zero,
            np.asarray(1.0, dtype=dtype),
            np.asarray(0, dtype=np.int64),
            np.zeros(2, dtype=dtype),
            powers,
            np.ones(2, dtype=dtype),
            np.ones(2, dtype=np.bool_),
            np.zeros((4, 2), dtype=dtype),
            exponents,
        )
