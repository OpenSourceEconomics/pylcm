"""Shock declarations keep base innovations separate from laws and node rules."""

import math
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr

import lcm
from lcm import shocks as sh
from lcm.exceptions import ShockDeclarationError
from lcm.typing import FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


@lcm.categorical(ordered=False)
class Offer:
    absent: ScalarInt
    present: ScalarInt


def _tol() -> float:
    return 10.0**-DECIMAL_PRECISION


def _normal_group(*, nodes: sh.NodeRule | None = None) -> sh.ShockGroup:
    return sh.ShockGroup(
        shocks={"eps": sh.Normal(mu=2.0, sigma=3.0)},
        nodes=nodes or sh.Product(eps=sh.GaussHermite(5)),
    )


def test_import_owns_worktree() -> None:
    """Execution imports the source that owns this test."""
    assert (
        Path(lcm.__file__).resolve().parents[2] == Path(__file__).resolve().parents[1]
    )


@pytest.mark.parametrize("rule", [sh.GaussHermite(5), sh.EqualSpaced(7, n_std=3.0)])
def test_normal_moments_and_transform(*, rule: sh.ScalarRule) -> None:
    group = _normal_group(nodes=sh.Product(eps=rule))
    nodes = group.resolve_nodes(group_name="wage", age=Fraction(70, 3))
    result = group.realise(nodes=nodes)
    result.raise_if_invalid()
    np.testing.assert_allclose(jnp.sum(result.weights), 1.0, atol=_tol())
    np.testing.assert_allclose(
        jnp.sum(result.values["eps"] * result.weights), 2.0, atol=_tol()
    )
    if isinstance(rule, sh.GaussHermite):
        np.testing.assert_allclose(
            jnp.sum((result.values["eps"] - 2.0) ** 2 * result.weights),
            9.0,
            rtol=_tol(),
        )


def test_product_order_and_correlation() -> None:
    group = sh.ShockGroup(
        shocks={"a": sh.Normal(mu=0, sigma=2), "b": sh.Normal(mu=1, sigma=3)},
        correlation=((1.0, 0.5), (0.5, 1.0)),
        nodes=sh.Product(a=sh.GaussHermite(3), b=sh.GaussHermite(5)),
    )
    nodes = group.resolve_nodes(group_name="wage", age=0)
    result = group.realise(nodes=nodes)
    result.raise_if_invalid()
    assert nodes.base_nodes.shape == (15, 2)
    np.testing.assert_allclose(
        jnp.sum(result.weights * result.values["a"] * (result.values["b"] - 1)),
        3.0,
        rtol=_tol(),
    )
    np.testing.assert_allclose(nodes.base_nodes[:5, 0], nodes.base_nodes[0, 0], atol=0)


def test_laws_runtime_callables_and_derivatives() -> None:
    def scale(*, health: FloatND, multiplier: FloatND) -> FloatND:
        return health * multiplier

    group = sh.ShockGroup(
        shocks={"eps": sh.Normal(mu=None, sigma=scale)},
        nodes=sh.Product(eps=sh.GaussHermite(3)),
    )
    nodes = group.resolve_nodes(group_name="wage", age=0)

    def evaluate(multiplier: FloatND) -> FloatND:
        result = group.realise(
            nodes=nodes,
            params={"eps__mu": jnp.asarray(2.0)},
            inputs={"health": jnp.asarray(2.0), "multiplier": multiplier},
        )
        return jnp.sum(result.weights * result.values["eps"] ** 2)

    for function in (evaluate, jax.jit(evaluate)):
        np.testing.assert_allclose(function(jnp.asarray(3.0)), 40.0, rtol=_tol())
    np.testing.assert_allclose(jax.grad(evaluate)(jnp.asarray(3.0)), 24.0, rtol=_tol())
    np.testing.assert_allclose(
        jax.vmap(evaluate)(jnp.asarray([1.0, 2.0])), [8.0, 20.0], rtol=_tol()
    )


@pytest.mark.parametrize(
    ("law", "base", "expected"),
    [
        (sh.LogNormal(mu=0, sigma=1), 0.0, 1.0),
        (sh.Normal(mu=2, sigma=0), 9.0, 2.0),
        (sh.LogNormal(mu=2, sigma=0), 9.0, np.exp(2.0)),
        (sh.Uniform(start=2, stop=6), 0.25, 3.0),
        (sh.Uniform(start=2, stop=2), 0.25, 2.0),
    ],
)
def test_law_transform(*, law: sh.Law, base: float, expected: float) -> None:
    rule = (
        sh.EqualSpacedUniform(3) if isinstance(law, sh.Uniform) else sh.GaussHermite(3)
    )
    group = sh.ShockGroup(shocks={"eps": law}, nodes=sh.Product(eps=rule))
    for function in (group.transform, jax.jit(group.transform)):
        result = function(base_nodes=jnp.asarray([[base]]))
        result.raise_if_invalid()
        np.testing.assert_allclose(result.values["eps"], [expected], rtol=_tol())


def test_mixture_inverse_cdf_and_implicit_derivative() -> None:
    group = sh.ShockGroup(
        shocks={
            "eps": sh.NormalMixture(p1=0.5, mu1=None, sigma1=2.0, mu2=3.0, sigma2=2.0)
        },
        nodes=sh.Product(eps=sh.GaussHermite(3)),
    )

    def quantile(mu: FloatND) -> FloatND:
        return group.transform(
            base_nodes=jnp.asarray([[0.0]]), params={"eps__mu1": mu}
        ).values["eps"][0]

    np.testing.assert_allclose(jax.jit(quantile)(jnp.asarray(-3.0)), 0.0, atol=_tol())
    np.testing.assert_allclose(jax.grad(quantile)(jnp.asarray(-3.0)), 0.5, rtol=_tol())
    result = group.transform(
        base_nodes=jnp.asarray([[-10.0], [10.0]]),
        params={"eps__mu1": jnp.asarray(-3.0)},
    )
    result.raise_if_invalid()
    assert np.isfinite(np.asarray(result.values["eps"])).all()


def test_exact_discrete_and_zero_masses() -> None:
    group = sh.ShockGroup(
        shocks={
            "offer": sh.Discrete(
                categories=Offer,
                probs=lambda probability: jnp.stack(
                    (1 - probability, probability), axis=-1
                ),
            )
        }
    )
    nodes = group.resolve_nodes(group_name="offer", age=0)
    result = group.realise(
        nodes=nodes, inputs={"probability": jnp.asarray([0.0, 0.25, 1.0])}
    )
    result.raise_if_invalid()
    np.testing.assert_allclose(
        result.weights, [[1, 0], [0.75, 0.25], [0, 1]], atol=_tol()
    )
    drawn = group.transform(
        base_nodes=jnp.asarray([[0.0], [0.75], [0.99]]),
        inputs={"probability": jnp.asarray(0.25)},
    )
    drawn.raise_if_invalid()
    np.testing.assert_array_equal(drawn.values["offer"], [0, 1, 1])


@pytest.mark.parametrize(
    "rule", [sh.MonteCarlo(16, seed=17), sh.QuasiMonteCarlo(16, seed=17)]
)
def test_random_rules_age_and_parameter_identity(*, rule: sh.NodeRule) -> None:
    group = _normal_group(nodes=rule)
    a = group.resolve_nodes(group_name="wage", age=Fraction(70, 3))
    b = group.resolve_nodes(group_name="wage", age=Fraction(70, 3))
    c = group.resolve_nodes(group_name="wage", age=Fraction(71, 3))
    np.testing.assert_array_equal(a.base_nodes, b.base_nodes)
    assert not np.array_equal(a.base_nodes, c.base_nodes)
    np.testing.assert_allclose(a.weights, np.full(16, 1 / 16), atol=0)
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint


@pytest.mark.parametrize("family", ["normal", "uniform"])
@pytest.mark.parametrize("level", [0, 1])
def test_smolyak_constant_and_second_moments(*, family: str, level: int) -> None:
    law = (
        sh.Normal(mu=0, sigma=1) if family == "normal" else sh.Uniform(start=0, stop=1)
    )
    group = sh.ShockGroup(shocks={"a": law, "b": law}, nodes=sh.Smolyak(level))
    nodes = group.resolve_nodes(group_name="joint", age=0)
    result = group.realise(nodes=nodes)
    result.raise_if_invalid()
    np.testing.assert_allclose(jnp.sum(result.weights), 1.0, atol=_tol())
    if level > 0:
        expected = 1.0 if family == "normal" else 1 / 3
        np.testing.assert_allclose(
            jnp.sum(result.weights * result.values["a"] ** 2), expected, rtol=_tol()
        )
    assert len(set(map(tuple, np.asarray(nodes.base_nodes)))) == len(nodes.weights)


def test_user_nodes_with_negative_weights_are_refused() -> None:
    """A node rule with a negative weight never reaches a model."""
    with pytest.raises(ShockDeclarationError, match=r"UserNodes.*negative weights"):
        sh.UserNodes(
            nodes=np.asarray([[-1.0], [0.0], [1.0]]),
            weights=np.asarray([-0.1, 1.2, -0.1]),
        )


@pytest.mark.parametrize(
    ("family", "dimension", "level"),
    [
        ("normal", 1, 3),
        ("normal", 2, 2),
        ("normal", 3, 1),
        ("normal", 4, 1),
        ("uniform", 2, 2),
    ],
)
def test_smolyak_with_negative_weights_is_refused_at_declaration(
    *, family: str, dimension: int, level: int
) -> None:
    """Higher Smolyak levels have negative weights and are refused by the group."""
    law = (
        sh.Normal(mu=0, sigma=1) if family == "normal" else sh.Uniform(start=0, stop=1)
    )
    with pytest.raises(
        ShockDeclarationError, match=rf"Smolyak\(level={level}\).*negative weights"
    ):
        sh.ShockGroup(
            shocks={f"x{i}": law for i in range(dimension)},
            nodes=sh.Smolyak(level),
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: sh.Normal(mu=0, sigma=-1),
        lambda: sh.Uniform(start=2, stop=1),
        lambda: sh.NormalMixture(p1=0.5, mu1=0, sigma1=0, mu2=0, sigma2=1),
        lambda: sh.QuasiMonteCarlo(15, seed=0),
        lambda: sh.UserNodes(nodes=np.ones((2, 1)), weights=np.asarray([np.nan, 1.0])),
        lambda: sh.UserNodes(nodes=np.ones((2, 1)), weights=np.asarray([-1.0, 2.0])),
        lambda: sh.UserNodes(nodes=np.ones((2, 1)), weights=np.asarray([0.1, 0.2])),
        lambda: sh.ShockGroup(
            shocks={"a": sh.Normal(mu=0, sigma=1)},
            nodes=sh.UserNodes(nodes=np.ones((2, 2)), weights=np.asarray([0.5, 0.5])),
        ),
        lambda: sh.ShockGroup(
            shocks={"a": sh.Uniform(start=0, stop=1)},
            nodes=sh.Product(a=sh.GaussHermite(3)),
        ),
    ],
)
def test_invalid_declarations(
    *, factory: Callable[[], sh.Law | sh.NodeRule | sh.ShockGroup]
) -> None:
    with pytest.raises(ShockDeclarationError):
        factory()


def test_invalid_runtime_law_has_explicit_status() -> None:
    group = sh.ShockGroup(
        shocks={"eps": sh.Normal(mu=0, sigma=None)},
        nodes=sh.Product(eps=sh.GaussHermite(3)),
    )
    result = jax.jit(group.transform)(
        base_nodes=jnp.asarray([[0.0]]), params={"eps__sigma": jnp.asarray(-1.0)}
    )
    with pytest.raises(ShockDeclarationError):
        result.raise_if_invalid()


def test_callable_dependency_closure() -> None:
    group = sh.ShockGroup(
        shocks={"eps": sh.Normal(mu=0, sigma=lambda scale: scale)},
        nodes=sh.Product(eps=sh.GaussHermite(3)),
    )
    assert group.validate_dependencies(
        states=frozenset({"health"}),
        actions=frozenset({"work"}),
        functions={"scale": lambda health: health},
    ) == frozenset({"health"})
    with pytest.raises(ShockDeclarationError, match="work"):
        group.validate_dependencies(
            states=frozenset({"health"}),
            actions=frozenset({"work"}),
            functions={"scale": lambda work: work},
        )
    with pytest.raises(ShockDeclarationError, match="eps"):
        group.validate_dependencies(
            states=frozenset(),
            actions=frozenset(),
            functions={"scale": lambda eps: eps},
        )


def test_runtime_correlation_and_fingerprint_separation() -> None:
    fixed = _normal_group()
    runtime = sh.ShockGroup(shocks=fixed.shocks, nodes=fixed.nodes, correlation=None)
    nodes = runtime.resolve_nodes(group_name="wage", age=0)
    with pytest.raises(ShockDeclarationError, match="corr"):
        runtime.realise(nodes=nodes)
    result = runtime.realise(nodes=nodes, params={"corr": jnp.ones((1, 1))})
    result.raise_if_invalid()
    assert fixed.fingerprint != runtime.fingerprint


def test_smolyak_level_limit() -> None:
    with pytest.raises(ShockDeclarationError, match="level"):
        _normal_group(nodes=sh.Smolyak(4))


def test_discrete_probability_refusals() -> None:
    with pytest.raises(ShockDeclarationError):
        sh.Discrete(categories=Offer, probs=np.asarray([0.8, 0.3]))
    group = sh.ShockGroup(shocks={"offer": sh.Discrete(categories=Offer, probs=None)})
    result = jax.jit(group.realise)(
        nodes=group.resolve_nodes(group_name="offer", age=0),
        params={"offer__probs": jnp.asarray([1.1, -0.1])},
    )
    with pytest.raises(ShockDeclarationError):
        result.raise_if_invalid()


def test_singular_and_invalid_correlation() -> None:
    group = sh.ShockGroup(
        shocks={"a": sh.Normal(mu=0, sigma=1), "b": sh.Normal(mu=0, sigma=1)},
        correlation=((1.0, 1.0), (1.0, 1.0)),
        nodes=sh.Product(a=sh.GaussHermite(3), b=sh.GaussHermite(3)),
    )
    result = group.realise(nodes=group.resolve_nodes(group_name="wage", age=0))
    result.raise_if_invalid()
    np.testing.assert_array_equal(result.values["a"], result.values["b"])
    with pytest.raises(ShockDeclarationError):
        sh.ShockGroup(
            shocks=group.shocks, correlation=((1.0, 1.1), (1.1, 1.0)), nodes=group.nodes
        )


def test_conditioning_uses_category_codes() -> None:
    group = sh.ShockGroup(
        shocks={
            "eps": sh.Normal(
                mu=0,
                sigma=sh.StateConditioned(
                    on="offer", by={"present": 2.0, "absent": 1.0}
                ),
            )
        },
        nodes=sh.Product(eps=sh.GaussHermite(3)),
    )
    bound = group.bind_conditioning(state_grids={"offer": lcm.DiscreteGrid(Offer)})
    result = bound.transform(
        base_nodes=jnp.asarray([[1.0], [1.0]]),
        inputs={"offer": jnp.asarray([0, 1], dtype=jnp.int32)},
    )
    result.raise_if_invalid()
    np.testing.assert_allclose(result.values["eps"], [1.0, 2.0], atol=_tol())


def test_user_nodes_fingerprint_and_owned_inputs() -> None:
    raw = np.asarray([[-1.0], [1.0]])
    rule = sh.UserNodes(nodes=raw, weights=np.asarray([0.5, 0.5]))
    raw[:] = 999
    first = _normal_group(nodes=rule).resolve_nodes(group_name="wage", age=0)
    second = _normal_group(
        nodes=sh.UserNodes(
            nodes=np.asarray([[0.0], [1.0]]), weights=np.asarray([0.5, 0.5])
        )
    ).resolve_nodes(group_name="wage", age=0)
    np.testing.assert_allclose(first.base_nodes[:, 0], [-1, 1], atol=0)
    assert first.fingerprint != second.fingerprint


def test_mixture_probability_derivative_keeps_positive_tail() -> None:
    """The mixture-weight slope retains survival mass beyond rounded unit CDFs."""

    z = 40.0 if jax.config.jax_enable_x64 else 15.0
    group = sh.ShockGroup(
        shocks={"eps": sh.NormalMixture(p1=None, mu1=0, sigma1=1, mu2=-1, sigma2=1)},
        nodes=sh.Product(eps=sh.GaussHermite(3)),
    )

    def quantile(p: FloatND) -> FloatND:
        result = group.transform(base_nodes=jnp.asarray([[z]]), params={"eps__p1": p})
        return result.values["eps"][0]

    log_sf = float(log_ndtr(np.asarray(-z)))
    log_difference = log_sf + np.log(
        -np.expm1(float(log_ndtr(np.asarray(-(z + 1)))) - log_sf)
    )
    reference = np.exp(log_difference + z * z / 2 + np.log(2 * np.pi) / 2)
    derivative = jax.jit(jax.grad(quantile))(jnp.asarray(1.0))
    assert reference > 0
    np.testing.assert_allclose(derivative, reference, rtol=32 * _tol())


def test_discrete_terminal_plateau_skips_zero_mass() -> None:
    """A near-unit positive prefix never gives trailing zero outcomes mass."""

    @lcm.categorical(ordered=False)
    class Categories:
        first: ScalarInt
        second: ScalarInt
        empty: ScalarInt

    group = sh.ShockGroup(
        shocks={"draw": sh.Discrete(categories=Categories, probs=None)}
    )
    dtype = jnp.asarray(0.0).dtype
    half = jnp.asarray(0.5, dtype=dtype)
    below_half = jnp.nextafter(half, jnp.asarray(0.0, dtype=dtype))
    probs = jnp.asarray([below_half, below_half, 0.0])
    assert float(probs[-1]) == 0.0
    result = group.transform(
        base_nodes=jnp.asarray(
            [
                [
                    jnp.nextafter(
                        jnp.asarray(1.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype)
                    )
                ]
            ]
        ),
        params={"draw__probs": probs},
    )
    result.raise_if_invalid()
    np.testing.assert_array_equal(result.values["draw"], [1])


def test_fixed_discrete_probability_matrix_is_typed_error() -> None:
    with pytest.raises(ShockDeclarationError):
        sh.Discrete(categories=Offer, probs=np.asarray([[0.5, 0.5]]))


def test_runtime_correlation_gradient_keeps_parameter() -> None:
    """A prepared support plan keeps estimated correlation as a traced parameter."""
    group = sh.ShockGroup(
        shocks={"a": sh.Normal(mu=0, sigma=1), "b": sh.Normal(mu=0, sigma=1)},
        nodes=sh.Product(a=sh.GaussHermite(3), b=sh.GaussHermite(3)),
        correlation=None,
    ).bind_correlation(correlation=((1.0, 0.3), (0.3, 1.0)))
    nodes = group.resolve_nodes(group_name="wage", age=0)

    def moment(rho: FloatND) -> FloatND:
        corr = jnp.stack(
            (jnp.stack((jnp.asarray(1.0), rho)), jnp.stack((rho, jnp.asarray(1.0))))
        )
        result = group.realise(nodes=nodes, params={"corr": corr})
        return jnp.sum(result.weights * result.values["a"] * result.values["b"])

    np.testing.assert_allclose(jax.jit(moment)(jnp.asarray(0.7)), 0.7, rtol=_tol())
    np.testing.assert_allclose(
        jax.jit(jax.grad(moment))(jnp.asarray(0.7)), 1.0, rtol=_tol()
    )


@pytest.mark.parametrize(
    ("separation", "scale1", "scale2"),
    [(100.0, 1.0, 2.0), (60.0, 2.0, 1.0), (200.0, 3.0, 2.0)],
)
def test_mixture_quantile_resolves_cdf_plateau(
    *, separation: float, scale1: float, scale2: float
) -> None:
    """Separated-component medians retain physical quantile accuracy on CDF plateaus."""
    group = sh.ShockGroup(
        shocks={
            "eps": sh.NormalMixture(
                p1=0.5, mu1=0, sigma1=scale1, mu2=separation, sigma2=scale2
            )
        },
        nodes=sh.Product(eps=sh.GaussHermite(3)),
    )
    result = group.transform(base_nodes=jnp.asarray([[0.0]]))
    result.raise_if_invalid()
    expected = separation * scale1 / (scale1 + scale2)
    np.testing.assert_allclose(result.values["eps"], [expected], rtol=8 * _tol())


@pytest.mark.parametrize(
    "p", [np.nextafter(0.5, 0.0), 0.1, 0.9, np.nextafter(0.5, 1.0)]
)
@pytest.mark.parametrize("separation", [20.0, 100.0])
def test_mixture_quantile_around_plateau_mass(*, p: float, separation: float) -> None:
    """An independent scalar tail equation covers probabilities beside plateau mass."""

    # The actual floating dtype is the law's operand, including near-half probabilities.
    probability = float(jnp.asarray(p))
    group = sh.ShockGroup(
        shocks={
            "eps": sh.NormalMixture(
                p1=probability, mu1=0, sigma1=1, mu2=separation, sigma2=2
            )
        },
        nodes=sh.Product(eps=sh.GaussHermite(3)),
    )
    if probability == 0.5:
        reference = separation / 3
    else:

        def residual(q: float) -> float:
            return math.fsum(
                (
                    probability - 0.5,
                    (1 - probability) * float(ndtr(np.asarray((q - separation) / 2))),
                    -probability * float(ndtr(np.asarray(-q))),
                )
            )

        reference = brentq(residual, 0.0, separation, xtol=1e-13)
    result = jax.jit(group.transform)(base_nodes=jnp.asarray([[0.0]]))
    result.raise_if_invalid()
    np.testing.assert_allclose(result.values["eps"], [reference], rtol=8 * _tol())


def test_runtime_correlation_support_requires_rebinding() -> None:
    group = sh.ShockGroup(
        shocks={"a": sh.Normal(mu=0, sigma=1), "b": sh.Normal(mu=0, sigma=1)},
        nodes=sh.Product(a=sh.GaussHermite(3), b=sh.GaussHermite(3)),
        correlation=None,
    ).bind_correlation(correlation=((1.0, 0.0), (0.0, 1.0)))
    with pytest.raises(ShockDeclarationError, match="support"):
        group.validate_runtime_correlation(correlation=((1.0, 1.0), (1.0, 1.0)))


def test_mixture_indeterminate_finite_tail_reports_status() -> None:
    """Finite physical bounds do not license an indeterminate logarithmic comparison."""
    z = 1e200 if jax.config.jax_enable_x64 else 1e20
    group = sh.ShockGroup(
        shocks={"eps": sh.NormalMixture(p1=0.5, mu1=0, sigma1=1, mu2=1, sigma2=2)},
        nodes=sh.Product(eps=sh.GaussHermite(3)),
    )
    result = jax.jit(group.transform)(base_nodes=jnp.asarray([[z]]))
    with pytest.raises(ShockDeclarationError):
        result.raise_if_invalid()


def test_runtime_scalar_parameter_shape_is_typed_error() -> None:
    group = sh.ShockGroup(
        shocks={"eps": sh.Normal(mu=None, sigma=None)},
        nodes=sh.Product(eps=sh.GaussHermite(3)),
    )
    with pytest.raises(ShockDeclarationError):
        group.transform(
            base_nodes=jnp.zeros((2, 1)),
            params={"eps__mu": jnp.ones(2), "eps__sigma": jnp.ones(3)},
        )


def test_runtime_singular_correlation_uses_exact_support() -> None:
    """A duplicate represented row keeps its exact zero pivot under compilation."""
    rho = 0.167
    matrix = ((1.0, rho, rho), (rho, 1.0, 1.0), (rho, 1.0, 1.0))
    group = sh.ShockGroup(
        shocks={name: sh.Normal(mu=0, sigma=1) for name in ("a", "b", "c")},
        nodes=sh.Product(**{name: sh.GaussHermite(3) for name in ("a", "b", "c")}),
        correlation=None,
    ).bind_correlation(correlation=matrix)
    nodes = group.resolve_nodes(group_name="joint", age=0)
    for function in (group.realise, jax.jit(group.realise)):
        result = function(nodes=nodes, params={"corr": jnp.asarray(matrix)})
        result.raise_if_invalid()
        np.testing.assert_array_equal(result.values["b"], result.values["c"])


@pytest.mark.parametrize(("level", "degree"), [(0, 1), (1, 5), (2, 15)])
def test_genz_keister_analytic_moment_degree(*, level: int, degree: int) -> None:
    """Normal-rule moments agree with the analytic double-factorial identity."""
    group = sh.ShockGroup(
        shocks={"eps": sh.Normal(mu=0, sigma=1)}, nodes=sh.Smolyak(level)
    )
    result = group.realise(nodes=group.resolve_nodes(group_name="normal", age=0))
    result.raise_if_invalid()
    for power in range(0, degree + 1, 2):
        reference = math.prod(range(1, power, 2))
        np.testing.assert_allclose(
            jnp.sum(result.weights * result.values["eps"] ** power),
            reference,
            rtol=8 * _tol(),
        )


def test_law_identity_excludes_rule_and_scalar_spelling() -> None:
    first = sh.ShockGroup(
        shocks={"eps": sh.Normal(mu=0, sigma=1)},
        nodes=sh.Product(eps=sh.GaussHermite(3)),
    )
    second = sh.ShockGroup(
        shocks={"eps": sh.Normal(mu=0.0, sigma=1.0)},
        nodes=sh.Product(eps=sh.GaussHermite(5)),
    )
    assert first.has_same_laws_as(second)
    assert first.fingerprint != second.fingerprint


def test_full_realisation_is_vmap_pytree() -> None:
    """Batched transformed calls retain values, weights and per-call validity."""
    group = _normal_group()
    result = jax.jit(
        jax.vmap(lambda point: group.transform(base_nodes=point[None, None]))
    )(jnp.asarray([-1.0, 0.0, 1.0]))
    np.testing.assert_allclose(result.values["eps"][:, 0], [-1, 2, 5], atol=_tol())
    np.testing.assert_array_equal(result.valid, np.asarray([True, True, True]))
    np.testing.assert_allclose(result.weights, np.ones((3, 1)), atol=0)


def test_runtime_complex_law_parameter_is_typed_error() -> None:
    group = sh.ShockGroup(
        shocks={"eps": sh.Normal(mu=0, sigma=None)},
        nodes=sh.Product(eps=sh.GaussHermite(3)),
    )
    with pytest.raises(ShockDeclarationError):
        group.transform(
            base_nodes=jnp.asarray([[0.0]]), params={"eps__sigma": np.asarray(1 + 1j)}
        )


def test_named_correlation_identity_keeps_column_order() -> None:
    """Equal numeric correlation matrices can represent different named joint laws."""
    corr = ((1.0, 0.2, 0.3), (0.2, 1.0, 0.4), (0.3, 0.4, 1.0))

    def declaration(names: tuple[str, ...]) -> sh.ShockGroup:
        return sh.ShockGroup(
            shocks={name: sh.Normal(mu=0, sigma=1) for name in names},
            correlation=corr,
            nodes=sh.Product(**{name: sh.GaussHermite(3) for name in names}),
        )

    first, second = declaration(("a", "b", "c")), declaration(("b", "a", "c"))
    assert not first.has_same_laws_as(second)
    assert first.fingerprint != second.fingerprint


def test_callable_law_identity_uses_captured_semantics() -> None:
    def scale_factory(multiplier: float) -> sh.LawCallable:
        def scale(*, health: FloatND) -> FloatND:
            return multiplier * health

        return scale

    def declaration(multiplier: float) -> sh.ShockGroup:
        return sh.ShockGroup(
            shocks={"eps": sh.Normal(mu=0, sigma=scale_factory(multiplier))},
            nodes=sh.Product(eps=sh.GaussHermite(3)),
        )

    first, same, different = declaration(0.2), declaration(0.2), declaration(0.3)
    assert first.has_same_laws_as(same)
    assert not first.has_same_laws_as(different)


@pytest.mark.parametrize(
    ("nodes", "weights", "match"),
    [
        ([[0.0], [1.0]], [1.5, -0.5], "UserNodes has negative weights"),
        ([[0.0], [1.0]], [0.5, 0.25], "Weights must sum to unity"),
        ([[0.0], [np.nan]], [0.5, 0.5], "finite nonempty"),
        ([[0.0], [np.inf]], [0.5, 0.5], "finite nonempty"),
    ],
)
def test_current_iid_law_refuses_malformed_nodes_before_any_model(
    *, nodes: list[list[float]], weights: list[float], match: str
) -> None:
    """A malformed current-IID law is refused at its declaration."""
    with pytest.raises(ShockDeclarationError, match=match):
        sh.UserNodes(nodes=np.array(nodes), weights=np.array(weights))
