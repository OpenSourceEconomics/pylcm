"""Joint-normal covariance factors keep the declared law at every exponent scale.

The reference is analytic. For independent standard normals `Z0, Z1` and an exact
power-of-two scale `s`, `X = s Z0` and `Y = s (b Z0 + a Z1)` have covariance
`s**2 [[1, b], [b, b**2 + a**2]]` and lower-triangular factor `s [[1, 0], [b, a]]`.
Neither the expected factor nor the expected moments use a production
factorization. Scales sit near the exponent range of the active precision, where
an intermediate product of two covariance entries leaves that range although every
input and every factor entry is representable.
"""

from fractions import Fraction
from itertools import permutations, product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcm
import lcm.typing
from lcm.exceptions import GridInitializationError, InvalidInitialConditionsError


def _dtype_and_extreme_exponent() -> tuple[type[np.floating], int]:
    if jax.config.read("jax_enable_x64"):
        return np.float64, 600
    return np.float32, 80


def _scaled_law(*, exponent_sign: int, matrix: np.ndarray) -> tuple[float, np.ndarray]:
    dtype, extreme = _dtype_and_extreme_exponent()
    exponent = extreme * exponent_sign
    scale = float(np.ldexp(dtype(1), exponent // 2))
    cov = np.ldexp(np.asarray(matrix, dtype=dtype), exponent)
    return scale, cov


def _process(*, cov: np.ndarray | None, dimensions: int) -> lcm.JointNormalIIDProcess:
    return lcm.JointNormalIIDProcess(
        states=tuple(f"shock{index}" for index in range(dimensions)),
        n_points=3,
        gauss_hermite=True,
        mean=(0.0,) * dimensions,
        cov=None if cov is None else tuple(tuple(float(v) for v in row) for row in cov),
    )


def _exact_triangular_law(
    *, exponent: int, b: int, a: int
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[tuple[Fraction, ...], ...]]:
    """Return the covariance and factor of `X = s Z0, Y = s (b Z0 + a Z1)`."""
    scale = Fraction(2) ** (exponent // 2)
    factor = ((scale, Fraction(0)), (b * scale, a * scale))
    covariance = tuple(
        tuple(
            sum((factor[row][k] * factor[column][k] for k in range(2)), Fraction(0))
            for column in range(2)
        )
        for row in range(2)
    )
    return covariance, factor


@pytest.mark.parametrize("exponent", [-600, -80, 0, 80, 600])
@pytest.mark.parametrize(("b", "a"), [(1, 1), (-1, 1), (2, 0), (0, 2)])
def test_exact_reference_law_reproduces_its_covariance(
    *, exponent: int, b: int, a: int
) -> None:
    """The reference factor times its transpose is the stated covariance exactly."""
    covariance, _ = _exact_triangular_law(exponent=exponent, b=b, a=a)
    unit = Fraction(2) ** exponent
    assert covariance == ((unit, b * unit), (b * unit, (b * b + a * a) * unit))


@pytest.mark.parametrize("exponent_sign", [-1, 0, 1])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_second_basis_vector_maps_to_its_exact_factor_column(
    *, exponent_sign: int, fixed: bool, compiled: bool
) -> None:
    """For `s**2 [[1, 1], [1, 2]]` the factor maps `(0, 1)` to exactly `(0, s)`."""
    scale, cov = _scaled_law(
        exponent_sign=exponent_sign, matrix=np.array(((1, 1), (1, 2)))
    )
    process = _process(cov=cov if fixed else None, dimensions=2)
    process.validate_covariance(cov)
    coordinates = jnp.asarray((0.0, 1.0))

    def call(*, standard: jax.Array, covariance: jax.Array) -> jax.Array:
        return process.transform(standard=standard, cov=None if fixed else covariance)

    actual = (jax.jit(call) if compiled else call)(
        standard=coordinates, covariance=jnp.asarray(cov)
    )
    np.testing.assert_array_equal(np.asarray(actual) / scale, (0.0, 1.0))


_BASE_COVARIANCES = (
    pytest.param(np.array(((1, 1), (1, 2))), id="full-rank-positive"),
    pytest.param(np.array(((1, -1), (-1, 2))), id="full-rank-negative"),
    pytest.param(np.array(((1, 1), (1, 1))), id="rank-one"),
    pytest.param(np.array(((0, 0), (0, 1))), id="zero-pivot"),
    pytest.param(np.array(((1, 1, 0), (1, 2, 1), (0, 1, 2))), id="full-rank-3d"),
    pytest.param(np.array(((1, 1, 0), (1, 2, 1), (0, 1, 1))), id="rank-two-3d"),
)


@pytest.mark.parametrize("exponent_sign", [-1, 0, 1])
@pytest.mark.parametrize("matrix", _BASE_COVARIANCES)
def test_scaled_and_permuted_quadrature_keeps_the_declared_covariance(
    *, exponent_sign: int, matrix: np.ndarray
) -> None:
    """Every axis permutation of a scaled law integrates to its covariance."""
    dtype, _ = _dtype_and_extreme_exponent()
    # The envelope covers the rounded moments of the tiny rule only; rank and
    # PSD decisions are exact.
    moment_atol = 64 * float(np.finfo(dtype).eps)
    for order in permutations(range(len(matrix))):
        reference = matrix[np.ix_(order, order)]
        scale, cov = _scaled_law(exponent_sign=exponent_sign, matrix=reference)
        # Several laws are singular, which only a fixed covariance may declare.
        process = _process(cov=cov, dimensions=len(matrix))
        nodes = np.asarray(jax.jit(process.compute_gridpoints)())
        normalized = nodes / scale
        weights = np.asarray(process.get_weights())
        np.testing.assert_allclose(
            normalized.T @ (weights[:, None] * normalized),
            reference,
            atol=moment_atol,
            rtol=0,
            err_msg=f"axis order {order}",
        )


@pytest.mark.parametrize("exponent_sign", [-1, 0, 1])
def test_adjacent_positive_pivot_stays_positive(*, exponent_sign: int) -> None:
    """A pivot one unit in the last place above zero keeps its own direction."""
    dtype, _ = _dtype_and_extreme_exponent()
    one = dtype(1.0)
    next_one = np.nextafter(one, dtype(2.0))
    matrix = np.array(((one, one), (one, next_one)), dtype=dtype)
    scale, cov = _scaled_law(exponent_sign=exponent_sign, matrix=matrix)
    process = _process(cov=None, dimensions=2)
    process.validate_covariance(cov)
    actual = (
        np.asarray(
            process.transform(standard=jnp.asarray((0.0, 1.0)), cov=jnp.asarray(cov))
        )
        / scale
    )
    expected = np.sqrt(float(next_one) - 1.0)
    np.testing.assert_allclose(
        actual, (0.0, expected), atol=0, rtol=4 * float(np.finfo(dtype).eps)
    )


@pytest.mark.parametrize("axis_directions", [(1, -1), (-1, 1), (1, 0), (0, -1)])
@pytest.mark.parametrize("correlation_sign", [-1, 1])
@pytest.mark.parametrize("translated", [False, True])
def test_anisotropic_scales_and_translation_keep_normalized_moments(
    *, axis_directions: tuple[int, int], correlation_sign: int, translated: bool
) -> None:
    """Each physical row keeps its own scale, sign and mean."""
    dtype, extreme = _dtype_and_extreme_exponent()
    exponents = np.asarray(axis_directions) * (extreme // 2)
    scales = np.ldexp(np.ones(2, dtype=dtype), exponents)
    reference = np.asarray(((1, correlation_sign), (correlation_sign, 2)), dtype=dtype)
    cov = np.ldexp(reference, exponents[:, None] + exponents[None, :])
    normalized_mean = np.asarray((1.0, -2.0) if translated else (0.0, 0.0), dtype=dtype)
    mean = scales * normalized_mean
    process = lcm.JointNormalIIDProcess(
        states=("x", "y"), n_points=(3, 5), gauss_hermite=True, mean=None, cov=None
    )
    process.validate_mean(mean)
    process.validate_covariance(cov)
    nodes = np.asarray(
        jax.jit(process.compute_gridpoints)(
            mean=jnp.asarray(mean), cov=jnp.asarray(cov)
        )
    )
    normalized = nodes / scales - normalized_mean
    weights = np.asarray(process.get_weights())
    atol = 128 * float(np.finfo(dtype).eps)
    moments = np.concatenate(
        (weights @ normalized, (normalized.T @ (weights[:, None] * normalized)).ravel())
    )
    np.testing.assert_allclose(
        moments, np.concatenate((np.zeros(2), reference.ravel())), atol=atol, rtol=0
    )


def test_triangular_laws_survive_the_scale_sign_and_rank_neighborhood() -> None:
    """Exponents near the range edges, correlation signs and zero pivots are exact."""
    dtype, extreme = _dtype_and_extreme_exponent()
    process = lcm.JointNormalIIDProcess(
        states=("x", "y"), n_points=3, mean=(0.0, 0.0), cov=None, gauss_hermite=True
    )

    def image_of_basis(*, cov: jax.Array) -> jax.Array:
        return jax.vmap(lambda z: process.transform(standard=z, cov=cov))(jnp.eye(2))

    compiled = jax.jit(image_of_basis)
    exponents = (
        -extreme - 2,
        -extreme,
        -extreme + 2,
        0,
        extreme - 2,
        extreme,
        extreme + 2,
    )
    mismatches = []
    for exponent, b, a in product(exponents, (-2, -1, 0, 1, 2), (0, 1, 2)):
        covariance, factor = _exact_triangular_law(exponent=exponent, b=b, a=a)
        cov = np.asarray([[float(v) for v in row] for row in covariance], dtype=dtype)
        process.validate_covariance(cov)
        # Row `i` of the image is `L e_i`, so the image is the factor's transpose.
        expected = np.asarray(
            [[float(factor[column][row]) for column in range(2)] for row in range(2)],
            dtype=dtype,
        )
        actual = np.asarray(compiled(cov=jnp.asarray(cov)))
        if not np.array_equal(actual, expected):
            mismatches.append((exponent, b, a))
    assert mismatches == []


@lcm.categorical(ordered=False)
class ScaleRegimeId:
    working: lcm.typing.ScalarInt
    done: lcm.typing.ScalarInt


def _spread_utility(
    *,
    x: lcm.typing.ContinuousState,
    y: lcm.typing.ContinuousState,
    shock_scale: lcm.typing.FloatND,
) -> lcm.typing.FloatND:
    return ((y - x) / shock_scale) ** 2


def _spread_model(*, scale: float, fold: bool) -> lcm.Model:
    process = lcm.JointNormalIIDProcess(
        states=("x", "y"),
        n_points=3,
        gauss_hermite=True,
        mean=(0.0, 0.0),
        cov=None,
        fold=fold,
    )
    regime = lcm.Regime(
        states={"joint": process}, functions={"utility": _spread_utility}
    )
    return lcm.Model(
        regimes={"working": regime, "done": regime},
        n_periods=2,
        regime_id_class=ScaleRegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "working"}),
        edges={"working": {"done": lcm.Periods(values=(0,))}},
        fixed_params={
            "working": {
                "utility": {"shock_scale": scale},
                "koopmans_aggregator": {"discount_factor": 0.9},
            },
            "done": {"utility": {"shock_scale": scale}},
        },
        execution_config=lcm.ExecutionConfig(device_memory_bytes=None),
    )


@pytest.mark.parametrize("exponent_sign", [-1, 0, 1])
@pytest.mark.parametrize("fold", [False, True])
def test_public_lookup_prices_the_independent_spread(
    *, exponent_sign: int, fold: bool
) -> None:
    """At physical `(0, s)` utility is 1, its expectation 1 and the value 1.9."""
    dtype, _ = _dtype_and_extreme_exponent()
    scale, cov = _scaled_law(
        exponent_sign=exponent_sign, matrix=np.array(((1, 1), (1, 2)))
    )
    model = _spread_model(scale=scale, fold=fold)
    params = {name: {"joint": {"cov": cov}} for name in ("working", "done")}
    solution = model.solve(params=params, runtime_checks=True, log_level="off")
    lookup = model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="working",
        states={"x": jnp.asarray((0.0,)), "y": jnp.asarray((scale,))},
        keep_intermediates=("Q", "F", "U", "CE"),
    )
    assert lookup.U is not None
    assert lookup.CE is not None
    np.testing.assert_allclose(
        np.concatenate(
            (np.asarray(lookup.U), np.asarray(lookup.CE), np.asarray(lookup.value))
        ),
        (1.0, 1.0, 1.9),
        atol=64 * float(np.finfo(dtype).eps),
        rtol=0,
    )


@pytest.mark.parametrize("exponent_sign", [-1, 0, 1])
@pytest.mark.parametrize("fold", [False, True])
def test_simulation_publishes_the_supplied_physical_initial_state(
    *, exponent_sign: int, fold: bool
) -> None:
    """The physical initial state `(0, s)` survives lowering and publication."""
    scale, cov = _scaled_law(
        exponent_sign=exponent_sign, matrix=np.array(((1, 1), (1, 2)))
    )
    model = _spread_model(scale=scale, fold=fold)
    params = {name: {"joint": {"cov": cov}} for name in ("working", "done")}
    solution = model.solve(params=params, runtime_checks=True, log_level="off")
    result = model.simulate(
        params=params,
        initial_conditions={
            "x": jnp.asarray((0.0,)),
            "y": jnp.asarray((scale,)),
            "period": jnp.zeros(1, dtype=jnp.int32),
            "regime_id": jnp.zeros(1, dtype=jnp.int32),
        },
        solution=solution,
        seed=521,
        runtime_checks=True,
        log_level="off",
    )
    frame = result.to_dataframe(use_labels=False)
    initial = frame.loc[frame["period"] == 0, ["x", "y"]].to_numpy() / scale
    np.testing.assert_array_equal(initial, ((0.0, 1.0),))


def _exactly_singular_integer_covariances() -> tuple[np.ndarray, ...]:
    """Return `M M^T` for integer 3x2 matrices `M` of rank two.

    Each covariance has exact integer entries, positive leading pivots and an
    exactly zero third pivot, so the declared law has no third direction. Products
    and quotients of their entries round, which leaves a nonzero third residual in
    an unguided elimination for about a quarter of them.
    """
    rng = np.random.default_rng(seed=1)
    covariances = []
    for _ in range(3000):
        loadings = rng.integers(-7, 8, size=(3, 2))
        cov = loadings @ loadings.T
        first = Fraction(int(cov[0, 0]))
        second = Fraction(int(cov[1, 1])) - (
            Fraction(int(cov[1, 0])) ** 2 / first if first else Fraction(0)
        )
        if np.linalg.matrix_rank(loadings) == 2 and first > 0 and second > 0:
            covariances.append(cov)
    return tuple(covariances)


_SINGULAR_COVARIANCES = _exactly_singular_integer_covariances()


def test_fixed_singular_covariance_has_no_spurious_direction() -> None:
    """A fixed exactly singular law maps the third coordinate to exactly zero."""
    dtype, _ = _dtype_and_extreme_exponent()
    spurious = []
    for index, cov in enumerate(_SINGULAR_COVARIANCES):
        process = _process(cov=cov.astype(dtype), dimensions=3)
        image = np.asarray(process.transform(standard=jnp.asarray((0.0, 0.0, 1.0))))
        if np.any(image != 0):
            spurious.append(index)
    assert spurious == []


@lcm.categorical(ordered=False)
class SingularRegimeId:
    working: lcm.typing.ScalarInt
    done: lcm.typing.ScalarInt


def _three_shock_utility(
    *,
    x: lcm.typing.ContinuousState,
    y: lcm.typing.ContinuousState,
    z: lcm.typing.ContinuousState,
) -> lcm.typing.FloatND:
    return x + y + z


def _three_shock_model(*, cov: np.ndarray | None) -> lcm.Model:
    process = lcm.JointNormalIIDProcess(
        states=("x", "y", "z"),
        n_points=3,
        gauss_hermite=True,
        mean=(0.0, 0.0, 0.0),
        cov=None if cov is None else tuple(tuple(float(v) for v in row) for row in cov),
    )
    regime = lcm.Regime(
        states={"wages": process}, functions={"utility": _three_shock_utility}
    )
    return lcm.Model(
        regimes={"working": regime, "done": regime},
        n_periods=2,
        regime_id_class=SingularRegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "working"}),
        edges={"working": {"done": lcm.Periods(values=(0,))}},
        fixed_params={"working": {"koopmans_aggregator": {"discount_factor": 0.9}}},
        execution_config=lcm.ExecutionConfig(device_memory_bytes=None),
    )


# Exactly singular, and its unguided elimination leaves a third pivot at either
# precision.
_SINGULAR = np.asarray(((10, 13, -5), (13, 25, 16), (-5, 16, 65)))


@pytest.mark.parametrize(
    "cov",
    [
        pytest.param(_SINGULAR, id="rank-two-3d"),
        pytest.param(
            np.asarray(((1, 1, 0), (1, 1, 0), (0, 0, 2))), id="rank-one-block"
        ),
    ],
)
def test_runtime_singular_covariance_is_refused_by_name(*, cov: np.ndarray) -> None:
    """A singular runtime covariance names its group and asks for a fixed one."""
    model = _three_shock_model(cov=None)
    params = {
        name: {"wages": {"cov": cov.astype(float)}} for name in ("working", "done")
    }
    with pytest.raises(GridInitializationError, match=r"'wages'.*singular.*fixed"):
        model.solve(params=params, log_level="off")


def test_runtime_positive_definite_covariance_is_accepted() -> None:
    """A nonsingular runtime covariance solves and prices `E[x+y+z]=0`."""
    model = _three_shock_model(cov=None)
    cov = np.asarray(((4.0, 1.0, 0.0), (1.0, 2.0, 0.5), (0.0, 0.5, 3.0)))
    params = {name: {"wages": {"cov": cov}} for name in ("working", "done")}
    solution = model.solve(params=params, log_level="off")
    lookup = model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="working",
        states={name: jnp.asarray((1.0,)) for name in ("x", "y", "z")},
    )
    np.testing.assert_allclose(lookup.value, (3.0,), atol=1e-5, rtol=0)


def test_fixed_singular_law_refuses_an_off_support_initial_state() -> None:
    """A physical point off the exact singular support is refused, not lowered."""
    model = _three_shock_model(cov=_SINGULAR)
    solution = model.solve(params={}, log_level="off")
    # (x, y) = (0, 0) forces z = 0 on the support; z = 0.5 lies off it.
    with pytest.raises(InvalidInitialConditionsError, match="singular"):
        model.simulate(
            params={},
            solution=solution,
            initial_conditions={
                "x": jnp.asarray((0.0,)),
                "y": jnp.asarray((0.0,)),
                "z": jnp.asarray((0.5,)),
                "period": jnp.zeros(1, dtype=jnp.int32),
                "regime_id": jnp.zeros(1, dtype=jnp.int32),
            },
            seed=521,
            log_level="off",
        )
