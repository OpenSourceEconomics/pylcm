"""Wide integers from user functions keep their value at both precisions.

A float-role output written as `2**32`, or as a NumPy `int64`, is the float a
host conversion gives, eagerly and under JIT. A code keeps its width until the
off-grid check; a host code the active precision cannot hold is refused.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from _lcm.dtypes import as_canonical_float, as_code
from lcm import AgeGrid, InitialNodes, IrregSpacedGrid, Model, Regime, categorical
from lcm.typing import ContinuousState, FloatND, ScalarInt


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    final: ScalarInt


_FLOAT_CASES = [
    pytest.param(0, 0, id="python-zero-control"),
    pytest.param(1, 1, id="python-one-control"),
    pytest.param(2**24, 2**24, id="python-power24-control"),
    pytest.param(-(2**31), -(2**31), id="python-negative-power31"),
    pytest.param(2**31, 2**31, id="python-positive-power31"),
    pytest.param(-(2**32), -(2**32), id="python-negative-power32"),
    pytest.param(2**32, 2**32, id="python-positive-power32"),
    pytest.param(np.int64(2**32 + 1), 2**32 + 1, id="numpy-scalar-lowword-one"),
    pytest.param(
        np.asarray(2**32 + 1, dtype=np.int64), 2**32 + 1, id="numpy-0d-lowword-one"
    ),
]


def _host_float_reference(exact: int) -> np.ndarray:
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    # float(exact) is exact for every integer selected above (all below 2**53).
    # The host dtype conversion then supplies the precision's required rounding.
    return np.asarray(float(exact), dtype=dtype)


def _identity_wealth(*, wealth: ContinuousState) -> ContinuousState:
    return wealth


def _two_period_model(*, raw: float | np.int64 | np.ndarray, enable_jit: bool) -> Model:
    # Standalone version of the fixture in tests/test_callback_constant_outputs.py;
    # no test-module import or module-level JAX constant is needed.
    grid = IrregSpacedGrid(points=jnp.asarray([1.0, 2.0]))

    def utility(*, wealth: ContinuousState) -> FloatND:  # noqa: ARG001
        return raw  # ty: ignore[invalid-return-type]

    return Model(
        regimes={
            "alive": Regime(
                states={"wealth": grid},
                state_transitions={"wealth": _identity_wealth},
                functions={"utility": utility},
            ),
            "final": Regime(states={"wealth": grid}, functions={"utility": utility}),
        },
        regime_id_class=RegimeId,
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes=InitialNodes(by_age={0: "alive"}),
        edges={"alive": {"final": 0}},
        enable_jit=enable_jit,
        durable_identity=False,
    )


@pytest.mark.parametrize(("raw", "exact"), _FLOAT_CASES)
@pytest.mark.parametrize("enable_jit", [False, True])
def test_public_utility_matches_host_float_twin(
    *, raw: int | np.int64 | np.ndarray, exact: int, enable_jit: bool
) -> None:
    params = {"alive": {"koopmans_aggregator": {"discount_factor": 1.0}}, "final": {}}
    expected = _host_float_reference(exact)
    expected_alive = np.add(expected, expected, dtype=expected.dtype)
    twin = (
        _two_period_model(raw=float(exact), enable_jit=enable_jit)
        .solve(params=params, log_level="off")
        .values
    )
    actual = (
        _two_period_model(raw=raw, enable_jit=enable_jit)
        .solve(params=params, log_level="off")
        .values
    )
    for period, regime, oracle in (
        (0, "alive", expected_alive),
        (1, "final", expected),
    ):
        got = np.asarray(actual[period][regime])
        assert got.dtype == expected.dtype
        np.testing.assert_array_equal(got, np.asarray(twin[period][regime]))
        np.testing.assert_array_equal(got, np.full(2, oracle, dtype=expected.dtype))


@pytest.mark.parametrize(("raw", "exact"), _FLOAT_CASES)
@pytest.mark.parametrize("traced", [False, True])
def test_float_output_adapter_matches_host_reference(
    *, raw: int | np.int64 | np.ndarray, exact: int, traced: bool
) -> None:
    def call() -> FloatND:
        return as_canonical_float(value=raw, source="wide integer utility")

    got = jax.jit(call)() if traced else call()
    expected = _host_float_reference(exact)
    assert np.asarray(got).dtype == expected.dtype
    np.testing.assert_array_equal(np.asarray(got), expected)


@pytest.mark.parametrize("exact", [0, 1, 2**24])
def test_code_output_small_integer_controls(exact: int) -> None:
    got = np.asarray(as_code(value=exact, source="integer code control"))
    assert got.dtype.kind in "iu"
    assert int(got.item()) == exact


@pytest.mark.parametrize("dtype", [np.uint8, np.uint32, np.uint64])
@pytest.mark.parametrize("form", ["host-scalar", "host-0d", "jax"])
@pytest.mark.parametrize("traced", [False, True])
@pytest.mark.parametrize("exact", [0, 1])
def test_unsigned_code_becomes_an_int32_code(
    *, dtype: type[np.unsignedinteger], form: str, traced: bool, exact: int
) -> None:
    """An unsigned code converts to `int32` like a signed one, at both precisions.

    `uint64` and `int32` promote to a float, so promoting rather than converting
    would leave a `uint64` code as a float at double precision.
    """
    if form == "jax" and dtype is np.uint64 and not jax.config.x64_enabled:
        pytest.skip("JAX has no uint64 at single precision.")
    raw = {
        "host-scalar": lambda: dtype(exact),
        "host-0d": lambda: np.asarray(exact, dtype=dtype),
        "jax": lambda: jnp.asarray(exact, dtype=dtype),
    }[form]()

    def call() -> Array:
        return as_code(value=raw, source="Law 'unsigned_code'")

    got = (jax.jit(call) if traced else call)()
    assert got.dtype == jnp.int32
    assert int(got) == exact
