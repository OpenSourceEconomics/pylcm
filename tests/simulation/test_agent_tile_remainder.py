"""An agent population that is not a multiple of the tile width runs one body shape."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation.programs import AGENT_WIDTH_KEYWORD, _AgentTiled
from _lcm.solution.feasibility import BoundedResult, BoundInputStatus, QueryInputStatus
from lcm.typing import FloatND, IntND


def test_agent_tiled_traces_its_body_at_the_tile_width_only() -> None:
    """The remainder tile is padded to the tile width, so the body is traced once.

    A second trace at the remainder's shape would compile a second program for
    the last agents, whose rounding can differ from the full tiles'.
    """
    shapes = []

    def body(*, x: jax.Array, scale: jax.Array) -> jax.Array:
        shapes.append(x.shape)
        return x * scale + 1.0

    tiled = _AgentTiled(func=body, agent_arg_names=("x",))
    x = jnp.arange(10, dtype=jnp.float32)
    jax.jit(lambda x, s: tiled(x=x, scale=s, **{AGENT_WIDTH_KEYWORD: 4}))(
        x, jnp.float32(3.0)
    )
    assert shapes == [()]


def test_agent_tiled_returns_every_agent_once_with_a_padded_remainder() -> None:
    """Padding the remainder leaves exactly the population's rows, in order."""
    tiled = _AgentTiled(
        func=lambda *, x, scale: x * scale + 1.0, agent_arg_names=("x",)
    )
    x = jnp.arange(10, dtype=jnp.float32)
    out = tiled(x=x, scale=jnp.float32(3.0), **{AGENT_WIDTH_KEYWORD: 4})
    np.testing.assert_array_equal(np.asarray(out), np.arange(10) * 3.0 + 1.0)


@pytest.mark.parametrize("width", [2])
@pytest.mark.parametrize("compiled", [True])
def test_agent_tiled_preserves_empty_payload_axes_and_original_rows(
    *, width: int, compiled: bool
) -> None:
    """Empty count rosters retain their agent rows beside ordinary outputs."""

    def body(*, x: FloatND) -> tuple[FloatND, IntND, IntND]:
        code = x.astype(jnp.int32)
        return x + 1.0, jnp.zeros((0,), dtype=jnp.int32), jnp.stack((code, code + 10))

    tiled = _AgentTiled(func=body, agent_arg_names=("x",))

    def evaluate(x: FloatND) -> tuple[FloatND, IntND, IntND]:
        return cast(
            "tuple[FloatND, IntND, IntND]",
            tiled(x=x, **{AGENT_WIDTH_KEYWORD: width}),
        )

    x = jnp.arange(5, dtype=jnp.float32)
    values, empty_counts, codes = (jax.jit(evaluate) if compiled else evaluate)(x)
    assert values.dtype == x.dtype
    assert empty_counts.dtype == codes.dtype == jnp.int32
    assert empty_counts.shape == (5, 0)
    expected = np.arange(5, dtype=np.int32)
    np.testing.assert_array_equal(values, expected + 1.0)
    np.testing.assert_array_equal(codes, np.stack((expected, expected + 10), axis=1))


@pytest.mark.parametrize("width", [2])
@pytest.mark.parametrize("compiled", [True])
@pytest.mark.parametrize(
    ("active", "expected"),
    [
        ((False, False, False, False, False), False),
        ((False, True, True, True, False), False),
        ((True, True, True, True, False), True),
        ((True, True, True, True, True), True),
    ],
)
def test_agent_tiled_integrity_checks_only_actual_regime_holders(
    *, width: int, compiled: bool, active: tuple[bool, ...], expected: bool
) -> None:
    """Placeholder rows cannot invalidate holders, and actual bad rows still fail."""

    def body(*, x: FloatND) -> BoundedResult[tuple[IntND, FloatND]]:
        bad = (x == 0) | (x == 4)
        return BoundedResult(
            output=(x.astype(jnp.int32), x + 10.0),
            bound_status=(
                BoundInputStatus(nonfinite_floor=bad, invalid_shifted_grid=bad),
            ),
            query_status=(
                QueryInputStatus(
                    nonfinite_floor=bad,
                    invalid_shifted_grid=bad,
                    coverage_missing=bad,
                ),
            ),
        )

    tiled = _AgentTiled(
        func=body, agent_arg_names=("x",), n_bound_records=1, n_query_records=1
    )
    assert tiled.agent_shard_arg_names == ("x",)

    def evaluate(*, x: FloatND, holders: jax.Array) -> tuple[jax.Array, ...]:
        return cast(
            "tuple[jax.Array, ...]",
            tiled(
                x=x,
                _lcm_active_agents=holders,
                **{AGENT_WIDTH_KEYWORD: width},
            ),
        )

    output = (jax.jit(evaluate) if compiled else evaluate)(
        x=jnp.arange(5, dtype=jnp.float32), holders=jnp.asarray(active)
    )
    np.testing.assert_array_equal(output[0], np.arange(5, dtype=np.int32))
    np.testing.assert_array_equal(output[1], np.arange(5) + 10.0)
    np.testing.assert_array_equal(np.asarray(output[2:]), np.full((5,), expected))


@pytest.mark.parametrize("compiled", [True])
@pytest.mark.parametrize("active", [(False, False), (False, True)])
def test_stateless_agent_integrity_keeps_scalar_outputs(
    *, compiled: bool, active: tuple[bool, ...]
) -> None:
    """A shared scalar decision owns its status exactly when an agent acts."""

    def body() -> BoundedResult[tuple[IntND, FloatND]]:
        return BoundedResult(
            output=(jnp.int32(0), jnp.float32(2.0)),
            bound_status=(),
            query_status=(
                QueryInputStatus(
                    nonfinite_floor=jnp.asarray(a=False),
                    invalid_shifted_grid=jnp.asarray(a=False),
                    coverage_missing=jnp.asarray(a=True),
                ),
            ),
        )

    tiled = _AgentTiled(
        func=body, agent_arg_names=("_lcm_active_agents",), n_query_records=1
    )
    assert tiled.agent_shard_arg_names == ()

    def evaluate(*, holders: jax.Array) -> tuple[jax.Array, ...]:
        return cast(
            "tuple[jax.Array, ...]",
            tiled(_lcm_active_agents=holders, **{AGENT_WIDTH_KEYWORD: 2}),
        )

    outputs = (jax.jit(evaluate) if compiled else evaluate)(holders=jnp.asarray(active))
    assert all(output.shape == () for output in outputs)
    np.testing.assert_array_equal(outputs[:2], [0, 2.0])
    np.testing.assert_array_equal(outputs[2:], [False, False, any(active)])
