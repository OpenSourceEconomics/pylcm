"""Every admitted runtime partition of one compiled envelope publishes identical rows.

The construction's claim is bit identity, not agreement to a tolerance: the
static compilation envelope (`max_*`, `pair_*`) is the whole compilation key, so
the ride and branch partition sizes reach the executable as scalar operands and
cannot change the vectorized kernel XLA emits. A partition that reassociated a
floating expression would show up here as an exact-comparison failure.

The two families below contribute 48 and 78 comparisons, 126 in total, and the
counts are asserted so that shrinking the sweep cannot quietly pass.
"""

import jax
import jax.numpy as jnp
import pytest

from tests.solution._fused_continuation_envelope_oracle import (
    FusedStreamingConfig,
    build_fused_continuation_envelope,
)
from tests.solution.test_fused_continuation_envelope import (
    assert_result_equal,
    callbacks,
    payloads,
)

pytestmark = pytest.mark.usefixtures("x64_enabled")

_DTYPES = (jnp.float32, jnp.float64)

# Every admitted partition of a 20x20 envelope at microtile 4x4, i.e. W = 16.
_W16_ENVELOPE = FusedStreamingConfig(20, 20, 4, 4)
_W16_PARTITIONS = tuple(
    (ride, branch) for ride in (4, 8, 12, 16, 20) for branch in (4, 8, 12, 16, 20)
)

# Wider microtiles and rectangles that are not square, at the widths an A100
# profile would choose between.
_NEIGHBOURHOOD = (
    (
        FusedStreamingConfig(20, 20, 4, 4),
        ((4, 4), (8, 4), (4, 8), (8, 12), (16, 4), (20, 20)),
    ),
    (
        FusedStreamingConfig(24, 20, 8, 4),
        ((8, 4), (16, 4), (8, 8), (24, 12), (24, 20)),
    ),
    (
        FusedStreamingConfig(64, 20, 32, 4),
        ((32, 4), (32, 8), (32, 20), (64, 4), (64, 20)),
    ),
)

# Leading shapes that exercise a full rectangle, a partial one, and a remainder.
_SHAPES = ((9, 7), (17, 13), (31, 20))


def _compiled(*, config: FusedStreamingConfig, dtype):
    """Compile one executable for a static envelope and microtile width."""
    continuation_batch, envelope_batch = callbacks(
        width=config.pair_vector_width, dtype=dtype
    )
    return jax.jit(
        build_fused_continuation_envelope(
            continuation_batch=continuation_batch,
            envelope_batch=envelope_batch,
            config=config,
        )
    )


def _disagreements(*, core, args, partitions) -> tuple[int, int]:
    """Compare every partition against the first, returning (compared, differing)."""
    baseline = core(*args, jnp.int32(partitions[0][0]), jnp.int32(partitions[0][1]))
    jax.block_until_ready(baseline)
    compared = 0
    differing = 0
    for ride_size, branch_size in partitions[1:]:
        result = core(*args, jnp.int32(ride_size), jnp.int32(branch_size))
        jax.block_until_ready(result)
        compared += 1
        try:
            assert_result_equal(result, baseline)
        except AssertionError:
            differing += 1
    return compared, differing


def test_every_admitted_partition_of_one_envelope_agrees_bit_for_bit() -> None:
    """All 24 non-baseline partitions of a 20x20 W=16 envelope agree, at both dtypes."""
    compared = 0
    differing = 0
    for dtype in _DTYPES:
        core = _compiled(config=_W16_ENVELOPE, dtype=dtype)
        args = payloads(n_rides=31, n_branches=20, dtype=dtype)
        one_compared, one_differing = _disagreements(
            core=core, args=args, partitions=_W16_PARTITIONS
        )
        compared += one_compared
        differing += one_differing
    assert (compared, differing) == (48, 0)


def test_partitions_agree_across_microtile_widths_and_remainders() -> None:
    """Wider microtiles, non-square rectangles and partial trailing blocks all agree."""
    compared = 0
    differing = 0
    for dtype in _DTYPES:
        for config, partitions in _NEIGHBOURHOOD:
            core = _compiled(config=config, dtype=dtype)
            for n_rides, n_branches in _SHAPES:
                args = payloads(n_rides=n_rides, n_branches=n_branches, dtype=dtype)
                one_compared, one_differing = _disagreements(
                    core=core, args=args, partitions=partitions
                )
                compared += one_compared
                differing += one_differing
    assert (compared, differing) == (78, 0)


def test_separately_compiled_envelopes_do_disagree() -> None:
    """Two executables for the same microtile disagree where one executable does not.

    The positive control for the two suites above. Holding the microtile at 4x4
    and compiling a 4x4 and an 8x4 static rectangle produces two compilation
    keys, which is the configuration whose reassociation the single-envelope
    construction exists to remove. If this ever reports agreement the exact
    comparison has stopped discriminating and the zero counts above are vacuous.
    """
    dtype = jnp.float64
    args = payloads(n_rides=9, n_branches=7, dtype=dtype)
    narrow = _compiled(config=FusedStreamingConfig(4, 4, 4, 4), dtype=dtype)
    wide = _compiled(config=FusedStreamingConfig(8, 4, 4, 4), dtype=dtype)
    from_narrow = narrow(*args, jnp.int32(4), jnp.int32(4))
    from_wide = wide(*args, jnp.int32(4), jnp.int32(4))
    jax.block_until_ready(from_narrow)
    jax.block_until_ready(from_wide)
    with pytest.raises(AssertionError):
        assert_result_equal(from_wide, from_narrow)
