"""A link with positive width may never be certified into an exact tie.

`certified_margin_sign` scales abscissae by one power of two chosen from the
largest of them, which is what keeps its products inside the range where the
error-free transforms are exact. The scaling is shared, so a link far narrower
than that largest abscissa can have both endpoints round to the same number:
its width becomes zero, its contribution to the determinant vanishes, and two
strictly separated links are reported as tied.

A tie is a certificate — it licenses the caller to treat either link as owner —
so it may not be issued on geometry the transform silently flattened. The
unresolved verdict exists for exactly this case.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.upper_envelope.certified_sign import (
    UNRESOLVED_SIGN,
    certified_margin_sign,
)
from tests.conftest import X64_ENABLED


def _sign_for_width_ratio(dtype, jax_dtype, large_exp: int, small_exp: int) -> int:
    """Certified sign of `A - B` where B is much narrower than A.

    Both links rise from `-0.5` to `0.5` across their own span, so at half of B's
    half-width B is exactly `0.25` above A, whatever the ratio between them.
    """
    large = dtype(np.ldexp(1.0, large_exp))
    small = dtype(np.ldexp(1.0, small_exp))
    return int(
        certified_margin_sign(
            a_x0=jnp.asarray(-large, dtype=jax_dtype),
            a_x1=jnp.asarray(large, dtype=jax_dtype),
            a_v0=jnp.asarray(dtype(-0.5), dtype=jax_dtype),
            a_v1=jnp.asarray(dtype(0.5), dtype=jax_dtype),
            b_x0=jnp.asarray(-small, dtype=jax_dtype),
            b_x1=jnp.asarray(small, dtype=jax_dtype),
            b_v0=jnp.asarray(dtype(-0.5), dtype=jax_dtype),
            b_v1=jnp.asarray(dtype(0.5), dtype=jax_dtype),
            x_query=jnp.asarray(small / dtype(2), dtype=jax_dtype),
        )
    )


@pytest.mark.parametrize(
    ("dtype", "jax_dtype", "large_exp", "small_exp"),
    [
        (np.float32, jnp.float32, 47, -103),
        (np.float64, jnp.float64, 400, -675),
    ],
)
def test_a_collapsed_width_is_unresolved_rather_than_tied(
    dtype, jax_dtype, large_exp, small_exp
) -> None:
    """Where the shared scaling flattens the narrow link, the sign is unresolved.

    The narrow link is strictly above at the query, so the honest verdicts are
    `-1` or the fail-loud sentinel. `0` would certify a tie that does not exist,
    and `BELOW_RESOLUTION_SIGN` would claim the two are within a rounding of each
    other when they are `0.25` apart.
    """
    if (jax_dtype is jnp.float64) != X64_ENABLED:
        pytest.skip("dtype is not the configured working precision")
    assert _sign_for_width_ratio(dtype, jax_dtype, large_exp, small_exp) in (
        -1,
        UNRESOLVED_SIGN,
    )


@pytest.mark.parametrize("unit_shift", [-10, 0, 10])
def test_the_verdict_does_not_depend_on_the_choice_of_units(unit_shift: int) -> None:
    """Rescaling both links by a power of two cannot turn a strict sign into a tie.

    The determinant is homogeneous, so a common power-of-two change of units
    multiplies it by a positive constant and leaves its sign alone.
    """
    dtype, jax_dtype = (
        (np.float64, jnp.float64) if X64_ENABLED else (np.float32, jnp.float32)
    )
    large_exp, small_exp = (400, -675) if X64_ENABLED else (47, -103)
    assert _sign_for_width_ratio(
        dtype, jax_dtype, large_exp + unit_shift, small_exp + unit_shift
    ) in (-1, UNRESOLVED_SIGN)


def test_ordinary_width_ratios_still_resolve_strictly() -> None:
    """The fence does not cost resolution on geometry the transforms handle.

    A width ratio a real grid could produce keeps its strict verdict; only ratios
    wide enough to flatten a link fall back to unresolved.
    """
    dtype, jax_dtype = (
        (np.float64, jnp.float64) if X64_ENABLED else (np.float32, jnp.float32)
    )
    assert _sign_for_width_ratio(dtype, jax_dtype, 3, -17) == -1
