"""The bridged read's cliff-cell error has no guaranteed sign under Hermite.

For a positive *linear* read, interpolating the pointwise-maximized (bridged)
value row can only overstate the true envelope `max_b I[V_b]` between nodes.
The production value read is monotone-limited cubic Hermite, which is not a
linear operator of the node values: the aggregate row inherits the winning
branch's node slopes, and the Fritsch-Carlson limiter can pull the aggregate
interpolant *below* every per-branch read. The configuration pinned here
exhibits that understatement, so the bridged mode's signed cliff-cell error is
an empirical quantity — only one-sided representation removes it by
construction.
"""

import jax.numpy as jnp

from _lcm.egm.interp import interp_on_padded_grid

_XP = jnp.asarray([0.0, 1.0, 2.0])
_QUERY = jnp.asarray([1.5])

_F_BRANCH_1 = jnp.asarray([0.5937488107877469, 0.871928750162716, 3.1371159946104377])
_F_BRANCH_2 = jnp.asarray([0.28782561949508034, 0.8941053870984388, 3.4259209469034624])
_SLOPES_1 = jnp.asarray([3.818460500549095, 5.968486512864201, 0.14918940477172948])
_SLOPES_2 = jnp.asarray([0.7988903116888513, 0.4719995646153583, 5.041586936324435])


def _hermite_read(fp: jnp.ndarray, slopes: jnp.ndarray) -> float:
    return float(
        interp_on_padded_grid(x_query=_QUERY, xp=_XP, fp=fp, fp_slopes=slopes)[0]
    )


def test_bridged_hermite_read_can_understate_the_branch_envelope() -> None:
    """A monotone two-branch node configuration exists where the Hermite read
    of the aggregate-max row falls strictly below both per-branch Hermite
    reads — the linear-read overstatement guarantee does not transfer."""
    value_winner_slopes = jnp.where(_F_BRANCH_1 >= _F_BRANCH_2, _SLOPES_1, _SLOPES_2)
    aggregate = _hermite_read(
        jnp.maximum(_F_BRANCH_1, _F_BRANCH_2), value_winner_slopes
    )
    branch_envelope = max(
        _hermite_read(_F_BRANCH_1, _SLOPES_1),
        _hermite_read(_F_BRANCH_2, _SLOPES_2),
    )
    assert aggregate < branch_envelope - 1.0
    assert abs(aggregate - 1.5888147) < 1e-4
    assert abs(branch_envelope - 2.7319345) < 1e-4
