"""A constrained segment's existence is a local, coordinate-free property.

Below the first endogenous point the borrowing limit binds and the policy is
the closed form `R - borrowing_limit`. `_compute_constrained_candidates`
samples that segment, and it declines to produce candidates in exactly one
circumstance: the interval is not an interval, because its width is not a
positive finite number.

That is the only admissible emptiness test, and these tests fence it against a
magnitude-based one — a rule deleting the segment when its consumption is small
relative to the resources on hand.

The distinction the fence turns on is between *which points get sampled* and
*whether there is anything to sample*. The two are not the same property and
only the second is semantic:

- **Where the sample lands is allowed to depend on the batch.** The span is
  capped at `min(first_endogenous_point, max(publish_resources))`, so a row
  carrying a distant resource samples the whole interval while a row carrying
  only a near query samples up to that query. That is adaptive resolution
  doing its job: it never samples above the true bound, and it never samples
  below the largest query it must serve. The row and scalar callers therefore
  produce different point sets for the same economics, legitimately.
- **Whether the segment exists at all may not depend on the batch.** A live
  interval must yield a full set of finite candidates whichever caller arrives
  and whatever else shares the row, every sampled point must carry the exact
  closed-form value of its own action, and no sampled point may leave the
  interval. Those three hold under every placement below.

A magnitude guard fails all three at once: it empties a live interval on
evidence drawn from elsewhere in the row. It also fails a fourth property no
test here needs to state, because it is visible by inspection — adding a common
level to every resource leaves every available-resource span untouched, yet
flips any predicate that compares a consumption against an absolute resource
magnitude.

The cost of getting this wrong is a wrong policy at full size, not a rounding.
Deleting a live segment removes its nodes from the refined envelope, and the
envelope is carried to the parent unfloored — `_publish_V_and_carry_rows`
floors only the published `V_row`. A parent query below the first surviving
node continues that bracket's secant (see `interp_on_padded_grid`), so the
continuation is read off an extrapolant instead of off the constrained branch.
Under log felicity, deleting the segment and reading the secant from the first
Euler bracket overstates the continuation at the segment's midpoint by exactly
`0.5 * log(2)`; a rival candidate placed between the two values reverses the
discrete choice.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.egm.step_core import _compute_constrained_candidates
from tests.conftest import DECIMAL_PRECISION

_N_CONSTRAINED = 64
_CONSTRAINED_RATIO = (1.0 / 1e-4) ** (1.0 / (_N_CONSTRAINED - 1))


def _utility(consumption: jnp.ndarray) -> jnp.ndarray:
    """Return log felicity, the case where the omission error is exactly known."""
    return jnp.log(consumption)


def _candidates(
    *,
    first_endogenous_point: float,
    publish_resources: float | list[float],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the constrained candidate actions and values at a zero limit."""
    return _compute_constrained_candidates(
        first_endogenous_point=jnp.asarray(first_endogenous_point),
        publish_resources=jnp.asarray(publish_resources),
        borrowing_limit=jnp.asarray(0.0),
        n_constrained=_N_CONSTRAINED,
        constrained_ratio=_CONSTRAINED_RATIO,
        utility_of_action=_utility,
        discounted_expected_value_at_limit=jnp.asarray(0.0),
    )


_FIRST_ENDOGENOUS_POINT = 1e-6
_QUERY_INSIDE_SEGMENT = 0.5 * _FIRST_ENDOGENOUS_POINT

_PLACEMENTS = {
    "scalar caller": _QUERY_INSIDE_SEGMENT,
    "row, query alone": [_QUERY_INSIDE_SEGMENT],
    "row, distant O(1)": [_QUERY_INSIDE_SEGMENT, 1.0],
    "row, distant O(1e7)": [_QUERY_INSIDE_SEGMENT, 1e7],
    "row, two distant": [_QUERY_INSIDE_SEGMENT, 1.0, 1e7],
}


@pytest.mark.parametrize("placement", list(_PLACEMENTS))
def test_the_segment_survives_every_batch_composition(placement: str) -> None:
    """A live segment yields a full set of finite candidates however it is called."""
    actions, values = _candidates(
        first_endogenous_point=_FIRST_ENDOGENOUS_POINT,
        publish_resources=_PLACEMENTS[placement],
    )

    assert np.isfinite(np.asarray(actions)).all()
    assert np.isfinite(np.asarray(values)).all()
    assert len(actions) == _N_CONSTRAINED


@pytest.mark.parametrize("placement", list(_PLACEMENTS))
def test_every_candidate_sits_on_the_exact_constrained_branch(placement: str) -> None:
    """Sampled candidates carry the closed-form value of their own action."""
    actions, values = _candidates(
        first_endogenous_point=_FIRST_ENDOGENOUS_POINT,
        publish_resources=_PLACEMENTS[placement],
    )

    aaae(np.asarray(values), np.log(np.asarray(actions)), decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize("placement", list(_PLACEMENTS))
def test_the_sample_never_leaves_the_constrained_interval(placement: str) -> None:
    """No sampled action exceeds the Euler-inverted consumption that bounds it.

    The top of the ladder is reached by `_N_CONSTRAINED` successive
    multiplications by the geometric ratio, so it carries that many rounding
    steps and lands within them of the bound rather than exactly on it.
    """
    actions, _ = _candidates(
        first_endogenous_point=_FIRST_ENDOGENOUS_POINT,
        publish_resources=_PLACEMENTS[placement],
    )

    sampled = np.asarray(actions)
    ladder_rounding = _N_CONSTRAINED * float(np.finfo(sampled.dtype).eps)

    assert (sampled > 0.0).all()
    assert sampled.max() <= _FIRST_ENDOGENOUS_POINT * (1.0 + ladder_rounding)
