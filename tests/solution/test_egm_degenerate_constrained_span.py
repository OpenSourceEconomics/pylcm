"""The credit-constrained candidate segment is empty when its span is degenerate.

The constrained segment covers resources between the borrowing limit and the
lowest endogenous point. When that interval is empty — the lowest endogenous
point sits at or below the limit, the exogenous grid never rises above it, or
the point is not a finite number — there are no credit-constrained candidates
at all, and the segment must be dead rather than sampled at a manufactured
width.

Sampling such a segment at the dtype's smallest normal produces actions around
`1e-42`, whose utility under a curved felicity is astronomically negative. Those
candidates are not economically meaningful, and their magnitude defeats the
upper envelope's certified-sign arithmetic, poisoning the whole row.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.step_core import _compute_constrained_candidates

N_CONSTRAINED = 20
CONSTRAINED_RATIO = 1.6237766


def _utility_of_action(action):
    """CRRA felicity at `gamma = 2`, so a near-zero action blows up."""
    return -1.0 / action


def _candidates(*, first_endogenous_point, publish_resources, borrowing_limit):
    return _compute_constrained_candidates(
        first_endogenous_point=jnp.asarray(first_endogenous_point),
        publish_resources=jnp.asarray(publish_resources),
        borrowing_limit=jnp.asarray(borrowing_limit),
        n_constrained=N_CONSTRAINED,
        constrained_ratio=CONSTRAINED_RATIO,
        utility_of_action=_utility_of_action,
        discounted_expected_value_at_limit=jnp.asarray(1.0),
    )


@pytest.mark.parametrize(
    ("case", "first_endogenous_point", "publish_resources"),
    [
        ("first endogenous point exactly at the limit", 0.01, [0.01, 5.0, 20.0]),
        ("first endogenous point below the limit", 0.005, [0.01, 5.0, 20.0]),
        ("exogenous grid never rises above the limit", 20.0, [0.002, 0.005, 0.01]),
        ("first endogenous point is not a number", np.nan, [0.01, 5.0, 20.0]),
    ],
)
def test_degenerate_span_yields_no_constrained_candidates(
    case, first_endogenous_point, publish_resources
):
    """A degenerate constrained interval contributes only dead candidates."""
    actions, values = _candidates(
        first_endogenous_point=first_endogenous_point,
        publish_resources=publish_resources,
        borrowing_limit=0.01,
    )

    assert np.isnan(np.asarray(actions)).all(), case
    assert np.isnan(np.asarray(values)).all(), case


def test_a_positive_span_still_samples_the_constrained_segment():
    """A genuine constrained interval keeps its geometrically spaced candidates."""
    actions, values = _candidates(
        first_endogenous_point=2.01,
        publish_resources=[0.01, 5.0, 20.0],
        borrowing_limit=0.01,
    )

    actions = np.asarray(actions)
    assert np.isfinite(actions).all()
    assert (np.diff(actions) > 0).all()
    np.testing.assert_allclose(actions[-1], 2.0, rtol=1e-5)
    assert np.isfinite(np.asarray(values)).all()
