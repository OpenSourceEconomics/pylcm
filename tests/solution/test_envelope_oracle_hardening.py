"""The host envelope oracle is robust enough to be the authoritative reference.

These pin the oracle hardening: folded branches, exact-maximum value reporting
independent of the tie tolerance, segment-label winners, loud failure on
malformed input, multivalued-branch rejection, right-continuous tie breaking,
and an event-complete query generator.
"""

import numpy as np
import pytest

from tests.solution._envelope_oracle import (
    TopologyError,
    envelope_event_points,
    exact_envelope,
)


def test_oracle_evaluates_a_folded_branch_as_a_polyline():
    """A non-monotone branch is the polyline through its vertices, in input order.

    The branch `(0, 0) -> (2, 10) -> (1, 0)` folds back in `x`. At `x = 1.5` the
    rising edge `(0, 0)-(2, 10)` gives `7.5` and the falling edge `(2, 10)-(1, 0)`
    gives `5`; the branch value is their maximum, `7.5` — not the `5` a sort-by-x
    would produce.
    """
    env_value, _policy, _winner = exact_envelope(
        endog_grid=np.array([0.0, 2.0, 1.0]),
        value=np.array([0.0, 10.0, 0.0]),
        policy=np.array([0.0, 0.0, 0.0]),
        segment_id=np.array([0.0, 0.0, 0.0]),
        x_query=np.array([1.5]),
    )
    np.testing.assert_allclose(env_value, [7.5], atol=1e-12)


def test_oracle_reports_exact_maximum_not_the_tie_winner_value():
    """The reported value is the exact branch maximum, not lowered by `tol`.

    Branch 1 exceeds branch 0 by `1e-10`, inside a `tol = 1e-9` tie band. The two
    are classified as tied, but the envelope value must still be the true maximum
    `1.0 + 1e-10`, not branch 0's `1.0`.
    """
    env_value, _policy, _winner = exact_envelope(
        endog_grid=np.array([0.0, 2.0, 0.0, 2.0]),
        value=np.array([1.0, 1.0, 1.0 + 1e-10, 1.0 + 1e-10]),
        policy=np.array([0.0, 0.0, 9.0, 9.0]),
        segment_id=np.array([0.0, 0.0, 1.0, 1.0]),
        x_query=np.array([1.0]),
        tol=1e-9,
    )
    np.testing.assert_allclose(env_value, [1.0 + 1e-10], atol=1e-13)


def test_oracle_winner_is_the_segment_label_not_the_list_index():
    """`winner` returns the segment label, so non-index labels round-trip."""
    _value, _policy, winner = exact_envelope(
        endog_grid=np.array([0.0, 2.0, 0.0, 2.0]),
        value=np.array([0.0, 0.0, 1.0, 1.0]),  # branch 9 dominates
        policy=np.array([0.0, 0.0, 5.0, 5.0]),
        segment_id=np.array([4.0, 4.0, 9.0, 9.0]),
        x_query=np.array([1.0]),
    )
    np.testing.assert_array_equal(winner, [9.0])


@pytest.mark.parametrize(
    ("value", "policy", "segment_id"),
    [
        ([0.0, np.nan], [0.0, 0.0], [0.0, 0.0]),  # non-finite value
        ([0.0, 1.0], [0.0, np.nan], [0.0, 0.0]),  # non-finite policy
        ([0.0, 1.0], [0.0, 0.0], [0.0, np.nan]),  # non-finite segment label
    ],
)
def test_oracle_fails_loudly_on_nonfinite_live_input(value, policy, segment_id):
    """A live candidate with a non-finite value/policy/label raises, not crashes."""
    with pytest.raises(TopologyError):
        exact_envelope(
            endog_grid=np.array([0.0, 1.0]),
            value=np.array(value),
            policy=np.array(policy),
            segment_id=np.array(segment_id),
            x_query=np.array([0.5]),
        )


def test_oracle_rejects_a_multivalued_branch():
    """A branch revisiting an abscissa with a different policy is a topology error."""
    with pytest.raises(TopologyError):
        exact_envelope(
            endog_grid=np.array([0.0, 1.0, 1.0]),
            value=np.array([0.0, 1.0, 1.0]),
            policy=np.array([0.0, 3.0, 7.0]),  # x=1 carries two policies
            segment_id=np.array([0.0, 0.0, 0.0]),
            x_query=np.array([0.5]),
        )


def test_oracle_breaks_an_exact_tie_right_continuously():
    """At an on-node crossing the winning policy is the branch that wins to the right.

    Branch 0 (policy 3) rises with slope 1/3; branch 1 (policy 0.5) rises with
    slope 2; they cross at `R = 11` with equal value 2. Just right of 11 branch 1
    is higher, so the published policy at the crossing is branch 1's 0.5 and the
    winner is segment 1 — the `side="right"` convention.
    """
    env_value, env_policy, winner = exact_envelope(
        endog_grid=np.array([10.0, 11.0, 12.0, 10.0, 11.0, 12.0]),
        value=np.array([5 / 3, 2.0, 7 / 3, 0.0, 2.0, 4.0]),
        policy=np.array([3.0, 3.0, 3.0, 0.5, 0.5, 0.5]),
        segment_id=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        x_query=np.array([10.5, 11.0, 11.5]),
    )
    np.testing.assert_allclose(env_value, [11 / 6, 2.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(env_policy, [3.0, 0.5, 0.5], atol=1e-12)
    np.testing.assert_array_equal(winner, [0.0, 1.0, 1.0])


def test_event_points_include_the_branch_crossing():
    """The event-complete query set contains the exact branch-crossing abscissa.

    Branch 0 is `V = R`, branch 1 is `V = 1 - R`; they cross at `R = 0.5`. The
    event generator must surface `0.5` so no sampling gap can hide the kink.
    """
    events = envelope_event_points(
        endog_grid=np.array([0.0, 1.0, 0.0, 1.0]),
        value=np.array([0.0, 1.0, 1.0, 0.0]),
        policy=np.array([0.0, 0.0, 10.0, 10.0]),
        segment_id=np.array([0.0, 0.0, 1.0, 1.0]),
    )
    assert np.any(np.isclose(events, 0.5, atol=1e-12))
