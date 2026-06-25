"""LTM and MSS must not bridge unrelated branches into a spurious envelope.

Both backends infer segment topology from the candidate ordering: LTM treats
every consecutive input pair as one envelope segment, and MSS starts a new
segment only where the grid or value decreases. Neither holds when a
discrete-choice switch raises both the endogenous resource and the value — the
two unrelated branch endpoints are then linked into a segment that never
existed, and the envelope is read off that phantom bridge. The exact envelope
(the host oracle, which consumes explicit branch labels) is the reference.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.upper_envelope import ltm, mss
from tests.solution._envelope_oracle import exact_envelope

# Three branches fed in the order A, B, C:
# - A: x in {0, 1}, V = {0, 1}, policy 0;
# - B: x in {2, 3}, V = {4, 5}, policy 10;
# - C: x in {1.5, 1.75}, V = {0.5, 0.5}, policy 5.
# At x = 1.5 only branch C is defined, with value 0.5. A topology that bridges
# A's endpoint (1, 1) to B's endpoint (2, 4) instead reports the phantom 2.5.
_ENDOG = jnp.array([0.0, 1.0, 2.0, 3.0, 1.5, 1.75])
_VALUE = jnp.array([0.0, 1.0, 4.0, 5.0, 0.5, 0.5])
_POLICY = jnp.array([0.0, 0.0, 10.0, 10.0, 5.0, 5.0])
_SEGMENT = jnp.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])


def _published_value_at(backend, x_query):
    grid, _policy, value, _n = backend.refine_envelope(
        endog_grid=_ENDOG, policy=_POLICY, value=_VALUE, n_refined=24
    )
    return float(
        interp_on_padded_grid(x_query=jnp.array([x_query]), xp=grid, fp=value)[0]
    )


@pytest.mark.parametrize("backend", [ltm, mss])
@pytest.mark.xfail(
    reason="LTM/MSS infer topology from ordering and bridge unrelated branches",
    strict=True,
)
def test_backend_matches_oracle_on_a_non_bridging_branch(backend):
    """The backend reports branch C's value at x=1.5, not the A-to-B bridge.

    The exact envelope at x=1.5 is branch C's 0.5; a backend that bridges A's
    endpoint to B's reports 2.5 instead.
    """
    oracle_value, _policy, _winner = exact_envelope(
        endog_grid=np.asarray(_ENDOG),
        value=np.asarray(_VALUE),
        policy=np.asarray(_POLICY),
        segment_id=np.asarray(_SEGMENT),
        x_query=np.array([1.5]),
    )
    np.testing.assert_allclose(oracle_value, [0.5], atol=1e-12)

    published = _published_value_at(backend, 1.5)
    np.testing.assert_allclose(published, 0.5, atol=1e-9)
