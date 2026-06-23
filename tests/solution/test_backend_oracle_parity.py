"""Parity of every upper-envelope backend against the exact branch-aware oracle.

The four production backends (FUES, RFC, LTM, MSS) must agree with the
independent host-side oracle on cases their algorithm is supposed to handle.
This closes the review's gap that MSS/LTM/RFC had no dedicated independent
oracle — they were cross-checked only against each other and a dense VFI on
solved values, never against an exact envelope on the candidate row itself.

The clean certification here is a *strictly dominated* second branch: one branch
lies entirely below the other over their overlap, so the correct envelope deletes
the dominated points with no crossing insertion — something every backend (even
the deletion-only RFC/LTM) must get right. Crossing-insertion cases where the
backends legitimately differ (RFC/LTM only delete points) are covered separately.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.upper_envelope.fues import refine_envelope as refine_fues
from _lcm.egm.upper_envelope.ltm import refine_envelope as refine_ltm
from _lcm.egm.upper_envelope.mss import refine_envelope as refine_mss
from _lcm.egm.upper_envelope.rfc import refine_envelope as refine_rfc
from tests.solution._envelope_oracle import exact_envelope


def _run_backend(backend, *, endog_grid, policy, value, marginal_utility, n_refined):
    """Refine one candidate row with the named backend via its native signature."""
    if backend == "fues":
        return refine_fues(
            endog_grid=endog_grid, policy=policy, value=value, n_refined=n_refined
        )
    if backend == "rfc":
        return refine_rfc(
            endog_grid=endog_grid,
            policy=policy,
            value=value,
            marginal_utility=marginal_utility,
            n_refined=n_refined,
            search_radius=n_refined,
            jump_thresh=2.0,
        )
    if backend == "ltm":
        return refine_ltm(
            endog_grid=endog_grid, policy=policy, value=value, n_refined=n_refined
        )
    return refine_mss(
        endog_grid=endog_grid, policy=policy, value=value, n_refined=n_refined
    )


@pytest.mark.parametrize("backend", ["fues", "rfc", "ltm", "mss"])
def test_backend_deletes_a_strictly_dominated_branch(backend):
    """Every backend recovers the dominant branch when the other is strictly below.

    Branch 0 is `V(R) = R` (policy `0.3 R`); branch 1 is `V(R) = R - 1`
    (policy 10), strictly below branch 0 at every shared abscissa. The exact
    envelope is branch 0, with all of branch 1 deleted. This is pure dominance —
    no crossing insertion — so every backend, including the deletion-only RFC and
    LTM, must reproduce the oracle.
    """
    seg0_x = [0.0, 1.0, 2.0, 3.0]
    seg1_x = [0.0, 1.0, 2.0, 3.0]
    endog_grid = jnp.asarray(seg0_x + seg1_x)
    value = jnp.asarray(seg0_x + [x - 1.0 for x in seg1_x])
    policy = jnp.asarray([0.3 * x for x in seg0_x] + [10.0 for _ in seg1_x])
    marginal_utility = jnp.ones_like(endog_grid)  # both branches have slope 1
    segment_id = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)

    grid, policy_out, value_out, _n = _run_backend(
        backend,
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        marginal_utility=marginal_utility,
        n_refined=32,
    )

    x_query = jnp.linspace(0.0, 3.0, 13)
    got_value = np.asarray(
        interp_on_padded_grid(x_query=x_query, xp=grid, fp=value_out)
    )
    got_policy = np.asarray(
        interp_on_padded_grid(x_query=x_query, xp=grid, fp=policy_out)
    )
    oracle_value, oracle_policy, _winner = exact_envelope(
        endog_grid=np.asarray(endog_grid),
        value=np.asarray(value),
        policy=np.asarray(policy),
        segment_id=segment_id,
        x_query=np.asarray(x_query),
    )

    np.testing.assert_allclose(got_value, oracle_value, atol=1e-6)
    np.testing.assert_allclose(got_policy, oracle_policy, atol=1e-6)


@pytest.mark.parametrize("backend", ["fues", "rfc", "ltm", "mss"])
def test_backend_is_identity_on_a_single_concave_branch(backend):
    """On a single concave branch every backend returns that branch's interpolant.

    A concave, increasing single branch has no dominated points and no crossings,
    so the refined envelope must equal the input function — the simplest possible
    certification, which every backend must pass.
    """
    endog_grid = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    value = jnp.asarray([0.0, 1.0, 1.7, 2.2, 2.5])  # concave, increasing
    policy = jnp.asarray([0.0, 0.6, 1.0, 1.3, 1.5])
    # Supgradient = local slope, falling (concave).
    marginal_utility = jnp.asarray([1.0, 0.85, 0.6, 0.45, 0.3])

    grid, _policy_out, value_out, _n = _run_backend(
        backend,
        endog_grid=endog_grid,
        policy=policy,
        value=value,
        marginal_utility=marginal_utility,
        n_refined=16,
    )

    x_query = jnp.linspace(0.0, 4.0, 17)
    got_value = np.asarray(
        interp_on_padded_grid(x_query=x_query, xp=grid, fp=value_out)
    )
    expected = np.interp(np.asarray(x_query), np.asarray(endog_grid), np.asarray(value))
    np.testing.assert_allclose(got_value, expected, atol=1e-6)
