"""G1 — kinked-toy parity for the NEGM solver (GPU-only).

NEGM solves the kinked two-asset toy (`tests/test_models/negm_kinked_toy.py`) by
an outer search over the durable post-decision `next_illiquid` and an inner 1-D
DC-EGM solve on `wealth`. The committed brute oracle for the equivalent spec
(`negm_phase0/kinked_toy_oracle.py`, §2 of
`negm_phase0/negm-phase0-findings.md`) restricts the continuous policy to the
action grid, so it is a lower bound on the off-grid NEGM value: the parity
criterion is **NEGM ≥ brute (weak improvement) and NEGM → dense-brute as the
brute action grids refine**, not bit-parity.

The whole module is skipped: solving this model OOMs the local box (DC-EGM /
NEGM solves are GPU-only, see `feedback_no_heavy_tests_local`). Run it on gpu-01.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_models import negm_kinked_toy

pytestmark = pytest.mark.skip(
    reason="gpu-01 only: NEGM/DC-EGM solve OOMs the local box"
)

# Period-0, regime `alive`, shape (N_X, N_Z) brute oracle from
# `negm_phase0/negm-phase0-findings.md` §2 (the equivalent-spec brute solve).
_ORACLE_CELLS = {
    (0, 0): -0.4002052501,
    (0, 6): -0.2250119814,
    (6, 0): -0.2334363467,
    (6, 6): -0.1619449719,
    (11, 11): -0.1331192126,
}
_ORACLE_MIN = -0.40020525
_ORACLE_MAX = -0.13311921

_PARAMS = {"discount_factor": 0.95, "alive": {}}


def _solve_period0_alive() -> jnp.ndarray:
    """Solve the kinked-toy NEGM model and return period-0 `alive` value array."""
    solution = negm_kinked_toy.build_model().solve(params=_PARAMS, log_level="off")
    return solution[0]["alive"]


def test_negm_value_is_a_weak_improvement_over_the_brute_oracle():
    """NEGM's off-grid value weakly dominates the action-grid brute oracle.

    Brute restricts the continuous policy to the action grid, so its value is a
    lower bound; NEGM's inner EGM puts consumption off-grid, so at every listed
    coordinate `V_negm >= V_brute` up to interpolation tolerance.
    """
    v0 = _solve_period0_alive()
    for (ix, iz), brute_value in _ORACLE_CELLS.items():
        negm_value = float(v0[ix, iz])
        assert negm_value >= brute_value - 1e-4, (
            f"NEGM value {negm_value} at ({ix}, {iz}) is below the brute oracle "
            f"lower bound {brute_value}."
        )


def test_negm_value_approaches_the_brute_oracle():
    """NEGM reproduces the brute oracle to a coarse-grid tolerance.

    With the mandatory outer candidate `s' = illiquid` and the inner savings
    grid split across the credit-card rate kink, NEGM tracks the dense-brute
    value; the committed brute cells are matched within the coarse-grid
    discretization band (it tightens as the brute action grids refine).
    """
    v0 = _solve_period0_alive()
    for (ix, iz), brute_value in _ORACLE_CELLS.items():
        np.testing.assert_allclose(float(v0[ix, iz]), brute_value, atol=2e-2)


def test_negm_value_is_monotone_increasing_in_both_assets():
    """The period-0 value rises with both liquid and illiquid wealth."""
    v0 = _solve_period0_alive()
    assert float(v0[0, 0]) <= _ORACLE_MAX
    assert float(v0[-1, -1]) >= _ORACLE_MIN
    row_diffs = jnp.diff(v0, axis=0)
    col_diffs = jnp.diff(v0, axis=1)
    assert bool(jnp.all(row_diffs >= -1e-6))
    assert bool(jnp.all(col_diffs >= -1e-6))


@pytest.mark.parametrize(
    ("marginal_continuation", "illiquid", "expected_consumption"),
    [
        (0.25, 0.0, 2.0),  # z = 0: no offset, c = m^{-1/2}
        (0.25, 6.0, 1.7),  # z = 6: c = 2.0 - iota * 6 = 2.0 - 0.3
        (1.0, 6.0, 0.7),  # m = 1: c = 1.0 - 0.3
    ],
)
def test_inverse_marginal_utility_subtracts_the_durable_flow_offset(
    marginal_continuation: float, illiquid: float, expected_consumption: float
):
    """`(u')^{-1}(m) = m^{-1/gamma} - iota*Z` at the durable node `Z`.

    Utility is `(c + iota*Z)^{1-gamma}/(1-gamma)`, so `u'(c) = (c + iota*Z)^{-gamma}`
    and inverting for the consumption action carries the `- iota*Z` offset the
    durable state contributes to the flow.
    """
    consumption = float(
        negm_kinked_toy.inverse_marginal_utility(
            marginal_continuation=marginal_continuation, illiquid=illiquid
        )
    )
    np.testing.assert_allclose(consumption, expected_consumption, atol=1e-12)
