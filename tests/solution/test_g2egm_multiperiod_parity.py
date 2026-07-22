"""Chaining the four-segment G2EGM step across periods reproduces the brute solve.

The single-segment (`ucon`-only) step cannot be chained: its constrained-corner
extrapolation poisons the next period's Euler inversion, so a backward loop collapses to
`NaN`. The four-segment upper envelope covers those corners and masks invalid
candidates, so the value it publishes can be fed back as the next period's continuation.
A backward loop `V_dead -> V1 -> V0` then matches the dense grid-search solve across the
pension interior at each period.

The top pension boundary is an intrinsic grid-extent limit, not an algorithm gap: a
target at the top of the pension grid grows its post-decision pension balance past the
grid maximum, so its optimal continuation is off-grid. The brute solve fills that edge
with a boundary-clamped value while the envelope marks it an uncovered `-inf` hole, and
that hole advances one pension column inward per backward period. The interior — every
column the hole has not reached — matches; the comparison excludes the boundary layer.
"""

import jax.numpy as jnp
import numpy as np

from _lcm.egm.two_asset_g2egm_step import g2egm_step
from tests.test_models.deterministic.two_asset import get_model, get_params

_P = {
    "discount_factor": 0.95,
    "crra": 2.0,
    "match_rate": 1.0,
    "return_liquid": 0.02,
    "return_pension": 0.06,
    "wage": 10.0,
}
_M_GRID = jnp.linspace(1.0, 100.0, 12)
_N_GRID = jnp.linspace(0.0, 50.0, 10)
_A_GRID = jnp.linspace(0.0, 85.0, 18)
_B_GRID = jnp.linspace(0.0, 46.0, 16)
_CONSUMPTION_GRID = jnp.linspace(0.5, 90.0, 18)


def _g2egm_step(next_value):
    return g2egm_step(
        next_value=next_value,
        m_grid=_M_GRID,
        n_grid=_N_GRID,
        a_grid=_A_GRID,
        b_grid=_B_GRID,
        consumption_grid=_CONSUMPTION_GRID,
        **_P,
    ).value


def _solve_chain():
    """Backward-induct the three-period two-asset model with the G2EGM step.

    Returns the chained EGM values `(V1, V0)` and the brute reference
    `(brute[1]["working"], brute[0]["working"])`.
    """
    model = get_model(n_periods=3)
    params = get_params(n_periods=3, pension_bequest_weight=0.5)
    brute = model.solve(params=params, log_level="off")
    v_dead = jnp.asarray(brute[2]["dead"])
    v1 = _g2egm_step(v_dead)
    v0 = _g2egm_step(v1)
    return (
        np.asarray(v1),
        np.asarray(v0),
        np.asarray(brute[1]["working"]),
        np.asarray(brute[0]["working"]),
    )


# V1 is one backward step from the terminal value, so only its top pension column is an
# uncovered hole; V0 is two steps, so its top two columns are. The interior is what
# remains once the hole has been excluded at each period.
_V1_INTERIOR = np.s_[:, :9]
_V0_INTERIOR = np.s_[:, :8]


def test_chained_g2egm_matches_brute_on_the_pension_interior():
    """Both the single and the chained step match brute where the grid covers."""
    v1, v0, b1, b0 = _solve_chain()

    assert np.isfinite(v1[_V1_INTERIOR]).all()
    rel1 = np.abs(v1[_V1_INTERIOR] - b1[_V1_INTERIOR]) / np.abs(b1[_V1_INTERIOR])
    assert np.median(rel1) < 0.01
    assert np.percentile(rel1, 95) < 0.10

    assert np.isfinite(v0[_V0_INTERIOR]).all()
    rel0 = np.abs(v0[_V0_INTERIOR] - b0[_V0_INTERIOR]) / np.abs(b0[_V0_INTERIOR])
    assert np.median(rel0) < 0.02
    assert np.percentile(rel0, 95) < 0.15


def test_chained_g2egm_value_is_monotone_in_both_assets_on_the_interior():
    """More liquid and more pension stay weakly valuable through the chained solve."""
    _v1, v0, _b1, _b0 = _solve_chain()
    interior = v0[_V0_INTERIOR]
    assert np.all(np.diff(interior, axis=0) >= -1e-6)
    assert np.all(np.diff(interior, axis=1) >= -1e-6)


def test_pension_hole_advances_one_column_inward_per_backward_period():
    """The uncovered top-pension hole grows by exactly one column each period.

    A target whose optimal post-decision pension exceeds the grid is an uncovered hole;
    reading that hole as next period's continuation makes the adjacent column uncovered
    too, so the boundary layer thickens by one column per backward step.
    """
    v1, v0, _b1, _b0 = _solve_chain()
    v1_hole = ~np.isfinite(v1)
    v0_hole = ~np.isfinite(v0)
    # V1: only the top pension column is a hole.
    assert v1_hole[:, 9].all()
    assert not v1_hole[:, :9].any()
    # V0: the top two pension columns are holes; everything interior is finite.
    assert v0_hole[:, 8:].all()
    assert not v0_hole[:, :8].any()
