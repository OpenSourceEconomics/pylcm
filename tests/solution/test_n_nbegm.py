"""`NNBEGM` — outer continuous grid search over an inner NB-EGM solve.

On a smooth two-asset model (no kinks, no jumps) the N-NB-EGM solve is
the same object as the nested DC-EGM solve: a keeper plus an outer sweep of
inner 1-D consumption-saving problems. The smooth toy pins the outer wrapper —
value agreement with `NEGM(inner=DCEGM)`, dense-brute consistency, invariance
to outer batching — before any breakpoint machinery enters.
"""

import numpy as np
import pytest

from lcm import NNBEGM, NormalIIDProcess
from lcm.exceptions import RegimeInitializationError
from lcm.solvers import NBEGM
from tests.test_models import n_nbegm_toy as toy

_PARAMS = {"discount_factor": 0.95}


def _nbegm_inner(**overrides: object) -> NBEGM:
    config: dict[str, object] = {
        "continuous_state": "wealth",
        "post_decision_function": "liquid_savings",
        "budget_target": "resources",
        "savings_grid": toy.SAVINGS_GRID,
    }
    config.update(overrides)
    return NBEGM(**config)  # ty: ignore[invalid-argument-type]


def test_rejects_outer_post_decision_equal_to_inner_post_decision() -> None:
    """The outer durable and inner liquid post-decision must be distinct."""
    with pytest.raises(RegimeInitializationError, match=r"post[-_]decision"):
        NNBEGM(
            inner=_nbegm_inner(),
            outer_action="illiquid_investment",
            outer_post_decision="liquid_savings",
            outer_grid=toy.OUTER_GRID,
        )


def test_rejects_stochastic_outer_grid() -> None:
    """The outer grid is an exogenous search grid, never a stochastic process."""
    with pytest.raises(RegimeInitializationError, match="stochastic"):
        NNBEGM(
            inner=_nbegm_inner(),
            outer_action="illiquid_investment",
            outer_post_decision="next_illiquid",
            outer_grid=NormalIIDProcess(
                n_points=5, gauss_hermite=True, mu=0.0, sigma=1.0
            ),
        )


def test_rejects_negative_outer_batch_size() -> None:
    """`outer_batch_size` is `0` (all nodes at once) or a positive chunk size."""
    with pytest.raises(RegimeInitializationError, match="outer_batch_size"):
        NNBEGM(
            inner=_nbegm_inner(),
            outer_action="illiquid_investment",
            outer_post_decision="next_illiquid",
            outer_grid=toy.OUTER_GRID,
            outer_batch_size=-1,
        )


def test_rejects_inner_without_explicit_continuous_state() -> None:
    """An inner NBEGM leaving `continuous_state` to inference is rejected."""
    with pytest.raises(RegimeInitializationError, match="continuous_state"):
        NNBEGM(
            inner=_nbegm_inner(continuous_state=None),
            outer_action="illiquid_investment",
            outer_post_decision="next_illiquid",
            outer_grid=toy.OUTER_GRID,
        )


def test_two_period_toy_agrees_with_nested_dcegm() -> None:
    """On the two-period smooth toy, `NNBEGM` tracks `NEGM(inner=DCEGM)`.

    A single alive period reads only the terminal carry, so this isolates the
    outer keeper/adjuster wrapper: both solvers sweep the identical candidate
    set with an exact 1-D inner solve, and the value functions agree up to the
    inner families' constrained-region representation at the poorest cells.
    """
    nested = toy.build_model(variant="n_nbegm", n_periods=2).solve(
        params=_PARAMS, log_level="off"
    )
    negm = toy.build_model(variant="negm", n_periods=2).solve(
        params=_PARAMS, log_level="off"
    )
    V_nested = np.asarray(nested[0]["alive"])
    V_negm = np.asarray(negm[0]["alive"])
    np.testing.assert_allclose(V_nested, V_negm, atol=0.15)
    # Away from the borrowing-constrained poorest cells and the state-grid
    # boundary, the two inner families integrate the same smooth Euler
    # equation, so agreement is tight.
    np.testing.assert_allclose(V_nested[2:-1, 1:-1], V_negm[2:-1, 1:-1], atol=2e-2)


def test_two_period_toy_tracks_dense_brute() -> None:
    """The nested solve closely tracks the dense two-action grid search.

    The two solvers optimize over different outer candidate sets — the nested
    solve enumerates a fixed post-decision grid of next-period stocks, brute
    an investment action grid whose induced next-stock candidates depend on
    the current state — so no directional (dominance) ordering exists between
    the values; agreement is asserted as an unsigned gap.
    """
    nested = toy.build_model(variant="n_nbegm", n_periods=2).solve(
        params=_PARAMS, log_level="off"
    )
    brute = toy.build_model(variant="brute", n_periods=2).solve(
        params=_PARAMS, log_level="off"
    )
    nested_V = np.asarray(nested[0]["alive"])
    brute_V = np.asarray(brute[0]["alive"])
    rel_gap = np.abs(nested_V - brute_V) / np.abs(brute_V)
    # A few cells near the adjust/no-adjust kink carry the candidate-set
    # difference at full size; agreement is a mean statement with a loose
    # per-cell cap.
    assert float(rel_gap.max()) < 0.25, f"max rel gap {float(rel_gap.max()):.4f}"
    assert float(rel_gap.mean()) < 0.01, f"mean rel gap {float(rel_gap.mean()):.4f}"


def test_three_period_toy_tracks_nested_dcegm_through_published_carries() -> None:
    """Chaining published nested carries, `NNBEGM` tracks `NEGM`.

    With two alive periods the parent reads the child's published outer
    envelope. Both solvers publish a bridged (finite-grid) envelope, so they
    share that approximation class and stay close everywhere — the gate for
    the topology-preserving publication is a separate, tighter deliverable.
    """
    nested = toy.build_model(variant="n_nbegm", n_periods=3).solve(
        params=_PARAMS, log_level="off"
    )
    negm = toy.build_model(variant="negm", n_periods=3).solve(
        params=_PARAMS, log_level="off"
    )
    for period in (0, 1):
        np.testing.assert_allclose(
            np.asarray(nested[period]["alive"]),
            np.asarray(negm[period]["alive"]),
            atol=0.2,
            err_msg=f"period {period}",
        )


@pytest.mark.parametrize("outer_batch_size", [1, 4, 100])
def test_outer_batch_size_is_value_invariant(outer_batch_size: int) -> None:
    """Chunking the outer sweep never changes the solved values."""
    reference = toy.build_model(variant="n_nbegm", n_periods=2).solve(
        params=_PARAMS, log_level="off"
    )
    chunked = toy.build_model(
        variant="n_nbegm", outer_batch_size=outer_batch_size, n_periods=2
    ).solve(params=_PARAMS, log_level="off")
    for period, regime_to_V in reference.items():
        for regime_name, V_arr in regime_to_V.items():
            np.testing.assert_array_equal(
                np.asarray(V_arr),
                np.asarray(chunked[period][regime_name]),
                err_msg=f"{regime_name} at period {period}",
            )
