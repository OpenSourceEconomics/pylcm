"""`NNBEGM` — outer continuous grid search over an inner NB-EGM solve.

On a smooth two-asset model (no kinks, no jumps) the N-NB-EGM solve is
the same object as the nested DC-EGM solve: a keeper plus an outer sweep of
inner 1-D consumption-saving problems. The smooth toy pins the outer wrapper —
value agreement with `NEGM(inner=DCEGM)`, dense-brute consistency, invariance
to outer batching — before any breakpoint machinery enters.
"""

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

import _lcm.solution.nnbegm as nnbegm_module
from _lcm.egm.published_policy import NNBEGMSimPolicy
from lcm import NormalIIDProcess
from lcm.exceptions import RegimeInitializationError
from lcm.solvers import NBEGM, NNBEGM, FiniteOuterGrid
from lcm.typing import ContinuousAction, ContinuousState
from tests.test_models import n_nbegm_toy as toy

_PARAMS = {"discount_factor": 0.95}


def _nbegm_inner() -> NBEGM:
    return NBEGM(
        savings_grid=toy.SAVINGS_GRID,
        envelope_arithmetic="ordinary",
    )


def _cubic_outer_target(
    illiquid: ContinuousState, illiquid_investment: ContinuousAction
) -> ContinuousState:
    return illiquid + illiquid_investment**3


def _zero_slope_outer_target(
    illiquid: ContinuousState, illiquid_investment: ContinuousAction
) -> ContinuousState:
    return illiquid + 0 * illiquid_investment


def _affine_outer_target(
    illiquid: ContinuousState, illiquid_investment: ContinuousAction
) -> ContinuousState:
    return illiquid + 2 * illiquid_investment


def test_public_nnbegm_contains_only_numerical_configuration() -> None:
    """DAG role names live on the regime-owned margins, not the solver."""
    solver = NNBEGM(
        inner=_nbegm_inner(), outer_search=FiniteOuterGrid(grid=toy.OUTER_GRID)
    )
    assert set(solver.__dataclass_fields__) == {
        "inner",
        "outer_search",
    }


def test_rejects_stochastic_outer_grid() -> None:
    """The outer grid is an exogenous search grid, never a stochastic process."""
    with pytest.raises(RegimeInitializationError, match="stochastic"):
        NNBEGM(
            inner=_nbegm_inner(),
            outer_search=FiniteOuterGrid(
                grid=NormalIIDProcess(n_points=5, gauss_hermite=True, mu=0.0, sigma=1.0)
            ),
        )


def test_constructing_nnbegm_with_a_non_nbegm_inner_is_refused() -> None:
    """`NNBEGM` cannot be built around an inner it does not support.

    Two mechanisms refuse it and which one fires depends on whether runtime
    type checking is active, so both are accepted here; that the explicit
    structural guard exists is asserted separately against the guard itself.
    """
    with pytest.raises((RegimeInitializationError, BeartypeCallHintParamViolation)):
        NNBEGM(
            inner=object(),  # ty: ignore[invalid-argument-type]
            outer_search=FiniteOuterGrid(grid=toy.OUTER_GRID),
        )


def test_negm_does_not_publish_a_keeper_only_simulation_policy() -> None:
    """The unrelated NEGM path remains on its existing no-policy contract."""
    _, policies = toy.build_model(variant="negm", n_periods=2).solve(
        params=_PARAMS,
        log_level="debug",
        return_simulation_policy=True,
    )
    assert all("alive" not in mapping for mapping in policies.values())


def test_nnbegm_publishes_every_ranked_candidate_inner_policy() -> None:
    """Replay carries keeper plus each outer-grid candidate, in solve order."""
    _, policies = toy.build_model(variant="n_nbegm", n_periods=2).solve(
        params=_PARAMS,
        log_level="debug",
        return_simulation_policy=True,
    )
    policy = policies[0]["alive"]
    assert isinstance(policy, NNBEGMSimPolicy)
    n_candidates = 1 + toy.OUTER_GRID.to_jax().shape[0]
    assert policy.candidate_inner_action.shape[0] == n_candidates
    assert policy.candidate_value.shape == policy.candidate_inner_action.shape


def test_nnbegm_rejects_a_nonlinear_outer_action_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NNBEGM requires an affine map from outer action to its target."""
    monkeypatch.setattr(toy, "new_illiquid", _cubic_outer_target)
    model = toy.build_model(variant="n_nbegm", n_periods=2)

    with pytest.raises(RegimeInitializationError, match=r"affine.*outer action"):
        model.solve(params=_PARAMS, log_level="off")


def test_nnbegm_rejects_a_zero_slope_outer_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An outer post-decision target that ignores the outer action is refused.

    A constant map retains no information about the action that reached it, so
    no action can be recovered from a retained target.
    """
    monkeypatch.setattr(toy, "new_illiquid", _zero_slope_outer_target)
    model = toy.build_model(variant="n_nbegm", n_periods=2)

    with pytest.raises(
        RegimeInitializationError, match="does not depend on the outer action"
    ):
        model.solve(params=_PARAMS, log_level="off")


def test_nnbegm_retains_targets_for_a_nonunit_affine_outer_law(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid affine outer law retains keeper and adjuster target identities."""
    monkeypatch.setattr(toy, "new_illiquid", _affine_outer_target)
    _values, policies = toy.build_model(variant="n_nbegm", n_periods=2).solve(
        params=_PARAMS,
        log_level="off",
        return_simulation_policy=True,
    )
    policy = policies[0]["alive"]
    assert isinstance(policy, NNBEGMSimPolicy)
    expected = np.concatenate(([0.0], np.asarray(toy.OUTER_GRID.to_jax())))
    np.testing.assert_allclose(
        np.asarray(policy.candidate_outer_target)[:, -1, 0],
        expected,
    )


def test_two_period_toy_agrees_with_nested_dcegm() -> None:
    """On the two-period smooth toy, `NNBEGM` tracks `NEGM(inner=DCEGM)`.

    A single alive period reads only the terminal carry, so this isolates the
    outer keeper/adjuster wrapper: both solvers sweep the identical candidate
    set with an exact 1-D inner solve, and the value functions agree up to the
    inner families' constrained-region representation at the poorest cells.
    """
    nested = toy.build_model(variant="n_nbegm", n_periods=2).solve(
        params=_PARAMS, log_level="debug"
    )
    negm = toy.build_model(variant="negm", n_periods=2).solve(
        params=_PARAMS, log_level="debug"
    )
    V_nested = np.asarray(nested[0]["alive"])
    V_negm = np.asarray(negm[0]["alive"])
    np.testing.assert_allclose(V_nested, V_negm, atol=0.15)
    # Away from the borrowing-constrained poorest cells and the state-grid
    # boundary, the two inner families integrate the same smooth Euler
    # equation, so agreement is tight.
    np.testing.assert_allclose(V_nested[2:-1, 1:-1], V_negm[2:-1, 1:-1], atol=2e-2)


def test_two_period_toy_dominates_the_dense_brute_oracle() -> None:
    """Away from the poorest wealth row, `NNBEGM` weakly dominates the oracle.

    Both solvers now choose the next durable stock on `OUTER_GRID`, so the
    comparison is over one candidate set rather than two and a directional
    ordering *does* exist: the nested solve searches `OUTER_GRID` plus the
    keeper and solves the inner problem by EGM, while brute searches
    `OUTER_GRID` on a 30-point consumption grid. The nested value can therefore
    only be weakly higher, and on the interior it is — strictly, in all 72
    cells.

    The poorest wealth row is excluded rather than loosened, because there the
    ordering fails for a *stated* reason: both nested EGM families represent
    the constrained region by extending the inner value below its lowest
    endogenous node, which overstates a value that falls to negative infinity.
    An exact enumeration of the nested solver's own candidate set puts the true
    optimum at (wealth 0, illiquid 0) 6.7 utils below what `NNBEGM` reports,
    and 6.8 below what `NEGM` reports. That is a property of the inner
    families, not of the outer wrapper this file pins, so it is documented here
    and asserted nowhere.
    """
    nested = toy.build_model(variant="n_nbegm", n_periods=2).solve(
        params=_PARAMS, log_level="debug"
    )
    brute = toy.build_model(variant="brute", n_periods=2).solve(
        params=_PARAMS, log_level="debug"
    )
    nested_V = np.asarray(nested[0]["alive"])
    brute_V = np.asarray(brute[0]["alive"])

    interior = (slice(2, -1), slice(1, -1))
    gain = nested_V[interior] - brute_V[interior]
    assert float(gain.min()) >= 0.0, (
        f"nested fell below the oracle by {float(gain.min()):.6f} on the interior"
    )
    rel_gap = np.abs(nested_V[interior] - brute_V[interior]) / np.abs(brute_V[interior])
    # Measured 0.0176 at both precisions; the residual is the oracle's 30-point
    # consumption grid, which the nested inner solve does not pay.
    assert float(rel_gap.max()) < 0.03, f"max rel gap {float(rel_gap.max()):.4f}"


def test_three_period_toy_tracks_nested_dcegm_through_published_carries() -> None:
    """Chaining published nested carries, `NNBEGM` tracks `NEGM`.

    With two alive periods the parent reads the child's published outer
    envelope. Both solvers publish a bridged (finite-grid) envelope, so they
    share that approximation class and stay close everywhere — the gate for
    the topology-preserving publication is a separate, tighter deliverable.
    """
    nested = toy.build_model(variant="n_nbegm", n_periods=3).solve(
        params=_PARAMS, log_level="debug"
    )
    negm = toy.build_model(variant="negm", n_periods=3).solve(
        params=_PARAMS, log_level="debug"
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
        params=_PARAMS, log_level="debug"
    )
    chunked = toy.build_model(
        variant="n_nbegm", outer_batch_size=outer_batch_size, n_periods=2
    ).solve(params=_PARAMS, log_level="debug")
    for period, regime_to_V in reference.items():
        for regime_name, V_arr in regime_to_V.items():
            np.testing.assert_array_equal(
                np.asarray(V_arr),
                np.asarray(chunked[period][regime_name]),
                err_msg=f"{regime_name} at period {period}",
            )


def test_finite_outer_grid_does_not_materialize_a_candidate_bank(monkeypatch) -> None:
    """Finite batching folds each completed chunk without retaining every node."""

    def fail_if_bank_is_built(**_kwargs):
        raise AssertionError("FiniteOuterGrid retained a full candidate bank")

    monkeypatch.setattr(
        nnbegm_module,
        "build_outer_candidate_bank",
        fail_if_bank_is_built,
    )

    solution = toy.build_model(
        variant="n_nbegm",
        outer_search=FiniteOuterGrid(grid=toy.OUTER_GRID, batch_size=2),
        n_periods=2,
    ).solve(params=_PARAMS, log_level="debug")

    assert np.isfinite(np.asarray(solution[0]["alive"])).all()
