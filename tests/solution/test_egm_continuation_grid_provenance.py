"""EGM kernels read `V_{t+1}` on the target regime's period-`t+1` grid.

A period-`t` EGM kernel interpolates its continuation off a grid. Which grid is
correct is fixed by where the continuation lives, not by where the kernel
publishes its own result: it is the **target** regime's grid, at period
**`t+1`**. Two axes make those differ from the source's own current grid:

- **cross-regime** — the target is a different regime whose state is discretized
  on a different grid (the retired regime's liquid support need not match the
  working regime's, and the terminal regime's need not match either);
- **temporal** — the state's grid is age-specialized, so period `t+1`'s nodes
  differ from period `t`'s within one regime.

The oracle throughout is the same model solved by dense `GridSearch`, compared on
the covered interior. Grid search never interpolates a continuation, so it cannot
share the defect.

Every configuration here is reachable from the public API: the DS pension model
takes per-regime grid overrides, all defaulting to one shared grid, so the
negative control is exactly today's model.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.two_asset_post_decision import post_decision_value_and_grad
from lcm import (
    AgeGrid,
    AgeSpecializedGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.regime import Regime
from lcm.solvers import GridSearch, OneAssetEGM, TwoDimEGM
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)
from tests.conftest import DECIMAL_PRECISION
from tests.test_models.deterministic.ds_pension import get_model, get_params

_N_PERIODS = 5
_RETIREMENT_PERIOD = 3

_A_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=18)
_B_GRID = LinSpacedGrid(start=0.0, stop=30.0, n_points=16)
_CONSUMPTION_GRID = LinSpacedGrid(start=0.1, stop=20.0, n_points=18)
_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=40)

# The interior the two solvers are comparable on: the off-grid top-pension
# boundary layer thickens backward, and the steep low-liquid rows are where the
# dense brute is least reliable.
_WORKING_INTERIOR = {2: np.s_[3:, :9], 1: np.s_[3:, :8], 0: np.s_[3:, :7]}
_RETIRED_INTERIOR = np.s_[2:]

# The agreement the matched-grid configuration achieves. A kernel reading its
# continuation off the wrong grid misses these by a wide margin.
_WORKING_MEDIAN_TOL = 0.03
_WORKING_P90_TOL = 0.15
_RETIRED_MEDIAN_TOL = 0.01
_RETIRED_MAX_TOL = 0.05


def _two_asset_solvers(*, upper_envelope="g2egm"):
    """The prime-time solver assignment: G2EGM working, 1-D EGM retired."""
    return {
        "working": TwoDimEGM(
            a_grid=_A_GRID,
            b_grid=_B_GRID,
            consumption_grid=_CONSUMPTION_GRID,
            upper_envelope=upper_envelope,
        ),
        "retired": OneAssetEGM(savings_grid=_SAVINGS_GRID),
    }


def _solve_pair(**grid_overrides):
    """Solve one grid configuration with the EGM solvers and with dense brute."""
    solvers = grid_overrides.pop("solvers")
    egm = get_model(n_periods=_N_PERIODS, solvers=solvers, **grid_overrides).solve(
        params=get_params(), log_level="debug"
    )
    brute = get_model(n_periods=_N_PERIODS, n_consumption=200, **grid_overrides).solve(
        params=get_params(), log_level="debug"
    )
    return egm, brute


def _assert_working_matches_brute(egm, brute, period):
    """The working value agrees with brute on the covered pension interior."""
    sl = _WORKING_INTERIOR[period]
    egm_v = np.asarray(egm[period]["working"])[sl]
    brute_v = np.asarray(brute[period]["working"])[sl]
    assert np.isfinite(egm_v).all()
    rel = np.abs(egm_v - brute_v) / np.abs(brute_v)
    assert np.median(rel) < _WORKING_MEDIAN_TOL
    assert np.percentile(rel, 90) < _WORKING_P90_TOL


def _assert_retired_matches_brute(egm, brute, period=_RETIREMENT_PERIOD):
    """The retired value agrees with brute on the unconstrained liquid interior."""
    egm_v = np.asarray(egm[period]["retired"])[_RETIRED_INTERIOR]
    brute_v = np.asarray(brute[period]["retired"])[_RETIRED_INTERIOR]
    assert np.isfinite(egm_v).all()
    rel = np.abs(egm_v - brute_v) / np.abs(brute_v)
    assert np.median(rel) < _RETIRED_MEDIAN_TOL
    assert np.max(rel) < _RETIRED_MAX_TOL


def _moving_liquid_grid(*, start, stop_at_age, n_points):
    """A liquid grid whose ceiling moves with age, shape held fixed."""
    return AgeSpecializedGrid(
        build=lambda age: LinSpacedGrid(
            start=start, stop=stop_at_age(age), n_points=n_points
        ),
        signature=lambda age: float(stop_at_age(age)),
    )


def test_the_affine_probe_distinguishes_the_grid_a_continuation_is_read_on():
    """An affine `V` separates a right grid, a wrong grid, and swapped axes.

    With `V(m, n) = 2m + 3n`, reading the continuation at a post-decision node
    recovers the value and both partials in closed form, so each way of getting
    the grid wrong lands on a distinct, hand-checkable triple. This validates the
    instrument the two-asset witnesses below rely on; it is not itself a witness.
    """
    target_m = jnp.array([0.0, 1.0, 2.0])
    target_n = jnp.array([0.0, 2.0, 4.0])
    source_n = jnp.array([0.0, 1.0, 2.0])
    next_value = 2.0 * target_m[:, None] + 3.0 * target_n[None, :]
    probe = {"a": jnp.array(0.5), "b": jnp.array(1.5)}
    flat = {"return_liquid": 0.0, "return_pension": 0.0, "wage": 0.0}

    right = post_decision_value_and_grad(
        next_value=next_value, m_grid=target_m, n_grid=target_n, **probe, **flat
    )
    wrong_n = post_decision_value_and_grad(
        next_value=next_value, m_grid=target_m, n_grid=source_n, **probe, **flat
    )
    swapped = post_decision_value_and_grad(
        next_value=next_value.T, m_grid=target_m, n_grid=target_n, **probe, **flat
    )

    for got, expected in (
        (right, (5.5, 2.0, 3.0)),
        (wrong_n, (10.0, 2.0, 6.0)),
        (swapped, (4.5, 6.0, 1.0)),
    ):
        np.testing.assert_array_almost_equal(
            [float(got.value), float(got.grad_a), float(got.grad_b)],
            expected,
            decimal=DECIMAL_PRECISION,
        )


def test_w7_matched_grids_agree_with_brute():
    """W7 (negative control). One shared liquid grid everywhere still agrees.

    This is today's configuration. It must hold both before and after the repair,
    which is what proves the repair changes only the cases it claims to.
    """
    egm, brute = _solve_pair(solvers=_two_asset_solvers())
    for period in (0, 1, 2):
        _assert_working_matches_brute(egm, brute, period)
    _assert_retired_matches_brute(egm, brute)


@pytest.mark.xfail(
    strict=True,
    reason="the one-asset kernel reads its continuation on its own liquid "
    "grid, not the target's",
)
def test_w1_retired_egm_reads_the_terminal_regime_on_the_terminal_grid():
    """W1 (cross-regime, one asset). The `dead` regime's own liquid grid is used.

    The retired regime's last living period reads its bequest continuation from
    the terminal regime. That regime discretizes `liquid` on its own grid, so a
    kernel that interpolates on the retired grid instead reads the wrong
    bequest.
    """
    egm, brute = _solve_pair(
        solvers=_two_asset_solvers(),
        dead_liquid_grid=LinSpacedGrid(start=0.1, stop=32.0, n_points=17),
    )
    _assert_retired_matches_brute(egm, brute, period=_N_PERIODS - 2)


@pytest.mark.xfail(
    strict=True,
    reason="the two-asset boundary reads the retired continuation on the working grid",
)
def test_w2_two_asset_boundary_reads_the_retired_regime_on_the_retired_grid():
    """W2 (cross-regime, two assets). The retirement boundary uses retired's grid.

    At the working->retired boundary the pension is paid out and the working
    kernel reads the 1-D retired continuation. That continuation lives on the
    retired regime's liquid grid, which need not be the working regime's.
    """
    egm, brute = _solve_pair(
        solvers=_two_asset_solvers(),
        retired_liquid_grid=LinSpacedGrid(start=0.1, stop=28.0, n_points=15),
    )
    _assert_working_matches_brute(egm, brute, _RETIREMENT_PERIOD - 1)


@pytest.mark.xfail(
    strict=True,
    reason="the G2EGM interior step reads next period's value on the "
    "current period's liquid grid",
)
def test_w3_two_asset_interior_reads_the_next_period_liquid_grid():
    """W3 (temporal, two assets, G2EGM). Age-specialized liquid support.

    Within the working regime, an age-specialized liquid grid makes period
    `t+1`'s nodes differ from period `t`'s. The interior step must read its own
    next-period value on `t+1`'s nodes while publishing on `t`'s.
    """
    egm, brute = _solve_pair(
        solvers=_two_asset_solvers(),
        working_liquid_grid=_moving_liquid_grid(
            start=0.1, stop_at_age=lambda age: 20.0 + 7.0 * float(age), n_points=12
        ),
    )
    for period in (0, 1):
        _assert_working_matches_brute(egm, brute, period)


@pytest.mark.xfail(
    strict=True,
    reason="the RFC interior step reads next period's value on the "
    "current period's liquid grid",
)
def test_w4_rfc_interior_reads_the_next_period_liquid_grid():
    """W4 (temporal, two assets, RFC). W3 on the rooftop-cut backend.

    The RFC step shares the conflated read/publish grid contract with G2EGM, so
    it is pinned separately rather than assumed to inherit the fix.
    """
    egm, brute = _solve_pair(
        solvers=_two_asset_solvers(upper_envelope="rfc"),
        working_liquid_grid=_moving_liquid_grid(
            start=0.1, stop_at_age=lambda age: 20.0 + 7.0 * float(age), n_points=12
        ),
    )
    for period in (0, 1):
        _assert_working_matches_brute(egm, brute, period)


@pytest.mark.xfail(
    strict=True,
    reason="the G2EGM interior step reads next period's value on the "
    "current period's pension grid",
)
def test_w8_two_asset_interior_reads_the_next_period_pension_grid():
    """W8 (temporal, pension axis, G2EGM). Only the pension nodes move.

    Isolating the pension axis separates the two grids the interior step
    conflates: a repair that threads the liquid grid correctly but leaves the
    pension grid tied to the current period still fails here.
    """
    egm, brute = _solve_pair(
        solvers=_two_asset_solvers(),
        working_pension_grid=AgeSpecializedGrid(
            build=lambda age: LinSpacedGrid(
                start=0.0, stop=15.0 + 7.0 * float(age), n_points=10
            ),
            signature=float,
        ),
    )
    for period in (0, 1):
        _assert_working_matches_brute(egm, brute, period)


@pytest.mark.xfail(
    strict=True,
    reason="the RFC interior step reads next period's value on the "
    "current period's pension grid",
)
def test_w9_rfc_interior_reads_the_next_period_pension_grid():
    """W9 (temporal, pension axis, RFC). W8 on the rooftop-cut backend."""
    egm, brute = _solve_pair(
        solvers=_two_asset_solvers(upper_envelope="rfc"),
        working_pension_grid=AgeSpecializedGrid(
            build=lambda age: LinSpacedGrid(
                start=0.0, stop=15.0 + 7.0 * float(age), n_points=10
            ),
            signature=float,
        ),
    )
    for period in (0, 1):
        _assert_working_matches_brute(egm, brute, period)


@pytest.mark.parametrize(
    "upper_envelope",
    [
        pytest.param(
            "g2egm",
            marks=pytest.mark.xfail(
                strict=True,
                reason="the G2EGM step extrapolates off the wrong grid and "
                "publishes non-finite values",
            ),
        ),
        "rfc",
    ],
)
def test_the_two_asset_solve_is_finite_on_every_moving_grid(upper_envelope):
    """No moving-grid configuration publishes NaN or Inf anywhere.

    A kernel reading off the wrong grid can extrapolate far outside support and
    poison the whole backward induction, which would mask the value comparisons
    above behind a NaN rather than a measurable disagreement.
    """
    solution, _ = _solve_pair(
        solvers=_two_asset_solvers(upper_envelope=upper_envelope),
        working_liquid_grid=_moving_liquid_grid(
            start=0.1, stop_at_age=lambda age: 20.0 + 7.0 * float(age), n_points=12
        ),
    )
    for period_solution in solution.values():
        for V_arr in period_solution.values():
            assert np.isfinite(np.asarray(V_arr)).all()


def _renamed_one_asset_model(*, solver):
    """A 1-D consumption--saving model whose state is `wealth`, not `liquid`.

    Structurally identical to the DS pension retired sub-problem, but the
    state's name differs from the EGM kernel's internal role vocabulary.
    Which name the modeller picks is their business; the kernel's private
    liquid/pension roles must be resolved from the solver's declaration, not
    matched against the user's spelling.
    """

    @categorical(ordered=False)
    class RenamedRegimeId:
        alive: ScalarInt
        gone: ScalarInt

    def utility(consumption: ContinuousAction, crra: float) -> FloatND:
        return consumption ** (1.0 - crra) / (1.0 - crra)

    def bequest(wealth: ContinuousState, crra: float) -> FloatND:
        return wealth ** (1.0 - crra) / (1.0 - crra)

    def next_wealth(
        wealth: ContinuousState,
        consumption: ContinuousAction,
        return_liquid: float,
        retirement_income: float,
    ) -> ContinuousState:
        return (1.0 + return_liquid) * (wealth - consumption) + retirement_income

    def feasible(wealth: ContinuousState, consumption: ContinuousAction) -> BoolND:
        return consumption <= wealth

    def prob_survive(age: int, last_age: float) -> FloatND:
        return jnp.where(age + 1 < last_age, 1.0, 0.0)

    def prob_gone(age: int, last_age: float) -> FloatND:
        return jnp.where(age + 1 >= last_age, 1.0, 0.0)

    wealth_grid = LinSpacedGrid(start=0.1, stop=20.0, n_points=12)
    ages = AgeGrid(start=0, stop=3, step="Y")
    alive = Regime(
        actions={"consumption": LinSpacedGrid(start=0.1, stop=20.0, n_points=14)},
        states={"wealth": wealth_grid},
        state_transitions={"wealth": {"alive": next_wealth, "gone": next_wealth}},
        constraints={"feasible": feasible},
        transition={
            "alive": MarkovTransition(prob_survive),
            "gone": MarkovTransition(prob_gone),
        },
        functions={"utility": utility},
        active=lambda age: age < 3,
        solver=solver,
    )
    gone = Regime(
        transition=None,
        states={"wealth": wealth_grid},
        functions={"utility": bequest},
        active=lambda age: age >= 3,
        solver=GridSearch(),
    )
    return Model(
        regimes={"alive": alive, "gone": gone},
        ages=ages,
        regime_id_class=RenamedRegimeId,
    )


def _renamed_one_asset_params():
    """Params for the renamed 1-D model, in the user's own vocabulary."""
    law = {"return_liquid": 0.02, "retirement_income": 0.5}
    return {
        "alive": {
            "utility": {"crra": 2.0},
            "H": {"discount_factor": 0.98},
            "alive": {"next_wealth": law, "next_regime": {"last_age": 3.0}},
            "gone": {"next_wealth": law, "next_regime": {"last_age": 3.0}},
        },
        "gone": {"utility": {"crra": 2.0}},
    }


@pytest.mark.xfail(
    strict=True,
    reason="the one-asset core demands a keyword named 'liquid' "
    "regardless of the state's name",
)
def test_w5_one_asset_egm_does_not_require_the_state_to_be_named_liquid():
    """W5 (D3, one asset). The kernel's private role vocabulary stays private.

    A modeller naming the single continuous state `wealth`, with a law of motion
    must get the same answer as the dense brute. Nothing about the EGM
    kernel's internal liquid role may surface as a requirement on the state's
    name.
    """
    egm = _renamed_one_asset_model(
        solver=OneAssetEGM(savings_grid=_SAVINGS_GRID)
    ).solve(params=_renamed_one_asset_params(), log_level="debug")
    brute = _renamed_one_asset_model(solver=GridSearch()).solve(
        params=_renamed_one_asset_params(), log_level="debug"
    )
    for period in (0, 1, 2):
        egm_v = np.asarray(egm[period]["alive"])[_RETIRED_INTERIOR]
        brute_v = np.asarray(brute[period]["alive"])[_RETIRED_INTERIOR]
        rel = np.abs(egm_v - brute_v) / np.abs(brute_v)
        assert np.max(rel) < _RETIRED_MAX_TOL
