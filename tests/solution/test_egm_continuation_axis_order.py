"""The two-asset kernel maps a target's value array into its private axis order.

The G2EGM step works in `(liquid, pension)`. A regime's value array follows the
order its states *resolve* in, which `batch_size` fixes independently of the
order they were declared in. Those are two separate facts about one adapter:

- the **output** permutation — from the step's private order into the order this
  regime publishes on;
- the **input** permutation — from the order the *target* regime published on
  back into the step's private order.

Normalizing only the output leaves every continuation transposed. With equal
axis lengths that is not a shape error, just wrong numbers, so the cases below
give the two axes the same length on purpose.

The structural oracle is the affine `V(m, n) = 2m + 3n`, for which bilinear
interpolation is exact: the value and both gradients are hand-computable, so any
deviation is layout and never approximation. The public oracle is the same model
solved by dense `GridSearch`.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.two_asset_post_decision import post_decision_value_and_grad
from _lcm.engine import StateActionSpace
from _lcm.solution.two_asset_egm import _TwoAssetEGMPeriodKernel
from lcm import AgeGrid, LinSpacedGrid
from lcm.solvers import EGM, TwoAssetEGM
from tests.conftest import X64_ENABLED
from tests.solution.test_egm_continuation_grid_provenance import (
    _A_GRID,
    _B_GRID,
    _CONSUMPTION_GRID,
    _G2EGM_BUDGET,
    _N_PERIODS,
    _SAVINGS_GRID,
    _assert_within_budget,
)
from tests.test_models.deterministic.ds_pension import get_model, get_params

_DTYPES = (jnp.float32, jnp.float64) if X64_ENABLED else (jnp.float32,)

# Equal lengths on both axes: a missing transpose then changes the numbers
# without changing the shape, which is the case a shape check cannot catch.
_N_BOTH = 12
# `batch_size == 0` sorts last, so a pension grid carrying any positive
# `batch_size` resolves ahead of the liquid grid whatever order they were
# declared in. This is the reachable, public way to induce a pension-first
# regime, and it does not depend on dict insertion order.
_PENSION_FIRST_GRID = LinSpacedGrid(
    start=0.0, stop=15.0, n_points=_N_BOTH, batch_size=1
)
_LIQUID_FIRST_GRID = LinSpacedGrid(start=0.0, stop=15.0, n_points=_N_BOTH)

# The affine probe. With both returns and the wage set to zero the post-decision
# balances pass straight through, so `(a, b) = (0.5, 1.5)` queries `(m, n) =
# (0.5, 1.5)`: V = 2(0.5) + 3(1.5) = 5.5, dV/da = 2, dV/db = 3. Reading the
# pension-first array as if it were liquid-first gives (4.5, 6, 1) instead.
_PROBE_LIQUID = (0.0, 1.0, 2.0)
_PROBE_PENSION = (0.0, 2.0, 4.0)
_PROBE_QUERY = (0.5, 1.5)
_PROBE_EXPECTED = (5.5, 2.0, 3.0)
# The adapter never reads the age grid on the interior branch; it only has to
# be a real one.
_PROBE_AGES = AgeGrid(start=0, stop=1, step="Y")


def _probe_state_action_space(*, liquid, pension):
    """A two-state, no-action space carrying just the grids the adapter reads."""
    return StateActionSpace(
        states=MappingProxyType({"liquid": liquid, "pension": pension}),
        discrete_actions=MappingProxyType({}),
        continuous_actions=MappingProxyType({}),
        state_and_discrete_action_names=("liquid", "pension"),
    )


def _solvers():
    return {
        "working": TwoAssetEGM(
            a_grid=_A_GRID, b_grid=_B_GRID, consumption_grid=_CONSUMPTION_GRID
        ),
        "retired": EGM(savings_grid=_SAVINGS_GRID),
    }


def _private_continuation_seen_by_the_core(*, dtype):
    """Run the adapter on a pension-first target and return what the core got."""
    liquid = jnp.asarray(_PROBE_LIQUID, dtype=dtype)
    pension = jnp.asarray(_PROBE_PENSION, dtype=dtype)
    private = 2.0 * liquid[:, None] + 3.0 * pension[None, :]
    seen: dict[str, object] = {}

    def core(**kwargs):
        seen.update(kwargs)
        return jnp.zeros_like(private)

    adapter = _TwoAssetEGMPeriodKernel(
        core=core,
        regime_name="working",
        continuation_target="working",
        is_boundary=False,
        transition_target_names=("working",),
        liquid_state="liquid",
        pension_state="pension",
        next_liquid_grid=liquid,
        next_pension_grid=pension,
        next_boundary_liquid_grid=liquid,
        publishes_pension_first=True,
        target_reads_pension_first=True,
    )
    adapter(
        compiled_cores=adapter.cores(),
        state_action_space=_probe_state_action_space(liquid=liquid, pension=pension),
        next_regime_to_V_arr=MappingProxyType({"working": private.T}),
        next_regime_to_continuation=MappingProxyType({}),
        flat_params=MappingProxyType({"working": MappingProxyType({})}),
        period=0,
        ages=_PROBE_AGES,
    )
    return seen["next_value_working"], private, liquid, pension


@pytest.mark.parametrize("dtype", _DTYPES)
def test_a_pension_first_target_reaches_the_private_kernel_liquid_first(dtype):
    """A pension-first target V arrives at the step in `(liquid, pension)` order.

    The step's own convention never changes, so the adapter owes it the private
    layout no matter which order the target regime happened to publish on.
    """
    got, private, _, _ = _private_continuation_seen_by_the_core(dtype=dtype)
    np.testing.assert_allclose(np.asarray(got), np.asarray(private))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_the_production_reader_returns_the_affine_triple_from_a_pension_first_target(
    dtype,
):
    """Value and both gradients survive the input permutation exactly.

    Both gradients feed the Euler and KKT channels, so a transposed
    continuation corrupts the policy as well as the value. Asserting the triple
    covers both channels at once.
    """
    got, _, liquid, pension = _private_continuation_seen_by_the_core(dtype=dtype)
    query_a, query_b = _PROBE_QUERY
    zero = jnp.asarray(0.0, dtype=dtype)
    post = post_decision_value_and_grad(
        next_value=got,
        m_grid=liquid,
        n_grid=pension,
        a=jnp.asarray(query_a, dtype=dtype),
        b=jnp.asarray(query_b, dtype=dtype),
        return_liquid=zero,
        return_pension=zero,
        wage=zero,
    )
    triple = (float(post.value), float(post.grad_a), float(post.grad_b))
    np.testing.assert_allclose(triple, _PROBE_EXPECTED, rtol=1e-5)


@pytest.mark.parametrize("enable_jit", [False, True])
def test_a_pension_first_regime_publishes_the_transpose_of_a_liquid_first_one(
    enable_jit,
):
    """Resolved axis order relabels a solve; it does not change its content.

    Two models identical but for the `batch_size` that decides which state
    resolves first must publish the same value function, one transposed against
    the other. Equality of the content, not a tolerance: the two runs execute
    the same arithmetic on the same nodes.
    """

    def solve(pension_grid):
        return get_model(
            n_periods=_N_PERIODS,
            n_liquid=_N_BOTH,
            n_pension=_N_BOTH,
            working_pension_grid=pension_grid,
            solvers=_solvers(),
            enable_jit=enable_jit,
        ).solve(params=get_params(), log_level="debug")

    liquid_first = solve(_LIQUID_FIRST_GRID)
    pension_first = solve(_PENSION_FIRST_GRID)
    compared = 0
    for period, regime_to_V in liquid_first.items():
        if "working" not in regime_to_V:
            continue
        expected = np.asarray(regime_to_V["working"])
        got = np.asarray(pension_first[period]["working"])
        assert got.shape == expected.shape[::-1]
        np.testing.assert_array_equal(got, expected.T)
        compared += 1
    assert compared > 0


def test_a_pension_first_two_asset_solve_matches_dense_grid_search():
    """A pension-first regime agrees with brute force on the covered interior.

    The public end of the same fact: transposing the continuation is not a
    labelling detail, it changes the solved value. Dense `GridSearch` never
    interpolates a continuation, so it cannot share the error.
    """
    egm = get_model(
        n_periods=_N_PERIODS,
        n_liquid=_N_BOTH,
        n_pension=_N_BOTH,
        working_pension_grid=_PENSION_FIRST_GRID,
        solvers=_solvers(),
    ).solve(params=get_params(), log_level="debug")
    brute = get_model(
        n_periods=_N_PERIODS,
        n_liquid=_N_BOTH,
        n_pension=_N_BOTH,
        working_pension_grid=_PENSION_FIRST_GRID,
        n_consumption=200,
    ).solve(params=get_params(), log_level="debug")
    # Pension resolves first, so the published axes are `(pension, liquid)`:
    # the interior keeps the same nodes as the liquid-first case, transposed.
    # Every working period is checked — a transposed continuation degrades
    # backward, so the earliest periods are where it shows first.
    for period, pension_stop in ((2, 9), (1, 8), (0, 7)):
        egm_v = np.asarray(egm[period]["working"])[:pension_stop, 3:]
        brute_v = np.asarray(brute[period]["working"])[:pension_stop, 3:]
        assert np.isfinite(egm_v).all()
        rel = np.abs(egm_v - brute_v) / np.abs(brute_v)
        _assert_within_budget(rel, _G2EGM_BUDGET)
