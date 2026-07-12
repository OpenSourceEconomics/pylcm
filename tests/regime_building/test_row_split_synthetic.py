"""Tests for the divorce row-split, synthetic mode (row-split PLAN, commit 1).

pylcm's forward simulation tracks INDIVIDUALS, not households: a married
individual's row carries its own state plus a DRAWN (synthetic) partner
block (slice-5/6 machinery). On divorce (`~gate`), the row must revert to
`single_<own role>` with its OWN projected state — the leg whose
`source_stakeholder` matches the row's own role — not unconditionally the
FIRST declared leg (`_lcm.simulation.gated_routing.route_gated_edges`'s
prior "primary leg" convention, still the `own_stakeholder=None` default).

This is EKL Appendix F's design: two independent, single-gender cohorts
(all-women, all-men), each simulated with a synthetic opposite-gender
partner, each correctly reverting to ITS OWN single regime on divorce. No
cross-row linkage, no matcher, no new per-subject array — `own_stakeholder`
is a single value for the whole `simulate()` call (see
`_lcm.simulation.gated_routing._select_own_leg`).

Reuses the divorce miniature from `test_collective_regime_simulate.py`
(`_make_divorce_regimes` / `_solve_divorce`): a collective `married` regime
with stakeholders `("f", "m")`, a divorce edge with two legs (`"f" ->
single_f`, `"m" -> single_m`), and a slice-3 IR mask that empties (`D=True`)
at `wage=2` on the `WAGE_3 = {1, 2, 3}` grid.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np

from _lcm.simulation.simulate import simulate
from _lcm.utils.logging import get_logger
from tests.regime_building.test_collective_regime_simulate import _solve_divorce


def _simulate_divorce_cohort(*, own_stakeholder: str | None):
    """Simulate the 3-subject divorce miniature with a fixed own-role."""
    ages, regimes, regime_names_to_ids, flat_params, solution, divorce_flags = (
        _solve_divorce()
    )
    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([1.0, 2.0, 3.0]),
            "age": jnp.array([0.0, 0.0, 0.0]),
            "regime_id": jnp.array(
                [regime_names_to_ids["married"]] * 3, dtype=jnp.int32
            ),
        }
    )
    return simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=regimes,
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="off"),
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_divorce_flags=divorce_flags,
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
        own_stakeholder=own_stakeholder,
    ), solution


# ----------------------------------------------------------------------------------
# Test (a): a synthetic-married individual whose slice-3 IR mask empties at t+1
# (D=True) reverts to `single_<own role>` with its own projected state.
# ----------------------------------------------------------------------------------


def test_synthetic_divorce_reverts_to_own_role_single_regime_for_m():
    """`own_stakeholder="m"` routes the divorced row to `single_m`, not `single_f`.

    D=True only at wage=2 (hand-verified in `test_gated_edges_collective_solve.py`
    / `test_collective_regime_simulate.py`'s divorce test). Under the PRIOR
    "first declared leg" convention this row would incorrectly become
    `single_f`; the row's own role here is "m".
    """
    result, solution = _simulate_divorce_cohort(own_stakeholder="m")

    married_ir = result.raw_results["married_ir"][1]
    single_f = result.raw_results["single_f"][1]
    single_m = result.raw_results["single_m"][1]

    # The divorce DECISION itself (the gate) does not depend on own_stakeholder.
    np.testing.assert_array_equal(np.asarray(married_ir.in_regime), [True, False, True])
    # But the ROUTING does: the row's own continuing membership is single_m now.
    np.testing.assert_array_equal(np.asarray(single_m.in_regime), [False, True, False])
    np.testing.assert_array_equal(np.asarray(single_f.in_regime), [False, False, False])

    # The row's own projected state (identity wage projection) is correct.
    np.testing.assert_array_equal(np.asarray(single_m.states["wage"])[1], 2.0)

    # And its value matches the solved single_m array at that same grid point
    # (WAGE_3 = {1, 2, 3}, index 1 = wage=2.0) -- ground truth independently
    # verified by test_gated_edges_collective_solve.py's IR miniature.
    np.testing.assert_allclose(
        np.asarray(single_m.V_arr)[1],
        np.asarray(solution[1]["single_m"])[1],
        rtol=1e-6,
    )


def test_synthetic_divorce_default_still_takes_first_declared_leg():
    """`own_stakeholder=None` (the default) is byte-identical to the prior behavior.

    Regression pin: omitting `own_stakeholder` must keep routing the divorced
    row to the FIRST declared leg's fallback (`single_f`, since `married`'s
    legs are declared in stakeholder order `("f", "m")`) -- exactly the
    pre-row-split convention `test_collective_regime_simulate.py`'s
    `test_divorce_edge_routes_primary_leg_to_own_single_regime` pins.
    """
    result, _solution = _simulate_divorce_cohort(own_stakeholder=None)

    single_f = result.raw_results["single_f"][1]
    single_m = result.raw_results["single_m"][1]
    np.testing.assert_array_equal(np.asarray(single_f.in_regime), [False, True, False])
    np.testing.assert_array_equal(np.asarray(single_m.in_regime), [False, False, False])


# ----------------------------------------------------------------------------------
# Test (b): two independent populations (EKL Appendix F) -- an all-women cohort
# and an all-men cohort, each correctly reverting on divorce, with no
# cross-population coupling.
# ----------------------------------------------------------------------------------


def test_two_independent_synthetic_populations_each_revert_correctly():
    """A women-role run and a men-role run each divorce-revert to their own single.

    Both runs share the identical input population (same wages, same solved
    value functions) and differ ONLY in `own_stakeholder`; their outputs
    differ ONLY in which single regime the divorced row (wage=2) lands in --
    everything else (the gate decision, the still-married rows' values) is
    identical, confirming the two populations are independent (no
    cross-population state leakage through shared solved arrays).
    """
    women_result, _ = _simulate_divorce_cohort(own_stakeholder="f")
    men_result, _ = _simulate_divorce_cohort(own_stakeholder="m")

    # Women cohort: divorced row (index 1) becomes single_f, not single_m.
    np.testing.assert_array_equal(
        np.asarray(women_result.raw_results["single_f"][1].in_regime),
        [False, True, False],
    )
    np.testing.assert_array_equal(
        np.asarray(women_result.raw_results["single_m"][1].in_regime),
        [False, False, False],
    )

    # Men cohort: divorced row becomes single_m, not single_f.
    np.testing.assert_array_equal(
        np.asarray(men_result.raw_results["single_m"][1].in_regime),
        [False, True, False],
    )
    np.testing.assert_array_equal(
        np.asarray(men_result.raw_results["single_f"][1].in_regime),
        [False, False, False],
    )

    # The divorce decision and the still-married rows' recorded values are
    # identical across the two cohorts (own_stakeholder never touches them).
    np.testing.assert_array_equal(
        np.asarray(women_result.raw_results["married_ir"][1].in_regime),
        np.asarray(men_result.raw_results["married_ir"][1].in_regime),
    )
    np.testing.assert_allclose(
        np.asarray(women_result.raw_results["married_ir"][1].V_arr),
        np.asarray(men_result.raw_results["married_ir"][1].V_arr),
        rtol=1e-6,
    )

    # Re-running the women cohort is deterministic and unaffected by having
    # just run the men cohort (pure functions, no shared mutable state).
    women_result_again, _ = _simulate_divorce_cohort(own_stakeholder="f")
    np.testing.assert_array_equal(
        np.asarray(women_result.raw_results["single_f"][1].in_regime),
        np.asarray(women_result_again.raw_results["single_f"][1].in_regime),
    )
