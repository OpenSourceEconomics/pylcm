"""Tests for the dissolution row-split, synthetic mode.

pylcm's forward simulation tracks INDIVIDUALS, not households: a married
individual's row carries its own state plus a DRAWN (synthetic) partner
block. On dissolution (the household's own exit), the row must revert to
`single_<own role>` with its OWN projected state — the leg whose
`source_member` matches the row's own role. A cohort that names no role
is refused rather than routed through some declared leg.

The cohorts below are uniform in role — an all-women run and an all-men run,
each simulated with a synthetic opposite-gender partner — which is the
single-gender design EKL Appendix F describes. The role itself is per row
(`initial_conditions["own_member"]`), so a uniform cohort is one case of
it rather than the only expressible one; a mixed cohort is exercised in
`tests/simulation/initial_conditions/test_own_member_per_row.py`.

Reuses the dissolution miniature from `test_collective_regime_simulate.py`
(`_make_dissolution_regimes` / `_solve_dissolution`): a collective `married` regime
with members `("f", "m")`, an exit with one target per member (`"f" ->
single_f`, `"m" -> single_m`), and a participation constraint that empties
(`D=True`) at `wage=2` on the `WAGE_3 = {1, 2, 3}` grid.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.simulation.simulate import simulate
from _lcm.typing import PeriodToRegimeToVArr
from _lcm.utils.logging import get_logger
from lcm.exceptions import InvalidInitialConditionsError
from lcm.result import SimulationResult
from tests.regime_building.test_collective_regime_simulate import _solve_dissolution
from tests.simulation.test_runtime_helpers import bind_eager_simulation


def _simulate_dissolution_cohort(
    *, own_member: str | None
) -> tuple[SimulationResult, PeriodToRegimeToVArr]:
    """Simulate the 3-agent dissolution miniature with one role for every row."""
    ages, regimes, regime_names_to_ids, flat_params, solution = _solve_dissolution()
    conditions = {
        "wage": jnp.array([1.0, 2.0, 3.0]),
        "age": jnp.array([0.0, 0.0, 0.0]),
        "regime_id": jnp.array([regime_names_to_ids["married"]] * 3, dtype=jnp.int32),
    }
    if own_member is not None:
        role = regimes["married"].member_names_to_ids[own_member]
        conditions["own_member"] = jnp.full(3, role, dtype=jnp.int32)
    initial_conditions = MappingProxyType(conditions)
    return simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=bind_eager_simulation(regimes=regimes),
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="off"),
        period_to_regime_to_V_arr=solution,
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    ), solution


def test_synthetic_dissolution_reverts_to_own_role_single_regime_for_m() -> None:
    """`own_member="m"` routes the dissolved row to `single_m`, not `single_f`.

    D=True only at wage=2 (hand-verified in `test_gated_edges_collective_solve.py`
    / `test_collective_regime_simulate.py`'s dissolution test). The leg is picked
    by the row's own role, "m" here -- not by declaration order, which would send
    it to `single_f`.
    """
    result, solution = _simulate_dissolution_cohort(own_member="m")

    married_ir = result.raw_results["married_ir"][1]
    single_f = result.raw_results["single_f"][1]
    single_m = result.raw_results["single_m"][1]

    # The dissolution DECISION itself (the gate) does not depend on own_member.
    np.testing.assert_array_equal(np.asarray(married_ir.in_regime), [True, False, True])
    # But the ROUTING does: the row's own continuing membership is single_m now.
    np.testing.assert_array_equal(np.asarray(single_m.in_regime), [False, True, False])
    np.testing.assert_array_equal(np.asarray(single_f.in_regime), [False, False, False])

    # The row's own projected state (identity wage projection) is correct.
    np.testing.assert_array_equal(np.asarray(single_m.states["wage"])[1], 2.0)

    # And its value is the solved single_m array at that same grid point
    # (WAGE_3 = {1, 2, 3}, index 1 = wage=2.0) less the dissolution cost
    # `delta_m = 0.2` that the household's own exit pays on the way out.
    np.testing.assert_allclose(
        np.asarray(single_m.V_arr)[1],
        np.asarray(solution[1]["single_m"])[1] - 0.2,
        rtol=1e-6,
    )


def test_synthetic_dissolution_without_a_role_is_refused() -> None:
    """Simulating this cohort without a role is refused, not defaulted.

    `married`'s legs are declared in member order `("f", "m")`, and
    picking the first of them for rows that never said which partner they are
    would simulate every divorced husband as his wife. The choice belongs to
    the caller, so omitting it is an error.
    """
    with pytest.raises(InvalidInitialConditionsError, match="own_member"):
        _simulate_dissolution_cohort(own_member=None)


def test_two_independent_synthetic_populations_each_revert_correctly() -> None:
    """A women-role run and a men-role run each dissolution-revert to their own single.

    Both runs share the identical input population (same wages, same solved
    value functions) and differ ONLY in the seeded role; their outputs
    differ ONLY in which single regime the dissolved row (wage=2) lands in --
    everything else (the gate decision, the still-married rows' values) is
    identical, confirming the two populations are independent (no
    cross-population state leakage through shared solved arrays).
    """
    women_result, _ = _simulate_dissolution_cohort(own_member="f")
    men_result, _ = _simulate_dissolution_cohort(own_member="m")

    # Women cohort: dissolved row (index 1) becomes single_f, not single_m.
    np.testing.assert_array_equal(
        np.asarray(women_result.raw_results["single_f"][1].in_regime),
        [False, True, False],
    )
    np.testing.assert_array_equal(
        np.asarray(women_result.raw_results["single_m"][1].in_regime),
        [False, False, False],
    )

    # Men cohort: dissolved row becomes single_m, not single_f.
    np.testing.assert_array_equal(
        np.asarray(men_result.raw_results["single_m"][1].in_regime),
        [False, True, False],
    )
    np.testing.assert_array_equal(
        np.asarray(men_result.raw_results["single_f"][1].in_regime),
        [False, False, False],
    )

    # The dissolution decision and the still-married rows' recorded values are
    # identical across the two cohorts (the role never touches them).
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
    women_result_again, _ = _simulate_dissolution_cohort(own_member="f")
    np.testing.assert_array_equal(
        np.asarray(women_result.raw_results["single_f"][1].in_regime),
        np.asarray(women_result_again.raw_results["single_f"][1].in_regime),
    )
