"""Regression tests for the SIMULATE-subsystem boundary guards (F3-F7).

Five findings from an audit of the collective-regimes forward-simulation
path (`_lcm.simulation.gated_routing`, `_lcm.simulation.result_dataframe`,
`_lcm.simulation.simulate`):

- F3: a gated edge's target is solved at `period + 1` but a declared
  reference regime (fallback / gate ref) is not -- a malformed ACTIVE edge
  that `substitute_gated_edge_continuations` silently no-opped instead of
  raising.
- F4: two legs of one gated edge sharing the same fallback regime -- the
  later leg's projected state silently overwrites the earlier leg's in
  `route_gated_edges`. Rejected at model construction instead.
- F5: `_select_own_leg` returning `legs[0]` for an `own_stakeholder` that
  matches no leg of a genuinely multi-leg (collective) source.
- F6: `to_dataframe()` on a result where every populated regime is
  collective (no scalar `value` column anywhere) raising `KeyError`.
- F7: a STATELESS collective regime's simulated `V_arr` / argmax index
  missing the subject axis a stateful collective or a stateless singleton
  regime always carries.

Each test proves the finding reproduces against the pre-fix code (see the
module-level PR description / audit notes for the pre-fix evidence
captured while writing these tests), then pins the post-fix behavior.
Byte-identical regression coverage for the untouched paths (stateful
collective, stateless singleton, mixed-topology dataframe, distinct-
fallback dissolution, `own_stakeholder=None`/matching-leg routing) is
already carried by `test_collective_regime_simulate.py` and
`test_row_split_synthetic.py`; this file focuses on the five guards
themselves plus their required negative controls.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.gated_edges import ResolvedEdgeLeg
from _lcm.regime_building.Q_and_F import ResolvedSamePeriodRef
from _lcm.simulation.gated_routing import (
    _select_own_leg,
    substitute_gated_edge_continuations,
)
from _lcm.simulation.simulate import simulate
from _lcm.solution.backward_induction import solve
from _lcm.utils.logging import get_logger
from lcm import DiscreteGrid, LinSpacedGrid, categorical, fixed_transition
from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.regime import EdgeLeg, GatedEdge, Regime, SamePeriodRef
from lcm.transition import MarkovTransition
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt
from tests.regime_building.test_collective_regime_simulate import (
    _make_dissolution_regimes,
    _solve_and_process,
    _solve_dissolution,
)

_BETA = 0.95


@categorical(ordered=True)
class Work:
    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


# ------------------------------------------------------------------------------
# F3: a gated edge's target is solved at period+1 but a declared reference
# regime is not -- must raise, not silently no-op.
# ------------------------------------------------------------------------------


def test_missing_reference_regime_at_target_period_raises():
    """Repro (see report): dropping `single_f` from period 1's solution while
    keeping `married_ir` (the edge's target) present currently returns the
    substitution UNCHANGED and no gate array -- a silent ungate. Post-fix
    this raises `ModelInitializationError` naming the missing reference.
    """
    _ages, regimes, _ids, flat_params, solution, dissolution_flags = (
        _solve_dissolution()
    )
    married = regimes["married"]
    edge = married.gated_edges["married_ir"]
    assert edge.reference_regimes == ("single_f", "single_m")

    base_state_action_spaces = {
        name: regime.solution.state_action_space(regime_params=flat_params[name])
        for name, regime in regimes.items()
    }
    period1 = dict(solution[1])
    del period1["single_f"]
    filtered_period_to_V = MappingProxyType({1: MappingProxyType(period1)})

    with pytest.raises(ModelInitializationError, match="single_f"):
        substitute_gated_edge_continuations(
            regime=married,
            regime_name="married",
            period=0,
            next_regime_to_V_arr=solution[1],
            base_state_action_spaces=base_state_action_spaces,
            period_to_regime_to_V_arr=filtered_period_to_V,
            period_to_regime_to_dissolution_flags=MappingProxyType(
                {1: dissolution_flags[1]}
            ),
            flat_params=flat_params,
        )


def test_target_absent_at_next_period_is_still_a_legitimate_no_op():
    """Negative control: the TARGET (not a reference) missing at period+1 stays
    a silent no-op -- the legitimate repeating/one-shot boundary case (F3's
    `continue` this finding must NOT touch). Mirrors
    `test_repeating_self_loop_gated_edge_simulates_past_activity_boundary`'s
    scenario at the kernel level: an empty `period_to_regime_to_V_arr` for
    period+1 (no target, hence no references either) must not raise.
    """
    _ages, regimes, _ids, flat_params, solution, _dissolution_flags = (
        _solve_dissolution()
    )
    married = regimes["married"]
    base_state_action_spaces = {
        name: regime.solution.state_action_space(regime_params=flat_params[name])
        for name, regime in regimes.items()
    }
    substituted, gate_arrays = substitute_gated_edge_continuations(
        regime=married,
        regime_name="married",
        period=0,
        next_regime_to_V_arr=solution[1],
        base_state_action_spaces=base_state_action_spaces,
        period_to_regime_to_V_arr=MappingProxyType({}),  # period+1 wholly absent
        period_to_regime_to_dissolution_flags=MappingProxyType({}),
        flat_params=flat_params,
    )
    assert dict(gate_arrays) == {}
    np.testing.assert_array_equal(
        np.asarray(substituted["married_ir"]), np.asarray(solution[1]["married_ir"])
    )


def test_dissolution_fixture_with_all_references_present_is_unaffected():
    """Regression pin: the ordinary (fully-solved) dissolution fixture never
    hits the new raise -- `test_dissolution_edge_routes_primary_leg_to_own_
    single_regime` (test_collective_regime_simulate.py) still passes end to
    end through `simulate()`, exercised again here at the kernel level.
    """
    _ages, regimes, _ids, flat_params, solution, dissolution_flags = (
        _solve_dissolution()
    )
    married = regimes["married"]
    base_state_action_spaces = {
        name: regime.solution.state_action_space(regime_params=flat_params[name])
        for name, regime in regimes.items()
    }
    _substituted, gate_arrays = substitute_gated_edge_continuations(
        regime=married,
        regime_name="married",
        period=0,
        next_regime_to_V_arr=solution[1],
        base_state_action_spaces=base_state_action_spaces,
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=dissolution_flags,
        flat_params=flat_params,
    )
    assert "married_ir" in gate_arrays


# ------------------------------------------------------------------------------
# F4: two legs of one gated edge sharing a fallback regime must be rejected
# at model construction.
# ------------------------------------------------------------------------------

_WAGE_3 = LinSpacedGrid(start=1.0, stop=3.0, n_points=3)  # {1, 2, 3}


def _identity_wage(wage: ContinuousState) -> ContinuousState:
    return wage


def _reverse_wage(wage: ContinuousState) -> ContinuousState:
    return 4.0 - wage


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _u_zero_collective(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 0.0 * wage * work


def _no_dissolution_gate(D_target: BoolND) -> BoolND:
    return ~D_target


def _u_shared(wage: ContinuousState) -> FloatND:
    return 1.0 * wage


def _make_shared_fallback_regimes() -> dict[str, Regime]:
    """Same dissolution topology as `_make_dissolution_regimes`, but BOTH legs
    of `married`'s edge fall back to the SAME regime (`single_shared`) under
    different projections -- the F4 counterexample.
    """
    married = Regime(
        transition={"married_ir": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_3},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_zero_collective, "utility_m": _u_zero_collective},
        gated_edges={
            "married_ir": GatedEdge(
                gate=_no_dissolution_gate,
                legs={
                    "f": EdgeLeg(
                        target_stakeholder="f",
                        fallback=SamePeriodRef(
                            regime="single_shared", projection={"wage": _identity_wage}
                        ),
                    ),
                    "m": EdgeLeg(
                        target_stakeholder="m",
                        fallback=SamePeriodRef(
                            regime="single_shared", projection={"wage": _reverse_wage}
                        ),
                    ),
                },
            )
        },
    )
    married_ir = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_3},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_zero_collective, "utility_m": _u_zero_collective},
    )
    single_shared = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE_3},
        functions={"utility": _u_shared},
    )
    return {
        "married": married,
        "married_ir": married_ir,
        "single_shared": single_shared,
    }


def test_two_legs_sharing_a_fallback_regime_is_rejected_at_construction():
    """Repro (see report): pre-fix, `process_regimes` builds this model without
    complaint, and forward-simulating it shows `single_shared`'s stored state
    is entirely the "m" leg's (reverse-projected) values -- the "f" leg's own
    write is silently clobbered, regardless of which leg `own_stakeholder`
    would actually select. Post-fix, construction raises
    `ModelInitializationError` naming the shared fallback regime.
    """
    ages = AgeGrid(start=0, stop=2, step="Y")
    regimes_dict = _make_shared_fallback_regimes()
    with pytest.raises(ModelInitializationError, match="single_shared"):
        _solve_and_process(
            regimes_dict=regimes_dict, ages=ages, regime_names=list(regimes_dict)
        )


def test_dissolution_fixture_has_distinct_fallbacks_and_still_constructs():
    """Negative control: `_make_dissolution_regimes` (single_f / single_m,
    DISTINCT fallback regimes) is exactly the EKL-shaped topology and must
    keep constructing without the new guard firing.
    """
    ages = AgeGrid(start=0, stop=3, step="Y")
    regimes_dict = _make_dissolution_regimes()
    married_edge = regimes_dict["married"].gated_edges["married_ir"]
    fallback_regimes = [leg.fallback.regime for leg in married_edge.legs.values()]
    assert len(fallback_regimes) == len(set(fallback_regimes))
    # Must not raise.
    _solve_and_process(
        regimes_dict=regimes_dict, ages=ages, regime_names=list(regimes_dict)
    )


# ------------------------------------------------------------------------------
# F5: an own_stakeholder that matches no leg of a multi-leg source must raise,
# not silently fall back to legs[0].
# ------------------------------------------------------------------------------


def _leg(source_stakeholder: str | None) -> ResolvedEdgeLeg:
    return ResolvedEdgeLeg(
        source_stakeholder=source_stakeholder,
        target_component_index=None,
        fallback=ResolvedSamePeriodRef(
            regime=f"single_{source_stakeholder}",
            projection={},
            stakeholder_index=None,
        ),
    )


def test_select_own_leg_unmatched_role_on_multi_leg_source_raises():
    """Reviewer's exact counterexample: a typo'd `own_stakeholder` on a
    two-leg (collective) source must raise, not silently return `legs[0]`.
    """
    legs = (_leg("f"), _leg("m"))
    with pytest.raises(ValueError, match="typo"):
        _select_own_leg(legs, "typo")


def test_select_own_leg_none_still_returns_first_leg():
    """Negative control: `own_stakeholder=None` (the legacy default) is
    untouched -- still `legs[0]`.
    """
    legs = (_leg("f"), _leg("m"))
    assert _select_own_leg(legs, None) is legs[0]


def test_select_own_leg_matching_role_still_returns_matching_leg():
    """Negative control: a genuinely matching role still resolves correctly."""
    legs = (_leg("f"), _leg("m"))
    assert _select_own_leg(legs, "m") is legs[1]


def test_select_own_leg_singleton_source_with_non_none_role_returns_sole_leg():
    """Negative control: a singleton source's sole leg (`source_stakeholder=
    None`) never matches a non-`None` own_stakeholder -- this is the common,
    correct case (an all-women/all-men cohort routing through a SINGLETON
    source's gated edge) and must keep returning the sole leg, not raise.
    """
    legs = (_leg(None),)
    assert _select_own_leg(legs, "f") is legs[0]


# ------------------------------------------------------------------------------
# F6: to_dataframe() on an all-collective result must not require a scalar
# `value` column.
# ------------------------------------------------------------------------------


def _u_couple_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    consumption = wage * work
    return consumption + 30.0 * (1.0 - work)


def _u_couple_m(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    consumption = wage * work
    return 2.0 * consumption


def _next_wage(work: DiscreteAction) -> ContinuousState:
    return 40.0 * work + 8.0 * (1.0 - work)


_WAGE_GRID_2 = LinSpacedGrid(start=8.0, stop=40.0, n_points=2)


def _make_all_collective_regimes() -> dict[str, Regime]:
    couple = Regime(
        transition=lambda: jnp.int32(1),
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_GRID_2},
        state_transitions={"wage": _next_wage},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_couple_f, "utility_m": _u_couple_m},
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_GRID_2},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_couple_f, "utility_m": _u_couple_m},
    )
    return {"couple": couple, "couple_terminal": couple_terminal}


def _solve_all_collective():
    ages = AgeGrid(start=0, stop=2, step="Y")
    regimes_dict = _make_all_collective_regimes()
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict, ages=ages, regime_names=list(regimes_dict)
    )
    flat_params = MappingProxyType(
        {
            "couple": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "couple_terminal": MappingProxyType({}),
        }
    )
    _bi_result = solve(
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    solution = _bi_result.value_functions
    _sim_policies = _bi_result.simulation_policies
    dissolution_flags = _bi_result.dissolution_flags
    return ages, regimes, regime_names_to_ids, flat_params, solution, dissolution_flags


def test_to_dataframe_all_collective_result_has_no_scalar_value_column():
    """Repro (see report): pre-fix, this raised `KeyError: ['value'] not in
    index` inside `_reorder_columns` -- every populated regime here is
    collective, so `_process_regime` never writes a scalar `value` column.
    Post-fix, `to_dataframe()` returns `value_f`/`value_m` and no `value`.
    """
    ages, regimes, regime_names_to_ids, flat_params, solution, dissolution_flags = (
        _solve_all_collective()
    )
    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([8.0, 40.0]),
            "age": jnp.array([0.0, 0.0]),
            "regime_id": jnp.array([0, 0], dtype=jnp.int32),
        }
    )
    result = simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=regimes,
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="off"),
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=dissolution_flags,
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    )
    df = result.to_dataframe()
    assert "value" not in df.columns
    assert {"value_f", "value_m"} <= set(df.columns)
    assert len(df) == 4  # 2 subjects x 2 periods
    assert df["value_f"].notna().all()
    assert df["value_m"].notna().all()


# ------------------------------------------------------------------------------
# F7: a stateless collective regime's V_arr / argmax index must carry the
# subject axis.
# ------------------------------------------------------------------------------


def _u_stateless_f(work: DiscreteAction) -> FloatND:
    return 10.0 * work


def _u_stateless_m(work: DiscreteAction) -> FloatND:
    return 5.0 * (1.0 - work)


def _make_stateless_collective_regime() -> dict[str, Regime]:
    regime = Regime(
        transition=None,
        stakeholders=("f", "m"),
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_stateless_f, "utility_m": _u_stateless_m},
    )
    return {"stateless_couple": regime}


def test_stateless_collective_regime_simulate_carries_subject_axis():
    """Repro (see report): pre-fix this raised `ValueError: vmap was
    requested to map its argument along axis 0, which implies that its rank
    should be at least 1, but is only 0` -- `argmax_and_max_Q_over_a` has no
    per-subject state array to vmap over, so it returns a single
    `(n_stakeholders,)` V_arr and a 0-d argmax index, un-broadcast to the
    subject axis. Post-fix, `V_arr` is `(n_subjects, n_stakeholders)` and
    every subject's own values/actions round-trip correctly.
    """
    ages = AgeGrid(start=0, stop=1, step="Y")
    regimes_dict = _make_stateless_collective_regime()
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict, ages=ages, regime_names=list(regimes_dict)
    )
    flat_params = MappingProxyType({"stateless_couple": MappingProxyType({})})
    _bi_result = solve(
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    solution = _bi_result.value_functions
    _sim_policies = _bi_result.simulation_policies
    _dissolution_flags = _bi_result.dissolution_flags
    assert solution[0]["stateless_couple"].shape == (2,)  # solve: no subject axis

    n_subjects = 3
    initial_conditions = MappingProxyType(
        {
            "age": jnp.zeros(n_subjects),
            "regime_id": jnp.zeros(n_subjects, dtype=jnp.int32),
        }
    )
    result = simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=regimes,
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="off"),
        period_to_regime_to_V_arr=solution,
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    )
    period_0 = result.raw_results["stateless_couple"][0]
    assert period_0.V_arr.shape == (n_subjects, 2)
    np.testing.assert_allclose(
        np.asarray(period_0.V_arr), [[10.0, 0.0]] * n_subjects, rtol=1e-6
    )
    np.testing.assert_array_equal(np.asarray(period_0.actions["work"]), [1, 1, 1])
    assert period_0.in_regime.shape == (n_subjects,)


def test_stateless_collective_without_any_action_is_rejected_upstream():
    """Documents scope: a collective regime with NO discrete action at all
    (the OTHER stateless variant the finding names) is already rejected by
    `_validate_collective_regime` before this guard is ever reached -- "a
    collective regime must have at least one discrete action" -- so it can
    never reach `_simulate_regime_in_period` in the first place.
    """
    with pytest.raises(RegimeInitializationError, match="discrete action"):
        Regime(
            transition=None,
            stakeholders=("f", "m"),
            functions={
                "utility_f": lambda: jnp.asarray(3.0),
                "utility_m": lambda: jnp.asarray(7.0),
            },
        )


def test_stateful_collective_regime_simulate_shape_is_byte_identical():
    """Regression pin: a STATEFUL collective regime (has its own state axis,
    e.g. `_make_couple_regimes`'s `wage`) already carries the subject axis
    from its per-subject state arrays and must be untouched by the F7
    branch. Reuses `_make_all_collective_regimes` (has a `wage` state).
    """
    ages, regimes, regime_names_to_ids, flat_params, solution, dissolution_flags = (
        _solve_all_collective()
    )
    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([8.0, 40.0]),
            "age": jnp.array([0.0, 0.0]),
            "regime_id": jnp.array([0, 0], dtype=jnp.int32),
        }
    )
    result = simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=regimes,
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="off"),
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=dissolution_flags,
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    )
    period_0 = result.raw_results["couple"][0]
    assert period_0.V_arr.shape == (2, 2)
    np.testing.assert_allclose(
        np.asarray(period_0.V_arr), [[46.0, 92.0], [78.0, 156.0]], rtol=1e-6
    )
