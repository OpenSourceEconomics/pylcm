"""Regression: the gated-edge GATE reader with a NON-FOLDED process-state TARGET.

Companion to `test_same_period_ref_process_state_interpolation.py`. That fix
(`_lcm.regime_building.V.get_V_interpolator`'s `interpolate_process_axes` mode,
auto-selected by `_build_same_period_ref_reader` in `Q_and_F.py`) covers every
reader built off `_build_same_period_ref_reader`: the solve-time fold's own
identity-projection reads of the target's `V_target_<s>` / `D_target` and the
gate's `gate_refs`, and the simulate-side FALLBACK-leg projector.

One reader was NOT covered: the gated-edge's own GATE array reader —
`gate_interpolators[target_name]` built in
`_lcm.regime_building.processing._attach_gated_edge_folds` (around line 451,
stored as `gated_edge_gate_interpolators` on the canonical regime) via a plain
`get_V_interpolator(..., V_arr_name=GATE_ARR_NAME)` call, unconditionally
`interpolate_process_axes=False`. `_lcm.simulation.gated_routing.route_gated_edges`
calls this interpolator at the REALIZED candidate target-state draw
(`calculate_next_states`' output for the target regime) to decide ACTUAL
routing. When the target regime carries a non-folded process axis (e.g. EKL's
wife's transitory wage shock persisting into `married`), that candidate draw
is a genuine VALUE for the process axis (one of its own discretized nodes, but
a raw float, never an integer node INDEX) — hitting the same
`ValueError: Indexer must have integer or boolean type` `V.py` raises for any
axis fed a value instead of an index, this time from `processing.py`'s
GATE_ARR_NAME reader rather than `Q_and_F.py`'s same-period-ref reader.

Fix: `_attach_gated_edge_folds` now auto-selects `interpolate_process_axes=True`
for the gate interpolator whenever the TARGET regime's own
`VInterpolationInfo.discrete_states` carries a `_ContinuousStochasticProcess`
axis — the identical auto-select `_build_same_period_ref_reader` already uses,
applied to this second, previously-missed reader.

UPDATE (simulate F1 fix). `route_gated_edges` no longer interpolates a baked
boolean `gate` array at all — `gated_edge_gate_interpolators` /
`GATE_ARR_NAME` are gone; `route_gated_edges` now RECOMPUTES the gate from
interpolated VALUE operands via `get_edge_simulate_gate_evaluator`
(`_lcm.regime_building.gated_edges`), stored as
`gated_edge_simulate_gate_evaluators`. The `interpolate_process_axes`
auto-select this test exercises now lives on the TARGET-V-component and `D`
interpolators inside that evaluator instead (same auto-select condition,
same target-grid process axis). This test's own assertions are UNCHANGED by
that fix and continue to pass: `_consent_gate`'s two operands are engineered
to be LINEAR in `shock` on both sides of the comparison (`_u_married_f`/`_m`
and the terminal single references), so linearly interpolating the VALUES
and applying the strict inequality is, for this specific fixture, exactly
equivalent to linearly interpolating the boolean predicate's own {0,1} grid
and thresholding at 0.5 (`_hand_computed_gate`'s derivation below still
holds — see the recomputed derivation in the module docstring's tail, kept
for the record). This is a special-case coincidence of a fixture engineered
to isolate the process-axis interpolation machinery in isolation, not a
general property — see `test_gated_edge_simulate_operand_recompute.py` for a
fixture where interpolate-then-threshold and recompute-then-predicate
genuinely DISAGREE.

This test builds a singleton source (`single_f`) with a mutual-consent gated
edge into a collective target (`married_terminal`, stakeholders `f`/`m`) that
carries a non-folded `NormalIIDProcess` shock IN ADDITION to `wage`. Utility
functions are engineered so the consent gate, EVALUATED ON THE TARGET'S OWN
DISCRETIZED GRID (nodes `[-1, 0, 1]`), is `[False, False, True]` at every
wage -- independent of wage (see `_consent_gate`'s derivation in the module
comments below). `_create_continuous_stochastic_next_func` (the simulate-side
process draw) samples the underlying CONTINUOUS distribution directly, not
the discretized nodes, so every simulated household's candidate `shock` for
`married_terminal` is a genuinely off-grid value -- exactly the case that
needs the gate reader's `interpolate_process_axes` fix, not merely an
integer-valued edge case. Clamped LINEAR interpolation of the boolean-as-float
grid `[0, 0, 1]` at nodes `[-1, 0, 1]` collapses to a clean closed form: 0 for
`shock <= 0`, `shock` for `shock` in `[0, 1]`, 1 for `shock >= 1` -- so
"interpolated value `> 0.5`" is exactly "`shock > 0.5`" (`_hand_computed_gate`).

(a) the model SOLVES and SIMULATES two periods without the integer-indexer
    crash (pre-fix: `ValueError` at the second period's routing step);
(b) EVERY simulated household's ACTUAL routing (`married_terminal` vs. its
    `single_f_terminal` fallback) matches the hand-computed
    `shock > 0.5` gate evaluated from that household's own realized
    (candidate, off-grid) shock draw -- reading the process axis off
    `states["shock"]` on the un-gated candidate target-state record,
    independent of the fix under test;
(c) the routed households' recorded values match the OPEN / CLOSED branch's
    hand-computed formula exactly, at each household's own off-grid shock;
(d) both branches are non-empty (a genuine test of routing on both sides of
    the process-axis threshold, not a degenerate single-outcome run).
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np

from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from _lcm.simulation.simulate import simulate
from _lcm.solution.backward_induction import solve
from _lcm.utils.logging import get_logger
from lcm import (
    DiscreteGrid,
    LinSpacedGrid,
    NormalIIDProcess,
    categorical,
    fixed_transition,
)
from lcm.ages import AgeGrid
from lcm.regime import EdgeLeg, GatedEdge, Regime, SamePeriodRef
from lcm.transition import MarkovTransition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)


@categorical(ordered=True)
class Work:
    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


_BETA = 0.95
_WAGE = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)  # {1.0, 2.0}
# Nodes: linspace(mu - n_std*sigma, mu + n_std*sigma, n_points) = [-1, 0, 1].
_SHOCK = NormalIIDProcess(n_points=3, gauss_hermite=False, mu=0.0, sigma=1.0, n_std=1.0)
_AGES = AgeGrid(start=0, stop=2, step="Y")
_REGIME_NAMES_TO_IDS = MappingProxyType(
    {
        "single_f": jnp.int32(0),
        "single_f_terminal": jnp.int32(1),
        "single_m_terminal": jnp.int32(2),
        "married_terminal": jnp.int32(3),
    }
)


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _identity_wage(wage: ContinuousState) -> ContinuousState:
    return wage


def _u_single_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return wage * work


def _u_single_f_terminal(wage: ContinuousState) -> FloatND:
    return wage + 0.5  # {1.5, 2.5}


def _u_single_m_terminal(wage: ContinuousState) -> FloatND:
    return wage + 0.5  # {1.5, 2.5}


def _u_married_f(
    wage: ContinuousState, shock: FloatND, work: DiscreteAction
) -> FloatND:
    # 2*wage + shock ranges over [1, 5] on this grid -> work=1 always optimal.
    return (2.0 * wage + shock) * work


def _u_married_m(
    wage: ContinuousState, shock: FloatND, work: DiscreteAction
) -> FloatND:
    # wage + shock ranges over [0, 3] on this grid -> work=1 always (weakly) optimal.
    return (wage + shock) * work


def _consent_gate(
    V_target_f: FloatND,
    V_target_m: FloatND,
    V_single_f_ref: FloatND,
    V_single_m_ref: FloatND,
) -> BoolND:
    # EKL eq. 27: strict, unanimous mutual consent.
    return (V_target_f > V_single_f_ref) & (V_target_m > V_single_m_ref)


# Derivation of the ON-GRID gate, a pure function of `shock` (independent of
# wage), evaluated exactly at the target's three discretized shock nodes.
# The wife's marriage value is "2 times wage plus shock" against her single
# reference "wage plus 0.5", so her consent needs "wage plus shock" above 0.5:
# true for shock in {0, 1} at wage 1 and for all shock at wage 2. The husband's
# marriage value is "wage plus shock" against the same "wage plus 0.5"
# reference, so his consent needs shock above 0.5: true only at shock 1. Their
# conjunction (strict, unanimous) is therefore "shock equals 1" at every wage,
# i.e. the boolean gate as a float grid over nodes [-1, 0, 1] is [0, 0, 1].
#
# The GATE reader interpolates this float grid at a genuinely OFF-GRID shock
# value via clamped linear interpolation (this fix's `interpolate_process_axes`
# mode): flat at 0 across [-1, 0], a straight ramp 0 -> 1 across [0, 1],
# clamped to 0 / 1 outside [-1, 1]. Thresholding the interpolated float at
# `GATE_THRESHOLD` (0.5) therefore collapses to the clean closed form
# "shock above 0.5", independent of wage and of which two nodes bracket a draw.
def _hand_computed_gate(shock: np.ndarray) -> np.ndarray:
    return shock > 0.5


def _make_regimes() -> dict[str, Regime]:
    single_f = Regime(
        transition={"married_terminal": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"wage": _WAGE, "shock": _SHOCK},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_single_f},
        gated_edges={
            "married_terminal": GatedEdge(
                gate=_consent_gate,
                legs={
                    "f": EdgeLeg(
                        target_stakeholder="f",
                        fallback=SamePeriodRef(
                            regime="single_f_terminal",
                            projection={"wage": _identity_wage},
                        ),
                    )
                },
                gate_refs={
                    "V_single_f_ref": SamePeriodRef(
                        regime="single_f_terminal",
                        projection={"wage": _identity_wage},
                    ),
                    "V_single_m_ref": SamePeriodRef(
                        regime="single_m_terminal",
                        projection={"wage": _identity_wage},
                    ),
                },
            )
        },
    )
    single_f_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE},
        functions={"utility": _u_single_f_terminal},
    )
    single_m_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE},
        functions={"utility": _u_single_m_terminal},
    )
    married_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE, "shock": _SHOCK},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_married_f, "utility_m": _u_married_m},
    )
    return {
        "single_f": single_f,
        "single_f_terminal": single_f_terminal,
        "single_m_terminal": single_m_terminal,
        "married_terminal": married_terminal,
    }


def _flat_params() -> MappingProxyType:
    return MappingProxyType(
        {
            "single_f": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "single_f_terminal": MappingProxyType({}),
            "single_m_terminal": MappingProxyType({}),
            "married_terminal": MappingProxyType({}),
        }
    )


def _build_solve_and_simulate(*, n_subjects: int, seed: int):
    regimes = process_regimes(
        user_regimes=finalize_regimes(
            user_regimes=_make_regimes(), derived_categoricals={}
        ),
        ages=_AGES,
        regime_names_to_ids=_REGIME_NAMES_TO_IDS,
        enable_jit=False,
    )
    flat_params = _flat_params()
    _bi_result = solve(
        flat_params=flat_params,
        ages=_AGES,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    solution = _bi_result.value_functions
    _sim_policies = _bi_result.simulation_policies
    dissolution_flags = _bi_result.dissolution_flags
    wages = jnp.array([1.0 if i % 2 == 0 else 2.0 for i in range(n_subjects)])
    initial_conditions = MappingProxyType(
        {
            "wage": wages,
            "shock": jnp.zeros(n_subjects),  # single_f's own period-0 draw; unused
            "age": jnp.zeros(n_subjects),
            "regime_id": jnp.array(
                [_REGIME_NAMES_TO_IDS["single_f"]] * n_subjects, dtype=jnp.int32
            ),
        }
    )
    result = simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=regimes,
        regime_names_to_ids=_REGIME_NAMES_TO_IDS,
        logger=get_logger(log_level="off"),
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=dissolution_flags,
        ages=_AGES,
        simulation_output_dtypes={},
        seed=seed,
    )
    return wages, result


def test_gate_reader_solves_and_simulates_with_a_nonfolded_process_target():
    """(a) No crash: pre-fix this raised `ValueError` at the period-1 routing step."""
    _wages, result = _build_solve_and_simulate(n_subjects=40, seed=0)
    married = result.raw_results["married_terminal"][1]
    single_f_term = result.raw_results["single_f_terminal"][1]
    # Every household lands in exactly one of the two branches.
    np.testing.assert_array_equal(
        np.asarray(married.in_regime) | np.asarray(single_f_term.in_regime),
        np.ones(40, dtype=bool),
    )
    np.testing.assert_array_equal(
        np.asarray(married.in_regime) & np.asarray(single_f_term.in_regime),
        np.zeros(40, dtype=bool),
    )


def test_gate_reader_routing_matches_hand_computed_gate_and_both_branches_fire():
    """(b)-(d): routing follows `shock == 1` exactly; both branches are populated."""
    wages, result = _build_solve_and_simulate(n_subjects=40, seed=0)
    married = result.raw_results["married_terminal"][1]
    single_f_term = result.raw_results["single_f_terminal"][1]

    # The candidate draw for EVERY household is recorded on the target's own
    # state slot regardless of whether the gate routed them there (module
    # docstring of `_lcm.simulation.gated_routing`): read the shock draw
    # this way so the check is independent of the fix under test.
    shock_draws = np.asarray(married.states["shock"])
    expected_open = _hand_computed_gate(shock_draws)

    np.testing.assert_array_equal(np.asarray(married.in_regime), expected_open)
    np.testing.assert_array_equal(np.asarray(single_f_term.in_regime), ~expected_open)

    # (d) both branches actually fire on this draw (n_points=3, seed=0, n=40).
    assert expected_open.any()
    assert (~expected_open).any()

    # (c) routed households' recorded values match the hand-computed formula.
    wages_np = np.asarray(wages)
    married_V = np.asarray(married.V_arr)  # (n, 2): trailing stakeholder axis.
    np.testing.assert_allclose(
        married_V[expected_open, 0],
        2.0 * wages_np[expected_open] + shock_draws[expected_open],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        married_V[expected_open, 1],
        wages_np[expected_open] + shock_draws[expected_open],
        atol=1e-6,
    )
    single_f_term_V = np.asarray(single_f_term.V_arr)
    np.testing.assert_allclose(
        single_f_term_V[~expected_open],
        wages_np[~expected_open] + 0.5,
        atol=1e-6,
    )
