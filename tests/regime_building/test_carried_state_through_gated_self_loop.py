"""Does a CARRIED-ONLY state compose with a repeating self-loop `GatedEdge`?

The carried-state feature (`Phased(solve=impute, simulate=Grid)`,
`tests/test_carried_states.py`) is exercised only on ORDINARY regime crossings.
The whole pylcm test suite has NO case combining a carried state with
`gated_edges` (verified 2026-07-22). Yet the EKL "Career and Family Decisions"
replication needs exactly this: a simulate-only uncapped `raw_experience`
accumulator living in `single_f`/`single_m` — regimes whose declined-marriage
path is a REPEATING self-loop gated edge (mutual-consent, eq. 27) — and in the
collective `married` regime. Before wiring six coupled EKL regimes on the
assumption the mechanism composes, pin the interaction here in isolation.

Topology (the singleton half of EKL's `single_f`): `src` is active over ages
0-1 with a repeating self-loop `GatedEdge` back to itself, fallback into a
terminal `src_fallback`; past its activity boundary the ordinary transition
routes it to `src_exit`. This mirrors
`test_repeating_self_loop_gated_edge_simulates_past_activity_boundary` exactly,
plus a carried `career` state that accumulates `+1` each period and is read by
utility with a ZERO coefficient — so the value function (and hence the gate
routing) is byte-identical to that test, isolating the carried-state machinery.

The gate is OPEN at wage=2 (stays in `src` for the genuine self-loop repeat)
and CLOSED at wage=1 (routed to `src_fallback`, which does not carry `career`).
The two questions with no existing coverage:

(a) does the model BUILD + solve + simulate at all with a carried state inside a
    gated self-loop regime, and one ordinary target (`src_exit`) that does NOT
    declare the carried state — the exact EKL shape where `single_f` carries
    `raw_experience` but `single_f_terminal` does not;
(b) does the carried value ACCUMULATE correctly through the gate-OPEN self-loop
    repeat (career 20 -> 21), rather than being frozen or reset to its
    solve-phase imputation.
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
    Phased,
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

_BETA = 0.95
_WAGE = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)  # {1.0, 2.0}
_CAREER = LinSpacedGrid(start=0.0, stop=30.0, n_points=4)  # carried-only range
_REPEAT_GATE_THRESHOLD = 2.0


@categorical(ordered=True)
class Work:
    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


def _identity_wage(wage: ContinuousState) -> ContinuousState:
    return wage


def _impute_career() -> FloatND:
    """Solve-phase career: a constant imputation (never a solve grid axis)."""
    return jnp.asarray(0.0)


def _next_career(career: FloatND) -> FloatND:
    """Simulate-phase carried law: accumulates +1 each period, uncapped."""
    return career + 1.0


def _u_src_repeat(
    wage: ContinuousState, work: DiscreteAction, career: FloatND
) -> FloatND:
    # `career` enters with a ZERO coefficient: the DAG reads it (so it is a used
    # state) but the value function is numerically identical to the carried-free
    # fixture, so the gate routing is unchanged.
    return wage + 0.0 * work + 0.0 * career


def _u_src_exit(wage: ContinuousState) -> FloatND:
    return 0.5 * wage


def _u_src_fallback(wage: ContinuousState) -> FloatND:
    return 0.1 * wage


def _prob_stay(age: FloatND) -> FloatND:
    return jnp.where(age < 1.0, 1.0, 0.0)


def _prob_exit_boundary(age: FloatND) -> FloatND:
    return jnp.where(age < 1.0, 0.0, 1.0)


def _repeat_gate(V_target: FloatND) -> BoolND:
    return V_target > _REPEAT_GATE_THRESHOLD


def _make_regimes() -> dict[str, Regime]:
    src = Regime(
        transition={
            "src": MarkovTransition(_prob_stay),
            "src_exit": MarkovTransition(_prob_exit_boundary),
        },
        active=lambda age: age < 2,
        states={
            "wage": _WAGE,
            "career": Phased(solve=_impute_career, simulate=_CAREER),
        },
        state_transitions={
            "wage": fixed_transition("wage"),
            # A carried state does not support a per-target dict law (rejected
            # by phases.py). So the law is PLAIN (all-target): it produces a
            # next value for every ordinary target, including `src_exit`, which
            # does NOT declare `career`. This is the EKL shape: `single_f`
            # carries `raw_experience` but its terminal target does not. The
            # question is whether the plain law tolerates the non-carrying
            # target.
            "career": _next_career,
        },
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_src_repeat},
        gated_edges={
            "src": GatedEdge(
                gate=_repeat_gate,
                legs={
                    "only": EdgeLeg(
                        fallback=SamePeriodRef(
                            regime="src_fallback",
                            projection={"wage": _identity_wage},
                        ),
                    )
                },
            )
        },
    )
    src_exit = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE},
        functions={"utility": _u_src_exit},
    )
    src_fallback = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE},
        functions={"utility": _u_src_fallback},
    )
    return {"src": src, "src_exit": src_exit, "src_fallback": src_fallback}


def _solve_and_simulate():
    ages = AgeGrid(start=0, stop=3, step="Y")
    regimes_dict = _make_regimes()
    regime_names = list(regimes_dict)
    regime_names_to_ids = MappingProxyType(
        {name: jnp.int32(i) for i, name in enumerate(regime_names)}
    )
    regimes = process_regimes(
        user_regimes=finalize_regimes(
            user_regimes=regimes_dict, derived_categoricals={}
        ),
        ages=ages,
        regime_names_to_ids=regime_names_to_ids,
        enable_jit=False,
    )
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "src_exit": MappingProxyType({}),
            "src_fallback": MappingProxyType({}),
        }
    )
    bi_result = solve(
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    initial_conditions = MappingProxyType(
        {
            "wage": jnp.array([1.0, 2.0]),
            "career": jnp.array([10.0, 20.0]),
            "age": jnp.array([0.0, 0.0]),
            "regime_id": jnp.array([regime_names_to_ids["src"]] * 2, dtype=jnp.int32),
        }
    )
    return simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=regimes,
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="off"),
        period_to_regime_to_V_arr=bi_result.value_functions,
        period_to_regime_to_dissolution_flags=bi_result.dissolution_flags,
        ages=ages,
        simulation_output_dtypes={},
        seed=0,
    )


def test_carried_state_builds_and_solves_inside_gated_self_loop():
    """(a) The model with a carried state in a gated self-loop regime builds,
    solves, and simulates without error, and the added zero-coefficient carried
    state leaves the value function (hence routing) byte-identical to the
    carried-free fixture."""
    result = _solve_and_simulate()
    period_0 = result.raw_results["src"][0]
    # Identical to `test_repeating_self_loop_gated_edge_...`'s V_0 = [1.095,
    # 4.8025]: the carried state did not perturb the solve.
    np.testing.assert_allclose(np.asarray(period_0.V_arr), [1.095, 4.8025], rtol=1e-6)

    # Routing unchanged: wage=1 -> fallback, wage=2 -> self-loop repeat.
    fallback_1 = result.raw_results["src_fallback"][1]
    src_1 = result.raw_results["src"][1]
    np.testing.assert_array_equal(np.asarray(fallback_1.in_regime), [True, False])
    np.testing.assert_array_equal(np.asarray(src_1.in_regime), [False, True])


def test_carried_state_accumulates_through_gate_open_self_loop():
    """(b) The wage=2 household stays in `src` for the genuine self-loop repeat;
    its carried `career` accumulated 20 -> 21 via `_next_career`, i.e. the true
    carried value was evolved, not frozen at or reset to the solve imputation
    (0.0)."""
    result = _solve_and_simulate()
    src_1 = result.raw_results["src"][1]
    career_1 = np.asarray(src_1.states["career"])
    # Household index 1 (wage=2) is the one that stays in `src` at period 1.
    np.testing.assert_allclose(career_1[1], 21.0, rtol=1e-6)
