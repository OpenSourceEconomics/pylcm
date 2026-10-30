"""A CARRIED-ONLY state composes with a repeating self-loop arrival decision.

The carried-state feature (`Phased(solve=impute, simulate=Grid)`,
`tests/test_carried_states.py`) is otherwise exercised on ordinary regime
crossings. The EKL "Career and Family Decisions" replication needs a
simulate-only uncapped `raw_experience` accumulator living in regimes whose
declined-marriage path is a REPEATING self-loop value decision, so the
interaction is pinned here in isolation.

Topology (the singleton half of EKL's `single_f`): `src` is covered over ages
0-1 with a repeating self-arrival back to itself, fallback into a terminal
`src_fallback`; at its last covered age the ordinary transition routes it to
`src_exit`. The occupied-boundary counterpart is
`test_repeating_arrival_boundary.py`,
`test_public_repeating_arrival_stops_at_the_sources_last_supported_date`.
A carried `career` state accumulates `+1` each period and is read by utility
with a ZERO coefficient, isolating the carried-state effect on the occupied
value and routing.

The decision is OPEN at wage=2 (stays in `src` for the genuine self-loop repeat)
and CLOSED at wage=1 (routed to `src_fallback`, which does not carry `career`).
The model builds, solves and simulates with one ordinary target (`src_exit`)
that does NOT declare the carried state, and the carried value accumulates
through the kept self-arrival (career 20 -> 21) rather than being frozen or
reset to its solve-phase imputation.
"""

import jax.numpy as jnp
import numpy as np

from lcm import (
    ArrivalDecision,
    ArrivalRead,
    ByAge,
    DiscreteGrid,
    Gate,
    LinSpacedGrid,
    Model,
    Phased,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    categorical,
    fixed_transition,
)
from lcm.ages import AgeGrid
from lcm.solver_api import ResultRetention
from lcm.transition import StochasticTransition, Transition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)
from tests.conftest import DECIMAL_PRECISION

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
    *, wage: ContinuousState, work: DiscreteAction, career: FloatND
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


def _repeat_at_high_wage(wage: ContinuousState) -> BoolND:
    return wage > 1.5


def _src_transition() -> Transition:
    """Return `src`'s transition: a gated self-loop until age 2, then `src_exit`.

    `tests/regime_building/test_invariant_components.py` reads it as a gated edge
    touching a carrier. The gate states on the wage what the public test's arrival
    decision computes from the value: the repeat stays open at wage 2 only.
    """
    exit_cell = StochasticTransition(func=_prob_exit_boundary)
    return Transition(
        law=ByAge.until(
            stop_age_exclusive=2,
            law={"src": StochasticTransition(func=_prob_stay), "src_exit": exit_cell},
            then={"src_exit": exit_cell},
        ),
        gates={
            "src": Gate(
                predicate=_repeat_at_high_wage,
                routes={
                    "only": StakeholderRoute(
                        fallback=ProjectedRegimeValue(
                            regime="src_fallback",
                            projection={"wage": _identity_wage},
                        ),
                    )
                },
            )
        },
    )


def _make_regimes() -> dict[str, Regime]:
    """Return the regimes without their laws; `_src_transition` declares `src`'s."""
    src = Regime(
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
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _u_src_repeat},
    )
    src_exit = Regime(states={"wage": _WAGE}, functions={"utility": _u_src_exit})
    src_fallback = Regime(
        states={"wage": _WAGE}, functions={"utility": _u_src_fallback}
    )
    return {"src": src, "src_exit": src_exit, "src_fallback": src_fallback}


@categorical(ordered=False)
class _PublicRegimeId:
    src: ScalarInt
    src_exit: ScalarInt
    src_fallback: ScalarInt


def test_public_repeated_arrival_carries_career_once_before_the_ordinary_exit() -> None:
    """A kept self-arrival increments true career once and then exits ordinarily."""
    exit_cell = StochasticTransition(func=_prob_exit_boundary)
    model = Model(
        regimes=_make_regimes(),
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=_PublicRegimeId,
        initial_nodes={0: "src"},
        edges={
            "src": Transition(
                targets={"src": 0, "src_exit": (0, 1), "src_fallback": 0},
                law=ByAge.until(
                    stop_age_exclusive=2,
                    law={
                        "src": StochasticTransition(func=_prob_stay),
                        "src_exit": exit_cell,
                    },
                    then={"src_exit": exit_cell},
                ),
                arrivals={
                    "src": ArrivalDecision(
                        predicate=_repeat_gate,
                        reads={
                            "V_target": ArrivalRead(
                                regime="src",
                                kind="IntegratedFull",
                                projection={"wage": _identity_wage},
                            )
                        },
                        routes={
                            "self": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="src_fallback",
                                    projection={"wage": _identity_wage},
                                )
                            )
                        },
                    )
                },
            )
        },
    )
    assert "career" not in model.user_regimes["src_fallback"].states
    assert "career" not in model.user_regimes["src_exit"].states
    params = {"discount_factor": _BETA}
    solution = model.solve(
        params=params, retention=ResultRetention.VALUES, log_level="off"
    )
    np.testing.assert_array_almost_equal(
        solution.values[0]["src"], [1.095, 4.8025], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        solution.values[1]["src"], [1.475, 2.95], decimal=DECIMAL_PRECISION
    )
    assert "src" not in solution.values[2]
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wage": jnp.asarray([1.0, 2.0]),
            "career": jnp.asarray([10.0, 20.0]),
            "age": jnp.asarray([0.0, 0.0]),
            "regime_id": jnp.full(2, model.regime_names_to_ids["src"], dtype=jnp.int32),
        },
        seed=0,
        log_level="off",
    )
    frame = result.to_dataframe()
    original = frame.loc[frame["period"] == 0].sort_values("subject_id")
    np.testing.assert_array_equal(original["subject_id"], [0, 1])
    np.testing.assert_array_equal(original["career"], [10.0, 20.0])
    landed = frame.loc[frame["period"] == 1].sort_values("subject_id")
    np.testing.assert_array_equal(landed["subject_id"], [0, 1])
    np.testing.assert_array_equal(landed["regime_name"], ["src_fallback", "src"])
    np.testing.assert_array_equal(landed["wage"], [1.0, 2.0])
    np.testing.assert_array_equal(
        landed.loc[landed["regime_name"] == "src", "career"], [21.0]
    )
    np.testing.assert_array_almost_equal(
        landed["value"], [0.1, 2.95], decimal=DECIMAL_PRECISION
    )
    assert "career" not in result.raw_results["src_fallback"][1].states
    exited = frame.loc[(frame["period"] == 2) & (frame["regime_name"] == "src_exit")]
    np.testing.assert_array_equal(exited["subject_id"], [1])
    np.testing.assert_array_equal(exited["wage"], [2.0])
    np.testing.assert_array_equal(exited["value"], [1.0])
    assert "career" not in result.raw_results["src_exit"][2].states
