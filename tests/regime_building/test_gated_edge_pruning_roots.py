"""A gated edge's reads keep the target regime's variables alive.

A gated edge is declared on the SOURCE regime but its gate and its gate-reference
projections are evaluated on the TARGET regime's grid. The target regime's own
DAG therefore never mentions them, and any check that walks one regime in
isolation misses the read: a broadcast variable the gate needs is pruned out from
under it, and a target state the projection needs is reported as never used.

Both models below are the same three-regime lifecycle — `worker` reaches
`retired` through a gated edge whose closed branch falls back to `outside` — and
differ only in which gated-edge function does the reading.
"""

import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal as aaae

from lcm import (
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
)
from lcm.regime import EdgeLeg, GatedEdge, Regime, SamePeriodRef
from lcm.transition import MarkovTransition
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt
from tests.collective_fixtures import AGES, Work
from tests.conftest import DECIMAL_PRECISION


@categorical(ordered=False)
class GatedRegimeId:
    """Regime ids of both models in this module."""

    worker: ScalarInt  # code 0
    retired: ScalarInt  # code 1
    outside: ScalarInt  # code 2


# The wage state every regime carries, on nodes {1.0, 2.0}.
_WAGE_GRID = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)

# The second state, on nodes {0.0, 1.0}, whose only consumer is a gated-edge read.
_BONUS_GRID = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)

# Discount factor of `worker`, the only regime with a continuation.
_DISCOUNT_FACTOR = 0.9

# `worker`'s period-0 value function of the projection model, indexed
# (wage, bonus). Gate-open cells take `retired`'s value, gate-closed cells
# `outside`'s: with `V_target = 1.5 * wage` and the reference read at the
# projected wage `1 + bonus` of `V_outside = 2 * wage`, the gate
# `V_target > V_outside_ref` is open at (wage=2, bonus=0) alone, so the folded
# continuation is ((2, 2), (3, 4)). Working dominates everywhere, hence
# `V = wage + bonus + 0.9 * continuation`.
_PROJECTION_MODEL_V_PERIOD_0 = ((2.8, 3.8), (4.7, 6.6))


def test_broadcast_state_read_by_an_incoming_gate_survives_in_the_target():
    """A model-level state a gated edge's gate reads is kept by the edge's target.

    The gate is evaluated pointwise on the target regime's grid, so the target
    must carry every state the gate names even when nothing else in that regime
    reads it. A regime no edge reaches still drops the broadcast state.
    """
    model = _make_model_with_a_gate_reading_a_broadcast_state()

    assert model.pruned_variables == {
        "worker": frozenset(),
        "retired": frozenset(),
        "outside": frozenset({"bonus"}),
    }


def test_target_state_read_only_by_an_incoming_projection_counts_as_used():
    """A target state a gate reference projects through is a used variable.

    The projection maps the target regime's own state cell into the reference
    regime's coordinates, so a target state read there is a decision input of
    the model and the target regime may declare it without declaring a second
    consumer.
    """
    model = _make_model_with_a_projection_reading_a_target_state()

    solution = model.solve(
        params={
            "worker": {"koopmans_aggregator": {"discount_factor": _DISCOUNT_FACTOR}}
        },
        log_level="debug",
    )

    aaae(
        solution[0]["worker"],
        jnp.asarray(_PROJECTION_MODEL_V_PERIOD_0),
        decimal=DECIMAL_PRECISION,
    )


def _make_model_with_a_gate_reading_a_broadcast_state() -> Model:
    """Build the lifecycle whose gate reads the model-level `bonus` state.

    `bonus` is declared once at model level and broadcast to all three regimes.
    `worker`'s utility reads it, the gate on the `worker -> retired` edge reads
    it, and nothing in `retired` or `outside` does.
    """
    worker = Regime(
        transition={"retired": MarkovTransition(_probability_one)},
        active=lambda age: age < 1,
        states={"wage": _WAGE_GRID},
        state_transitions={"wage": _next_wage},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility_worker},
        gated_edges={
            "retired": GatedEdge(
                gate=_gate_reading_bonus,
                legs={
                    "self": EdgeLeg(
                        target_stakeholder=None,
                        fallback=SamePeriodRef(
                            regime="outside",
                            projection={"wage": _project_wage_identically},
                        ),
                    )
                },
                gate_refs={
                    "V_outside_ref": SamePeriodRef(
                        regime="outside",
                        projection={"wage": _project_wage_identically},
                    )
                },
            )
        },
    )
    return Model(
        regimes={
            "worker": worker,
            "retired": _make_retired_regime(states={"wage": _WAGE_GRID}),
            "outside": _make_outside_regime(),
        },
        ages=AGES,
        regime_id_class=GatedRegimeId,
        states={"bonus": _BONUS_GRID},
        state_transitions={"bonus": fixed_transition("bonus")},
    )


def _make_model_with_a_projection_reading_a_target_state() -> Model:
    """Build the lifecycle whose gate-reference projection reads `bonus`.

    `bonus` is declared on `worker` and on `retired` directly. `worker`'s
    utility reads it; in `retired` its only consumer is the projection under
    which the incoming edge's gate reference reads `outside`'s value.
    """
    worker = Regime(
        transition={"retired": MarkovTransition(_probability_one)},
        active=lambda age: age < 1,
        states={"wage": _WAGE_GRID, "bonus": _BONUS_GRID},
        state_transitions={"wage": _next_wage, "bonus": fixed_transition("bonus")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility_worker},
        gated_edges={
            "retired": GatedEdge(
                gate=_gate_comparing_values,
                legs={
                    "self": EdgeLeg(
                        target_stakeholder=None,
                        fallback=SamePeriodRef(
                            regime="outside",
                            projection={"wage": _project_wage_identically},
                        ),
                    )
                },
                gate_refs={
                    "V_outside_ref": SamePeriodRef(
                        regime="outside",
                        projection={"wage": _project_wage_from_bonus},
                    )
                },
            )
        },
    )
    return Model(
        regimes={
            "worker": worker,
            "retired": _make_retired_regime(
                states={"wage": _WAGE_GRID, "bonus": _BONUS_GRID}
            ),
            "outside": _make_outside_regime(),
        },
        ages=AGES,
        regime_id_class=GatedRegimeId,
    )


def _make_retired_regime(*, states: dict[str, LinSpacedGrid]) -> Regime:
    """Build the gated edge's target regime over the given states."""
    return Regime(
        transition=None,
        active=lambda age: age >= 1,
        states=states,
        functions={"utility": _utility_retired},
    )


def _make_outside_regime() -> Regime:
    """Build the reference regime the gate-closed branch falls back to."""
    return Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE_GRID},
        functions={"utility": _utility_outside},
    )


def _probability_one(age: FloatND) -> FloatND:
    """Regime transition: `worker` reaches `retired` with probability one."""
    return jnp.ones_like(age, dtype=float)


def _utility_worker(
    wage: ContinuousState, work: DiscreteAction, bonus: ContinuousState
) -> FloatND:
    """Working earns the wage; the bonus accrues either way."""
    return wage * work + bonus


def _utility_retired(wage: ContinuousState) -> FloatND:
    """Retirement pays one and a half times the wage."""
    return 1.5 * wage


def _utility_outside(wage: ContinuousState) -> FloatND:
    """The outside option pays twice the wage."""
    return 2.0 * wage


def _next_wage(wage: ContinuousState) -> ContinuousState:
    """The wage is carried forward unchanged."""
    return wage


def _project_wage_identically(wage: ContinuousState) -> ContinuousState:
    """Read the reference regime at the same wage the target cell sits on."""
    return wage


def _project_wage_from_bonus(bonus: ContinuousState) -> ContinuousState:
    """Read the reference regime at the wage the target cell's bonus buys."""
    return 1.0 + bonus


def _gate_reading_bonus(
    V_target: FloatND, V_outside_ref: FloatND, bonus: ContinuousState
) -> BoolND:
    """Retirement is open when it beats the outside option and a bonus is due."""
    return (V_target > V_outside_ref) & (bonus > 0.5)


def _gate_comparing_values(V_target: FloatND, V_outside_ref: FloatND) -> BoolND:
    """Retirement is open when it beats the outside option."""
    return V_target > V_outside_ref
