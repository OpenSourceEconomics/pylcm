"""A gated edge is gated where the source lands, not on the target's nodes.

A gate is a predicate, and a predicate does not commute with interpolation.
Baking it into a surface on the target's grid and interpolating that surface at
the realized landing point reports, in every cell whose corners fall on opposite
sides of the gate, a blend of the open and the closed branch — a number neither
branch pays, and one that can rank a source action above the one it should have
taken. Reading each operand at the landing point and gating it there is the
order forward simulation routes in, so the value the source maximizes and the
branch a subject is sent down agree.

The model below is exactly hand-computable, so every expected number is an
integer or a half-integer rather than a tolerance-bounded approximation.
"""

from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    ValueDependentTransition,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.transition import MarkovTransition
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from tests.conftest import DECIMAL_PRECISION

# Nodes 0, 1, 2. The gate opens strictly above 1.0, so the cell [1, 2] straddles
# it and is where gating on the nodes and gating at the landing point differ.
_X = LinSpacedGrid(start=0.0, stop=2.0, n_points=3)

# 1.5 lands inside the straddled cell; 2.0 lands exactly on a node.
_SAVING = IrregSpacedGrid(points=(1.5, 2.0))

_ON_GRID_SAVING = IrregSpacedGrid(points=(1.0, 2.0))


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt
    fallback: ScalarInt


def _certain_target(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _utility_source(*, x: ContinuousState, saving: ContinuousAction) -> FloatND:
    return -2.0 * saving + 0.0 * x


def _next_x(saving: ContinuousAction) -> ContinuousState:
    return saving


def _utility_target(x: ContinuousState) -> FloatND:
    return 10.0 + x


def _utility_fallback(x: ContinuousState) -> FloatND:
    return 0.0 * x


def _identity_x(x: ContinuousState) -> ContinuousState:
    return x


def _gate(x: ContinuousState) -> BoolND:
    return x > 1.0


def _params() -> dict[str, dict[str, dict[str, float]]]:
    return {
        "source": {"koopmans_aggregator": {"discount_factor": 1.0}},
        "target": {},
        "fallback": {},
    }


def _make_model(
    *,
    saving_grid=_SAVING,
    off_grid: Literal["pointwise", "reject"] = "pointwise",
) -> Model:
    return Model(
        regimes={
            "source": Regime(
                transition={
                    "target": ValueDependentTransition(
                        probability=MarkovTransition(_certain_target),
                        gate=_gate,
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback",
                                    projection={"x": _identity_x},
                                )
                            )
                        },
                        off_grid=off_grid,
                    )
                },
                active=lambda age: age < 1,
                states={"x": _X},
                state_transitions={"x": _next_x},
                actions={"saving": saving_grid},
                functions={"utility": _utility_source},
            ),
            "target": Regime(
                transition=None,
                active=lambda age: age >= 1,
                states={"x": _X},
                functions={"utility": _utility_target},
            ),
            "fallback": Regime(
                transition=None,
                active=lambda age: age >= 1,
                states={"x": _X},
                functions={"utility": _utility_fallback},
            ),
        },
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
    )


def test_source_value_is_a_value_one_branch_actually_pays() -> None:
    """The source publishes `8.5`, the best gated value at a realized landing.

    Terminal values are `V_target(x) = 10 + x` and `V_fallback(x) = 0`. Saving
    `1.5` lands inside the straddled cell, where the gate is open, so the branch
    pays `V_target(1.5) = 11.5` against a flow cost of `-3`, i.e. `8.5`. Saving
    `2.0` pays `V_target(2) = 12` against `-4`, i.e. `8.0`.

    Gating on the nodes first turns the continuation into `[0, 0, 12]`, whose
    interpolant at `1.5` is `6` — halfway between the closed branch at `x = 1`
    and the open branch at `x = 2`, an amount no branch delivers — and would
    publish `8.0`.
    """
    solution = _make_model().solve(params=_params(), log_level="debug")

    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["source"]),
        np.full(3, 8.5),
        decimal=DECIMAL_PRECISION,
    )


def test_the_action_taken_is_the_one_the_gated_value_ranks_first() -> None:
    """The simulated subject saves `1.5`, not `2.0`.

    The two candidates are separated by a half unit of value, so this is a
    genuine reversal of the argmax rather than a rounding difference, and it is
    asserted as the discrete decision it is.
    """
    model = _make_model()

    result = model.simulate(
        params=_params(),
        initial_conditions={
            "age": jnp.array([0.0]),
            "x": jnp.array([0.0]),
            "regime_id": jnp.array([RegimeId.source]),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )
    first_period = result.to_dataframe().query("period == 0")

    assert first_period["saving"].to_numpy().tolist() == [1.5]


def test_landings_that_all_fall_on_nodes_are_unaffected() -> None:
    """With every landing point on a node, the gate is read exactly as declared.

    Saving `1.0` lands on the closed side and pays the fallback `0` against a
    flow cost of `-2`; saving `2.0` lands on the open side and pays `12` against
    `-4`. The best is `8.0`. Without this control the number above could come
    from gating differently rather than from gating in a different place.
    """
    model = _make_model(saving_grid=_ON_GRID_SAVING)

    solution = model.solve(params=_params(), log_level="debug")

    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["source"]),
        np.full(3, 8.0),
        decimal=DECIMAL_PRECISION,
    )


def test_off_grid_reject_refuses_a_continuous_target() -> None:
    """`off_grid="reject"` names the target's continuous state and stops.

    The setting promises the gate is never asked about a point the target's grid
    does not hold. A continuous target state breaks that promise, and building
    the model anyway would publish an interpolated gate under a declaration that
    says none is possible.
    """
    with pytest.raises(ModelInitializationError) as excinfo:
        _make_model(off_grid="reject")

    message = str(excinfo.value)
    assert "off_grid" in message
    assert "'x'" in message


def test_off_grid_reject_accepts_a_target_reached_exactly() -> None:
    """A target carrying only a discrete state satisfies the promise and builds.

    Every landing point is a node there, so the gate is read exactly where it
    was declared. Without this control the rejection above could come from
    declaring `off_grid="reject"` at all rather than from the target's grid.
    """
    model = _make_discrete_target_model(off_grid="reject")

    solution = model.solve(params=_params(), log_level="debug")

    # `sick` is closed and pays the fallback 0; `healthy` is open and pays 7.
    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["source"]),
        np.array([0.0, 7.0]),
        decimal=DECIMAL_PRECISION,
    )


@categorical(ordered=False)
class Health:
    sick: ScalarInt
    healthy: ScalarInt


def _utility_zero() -> FloatND:
    return jnp.asarray(0.0)


def _fallback_utility(health: DiscreteState) -> FloatND:
    """Zero either way, but read through `health` so the state is live."""
    return jnp.where(health == Health.healthy, 0.0, 0.0)


def _keep_health(health: DiscreteState) -> DiscreteState:
    return health


def _utility_healthy_target(health: DiscreteState) -> FloatND:
    return jnp.where(health == Health.healthy, 7.0, 3.0)


def _healthy_gate(health: DiscreteState) -> BoolND:
    return health == Health.healthy


def _make_discrete_target_model(*, off_grid: Literal["pointwise", "reject"]) -> Model:
    health = DiscreteGrid(category_class=Health)
    return Model(
        regimes={
            "source": Regime(
                transition={
                    "target": ValueDependentTransition(
                        probability=MarkovTransition(_certain_target),
                        gate=_healthy_gate,
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback",
                                    projection={"health": _keep_health},
                                )
                            )
                        },
                        off_grid=off_grid,
                    )
                },
                active=lambda age: age < 1,
                states={"health": health},
                state_transitions={"health": _keep_health},
                functions={"utility": _utility_zero},
            ),
            "target": Regime(
                transition=None,
                active=lambda age: age >= 1,
                states={"health": health},
                functions={"utility": _utility_healthy_target},
            ),
            "fallback": Regime(
                transition=None,
                active=lambda age: age >= 1,
                states={"health": health},
                functions={"utility": _fallback_utility},
            ),
        },
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
    )


# The canonical off-grid witness, in exact halves and fifths: a target whose
# value crosses its gate reference between the two nodes, so the gate is open at
# one node and closed at the other and every intermediate point is a cell the
# nodewise surface cannot describe.
_WITNESS_BETA = 0.95
_WITNESS_LANDING = 0.6
_WITNESS_X = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)


@categorical(ordered=True)
class Risk:
    safe: ScalarInt
    risky: ScalarInt


def _witness_utility(*, risk: DiscreteState, x: ContinuousState) -> FloatND:
    """The safe action pays one unit now, whatever the source's own state."""
    return jnp.where(risk == Risk.safe, 1.0, 0.0) + 0.0 * x


def _witness_next_x(risk: DiscreteState) -> ContinuousState:
    """The safe action lands on the closed node, the risky one inside the cell."""
    return jnp.where(risk == Risk.safe, 0.0, _WITNESS_LANDING)


def _witness_target(x: ContinuousState) -> FloatND:
    """`V_target = (1, 3)` on the two nodes."""
    return 1.0 + 2.0 * x


def _witness_reference(x: ContinuousState) -> FloatND:
    """`V_reference = (2, 2.5)` on the two nodes, so the two cross inside."""
    return 2.0 + 0.5 * x


def _witness_fallback(x: ContinuousState) -> FloatND:
    return 0.0 * x


def _witness_gate(*, V_target: FloatND, V_reference: FloatND) -> BoolND:
    return V_target > V_reference


@categorical(ordered=False)
class WitnessRegimeId:
    source: ScalarInt
    target: ScalarInt
    reference: ScalarInt
    fallback: ScalarInt


def _witness_params() -> dict[str, dict[str, dict[str, float]]]:
    return {
        "source": {"koopmans_aggregator": {"discount_factor": _WITNESS_BETA}},
        "target": {},
        "reference": {},
        "fallback": {},
    }


def _witness_model() -> Model:
    return Model(
        regimes={
            "source": Regime(
                transition={
                    "target": ValueDependentTransition(
                        probability=MarkovTransition(_certain_target),
                        gate=_witness_gate,
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback",
                                    projection={"x": _identity_x},
                                )
                            )
                        },
                        gate_references={
                            "V_reference": ProjectedRegimeValue(
                                regime="reference", projection={"x": _identity_x}
                            )
                        },
                    )
                },
                active=lambda age: age < 1,
                states={"x": _WITNESS_X},
                state_transitions={"x": _witness_next_x},
                actions={"risk": DiscreteGrid(category_class=Risk)},
                functions={"utility": _witness_utility},
            ),
            "target": Regime(
                transition=None,
                active=lambda age: age >= 1,
                states={"x": _WITNESS_X},
                functions={"utility": _witness_target},
            ),
            "reference": Regime(
                transition=None,
                active=lambda age: age >= 1,
                states={"x": _WITNESS_X},
                functions={"utility": _witness_reference},
            ),
            "fallback": Regime(
                transition=None,
                active=lambda age: age >= 1,
                states={"x": _WITNESS_X},
                functions={"utility": _witness_fallback},
            ),
        },
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=WitnessRegimeId,
    )


def test_a_gate_that_flips_inside_a_cell_does_not_reward_the_risky_action() -> None:
    """The source takes the safe action worth `1.0`, not the risky one worth `0`.

    At the landing point `x = 0.6` the operands read `V_target = 2.2` and
    `V_reference = 2.3`, so the gate is closed and the branch pays the fallback
    `0`. The nodewise surface `where(V_target > V_reference, V_target, 0)` is
    `(0, 3)` instead, whose interpolant at `0.6` is `1.8` — neither branch's
    value — and discounting it gives the risky action `0.95 * 1.8 = 1.71`,
    above the safe action's `1.0`. The reversal is a whole unit of value wide,
    so it is the argmax that is asserted.
    """
    model = _witness_model()

    solution = model.solve(params=_witness_params(), log_level="debug")

    # The nodewise surrogate, computed from the published terminal values: it
    # ranks the risky action above the safe one, so the fixture discriminates
    # between the two orders rather than agreeing under both.
    terminal = solution[1]
    nodewise = np.where(
        np.asarray(terminal["target"]) > np.asarray(terminal["reference"]),
        np.asarray(terminal["target"]),
        np.asarray(terminal["fallback"]),
    )
    surrogate = _WITNESS_BETA * np.interp(_WITNESS_LANDING, [0.0, 1.0], nodewise)
    np.testing.assert_almost_equal(surrogate, 1.71, decimal=DECIMAL_PRECISION)

    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["source"]),
        np.full(2, 1.0),
        decimal=DECIMAL_PRECISION,
    )
