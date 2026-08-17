"""Per-subject evaluation of a gated edge's leg fallback projector.

A leg's fallback projection is an ordinary model function: the solve-side fold
evaluates it at ONE target grid cell's scalar state coordinates, and forward
simulation must evaluate the same function at ONE subject's scalar realized
coordinates. The two then agree by construction, which is the whole point of
building both from the same projection declaration — a household enters its
fallback regime at the state the solved policy priced.

The projection here is deliberately NOT elementwise: it indexes a settlement
table with the target's `health` state and totals that row's components. Such a
function returns one number per household when it is given one household's
scalar state, and one number for the whole population when it is handed the
population's whole state column instead. An elementwise projection cannot tell
those two evaluations apart, so only a non-elementwise one pins the contract.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.certainty_equivalent import LinearExpectation
from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from _lcm.simulation.gated_routing import (
    route_gated_edges,
    substitute_gated_edge_continuations,
)
from _lcm.solution.backward_induction import solve
from _lcm.utils.logging import get_logger
from lcm import (
    DiscreteGrid,
    EdgeLeg,
    GatedEdge,
    LinSpacedGrid,
    Regime,
    SamePeriodRef,
    categorical,
)
from lcm.ages import AgeGrid
from lcm.koopmans_aggregation import LinearAggregator
from lcm.transition import MarkovTransition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from tests.conftest import DECIMAL_PRECISION, build_prepared_structure

_BETA = 0.95

# Two periods: the source is active at age 0, its target and fallback from age 1.
_AGES = AgeGrid(start=0, stop=2, step="Y")

# Transferable settlement components (housing, pension), one row per health state.
_SETTLEMENT_COMPONENTS = ((1.0, 2.0), (3.0, 4.0))

# What each health state's row totals to, i.e. the settlement the projection
# owes each household: 3.0 for the frail one, 7.0 for the robust one.
_SETTLEMENT_BY_HEALTH = (3.0, 7.0)

# The settlement carried in the fallback regime's slot before the edge is
# routed, so a row the router never writes is visible as such.
_UNWRITTEN_SETTLEMENT = -1.0


@categorical(ordered=True)
class Work:
    """The source regime's single binary action."""

    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


@categorical(ordered=False)
class Health:
    """The health status the settlement table is indexed by."""

    frail: ScalarInt  # code 0
    robust: ScalarInt  # code 1


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _next_health(health: DiscreteState) -> DiscreteState:
    """Health carries over into the target regime unchanged."""
    return health


def _utility_source(work: DiscreteAction) -> FloatND:
    """Flow payoff of the source period, flat in the action."""
    return work * 0.0


def _utility_target(health: DiscreteState) -> FloatND:
    """Target payoff, strictly positive so the gate below is closed everywhere."""
    return health * 1.0 + 1.0


def _utility_fallback(settlement: ContinuousState) -> FloatND:
    """`V_fallback(settlement) = settlement`.

    Linear, hence interpolated exactly on the fallback's grid, so the fold's
    `Wbar` under a closed gate IS the coordinate the fold projected — read off
    the solved arrays rather than recomputed by the test.
    """
    return settlement


def _gate_dissolves_everywhere(V_target: FloatND) -> BoolND:
    """A gate that is closed on the whole target grid.

    `V_target` is the target's own value, which `_utility_target` keeps at 1.0
    and 2.0, so this predicate is false at every cell and every household takes
    the leg's fallback.
    """
    return V_target < 0.0


def _settlement_from_health(health: DiscreteState) -> FloatND:
    """Wealth the leaving partner keeps: her health row's components, totalled.

    Written for one household, like every model function — `health` is that
    household's own scalar state and `_SETTLEMENT_COMPONENTS[health]` its own
    row of transferable components, so the total is that household's own
    settlement rather than the population's.
    """
    components = jnp.asarray(_SETTLEMENT_COMPONENTS)
    return jnp.sum(components[health])


def _make_regimes() -> dict[str, Regime]:
    """Source with a dissolution edge, its target, and the leg's fallback."""
    source = Regime(
        transition={"target": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"health": DiscreteGrid(Health)},
        state_transitions={"health": _next_health},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility_source},
        gated_edges={
            "target": GatedEdge(
                gate=_gate_dissolves_everywhere,
                legs={
                    "own": EdgeLeg(
                        fallback=SamePeriodRef(
                            regime="fallback",
                            projection={"settlement": _settlement_from_health},
                        )
                    )
                },
            )
        },
    )
    target = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"health": DiscreteGrid(Health)},
        functions={"utility": _utility_target},
    )
    fallback = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"settlement": LinSpacedGrid(start=0.0, stop=10.0, n_points=11)},
        functions={"utility": _utility_fallback},
    )
    return {"src": source, "target": target, "fallback": fallback}


def _flat_params() -> MappingProxyType:
    return MappingProxyType(
        {
            "src": MappingProxyType(
                {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
            ),
            "target": MappingProxyType({}),
            "fallback": MappingProxyType({}),
        }
    )


def _solve_fixture():
    """Process and solve the three regimes, kernel-level."""
    regimes_dict = _make_regimes()
    regime_names_to_ids = MappingProxyType(
        {name: jnp.int32(index) for index, name in enumerate(regimes_dict)}
    )
    finalized = finalize_regimes(
        user_regimes=regimes_dict,
        derived_categoricals={},
        koopmans_aggregator=LinearAggregator(),
        certainty_equivalent=LinearExpectation(),
    )
    regimes = process_regimes(
        prepared_structure=build_prepared_structure(user_regimes=finalized, ages=_AGES),
        user_regimes=finalized,
        ages=_AGES,
        regime_names_to_ids=regime_names_to_ids,
        enable_jit=False,
    )
    flat_params = _flat_params()
    solution = solve(
        flat_params=flat_params,
        ages=_AGES,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    ).value_functions
    return regimes, regime_names_to_ids, flat_params, solution


def _fold_output(*, regimes, flat_params, solution):
    """The edge's folded `Wbar` on the target grid, and the value mappings."""
    base_state_action_spaces = {
        name: regime.solution.state_action_space(regime_params=flat_params[name])
        for name, regime in regimes.items()
    }
    substituted, mappings = substitute_gated_edge_continuations(
        regime=regimes["src"],
        regime_name="src",
        regimes=regimes,
        period=0,
        next_regime_to_V_arr=solution[1],
        base_state_action_spaces=base_state_action_spaces,
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=MappingProxyType({}),
        flat_params=flat_params,
    )
    return substituted["target"], mappings


def test_solve_side_fold_projects_one_settlement_per_health_state():
    """The fold projects each target grid cell's own settlement.

    With the gate closed on the whole target grid and
    `V_fallback(settlement) = settlement`, the folded `Wbar` at a cell is the
    settlement that cell's health state projects to: 3.0 at `frail`, 7.0 at
    `robust`. A projection collapsed across the grid would report one number
    at both cells, so this also establishes that the fixture discriminates.
    """
    regimes, _ids, flat_params, solution = _solve_fixture()
    wbar, _mappings = _fold_output(
        regimes=regimes, flat_params=flat_params, solution=solution
    )
    aaae(
        np.asarray(wbar),
        np.asarray(_SETTLEMENT_BY_HEALTH),
        decimal=DECIMAL_PRECISION,
    )


def test_router_writes_each_subject_its_own_projected_fallback_state():
    """Each dissolving household enters the fallback at its own settlement.

    Two households whose realized target health differs project to different
    settlements, and each must be written into the fallback regime's state slot
    with its own — the same coordinate the solve-side fold priced that
    household's dissolution at.
    """
    regimes, regime_names_to_ids, flat_params, solution = _solve_fixture()
    wbar, mappings = _fold_output(
        regimes=regimes, flat_params=flat_params, solution=solution
    )
    next_states = MappingProxyType(
        {
            "target": MappingProxyType(
                {"health": jnp.array([Health.frail, Health.robust], dtype=jnp.int32)}
            ),
            "fallback": MappingProxyType(
                {"settlement": jnp.full((2,), _UNWRITTEN_SETTLEMENT)}
            ),
        }
    )
    states, _routed_ids = route_gated_edges(
        regime=regimes["src"],
        same_period_mappings=mappings,
        next_states=next_states,
        regime_names_to_ids=regime_names_to_ids,
        new_subject_regime_ids=jnp.full(
            (2,), regime_names_to_ids["target"], dtype=jnp.int32
        ),
        subjects_in_regime=jnp.array([True, True]),
        flat_params=flat_params,
    )
    aaae(
        np.asarray(states["fallback"]["settlement"]),
        np.asarray(wbar),
        decimal=DECIMAL_PRECISION,
    )


def test_router_sends_every_dissolving_household_to_the_fallback_regime():
    """A closed gate routes both households out of the target regime.

    The settlement equality above is only a statement about the coordinate a
    dissolving household enters with, so the routing decision it presupposes is
    asserted here on its own, exactly.
    """
    regimes, regime_names_to_ids, flat_params, solution = _solve_fixture()
    _wbar, mappings = _fold_output(
        regimes=regimes, flat_params=flat_params, solution=solution
    )
    next_states = MappingProxyType(
        {
            "target": MappingProxyType(
                {"health": jnp.array([Health.frail, Health.robust], dtype=jnp.int32)}
            ),
            "fallback": MappingProxyType(
                {"settlement": jnp.full((2,), _UNWRITTEN_SETTLEMENT)}
            ),
        }
    )
    _states, routed_ids = route_gated_edges(
        regime=regimes["src"],
        same_period_mappings=mappings,
        next_states=next_states,
        regime_names_to_ids=regime_names_to_ids,
        new_subject_regime_ids=jnp.full(
            (2,), regime_names_to_ids["target"], dtype=jnp.int32
        ),
        subjects_in_regime=jnp.array([True, True]),
        flat_params=flat_params,
    )
    np.testing.assert_array_equal(
        np.asarray(routed_ids),
        np.full((2,), int(regime_names_to_ids["fallback"])),
    )
