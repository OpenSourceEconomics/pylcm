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

from dataclasses import replace
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.certainty_equivalent import LinearExpectation
from _lcm.regime_building.collective import NO_ROLE
from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from _lcm.regime_building.Q_and_F import (
    EDGE_CHANNELS_ARG,
    EDGE_REF_PARAMS_ARG,
    EDGE_REF_V_ARG,
    evaluate_projected_readers,
)
from _lcm.simulation.gated_routing import (
    route_gated_edges,
    substitute_gated_edge_continuations,
)
from _lcm.solution.backward_induction import solve
from _lcm.utils.dispatchers import productmap
from _lcm.utils.functools import get_union_of_args
from _lcm.utils.logging import get_logger
from lcm import (
    DiscreteGrid,
    LinSpacedGrid,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    ValueDependentTransition,
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
        transition={
            "target": ValueDependentTransition(
                probability=MarkovTransition(_prob_one),
                gate=_gate_dissolves_everywhere,
                routes={
                    "own": StakeholderRoute(
                        fallback=ProjectedRegimeValue(
                            regime="fallback",
                            projection={"settlement": _settlement_from_health},
                        )
                    )
                },
            )
        },
        active=lambda age: age < 1,
        states={"health": DiscreteGrid(category_class=Health)},
        state_transitions={"health": _next_health},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _utility_source},
    )
    target = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"health": DiscreteGrid(category_class=Health)},
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
    """The edge's `Wbar` at the target's own nodes, and the value mappings.

    The substituted leaf holds the edge's operand channels, which the source
    reads at the point it lands on and gates there. A target grid node IS such
    a point, so applying the edge's own combiner at the nodes is that same read.
    """
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
    edge = regimes["src"].gated_edges["target"]
    supplied = {
        **base_state_action_spaces["target"].states,
        **flat_params["src"],
        "period": jnp.int32(1),
        "age": jnp.asarray(_AGES.period_to_age(1)),
    }
    engine_values = {
        **supplied,
        EDGE_REF_V_ARG: MappingProxyType(dict(solution[1])),
        EDGE_REF_PARAMS_ARG: flat_params,
    }
    edge_fold = edge.combine_at(period=1)
    combine = edge_fold.combine
    # A projected reference is read AT one landing point, and a projection is
    # written for one household — its state is that household's own scalar.
    # The source's kernel supplies the point because it already runs per cell;
    # reproducing the fold at every target node therefore means mapping each
    # reader over the target's grid, which is what production's own dispatcher
    # does around it.
    mapped_readers = tuple(
        replace(
            reader,
            reader=productmap(
                func=reader.reader,
                variables=reader.state_args,
                batch_sizes=dict.fromkeys(reader.state_args, 0),
            ),
        )
        for reader in edge_fold.projected_readers
    )
    projected = evaluate_projected_readers(
        readers=mapped_readers,
        landing_states={
            arg: supplied[arg]
            for reader in edge_fold.projected_readers
            for arg in reader.state_args
        },
        other_values={
            arg: engine_values[arg]
            for reader in edge_fold.projected_readers
            for arg in reader.other_args
        },
    )
    wbar = combine(
        **{EDGE_CHANNELS_ARG: substituted["target"]},
        **projected,
        **{
            name: supplied[name]
            for name in get_union_of_args([combine])
            if name != EDGE_CHANNELS_ARG and name not in projected
        },
    )
    return wbar, mappings


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
    states, _routed_ids, _routed_roles = route_gated_edges(
        # The source is simulated at period 0, so the gate is decided on
        # the value it would enter at period 1.
        fold_period=1,
        regime=regimes["src"],
        same_period_mappings=mappings,
        next_states=next_states,
        regime_names_to_ids=regime_names_to_ids,
        new_subject_regime_ids=jnp.full(
            (2,), regime_names_to_ids["target"], dtype=jnp.int32
        ),
        subjects_in_regime=jnp.array([True, True]),
        flat_params=flat_params,
        own_stakeholder=jnp.full_like(
            jnp.array([True, True]), NO_ROLE, dtype=jnp.int32
        ),
        new_own_stakeholder=jnp.full_like(
            jnp.array([True, True]), NO_ROLE, dtype=jnp.int32
        ),
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
    _states, routed_ids, _routed_roles = route_gated_edges(
        # The source is simulated at period 0, so the gate is decided on
        # the value it would enter at period 1.
        fold_period=1,
        regime=regimes["src"],
        same_period_mappings=mappings,
        next_states=next_states,
        regime_names_to_ids=regime_names_to_ids,
        new_subject_regime_ids=jnp.full(
            (2,), regime_names_to_ids["target"], dtype=jnp.int32
        ),
        subjects_in_regime=jnp.array([True, True]),
        flat_params=flat_params,
        own_stakeholder=jnp.full_like(
            jnp.array([True, True]), NO_ROLE, dtype=jnp.int32
        ),
        new_own_stakeholder=jnp.full_like(
            jnp.array([True, True]), NO_ROLE, dtype=jnp.int32
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(routed_ids),
        np.full((2,), int(regime_names_to_ids["fallback"])),
    )
