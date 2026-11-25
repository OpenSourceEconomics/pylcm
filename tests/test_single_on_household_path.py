"""A regime declared with a plain utility function is the household of `"self"`.

Its one member `"self"` carries Pareto weight 1.0. Its value arrays carry a
trailing member axis of length one; its simulated frame and its names keep the
short forms: `value`, no `own_member`, `V_target`.
"""

import copy
import functools
from collections.abc import Callable
from typing import Literal, cast

import jax.numpy as jnp
import pytest

from _lcm.regime_building.collective import NO_ROLE, build_role_vocabulary
from _lcm.simulation.gated_routing import _role_code
from _lcm_regime_finder.model_adapter import adapt_snapshot
from _lcm_regime_finder.model_snapshot import read_model
from _lcm_regime_finder.records import JSONValue
from lcm import (
    DiscreteGrid,
    Exit,
    ExitTarget,
    InitialNodes,
    LinSpacedGrid,
    Model,
    PowerMean,
    Regime,
)
from lcm.exceptions import InvalidInitialConditionsError
from lcm.execution import ExecutionConfig
from lcm.exits import Unilateral
from lcm.result import SimulationResult
from lcm.solver_api import SolutionResult
from lcm.solvers import EGM
from lcm.typing import FloatND, IntND, UserParamsNode
from lcm_examples import (
    collective_household,
    epstein_zin,
    precautionary_savings_health,
    stochastic_volatility,
)
from tests import test_transition_roles as single_to_household
from tests.regime_building import test_additive_shock_transition as additive_shock
from tests.regime_building import test_gate_shared_fallback as household_to_single
from tests.regime_building import test_gated_edge_params_are_per_edge as single_gate
from tests.solution import test_decoupled as decoupled
from tests.solution import test_egm_solver as egm_closed_form
from tests.solution.test_mutual_consent_settlement import _Singleton
from tests.test_models import (
    n_nbegm_toy,
    nbegm_floor_toy,
    nbegm_ride_along_toy,
    negm_kinked_toy,
    taste_shocks_toy,
)
from tests.test_models.deterministic import base as dcegm_base
from tests.test_models.deterministic import dcegm_variants
from tests.test_models.nbegm_common import RegimeId as NBEGMRegimeId

type InitialConditions = dict[str, FloatND | IntND]
type Case = tuple[Model, dict[str, UserParamsNode], InitialConditions]

_N_SUBJECTS = 8
_SEED = 7


def _precautionary_savings_health() -> Case:
    model = precautionary_savings_health.get_model(retirement_age=20)
    initial_conditions: InitialConditions = {
        "age": jnp.full(_N_SUBJECTS, 18.0),
        "wealth": jnp.linspace(5.0, 40.0, _N_SUBJECTS),
        "health": jnp.linspace(0.2, 0.8, _N_SUBJECTS),
        "regime_id": jnp.full(
            _N_SUBJECTS, precautionary_savings_health.RegimeId.working_life
        ),
    }
    return (
        model,
        precautionary_savings_health.get_params(retirement_age=20),
        initial_conditions,
    )


def _stochastic_volatility() -> Case:
    # `get_model` is cached; the wrapped builder makes a model under the switch.
    model = stochastic_volatility.get_model.__wrapped__(
        n_periods=3, wealth_n_points=8, consumption_n_points=8, income_n_points=3
    )
    low_and_high = jnp.arange(_N_SUBJECTS) % 2
    initial_conditions: InitialConditions = {
        "age": jnp.full(_N_SUBJECTS, 20.0),
        "wealth": jnp.linspace(2.0, 30.0, _N_SUBJECTS),
        "uncertainty": low_and_high,
        "lagged_uncertainty": low_and_high,
        "regime_id": jnp.full(_N_SUBJECTS, stochastic_volatility.RegimeId.alive),
    }
    return model, stochastic_volatility.get_params(), initial_conditions


def _taste_shocks_toy() -> Case:
    initial_conditions: InitialConditions = {
        "age": jnp.full(_N_SUBJECTS, 40.0),
        "wealth": jnp.linspace(1.0, 10.0, _N_SUBJECTS),
        "regime_id": jnp.full(_N_SUBJECTS, taste_shocks_toy.ToyRegimeId.alive),
    }
    return (
        taste_shocks_toy.get_model(),
        taste_shocks_toy.get_params(scale=0.2),
        initial_conditions,
    )


def _additive_shock() -> Case:
    # The continuation is read through the value averaged over the shock.
    initial_conditions: InitialConditions = {
        "age": jnp.zeros(_N_SUBJECTS),
        "wealth": jnp.linspace(1.0, 10.0, _N_SUBJECTS),
        "health": jnp.arange(_N_SUBJECTS) % 2,
        "regime_id": jnp.zeros(_N_SUBJECTS, dtype=jnp.int32),
    }
    return (
        additive_shock._model(additive=True),
        cast("dict[str, UserParamsNode]", additive_shock._params()),
        initial_conditions,
    )


def _decoupled(*, taste_shocks: bool) -> Case:
    # A partially naive agent: it chooses with one aggregator, stores another,
    # and its acting self chooses with a third.
    initial_conditions: InitialConditions = {
        "wealth": jnp.arange(_N_SUBJECTS) % 4,
        "regime_id": jnp.full(_N_SUBJECTS, decoupled.RegimeId.decide),
        "period": jnp.zeros(_N_SUBJECTS, dtype=jnp.int32),
    }
    if taste_shocks:
        return (
            decoupled._ev1_model(),
            cast("dict[str, UserParamsNode]", decoupled._ev1_params(beta_tilde=0.7)),
            initial_conditions,
        )
    return (
        decoupled._model(),
        cast("dict[str, UserParamsNode]", decoupled._params(beta_tilde=0.7)),
        initial_conditions,
    )


def _aggregator_reads_state() -> Case:
    # The aggregator alone reads the state `mood`.
    return (
        decoupled._mood_model(koopmans_aggregator=decoupled.mood_dependent_choice),
        {"decide": {"koopmans_aggregator": {"delta": decoupled.DELTA}}},
        {
            "wealth": jnp.arange(_N_SUBJECTS) % 4,
            "mood": jnp.arange(_N_SUBJECTS) % 2,
            "regime_id": jnp.full(_N_SUBJECTS, decoupled.RegimeId.decide),
            "period": jnp.zeros(_N_SUBJECTS, dtype=jnp.int32),
        },
    )


def _epstein_zin() -> Case:
    # A nonlinear certainty equivalent over a stochastic regime and health draw.
    initial_conditions: InitialConditions = {
        "age": jnp.full(_N_SUBJECTS, 25.0),
        "wealth": jnp.linspace(1.0, 10.0, _N_SUBJECTS),
        "health": jnp.arange(_N_SUBJECTS) % 2,
        "regime_id": jnp.full(_N_SUBJECTS, epstein_zin.EZRegimeId.alive),
    }
    return (
        epstein_zin.get_model(certainty_equivalent=PowerMean()),
        epstein_zin.get_params(risk_aversion=4.0),
        initial_conditions,
    )


def _egm() -> Case:
    model = egm_closed_form._model(
        solver=EGM(savings_grid=egm_closed_form._SAVINGS_GRID)
    )
    initial_conditions: InitialConditions = {
        "age": jnp.zeros(_N_SUBJECTS),
        "wealth": jnp.linspace(3.0, 50.0, _N_SUBJECTS),
        "regime_id": jnp.full(_N_SUBJECTS, egm_closed_form.RegimeId.saving),
    }
    return model, egm_closed_form._params(), initial_conditions


def _dcegm() -> Case:
    # `get_full_model` is cached; the wrapped builder makes a model under the switch.
    model = dcegm_variants.get_full_model.__wrapped__(solver="dcegm", n_periods=3)
    initial_conditions: InitialConditions = {
        "age": jnp.full(_N_SUBJECTS, 40.0),
        "wealth": jnp.linspace(10.0, 300.0, _N_SUBJECTS),
        "regime_id": jnp.full(_N_SUBJECTS, dcegm_base.RegimeId.working_life),
    }
    return model, dcegm_variants.get_full_params(n_periods=3), initial_conditions


def _negm() -> Case:
    initial_conditions: InitialConditions = {
        "age": jnp.full(_N_SUBJECTS, 20.0),
        "wealth": jnp.linspace(1.0, 25.0, _N_SUBJECTS),
        "illiquid": jnp.linspace(0.0, 20.0, _N_SUBJECTS),
        "regime_id": jnp.full(_N_SUBJECTS, negm_kinked_toy.RegimeId.alive),
    }
    return (
        negm_kinked_toy.build_model(),
        {"discount_factor": 0.95, "alive": {}},
        initial_conditions,
    )


def _nbegm() -> Case:
    model = nbegm_floor_toy.build_model(
        variant="nbegm", n_periods=3, n_liquid=20, n_savings=20
    )
    initial_conditions: InitialConditions = {
        "age": jnp.zeros(_N_SUBJECTS),
        "liquid": jnp.linspace(0.5, 25.0, _N_SUBJECTS),
        "regime_id": jnp.full(_N_SUBJECTS, NBEGMRegimeId.alive),
    }
    return model, nbegm_floor_toy.build_params(), initial_conditions


def _nbegm_ride_along() -> Case:
    # The schedule route with a ride-along state, whose periods share one program.
    initial_conditions: InitialConditions = {
        "age": jnp.zeros(_N_SUBJECTS),
        "liquid": jnp.linspace(0.5, 25.0, _N_SUBJECTS),
        "kind": jnp.arange(_N_SUBJECTS) % 2,
        "regime_id": jnp.full(_N_SUBJECTS, NBEGMRegimeId.alive),
    }
    model = nbegm_ride_along_toy.build_model(
        variant="nbegm", n_liquid=8, n_savings=10, n_consumption=12
    )
    return model, nbegm_ride_along_toy.build_params(), initial_conditions


def _nnbegm() -> Case:
    initial_conditions: InitialConditions = {
        "age": jnp.full(_N_SUBJECTS, 20.0),
        "wealth": jnp.linspace(1.0, 25.0, _N_SUBJECTS),
        "illiquid": jnp.linspace(0.0, 15.0, _N_SUBJECTS),
        "regime_id": jnp.full(_N_SUBJECTS, n_nbegm_toy.RegimeId.alive),
    }
    return (
        n_nbegm_toy.build_model(variant="n_nbegm", n_periods=3),
        {"discount_factor": 0.95},
        initial_conditions,
    )


def _collective_household() -> Case:
    # Half the cohort starts single and half married, the poorest couples
    # included, so the simulation carries both a consented formation and a
    # unilateral dissolution.
    model = collective_household.get_model(
        n_periods=3, wealth_n_points=6, consumption_n_points=6
    )
    n_subjects = 64
    n_single = n_subjects // 2
    is_single = jnp.arange(n_subjects) < n_single
    is_woman = jnp.arange(n_subjects) % 2 == 0
    regime_ids = model.regime_names_to_ids
    member_ids = model.member_names_to_ids
    initial_conditions: InitialConditions = {
        "wealth": jnp.concatenate(
            [jnp.linspace(1.0, 20.0, n_single), jnp.linspace(2.0, 40.0, n_single)]
        ),
        "age": jnp.zeros(n_subjects),
        "regime_id": jnp.where(
            is_single,
            jnp.where(is_woman, regime_ids["single_f"], regime_ids["single_m"]),
            regime_ids["couple"],
        ).astype(jnp.int32),
        "own_member": jnp.where(
            is_single, NO_ROLE, jnp.where(is_woman, member_ids["f"], member_ids["m"])
        ).astype(jnp.int32),
    }
    return model, collective_household.get_params(), initial_conditions


def _gate_single_to_single() -> Case:
    # The gate opens at the upper landing point only.
    model = single_gate._build_model(with_bystander=False)
    params: dict[str, UserParamsNode] = {
        "source": {"koopmans_aggregator": {"discount_factor": 0.5}},
        "target": {},
        "fallback": {},
        "edges": {"source": {"target": {"predicate": {"marriage_bonus": 1.0}}}},
    }
    initial_conditions: InitialConditions = {
        "age": jnp.full(_N_SUBJECTS, 40.0),
        "x": jnp.linspace(0.0, 2.0, _N_SUBJECTS),
        "regime_id": jnp.full(_N_SUBJECTS, single_gate.RegimeId.source),
    }
    return model, params, initial_conditions


def _single_to_household(*, edge_form: single_to_household.EdgeForm) -> Case:
    params: dict[str, UserParamsNode] = dict(
        single_to_household._params(marriage_rate=0.5)
    )
    return (
        single_to_household.build_model(edge_form=edge_form),
        params,
        dict(single_to_household.INITIAL_CONDITIONS),
    )


def _gate_household_to_single() -> Case:
    # Her leg falls back to the survivor's regime; his is priced at another
    # single in the solve and lands in hers in simulation.
    model = household_to_single.build_model(route_form="survivor")
    roles = model.member_names_to_ids
    initial_conditions: InitialConditions = {
        "regime_id": jnp.zeros(4, dtype=jnp.int32),
        "age": jnp.zeros(4),
        "wealth": jnp.asarray([1.0, 2.0, 3.0, 2.0]),
        "own_member": jnp.asarray(
            [roles["f"], roles["f"], roles["m"], roles["m"]], dtype=jnp.int32
        ),
    }
    return model, dict(household_to_single.PARAMS), initial_conditions


def _stay_utility(*, wealth: FloatND, choice: IntND) -> FloatND:
    return choice * (wealth - 0.5)


def _outside_utility(wealth: FloatND) -> FloatND:
    return 1.0 + 0.0 * wealth


def _keep_wealth(wealth: FloatND) -> FloatND:
    return wealth


def _single_exit(*, compares: Literal["each_action", "optimum"]) -> Case:
    """A single leaves for another single where staying pays less than one.

    The stay optimum pays `max(0, wealth - 0.5)`, so the source leaves at wealth
    0 and 1 and stays at wealth 2, under either comparison point.
    """
    wealth = LinSpacedGrid(start=0.0, stop=2.0, n_points=3)
    model = Model(
        regimes={
            "source": Regime(
                states={"wealth": wealth},
                actions={"choice": DiscreteGrid(values=jnp.asarray([0, 1]))},
                functions={"utility": _stay_utility},
                exit=Exit(
                    targets={
                        "self": ExitTarget(
                            regime="target", projection={"wealth": _keep_wealth}
                        )
                    },
                    rule=Unilateral(compares=compares),
                ),
            ),
            "target": Regime(
                states={"wealth": wealth},
                functions={"utility": _outside_utility},
            ),
        },
        n_periods=1,
        regime_id_class=_Singleton,
        initial_nodes=InitialNodes(by_period={0: ("source", "target")}),
        edges={},
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )
    initial_conditions: InitialConditions = {
        "period": jnp.zeros(_N_SUBJECTS, dtype=jnp.int32),
        "regime_id": jnp.zeros(_N_SUBJECTS, dtype=jnp.int32),
        "wealth": jnp.linspace(0.0, 2.0, _N_SUBJECTS),
    }
    return model, {}, initial_conditions


_CASES: dict[str, Callable[[], Case]] = {
    "additive_shock": _additive_shock,
    "aggregator_reads_state": _aggregator_reads_state,
    "collective_household": _collective_household,
    "dcegm": _dcegm,
    "decoupled": functools.partial(_decoupled, taste_shocks=False),
    "decoupled_taste_shocks": functools.partial(_decoupled, taste_shocks=True),
    "egm": _egm,
    "epstein_zin": _epstein_zin,
    "gate_household_to_single": _gate_household_to_single,
    "gate_single_to_household_member_route": functools.partial(
        _single_to_household, edge_form="hand_gate"
    ),
    "gate_single_to_household_roles": functools.partial(
        _single_to_household, edge_form="roles"
    ),
    "gate_single_to_single": _gate_single_to_single,
    "nbegm": _nbegm,
    "nbegm_ride_along": _nbegm_ride_along,
    "negm": _negm,
    "nnbegm": _nnbegm,
    "single_exit_each_action": functools.partial(_single_exit, compares="each_action"),
    "single_exit_at_optimum": functools.partial(_single_exit, compares="optimum"),
    "precautionary_savings_health": _precautionary_savings_health,
    "stochastic_volatility": _stochastic_volatility,
    "taste_shocks_toy": _taste_shocks_toy,
}


type _Run = tuple[Model, SolutionResult, SimulationResult]


def _solve_and_simulate(*, name: str) -> _Run:
    """Build the case, then solve and simulate it."""
    model, params, initial_conditions = _CASES[name]()
    solution = model.solve(params=params, log_level="off")
    simulation = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=solution,
        log_level="off",
        seed=_SEED,
    )
    return model, solution, simulation


class _Runs(dict[str, _Run]):
    """Each case solved and simulated on first use, so one failing case fails alone."""

    def __missing__(self, key: str) -> _Run:
        self[key] = _solve_and_simulate(name=key)
        return self[key]


@pytest.fixture(scope="module")
def runs() -> dict[str, _Run]:
    """Each case solved and simulated once."""
    return _Runs()


@pytest.mark.parametrize("name", sorted(_CASES))
def test_every_value_array_ends_in_the_members_of_its_regime(
    *, name: str, runs: dict[str, _Run]
) -> None:
    """The trailing value axis is `member`, its coordinates the regime's members."""
    model, solution, _ = runs[name]
    members = {regime: user.members for regime, user in model.user_regimes.items()}

    got = {
        (period, regime): (
            solution.value_array(period=period, regime=regime).dims[-1],
            tuple(
                solution.value_array(period=period, regime=regime)
                .coords["member"]
                .values.tolist()
            ),
        )
        for period, values in solution.values.items()
        for regime in values
    }

    assert got == {
        (period, regime): ("member", members[regime]) for period, regime in got
    }


@pytest.mark.parametrize(
    "name",
    ["gate_single_to_single", "single_exit_each_action", "single_exit_at_optimum"],
)
def test_a_model_of_singles_simulates_with_value_and_no_own_member(
    *, name: str, runs: dict[str, _Run]
) -> None:
    """A model of singles publishes the column `value` and no `own_member`."""
    columns = set(runs[name][2].to_dataframe().columns)

    assert ("value" in columns, "own_member" in columns) == (True, False)


def test_the_engine_regime_of_a_single_carries_the_member_self() -> None:
    """The engine regime of a single carries the member axis `("self",)`."""
    model = taste_shocks_toy.get_model()

    assert model._regimes["alive"].member_axis == ("self",)


def test_role_vocabulary_gives_the_member_self_no_role() -> None:
    """The member `"self"` is no role; a household's members are, in order."""
    vocabulary = build_role_vocabulary(
        {"single": ("self",), "couple": ("f", "m"), "widow": ("self",)}
    )

    assert dict(vocabulary) == {"f": 0, "m": 1}


def test_gated_route_of_the_member_self_carries_no_role() -> None:
    """A leg naming the member `"self"` routes a row with the no-role code."""
    assert _role_code(name="self", role_ids={"f": 0, "m": 1}) == NO_ROLE


def _adapted_without_input_digest(snapshot: dict[str, JSONValue]) -> JSONValue:
    """The regime finder's problem for a snapshot, less the snapshot's own digest."""
    problem = adapt_snapshot(snapshot)
    del problem["source_adapter"]["snapshot_sha256"]
    return problem


@pytest.mark.parametrize(
    "name",
    ["gate_household_to_single", "gate_single_to_single", "single_exit_each_action"],
)
def test_regime_finder_reads_a_recorded_member_self_as_an_individual(
    name: str,
) -> None:
    """A regime recorded with the one member `"self"` adapts as an individual."""
    snapshot = read_model(_CASES[name]()[0])
    with_self = copy.deepcopy(snapshot)
    for regime in cast(
        "dict[str, dict[str, dict[str, JSONValue]]]", with_self["user_regimes"]
    ).values():
        if regime["fields"].get("member_axis") is None:
            regime["fields"]["member_axis"] = ["self"]

    adapted = _adapted_without_input_digest(with_self)

    assert adapted == _adapted_without_input_digest(snapshot)


def test_the_gate_component_of_a_single_target_is_v_target() -> None:
    """The gated edge into a single reads the target's value as `V_target`."""
    model = _gate_single_to_single()[0]
    folds = model._regimes["source"].gated_edges["target"].folds_by_period

    assert {fold.channels.component_names for fold in folds.values()} == {("V_target",)}


def _without_own_member(initial_conditions: InitialConditions) -> InitialConditions:
    return {k: v for k, v in initial_conditions.items() if k != "own_member"}


def test_singles_start_without_own_member() -> None:
    """Subjects that all start single need no `own_member`."""
    model, params, initial_conditions = _collective_household()
    singles = {k: v[:32] for k, v in _without_own_member(initial_conditions).items()}

    model.validate_initial_conditions(initial_conditions=singles, params=params)


def test_a_household_start_still_requires_own_member() -> None:
    """Subjects starting in a couple that can dissolve must name their role."""
    model, params, initial_conditions = _collective_household()

    with pytest.raises(InvalidInitialConditionsError, match="need an `own_member`"):
        model.validate_initial_conditions(
            initial_conditions=_without_own_member(initial_conditions), params=params
        )


def test_a_single_start_is_not_checked_against_a_role() -> None:
    """The member `"self"` is no role, so a single's row is not role-checked."""
    model, params, initial_conditions = _collective_household()
    wives = dict(initial_conditions)
    wives["own_member"] = jnp.full_like(
        initial_conditions["own_member"], model.member_names_to_ids["f"]
    )

    model.validate_initial_conditions(initial_conditions=wives, params=params)
