"""A regime declared with a plain utility function is the household of `"self"`.

Its one member `"self"` carries Pareto weight 1.0. Its value arrays carry a
trailing member axis of length one; its simulated frame and its names keep the
short forms: `value`, no `own_member`, `V_target`.
"""

import copy
from collections.abc import Callable
from typing import cast

import jax.numpy as jnp
import pytest

from _lcm.regime_building.collective import NO_ROLE, build_role_vocabulary
from _lcm.simulation.gated_routing import _role_code
from _lcm_regime_finder.model_adapter import adapt_snapshot
from _lcm_regime_finder.model_snapshot import read_model
from _lcm_regime_finder.records import JSONValue
from lcm import (
    Model,
)
from lcm.exceptions import InvalidInitialConditionsError
from lcm.result import SimulationResult
from lcm.solver_api import SolutionResult
from lcm.typing import FloatND, IntND, UserParamsNode
from lcm_examples import (
    collective_household,
)
from tests.regime_building import test_gate_shared_fallback as household_to_single
from tests.regime_building import test_gated_edge_params_are_per_edge as single_gate
from tests.test_models import (
    taste_shocks_toy,
)

type InitialConditions = dict[str, FloatND | IntND]
type Case = tuple[Model, dict[str, UserParamsNode], InitialConditions]

_N_AGENTS = 8
_SEED = 7


def _collective_household() -> Case:
    # Half the cohort starts single and half married, the poorest couples
    # included, so the simulation carries both a consented formation and a
    # unilateral dissolution.
    model = collective_household.get_model(
        n_periods=3, wealth_n_points=6, consumption_n_points=6
    )
    n_agents = 64
    n_single = n_agents // 2
    is_single = jnp.arange(n_agents) < n_single
    is_woman = jnp.arange(n_agents) % 2 == 0
    regime_ids = model.regime_names_to_ids
    member_ids = model.member_names_to_ids
    initial_conditions: InitialConditions = {
        "wealth": jnp.concatenate(
            [jnp.linspace(1.0, 20.0, n_single), jnp.linspace(2.0, 40.0, n_single)]
        ),
        "age": jnp.zeros(n_agents),
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
        "age": jnp.full(_N_AGENTS, 40.0),
        "x": jnp.linspace(0.0, 2.0, _N_AGENTS),
        "regime_id": jnp.full(_N_AGENTS, single_gate.RegimeId.source),
    }
    return model, params, initial_conditions


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


_CASES: dict[str, Callable[[], Case]] = {
    "collective_household": _collective_household,
    "gate_household_to_single": _gate_household_to_single,
    "gate_single_to_single": _gate_single_to_single,
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


@pytest.mark.parametrize("name", ["collective_household"])
def test_every_value_array_ends_in_the_members_of_its_regime(
    *, name: str, runs: dict[str, _Run]
) -> None:
    """The trailing value axis is `member`, its coordinates the regime's members."""
    model, solution, _ = runs[name]
    members = {regime: user.members for regime, user in model.user_regimes.items()}

    got = {
        (period, regime): (
            solution.value_array(step=period, regime=regime).dims[-1],
            tuple(
                solution.value_array(step=period, regime=regime)
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
    ["gate_single_to_single"],
)
def test_a_model_of_singles_simulates_with_value_and_no_own_member(
    *, name: str, runs: dict[str, _Run]
) -> None:
    """A model of singles publishes the column `value` and no `own_member`."""
    columns = set(runs[name][2].to_dataframe().columns)

    assert ("value" in columns, "own_member" in columns) == (True, False)


def test_the_engine_regime_of_a_single_carries_the_member_self() -> None:
    """The engine regime of a single carries the members `("self",)`."""
    model = taste_shocks_toy.get_model()

    assert model._regimes["alive"].members == ("self",)


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
    ["gate_household_to_single"],
)
def test_regime_finder_reads_a_recorded_member_self_as_an_individual(
    name: str,
) -> None:
    """A regime recorded with the one member `"self"` adapts as an individual."""
    snapshot = read_model(_CASES[name]()[0])
    without_self = copy.deepcopy(snapshot)
    singles = [
        regime["fields"]
        for regime in cast(
            "dict[str, dict[str, dict[str, JSONValue]]]",
            without_self["user_regimes"],
        ).values()
        if regime["fields"]["members"] == ["self"]
    ]
    for fields in singles:
        del fields["members"]

    adapted = _adapted_without_input_digest(without_self)

    assert singles
    assert adapted == _adapted_without_input_digest(snapshot)


def test_the_gate_component_of_a_single_target_is_v_target() -> None:
    """The gated edge into a single reads the target's value as `V_target`."""
    model = _gate_single_to_single()[0]
    folds = model._regimes["source"].gated_edges["target"].folds_by_period

    assert {fold.channels.component_names for fold in folds.values()} == {("V_target",)}


def _without_own_member(initial_conditions: InitialConditions) -> InitialConditions:
    return {k: v for k, v in initial_conditions.items() if k != "own_member"}


def test_singles_start_without_own_member() -> None:
    """Agents that all start single need no `own_member`."""
    model, params, initial_conditions = _collective_household()
    singles = {k: v[:32] for k, v in _without_own_member(initial_conditions).items()}

    model.validate_initial_conditions(initial_conditions=singles, params=params)


def test_a_household_start_still_requires_own_member() -> None:
    """Agents starting in a couple that can dissolve must name their role."""
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
