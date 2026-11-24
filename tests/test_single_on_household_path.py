"""A single regime built through the household kernels as the household of `"self"`.

The private switch `_SINGLE_ON_HOUSEHOLD_PATH` in
`_lcm.regime_building.processing` builds every regime declared with a plain
utility function as a household whose one member `"self"` carries Pareto weight
1.0. Its value arrays then carry a trailing member axis of length one; its
simulated frame keeps the short names of a single.
"""

import copy
import functools
from collections.abc import Callable
from types import MappingProxyType
from typing import Literal, cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.regime_building import processing
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
    Regime,
)
from lcm.exceptions import InvalidInitialConditionsError
from lcm.execution import ExecutionConfig
from lcm.exits import Unilateral
from lcm.result import SimulationResult
from lcm.solver_api import SolutionResult
from lcm.typing import FloatND, IntND, RegimeName, UserParamsNode
from lcm_examples import (
    collective_household,
    precautionary_savings_health,
    stochastic_volatility,
)
from tests import test_transition_roles as single_to_household
from tests.conftest import assert_agrees_to_ulp
from tests.regime_building import test_gate_shared_fallback as household_to_single
from tests.regime_building import test_gated_edge_params_are_per_edge as single_gate
from tests.solution.test_fingerprint_pin import _MODELS as PINNED_MODELS
from tests.solution.test_fingerprint_pin import _pinned
from tests.solution.test_mutual_consent_settlement import _Singleton
from tests.test_models import taste_shocks_toy

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
                states={"wealth": wealth}, functions={"utility": _outside_utility}
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
    "collective_household": _collective_household,
    "gate_household_to_single": _gate_household_to_single,
    "gate_single_to_household_member_route": functools.partial(
        _single_to_household, edge_form="hand_gate"
    ),
    "gate_single_to_household_roles": functools.partial(
        _single_to_household, edge_form="roles"
    ),
    "gate_single_to_single": _gate_single_to_single,
    "single_exit_each_action": functools.partial(_single_exit, compares="each_action"),
    "single_exit_at_optimum": functools.partial(_single_exit, compares="optimum"),
    "precautionary_savings_health": _precautionary_savings_health,
    "stochastic_volatility": _stochastic_volatility,
    "taste_shocks_toy": _taste_shocks_toy,
}


def _solve_and_simulate(
    *, name: str, on_household_path: bool
) -> tuple[SolutionResult, SimulationResult]:
    """Build the case afresh with the switch set, then solve and simulate it."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(processing, "_SINGLE_ON_HOUSEHOLD_PATH", on_household_path)
        model, params, initial_conditions = _CASES[name]()
        solution = model.solve(params=params, log_level="off")
        simulation = model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=solution,
            log_level="off",
            seed=_SEED,
        )
    return solution, simulation


class _Runs(dict[tuple[str, bool], tuple[SolutionResult, SimulationResult]]):
    """Each case solved and simulated on first use, so one failing case fails alone."""

    def __missing__(
        self, key: tuple[str, bool]
    ) -> tuple[SolutionResult, SimulationResult]:
        name, on_household_path = key
        self[key] = _solve_and_simulate(name=name, on_household_path=on_household_path)
        return self[key]


@pytest.fixture(scope="module")
def runs() -> dict[tuple[str, bool], tuple[SolutionResult, SimulationResult]]:
    """Each case solved and simulated once with the switch off and once on."""
    return _Runs()


def _value_shapes(
    solution: SolutionResult,
) -> MappingProxyType[int, MappingProxyType[str, tuple[int, ...]]]:
    return MappingProxyType(
        {
            period: MappingProxyType(
                {regime: tuple(np.shape(v)) for regime, v in values.items()}
            )
            for period, values in solution.values.items()
        }
    )


def _is_declared_household(
    *, solution: SolutionResult, period: int, regime: RegimeName
) -> bool:
    """Whether the switch-off value array already carries the member axis."""
    return "member" in solution.value_array(period=period, regime=regime).dims


@pytest.mark.parametrize("name", sorted(_CASES))
def test_single_on_household_path_values_carry_a_member_axis_of_length_one(
    *,
    name: str,
    runs: dict[tuple[str, bool], tuple[SolutionResult, SimulationResult]],
) -> None:
    """Every single's value array gains one trailing axis of length one.

    A declared household's array keeps its shape.
    """
    solution = runs[name, False][0]
    single = _value_shapes(solution)
    expected = {
        period: {
            regime: shape
            if _is_declared_household(solution=solution, period=period, regime=regime)
            else (*shape, 1)
            for regime, shape in shapes.items()
        }
        for period, shapes in single.items()
    }

    got = _value_shapes(runs[name, True][0])

    assert got == expected


@pytest.mark.parametrize("name", sorted(_CASES))
def test_single_on_household_path_names_its_member_self(
    *,
    name: str,
    runs: dict[tuple[str, bool], tuple[SolutionResult, SimulationResult]],
) -> None:
    """The trailing value axis is the dimension `member` with coordinate `self`."""
    single = runs[name, False][0]
    solution = runs[name, True][0]
    period, regime = min(
        (period, regime)
        for period, values in single.values.items()
        for regime in values
        if not _is_declared_household(solution=single, period=period, regime=regime)
    )

    array = solution.value_array(period=period, regime=regime)

    assert array.coords["member"].values.tolist() == ["self"]


@pytest.mark.parametrize("name", sorted(_CASES))
def test_single_on_household_path_simulates_with_the_short_value_column(
    *,
    name: str,
    runs: dict[tuple[str, bool], tuple[SolutionResult, SimulationResult]],
) -> None:
    """The frame has the single's columns: `value`, no `value_self`, no `own_member`."""
    single = runs[name, False][1].to_dataframe()

    frame = runs[name, True][1].to_dataframe()

    assert list(frame.columns) == list(single.columns)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_single_on_household_path_simulates_the_same_discrete_choices(
    *,
    name: str,
    runs: dict[tuple[str, bool], tuple[SolutionResult, SimulationResult]],
) -> None:
    """With one seed, the discrete actions, states and regimes are the single's."""
    single = runs[name, False][1].to_dataframe()
    frame = runs[name, True][1].to_dataframe()
    discrete = [
        column
        for column in single.columns
        if isinstance(single[column].dtype, pd.CategoricalDtype)
    ]

    pd.testing.assert_frame_equal(frame[discrete], single[discrete])


def test_collective_household_cohort_forms_and_dissolves_couples(
    runs: dict[tuple[str, bool], tuple[SolutionResult, SimulationResult]],
) -> None:
    """The compared frames contain a consented formation and a dissolution."""
    frame = (
        runs["collective_household", False][1]
        .to_dataframe()
        .sort_values(["subject_id", "period"])
    )
    regime = frame["regime_name"].astype(str)
    previous = regime.groupby(frame["subject_id"]).shift()
    singles = ["single_f", "single_m"]

    assert (previous.isin(singles) & (regime == "couple")).any()
    assert ((previous == "couple") & regime.isin(singles)).any()


@pytest.mark.parametrize(
    ("name", "branches"),
    [
        ("gate_single_to_single", {"target", "fallback"}),
        ("single_exit_each_action", {"source", "target"}),
        ("single_exit_at_optimum", {"source", "target"}),
    ],
)
def test_compared_frames_take_both_branches(
    *,
    name: str,
    branches: set[str],
    runs: dict[tuple[str, bool], tuple[SolutionResult, SimulationResult]],
) -> None:
    """The compared frames hold rows down the open and the closed branch."""
    frame = runs[name, False][1].to_dataframe()

    assert branches <= set(frame["regime_name"].astype(str))


@pytest.mark.parametrize("name", sorted(_CASES))
def test_single_on_household_path_values_agree_with_the_single_to_16_ulp(
    *,
    name: str,
    runs: dict[tuple[str, bool], tuple[SolutionResult, SimulationResult]],
) -> None:
    """Every value is within 16 ULP, at the model's value scale, of the single's."""
    solution = runs[name, False][0]
    single = solution.values
    household = runs[name, True][0].values
    magnitude = max(
        float(np.nanmax(np.abs(np.where(np.isfinite(v), v, np.nan))))
        for values in single.values()
        for v in values.values()
        if np.isfinite(v).any()
    )

    for period, values in single.items():
        for regime, expected in values.items():
            got = np.asarray(household[period][regime])
            if not _is_declared_household(
                solution=solution, period=period, regime=regime
            ):
                got = got[..., 0]
            assert_agrees_to_ulp(
                got=got,
                expected=np.asarray(expected),
                n_ulp=16,
                operand_magnitude=magnitude,
                err_msg=f"period {period}, regime {regime!r}",
            )


def test_switch_off_keeps_the_pinned_fingerprint() -> None:
    """With the switch off, a single-agent model keeps its checked-in fingerprint."""
    fingerprint = PINNED_MODELS["multi_regime_normal"]()._model_structure_fingerprint

    assert fingerprint == _pinned()["multi_regime_normal"]


def test_switch_on_gives_the_engine_regime_the_member_self() -> None:
    """The engine regime of a single carries the member axis `("self",)`."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(processing, "_SINGLE_ON_HOUSEHOLD_PATH", True)
        model = taste_shocks_toy.get_model()

    assert model._regimes["alive"].member_axis == ("self",)


def _build(*, name: str, on_household_path: bool) -> Case:
    """Build the case afresh with the switch set, without solving it."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(processing, "_SINGLE_ON_HOUSEHOLD_PATH", on_household_path)
        return _CASES[name]()


_GATE_CASES = (
    "gate_household_to_single",
    "gate_single_to_household_member_route",
    "gate_single_to_household_roles",
    "gate_single_to_single",
)


def test_role_vocabulary_gives_the_member_self_no_role() -> None:
    """The member `"self"` is no role; a household's members are, in order."""
    vocabulary = build_role_vocabulary(
        {"single": ("self",), "couple": ("f", "m"), "widow": ("self",)}
    )

    assert dict(vocabulary) == {"f": 0, "m": 1}


def test_gated_route_of_the_member_self_carries_no_role() -> None:
    """A leg naming the member `"self"` routes a row with the no-role code."""
    assert _role_code(name="self", role_ids={"f": 0, "m": 1}) == NO_ROLE


@pytest.mark.parametrize("name", sorted(_CASES))
def test_switch_on_keeps_the_params_template(name: str) -> None:
    """A single's parameters keep their short keys, such as `utility__<name>`."""
    single = _build(name=name, on_household_path=False)[0].get_params_template()

    template = _build(name=name, on_household_path=True)[0].get_params_template()

    assert template == single


def _snapshot_records(model: Model) -> dict[str, JSONValue]:
    """The regime finder's records of a model, without the capture's timings."""
    records = read_model(model)
    del records["capture_report"]
    return records


@pytest.mark.parametrize("name", _GATE_CASES)
def test_switch_on_keeps_the_regime_finder_records(name: str) -> None:
    """The regime finder records a model the same with the switch on or off."""
    single = _snapshot_records(_build(name=name, on_household_path=False)[0])

    records = _snapshot_records(_build(name=name, on_household_path=True)[0])

    assert records == single


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
    snapshot = read_model(_build(name=name, on_household_path=False)[0])
    with_self = copy.deepcopy(snapshot)
    for regime in cast(
        "dict[str, dict[str, dict[str, JSONValue]]]", with_self["user_regimes"]
    ).values():
        if regime["fields"].get("member_axis") is None:
            regime["fields"]["member_axis"] = ["self"]

    adapted = _adapted_without_input_digest(with_self)

    assert adapted == _adapted_without_input_digest(snapshot)


@pytest.mark.parametrize(
    "name",
    ["gate_single_to_single", "single_exit_each_action", "single_exit_at_optimum"],
)
def test_switch_on_simulates_a_single_with_value_and_no_own_member(
    *,
    name: str,
    runs: dict[tuple[str, bool], tuple[SolutionResult, SimulationResult]],
) -> None:
    """A model of singles publishes the column `value` and no `own_member`."""
    columns = set(runs[name, True][1].to_dataframe().columns)

    assert ("value" in columns, "own_member" in columns) == (True, False)


def test_switch_on_names_the_gate_component_of_a_single_target_v_target() -> None:
    """The gated edge into a single reads the target's value as `V_target`."""
    model = _build(name="gate_single_to_single", on_household_path=True)[0]
    folds = model._regimes["source"].gated_edges["target"].folds_by_period

    assert {fold.channels.component_names for fold in folds.values()} == {("V_target",)}


def _without_own_member(initial_conditions: InitialConditions) -> InitialConditions:
    return {k: v for k, v in initial_conditions.items() if k != "own_member"}


@pytest.mark.parametrize("on_household_path", [False, True])
def test_singles_start_without_own_member(*, on_household_path: bool) -> None:
    """Subjects that all start single need no `own_member`."""
    model, params, initial_conditions = _build(
        name="collective_household", on_household_path=on_household_path
    )
    singles = {k: v[:32] for k, v in _without_own_member(initial_conditions).items()}

    model.validate_initial_conditions(initial_conditions=singles, params=params)


@pytest.mark.parametrize("on_household_path", [False, True])
def test_a_household_start_still_requires_own_member(
    *, on_household_path: bool
) -> None:
    """Subjects starting in a couple that can dissolve must name their role."""
    model, params, initial_conditions = _build(
        name="collective_household", on_household_path=on_household_path
    )

    with pytest.raises(InvalidInitialConditionsError, match="need an `own_member`"):
        model.validate_initial_conditions(
            initial_conditions=_without_own_member(initial_conditions), params=params
        )


@pytest.mark.parametrize("on_household_path", [False, True])
def test_a_single_start_is_not_checked_against_a_role(
    *, on_household_path: bool
) -> None:
    """The member `"self"` is no role, so a single's row is not role-checked."""
    model, params, initial_conditions = _build(
        name="collective_household", on_household_path=on_household_path
    )
    wives = dict(initial_conditions)
    wives["own_member"] = jnp.full_like(
        initial_conditions["own_member"], model.member_names_to_ids["f"]
    )

    model.validate_initial_conditions(initial_conditions=wives, params=params)
