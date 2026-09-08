"""Simulation declares its per-regime work as planner programs."""

import functools
import inspect
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import pytest

from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreProgram,
)
from _lcm.regime_building.max_Q_over_a import get_argmax_and_max_Q_over_a
from _lcm.simulation import programs as simulation_programs
from _lcm.simulation.program_types import SimulationPrograms
from _lcm.simulation.programs import (
    SUBJECT_AXIS,
    SUBJECT_WIDTH_KEYWORD,
    _ArgumentsBoundAtDispatch,
    _StreamedArgmaxQOverA,
    _SubjectTiled,
)
from _lcm.solution.contract import SolverBuildContext
from _lcm.typing import ArgmaxQOverAFunction, QAndFFunction
from benchmarks.asv._simulation_witnesses import WITNESSES
from lcm import AgeGrid, LinSpacedGrid, Model, categorical
from lcm.exceptions import ExecutionPlanningError
from lcm.regime import Regime as UserRegime
from lcm.solvers import ACTION_PRODUCT_AXIS
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    IntND,
    ScalarInt,
)
from tests.conftest import assert_agrees_to_ulp
from tests.test_models import taste_shocks_toy

# A regime whose solve kernel streams its action product, and one whose
# collective kernel keeps the canonical dense reducer and whose routing a host
# loop drives through the gated-edge folds.
_STREAMED = ("multi_regime", "work")
_COLLECTIVE = ("dissolution", "married")


def _programs(*, witness: str, regime: str) -> SimulationPrograms:
    """Return one regime's declared program families."""
    model, _, _ = WITNESSES[witness]()
    return model._regimes[regime].simulation.programs


def _program(*, witness: str, regime: str, family: str) -> CoreProgram:
    """Return the first program of one regime's declared family."""
    family_programs = getattr(_programs(witness=witness, regime=regime), family)
    programs = cast("Mapping[int, CoreProgram]", family_programs)
    return programs[next(iter(programs))]


def _cell_body(*, program: CoreProgram) -> Callable[..., object]:
    """Return the per-subject body one tiled program evaluates."""
    return cast("_SubjectTiled", program.function).func


def test_decision_program_declares_action_product_and_subject_axes() -> None:
    """The decision program streams the action product and tiles subjects."""
    program = _program(witness=_STREAMED[0], regime=_STREAMED[1], family="decision")
    assert program.requirements.axis_names == (ACTION_PRODUCT_AXIS, SUBJECT_AXIS)


def test_decision_program_is_planned() -> None:
    """The engine, not the regime builder, owns the decision program's widths."""
    program = _program(witness=_STREAMED[0], regime=_STREAMED[1], family="decision")
    assert program.disposition is CoreExecutionDisposition.PLANNED


def test_collective_decision_program_declares_the_subject_axis_only() -> None:
    """A collective decision tiles subjects and keeps its action product dense."""
    program = _program(witness=_COLLECTIVE[0], regime=_COLLECTIVE[1], family="decision")
    assert program.requirements.axis_names == (SUBJECT_AXIS,)


def test_taste_shock_decision_program_declares_the_subject_axis_only() -> None:
    """An EV1 decision tiles subjects and keeps its action product dense."""
    model = taste_shocks_toy.get_model()
    programs = model._regimes["alive"].simulation.programs.decision
    assert programs[next(iter(programs))].requirements.axis_names == (SUBJECT_AXIS,)


def test_transition_program_declares_the_subject_axis_only() -> None:
    """A law of motion tiles subjects and reduces no axis."""
    program = _program(witness=_STREAMED[0], regime=_STREAMED[1], family="transition")
    assert program.requirements.axis_names == (SUBJECT_AXIS,)


def test_route_program_declares_the_subject_axis_only() -> None:
    """A regime-transition draw tiles subjects and reduces no axis."""
    program = _program(witness=_STREAMED[0], regime=_STREAMED[1], family="route")
    assert program.requirements.axis_names == (SUBJECT_AXIS,)


def test_gated_route_program_is_host_driven() -> None:
    """A gated regime cedes its routing width to the host loop that drives it."""
    program = _program(witness=_COLLECTIVE[0], regime=_COLLECTIVE[1], family="route")
    assert program.disposition is CoreExecutionDisposition.HOST_DRIVEN


def test_gated_route_program_declares_no_execution_axis() -> None:
    """A host-driven route names no axis, because it cedes every width."""
    program = _program(witness=_COLLECTIVE[0], regime=_COLLECTIVE[1], family="route")
    assert program.requirements.axis_names == ()


def test_terminal_regime_declares_no_route_program() -> None:
    """A regime that draws no successor publishes no routing program."""
    assert _programs(witness=_STREAMED[0], regime="dead").route == {}


def test_decision_program_declares_its_continuation_reads() -> None:
    """A decision names every stored value leaf it reads across the boundary."""
    program = _program(witness=_STREAMED[0], regime=_STREAMED[1], family="decision")
    assert [
        (read.target.kind.value, read.target.period, read.target.regime)
        for read in program.requirements.value_reads
    ] == [("regime_value", 1, "dead"), ("regime_value", 1, "work")]


def test_ungated_route_program_declares_no_value_read() -> None:
    """An ungated draw reads the subject's own states, and no stored value."""
    program = _program(witness=_STREAMED[0], regime=_STREAMED[1], family="route")
    assert program.requirements.value_reads == ()


def test_gated_route_program_declares_its_gate_references() -> None:
    """A gated draw names the continuation and the projections its gate reads."""
    program = _program(witness=_COLLECTIVE[0], regime=_COLLECTIVE[1], family="route")
    assert [
        (read.target.kind.value, read.target.regime, read.target.target_regime)
        for read in program.requirements.value_reads
    ] == [
        ("gated_continuation", "married", "married_with_participation"),
        ("regime_value", "single_f", None),
        ("regime_value", "single_m", None),
    ]


def test_a_simulation_program_refuses_to_bind_arguments_at_model_build() -> None:
    """Nothing binds a simulated population's arguments before the call runs."""
    build_arguments = _ArgumentsBoundAtDispatch(program_name="simulate_decision")
    with pytest.raises(ExecutionPlanningError, match="binds its arguments at dispatch"):
        build_arguments(
            CoreBuildContext(
                state_action_space=None,
                next_regime_to_V_arr={},
                next_regime_to_continuation={},
                flat_params={},
                period=0,
                ages=None,
            )
        )


@categorical(ordered=False)
class _BranchRegimeId:
    """Two regimes a subject can move between, and the one it ends in."""

    stay: ScalarInt
    switch: ScalarInt
    done: ScalarInt


def _branch_next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> FloatND:
    """Return next-period wealth after consuming out of it."""
    return wealth - consumption


def _branch_utility(*, consumption: ContinuousAction) -> FloatND:
    """Return the flow utility of one period's consumption."""
    return jnp.log(consumption)


def _branch_affordable(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> BoolND:
    """Return whether the chosen consumption is covered by current wealth."""
    return consumption <= wealth


def _branch_next_regime(*, wealth: ContinuousState) -> ScalarInt:
    """Return the regime a subject moves into, by how wealthy it is."""
    return jnp.where(wealth > 2.0, _BranchRegimeId.switch, _BranchRegimeId.stay)


def _branch_terminal_utility() -> FloatND:
    """Return the flow utility of the absorbing regime."""
    return jnp.asarray(0.0)


def _branching_regime() -> UserRegime:
    """Return one of the two regimes a subject moves between."""
    return UserRegime(
        transition=_branch_next_regime,
        active=lambda age: age < 2,
        states={"wealth": LinSpacedGrid(start=1.0, stop=5.0, n_points=4)},
        state_transitions={"wealth": _branch_next_wealth},
        actions={"consumption": LinSpacedGrid(start=0.5, stop=2.0, n_points=3)},
        constraints={"affordable": _branch_affordable},
        functions={"utility": _branch_utility},
    )


@functools.cache
def _two_target_model() -> Model:
    """Build a model whose first period reaches two state-carrying regimes."""
    return Model(
        regimes={
            "stay": _branching_regime(),
            "switch": _branching_regime(),
            "done": UserRegime(
                transition=None, functions={"utility": _branch_terminal_utility}
            ),
        },
        regime_id_class=_BranchRegimeId,
        ages=AgeGrid(start=0, stop=2, step="Y"),
    )


def _cell_arguments(
    *, model: Model, regime: str, period: int, body: Callable[..., object]
) -> dict[str, object]:
    """Bind one subject's cell of a simulation body, from the regime's grids."""
    grids = dict(model._regimes[regime].simulation.grids)
    arguments: dict[str, object] = {}
    for name in inspect.signature(body).parameters:
        if name == "period":
            arguments[name] = jnp.int32(period)
        elif name == "age":
            arguments[name] = model.ages.values[period]
        elif name.startswith("key_"):
            arguments[name] = jax.random.key(0)
        elif name in grids:
            arguments[name] = jnp.asarray(grids[name].to_jax())[0]
        else:
            msg = f"the toy body takes an argument the test cannot bind: {name!r}"
            raise AssertionError(msg)
    return arguments


def test_transition_program_declares_the_tree_its_body_returns() -> None:
    """A law of motion declares one role per next state, under its target regime."""
    model = _two_target_model()
    program = model._regimes["stay"].simulation.programs.transition[0]
    body = _cell_body(program=program)
    got = body(**_cell_arguments(model=model, regime="stay", period=0, body=body))
    assert jax.tree.structure(got) == jax.tree.structure(program.output_roles)


def test_a_transition_program_covers_every_target_reachable_that_period() -> None:
    """The declared tree spans every regime the period's law of motion writes."""
    model = _two_target_model()
    program = model._regimes["stay"].simulation.programs.transition[0]
    roles = cast("Mapping[str, str]", program.output_roles)
    assert sorted(roles) == ["stay", "switch"]


def test_route_program_declares_the_tree_its_body_returns() -> None:
    """A regime-transition draw declares one role per regime it can draw."""
    model = _two_target_model()
    program = model._regimes["stay"].simulation.programs.route[0]
    body = _cell_body(program=program)
    got = body(**_cell_arguments(model=model, regime="stay", period=0, body=body))
    assert jax.tree.structure(got) == jax.tree.structure(program.output_roles)


def test_a_route_program_declares_every_regime_a_branching_draw_reaches() -> None:
    """The declared tree spans every regime the branching model can draw."""
    model = _two_target_model()
    program = model._regimes["stay"].simulation.programs.route[0]
    roles = cast("Mapping[str, str]", program.output_roles)
    assert list(roles) == ["stay", "switch", "done"]


def test_a_route_program_covers_every_regime_the_draw_can_reach() -> None:
    """The declared tree spans more than one target regime on a branching model."""
    program = _program(witness=_STREAMED[0], regime=_STREAMED[1], family="route")
    roles = cast("Mapping[str, str]", program.output_roles)
    assert list(roles) == ["work", "retire", "dead"]


def test_decision_program_declares_one_role_per_published_array() -> None:
    """A decision publishes the chosen action's identity and the value it attains."""
    program = _program(witness=_STREAMED[0], regime=_STREAMED[1], family="decision")
    assert program.output_roles == ("action_index", "decision_value")


@pytest.mark.parametrize("route", ["ev1", "collective"])
def test_a_streamed_decision_is_refused_where_the_hard_max_is_not_the_reduction(
    *, route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A public model build catches a classifier selecting the wrong reduction."""
    original = simulation_programs._supports_action_streaming
    regime_name = "alive" if route == "ev1" else "married"

    def force_wrong_reduction(*, context: SolverBuildContext) -> bool:
        return context.regime_name == regime_name or original(context=context)

    monkeypatch.setattr(
        simulation_programs, "_supports_action_streaming", force_wrong_reduction
    )
    build_model = (
        taste_shocks_toy.get_model if route == "ev1" else WITNESSES["dissolution"]
    )
    with pytest.raises(ExecutionPlanningError, match=repr(regime_name)):
        build_model()


def test_a_streamed_decision_is_admitted_for_a_plain_hard_max_regime() -> None:
    """The guard admits the streamed decision of a public singleton model."""
    program = _program(witness=_STREAMED[0], regime=_STREAMED[1], family="decision")
    assert program.requirements.axis_names == (ACTION_PRODUCT_AXIS, SUBJECT_AXIS)


def _tiled_cell(*, position: FloatND, offset: FloatND) -> dict[str, FloatND | IntND]:
    """Return one subject's shifted position and the bucket it falls in."""
    return {
        "shifted": position * 2.0 + offset,
        "bucket": jnp.int32(jnp.floor(position)),
    }


@pytest.mark.parametrize(
    ("extent", "width"), [(8, 1), (8, 2), (8, 4), (8, 8), (7, 2), (7, 3), (7, 7)]
)
def test_subject_tiles_reproduce_the_whole_populations_float_output(
    *, extent: int, width: int
) -> None:
    """Tiling the subject axis at any width reports every subject's own value."""
    positions = jnp.linspace(0.0, 6.0, extent)
    offset = jnp.asarray(0.25)
    tiled = _SubjectTiled(func=_tiled_cell, subject_arg_names=("position",))
    got = cast(
        "Mapping[str, FloatND]",
        tiled(position=positions, offset=offset, **{SUBJECT_WIDTH_KEYWORD: width}),
    )
    expected = jnp.stack(
        [_tiled_cell(position=one, offset=offset)["shifted"] for one in positions]
    )
    assert_agrees_to_ulp(got=got["shifted"], expected=expected, n_ulp=1)


@pytest.mark.parametrize(
    ("extent", "width"), [(8, 1), (8, 2), (8, 4), (8, 8), (7, 2), (7, 3), (7, 7)]
)
def test_subject_tiles_reproduce_the_whole_populations_structural_output(
    *, extent: int, width: int
) -> None:
    """Tiling the subject axis at any width keeps every subject in its own row."""
    positions = jnp.linspace(0.0, 6.0, extent)
    offset = jnp.asarray(0.25)
    tiled = _SubjectTiled(func=_tiled_cell, subject_arg_names=("position",))
    got = cast(
        "Mapping[str, FloatND]",
        tiled(position=positions, offset=offset, **{SUBJECT_WIDTH_KEYWORD: width}),
    )
    expected = jnp.stack(
        [_tiled_cell(position=one, offset=offset)["bucket"] for one in positions]
    )
    assert jnp.array_equal(got["bucket"], expected)


def test_a_body_with_no_per_subject_argument_is_called_once() -> None:
    """A body no subject varies has no axis to tile and takes no leading axis."""
    tiled = _SubjectTiled(func=_tiled_cell, subject_arg_names=())
    got = cast(
        "Mapping[str, FloatND]",
        tiled(
            position=jnp.asarray(3.0),
            offset=jnp.asarray(0.25),
            **{SUBJECT_WIDTH_KEYWORD: 4},
        ),
    )
    assert got["shifted"].shape == ()


def _toy_Q_and_F(
    *,
    next_regime_to_V_arr: Mapping[str, FloatND],
    consumption: FloatND,
    work: FloatND,
    wealth: FloatND,
) -> tuple[FloatND, BoolND]:
    """Return a toy action value and its feasibility at one state cell."""
    value = (
        jnp.asarray(wealth + 2.0 * consumption - consumption**2 + 0.5 * work)
        + next_regime_to_V_arr["target"]
    )
    return value, jnp.asarray(consumption <= wealth)


_TOY_ACTIONS = {
    "work": jnp.array([0.0, 1.0]),
    "consumption": jnp.array([0.1, 0.4, 0.9, 1.3, 1.8, 2.4, 3.1]),
}


def _reducers() -> tuple[ArgmaxQOverAFunction, _StreamedArgmaxQOverA]:
    """Return the dense canonical reducer and its streamed counterpart."""
    toy = cast("QAndFFunction", _toy_Q_and_F)
    dense = get_argmax_and_max_Q_over_a(
        Q_and_F=toy,
        action_names=("work", "consumption"),
        state_names=("wealth",),
        n_discrete_action_axes=1,
    )
    streamed = _StreamedArgmaxQOverA(
        Q_and_F=toy,
        action_names=("work", "consumption"),
        q_and_f_arg_names=frozenset(
            {"next_regime_to_V_arr", "consumption", "work", "wealth"}
        ),
        action_width_keyword="_lcm_action_block_width",
    )
    return dense, streamed


@pytest.mark.parametrize("width", [1, 2, 3, 5, 14])
@pytest.mark.parametrize("wealth", [0.5, 1.0, 2.5])
def test_streamed_decision_chooses_the_dense_reducers_action(
    *, width: int, wealth: float
) -> None:
    """Streaming the action product at any width picks the dense winner."""
    dense, streamed = _reducers()
    continuation = MappingProxyType({"target": jnp.asarray(0.25)})
    dense_index, _ = dense(
        next_regime_to_V_arr=continuation, wealth=jnp.asarray(wealth), **_TOY_ACTIONS
    )
    streamed_index, _ = streamed(
        next_regime_to_V_arr=continuation,
        wealth=jnp.asarray(wealth),
        _lcm_action_block_width=width,
        **_TOY_ACTIONS,
    )
    assert int(streamed_index) == int(dense_index)


@pytest.mark.parametrize("width", [1, 2, 3, 5, 14])
@pytest.mark.parametrize("wealth", [0.5, 1.0, 2.5])
def test_streamed_decision_reports_the_dense_reducers_value(
    *, width: int, wealth: float
) -> None:
    """Streaming the action product at any width reports the dense maximum."""
    dense, streamed = _reducers()
    continuation = MappingProxyType({"target": jnp.asarray(0.25)})
    _, dense_value = dense(
        next_regime_to_V_arr=continuation, wealth=jnp.asarray(wealth), **_TOY_ACTIONS
    )
    _, streamed_value = streamed(
        next_regime_to_V_arr=continuation,
        wealth=jnp.asarray(wealth),
        _lcm_action_block_width=width,
        **_TOY_ACTIONS,
    )
    assert_agrees_to_ulp(got=streamed_value, expected=dense_value, n_ulp=1)


@pytest.mark.parametrize("width", [1, 3, 14])
def test_streamed_decision_publishes_identity_zero_when_nothing_is_feasible(
    *, width: int
) -> None:
    """An all-infeasible cell publishes the identity the dense reducer publishes."""
    dense, streamed = _reducers()
    continuation = MappingProxyType({"target": jnp.asarray(0.25)})
    infeasible = jnp.asarray(-1.0)
    dense_index, _ = dense(
        next_regime_to_V_arr=continuation, wealth=infeasible, **_TOY_ACTIONS
    )
    streamed_index, _ = streamed(
        next_regime_to_V_arr=continuation,
        wealth=infeasible,
        _lcm_action_block_width=width,
        **_TOY_ACTIONS,
    )
    assert int(streamed_index) == int(dense_index)


@pytest.mark.parametrize("wealth", [0.5, 1.0, 2.5])
def test_the_toy_decision_does_not_trivially_pick_the_first_action(
    *, wealth: float
) -> None:
    """Every level the oracle differential runs at has a non-trivial winner."""
    dense, _ = _reducers()
    dense_index, _ = dense(
        next_regime_to_V_arr=MappingProxyType({"target": jnp.asarray(0.25)}),
        wealth=jnp.asarray(wealth),
        **_TOY_ACTIONS,
    )
    assert int(dense_index) != 0
