"""Simulation declares its per-regime work as planner programs."""

import functools
import inspect
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax.core import ShapedArray
from jax.extend.core import Jaxpr
from numpy.testing import assert_array_equal

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
from _lcm.solution.grid_search import _ActionStreamingDisposition
from _lcm.typing import ArgmaxQOverAFunction, PytreeValue, QAndFFunction
from benchmarks.asv._simulation_witnesses import WITNESSES
from lcm import (
    AgeGrid,
    DeterministicTransition,
    LinSpacedGrid,
    Model,
    Transition,
    categorical,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.regime import Regime as UserRegime
from lcm.solvers import ACTION_PRODUCT_AXIS
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    IntND,
    RegimeName,
    ScalarInt,
)
from tests.conftest import assert_agrees_to_ulp
from tests.simulation._profile_comparison import assert_values_agree
from tests.single_member import SELF, SELF_PARETO_WEIGHTS, with_member_axis
from tests.test_models import taste_shocks_toy

# A single whose decision streams its action product, and a household whose
# decision streams too and whose routing a host loop drives through the
# gated-edge folds.
_STREAMED = ("multi_regime", "work")
_COLLECTIVE = ("dissolution", "married")


def _programs(*, witness: str, regime: RegimeName) -> SimulationPrograms:
    """Return one regime's declared program families."""
    model, _, _ = WITNESSES[witness]()
    return model._regimes[regime].simulation.programs


def _program(*, witness: str, regime: RegimeName, family: str) -> CoreProgram:
    """Return the first program of one regime's declared family."""
    family_programs = getattr(_programs(witness=witness, regime=regime), family)
    programs = cast("Mapping[int, CoreProgram]", family_programs)
    return programs[next(iter(programs))]


def _cell_body(*, program: CoreProgram) -> Callable[..., PytreeValue]:
    """Return the per-subject body one tiled program evaluates."""
    return cast("_SubjectTiled", program.function).func


@pytest.mark.parametrize(("witness", "regime"), [_STREAMED, _COLLECTIVE])
def test_decision_program_declares_action_product_and_subject_axes(
    *, witness: str, regime: RegimeName
) -> None:
    """A single's or a household's decision streams the action product and tiles
    subjects."""
    program = _program(witness=witness, regime=regime, family="decision")
    assert program.requirements.axis_names == (ACTION_PRODUCT_AXIS, SUBJECT_AXIS)


def test_decision_program_is_planned() -> None:
    """The engine, not the regime builder, owns the decision program's widths."""
    program = _program(witness=_STREAMED[0], regime=_STREAMED[1], family="decision")
    assert program.disposition is CoreExecutionDisposition.PLANNED


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


def test_gated_route_program_is_planned() -> None:
    """A gated regime leaves its routing width to the engine, like any other."""
    program = _program(witness=_COLLECTIVE[0], regime=_COLLECTIVE[1], family="route")
    assert program.disposition is CoreExecutionDisposition.PLANNED


def test_gated_route_program_declares_the_subject_axis_only() -> None:
    """A gated routing draw tiles subjects and reduces no axis."""
    program = _program(witness=_COLLECTIVE[0], regime=_COLLECTIVE[1], family="route")
    assert program.requirements.axis_names == (SUBJECT_AXIS,)


def test_terminal_regime_declares_no_route_program() -> None:
    """A regime that draws no successor publishes no routing program."""
    assert _programs(witness=_STREAMED[0], regime="dead").route == {}


def test_decision_program_declares_its_continuation_reads() -> None:
    """A decision names every stored value leaf it reads across the boundary.

    At the first period `work` declares only itself as a target, so that is the
    one leaf it reads.
    """
    program = _program(witness=_STREAMED[0], regime=_STREAMED[1], family="decision")
    assert [
        (read.target.kind.value, read.target.period, read.target.regime)
        for read in program.requirements.value_reads
    ] == [("regime_value", 1, "work")]


def test_ungated_route_program_declares_no_value_read() -> None:
    """An ungated draw reads the subject's own states, and no stored value."""
    program = _program(witness=_STREAMED[0], regime=_STREAMED[1], family="route")
    assert program.requirements.value_reads == ()


def test_gate_blind_probability_program_declares_no_value_reads() -> None:
    """The realized gate owns raw reads; the probability body has none."""
    program = _program(witness=_COLLECTIVE[0], regime=_COLLECTIVE[1], family="route")
    assert program.requirements.value_reads == ()


@pytest.mark.parametrize(
    ("witness", "regime", "family"),
    [(*_STREAMED, "decision"), (*_COLLECTIVE, "decision")],
)
def test_value_read_occurrences_name_the_simulation_program(
    *, witness: str, regime: RegimeName, family: str
) -> None:
    """Each read belongs to the actual forward core that consumes its argument."""
    programs = cast(
        "Mapping[int, CoreProgram]",
        getattr(_programs(witness=witness, regime=regime), family),
    )
    reads = [
        (period, program, read)
        for period, program in programs.items()
        for read in program.requirements.value_reads
    ]
    assert reads
    for period, program, read in reads:
        assert read.source.source_period == period
        assert read.source.source_regime == regime
        assert read.source.core_key == program.name


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
            "done": UserRegime(functions={"utility": _branch_terminal_utility}),
        },
        regime_id_class=_BranchRegimeId,
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        initial_nodes={0: "stay"},
        edges={
            source: Transition(
                targets={"stay": 0, "switch": 0, "done": (0, 1)},
                law=DeterministicTransition(func=_branch_next_regime),
            )
            for source in ("stay", "switch")
        },
    )


def _cell_arguments(
    *, model: Model, regime: RegimeName, period: int, body: Callable[..., PytreeValue]
) -> dict[str, jax.Array]:
    """Bind one subject's cell of a simulation body, from the regime's grids."""
    grids = dict(model._regimes[regime].simulation.grids)
    arguments: dict[str, jax.Array] = {}
    for name in inspect.signature(body).parameters:
        if name == "period":
            arguments[name] = jnp.int32(period)
        elif name == "age":
            assert model.ages is not None
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


def test_a_streamed_decision_is_refused_where_the_hard_max_is_not_the_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public model build catches a classifier streaming a taste-shock decision."""
    original = simulation_programs._classify_action_streaming

    def force_wrong_reduction(
        *, context: SolverBuildContext
    ) -> _ActionStreamingDisposition:
        if context.regime_name == "alive":
            return _ActionStreamingDisposition.STREAMED
        return original(context=context)

    monkeypatch.setattr(
        simulation_programs, "_classify_action_streaming", force_wrong_reduction
    )
    with pytest.raises(ExecutionPlanningError, match="'alive'"):
        taste_shocks_toy.get_model()


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
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
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


def _streamed_reducer(*, Q_and_F: Callable[..., PytreeValue]) -> _StreamedArgmaxQOverA:
    """Return a single's streamed decision over the toy action product."""
    return _StreamedArgmaxQOverA(
        Q_and_F=cast(
            "QAndFFunction",
            with_member_axis(cast("Callable[..., tuple[FloatND, BoolND]]", Q_and_F)),
        ),
        action_names=("work", "consumption"),
        q_and_f_arg_names=frozenset(
            {"next_regime_to_V_arr", "consumption", "work", "wealth"}
        ),
        action_width_keyword="_lcm_action_block_width",
        members=SELF,
        pareto_weights=SELF_PARETO_WEIGHTS,
    )


def _reducers() -> tuple[ArgmaxQOverAFunction, _StreamedArgmaxQOverA]:
    """Return the dense canonical reducer and its streamed counterpart."""
    dense = get_argmax_and_max_Q_over_a(
        Q_and_F=cast("QAndFFunction", with_member_axis(_toy_Q_and_F)),
        action_names=("work", "consumption"),
        state_names=("wealth",),
        n_discrete_action_axes=1,
        members=SELF,
        pareto_weights=SELF_PARETO_WEIGHTS,
    )
    return cast("ArgmaxQOverAFunction", dense), _streamed_reducer(Q_and_F=_toy_Q_and_F)


def _work_bonus_Q_and_F(
    *,
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
    consumption: FloatND,
    work: FloatND,
    wealth: FloatND,
) -> tuple[FloatND, BoolND]:
    """Return the toy action value plus a bonus `sin(work)` that reads only `work`."""
    value, feasible = _toy_Q_and_F(
        next_regime_to_V_arr=next_regime_to_V_arr,
        consumption=consumption,
        work=work,
        wealth=wealth,
    )
    return value + jnp.sin(work), feasible


def _tied_Q_and_F(
    *,
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
    consumption: FloatND,
    work: FloatND,
    wealth: FloatND,
) -> tuple[FloatND, BoolND]:
    """Return a value every consumption level ties on, feasible from 0.4 up to wealth.

    The maximum is attained by working at every feasible consumption level, so the
    lowest winning identity is the second consumption level under `work = 1`.
    """
    value = wealth - (work - 1.0) ** 2 + 0.0 * consumption
    value = value + next_regime_to_V_arr["target"]
    return value, jnp.asarray((consumption >= 0.4) & (consumption <= wealth))


def _nan_where_feasible_Q_and_F(
    *,
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
    consumption: FloatND,
    work: FloatND,
    wealth: FloatND,
) -> tuple[FloatND, BoolND]:
    """Return the toy value with NaN at the feasible action `work = 1, c = 1.3`."""
    value, feasible = _toy_Q_and_F(
        next_regime_to_V_arr=next_regime_to_V_arr,
        consumption=consumption,
        work=work,
        wealth=wealth,
    )
    poisoned = (work == 1.0) & (consumption == 1.3)
    return jnp.where(poisoned, jnp.nan, value), feasible


def _nan_where_infeasible_Q_and_F(
    *,
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
    consumption: FloatND,
    work: FloatND,
    wealth: FloatND,
) -> tuple[FloatND, BoolND]:
    """Return the toy value with NaN at every action above wealth, all infeasible."""
    value, feasible = _toy_Q_and_F(
        next_regime_to_V_arr=next_regime_to_V_arr,
        consumption=consumption,
        work=work,
        wealth=wealth,
    )
    return jnp.where(feasible, value, jnp.nan), feasible


def _primitive_output_shapes(*, jaxpr: Jaxpr, primitive: str) -> list[tuple[int, ...]]:
    """Return the output shape of every `primitive` equation, nested jaxprs included."""
    shapes: list[tuple[int, ...]] = []
    for equation in jaxpr.eqns:
        if equation.primitive.name == primitive:
            shapes.extend(
                cast("ShapedArray", var.aval).shape for var in equation.outvars
            )
        for param in equation.params.values():
            for nested in param if isinstance(param, tuple | list) else (param,):
                inner = getattr(nested, "jaxpr", nested)
                if hasattr(inner, "eqns"):
                    shapes.extend(
                        _primitive_output_shapes(jaxpr=inner, primitive=primitive)
                    )
    return shapes


def _decide(
    *,
    streamed: _StreamedArgmaxQOverA,
    wealth: float,
    width: int,
    actions: Mapping[str, jax.Array] = _TOY_ACTIONS,
) -> tuple[IntND, FloatND]:
    """Return the streamed decision's identity and value at one wealth level."""
    return cast(
        "tuple[IntND, FloatND]",
        streamed(
            next_regime_to_V_arr=MappingProxyType({"target": jnp.asarray(0.25)}),
            wealth=jnp.asarray(wealth),
            _lcm_action_block_width=width,
            **actions,
        ),
    )


@pytest.mark.parametrize("width", [14, 20])
def test_full_width_decision_evaluates_a_one_action_term_once_per_its_value(
    *, width: int
) -> None:
    """A block covering the product computes a `work`-only term per `work` value.

    The term is mapped over the two `work` levels, not over the fourteen actions of
    the product.
    """
    streamed = _streamed_reducer(Q_and_F=_work_bonus_Q_and_F)
    jaxpr = jax.make_jaxpr(
        lambda wealth: _decide(streamed=streamed, wealth=wealth, width=width)
    )(2.5)
    assert _primitive_output_shapes(jaxpr=jaxpr.jaxpr, primitive="sin") == [(2,)]


@pytest.mark.parametrize(
    ("Q_and_F", "wealth"),
    [
        (_work_bonus_Q_and_F, 2.5),
        (_tied_Q_and_F, 2.5),
        (_nan_where_feasible_Q_and_F, 2.5),
        (_nan_where_infeasible_Q_and_F, 1.0),
        (_toy_Q_and_F, -1.0),
    ],
    ids=["interior", "tied", "feasible-nan", "infeasible-nan", "nothing-feasible"],
)
@pytest.mark.parametrize("narrow_width", [1, 3])
def test_full_width_decision_publishes_the_narrow_streams_identity(
    *, Q_and_F: Callable[..., PytreeValue], wealth: float, narrow_width: int
) -> None:
    """A block covering the product names the action the narrow stream names.

    Ties go to the lowest identity, a feasible NaN publishes identity zero, a NaN at
    an infeasible action is ignored and an empty feasible set publishes -1.
    """
    streamed = _streamed_reducer(Q_and_F=Q_and_F)
    full_index, _ = _decide(streamed=streamed, wealth=wealth, width=14)
    narrow_index, _ = _decide(streamed=streamed, wealth=wealth, width=narrow_width)
    assert int(full_index) == int(narrow_index)


@pytest.mark.parametrize(
    ("Q_and_F", "wealth"),
    [
        (_tied_Q_and_F, 2.5),
        (_nan_where_feasible_Q_and_F, 2.5),
        (_nan_where_infeasible_Q_and_F, 1.0),
        (_toy_Q_and_F, -1.0),
    ],
    ids=["tied", "feasible-nan", "infeasible-nan", "nothing-feasible"],
)
@pytest.mark.parametrize("narrow_width", [1, 3])
def test_full_width_decision_publishes_the_narrow_streams_value_bits(
    *, Q_and_F: Callable[..., PytreeValue], wealth: float, narrow_width: int
) -> None:
    """On exactly representable values, the full block's value has the stream's bits.

    This covers the tie, both NaN placements and the empty feasible set's `-inf`.
    """
    streamed = _streamed_reducer(Q_and_F=Q_and_F)
    _, full_value = _decide(streamed=streamed, wealth=wealth, width=14)
    _, narrow_value = _decide(streamed=streamed, wealth=wealth, width=narrow_width)
    assert full_value.tobytes() == narrow_value.tobytes()


@pytest.mark.parametrize(
    ("Q_and_F", "wealth", "expected"),
    [
        (_tied_Q_and_F, 2.5, 8),
        (_nan_where_feasible_Q_and_F, 2.5, 0),
        (_toy_Q_and_F, -1.0, -1),
    ],
    ids=["tied", "feasible-nan", "nothing-feasible"],
)
def test_full_width_decision_publishes_the_identity_each_rule_names(
    *, Q_and_F: Callable[..., PytreeValue], wealth: float, expected: int
) -> None:
    """The tie, NaN and empty cases each publish the identity their rule names."""
    full_index, _ = _decide(
        streamed=_streamed_reducer(Q_and_F=Q_and_F), wealth=wealth, width=14
    )
    assert int(full_index) == expected


def _vector_valued_Q_and_F(
    *,
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
    consumption: FloatND,
    work: FloatND,
    wealth: FloatND,
) -> tuple[FloatND, BoolND]:
    """Return the toy action value three times over, one vector per action."""
    value, feasible = _toy_Q_and_F(
        next_regime_to_V_arr=next_regime_to_V_arr,
        consumption=consumption,
        work=work,
        wealth=wealth,
    )
    return jnp.stack([value, value, value]), feasible


@pytest.mark.parametrize("width", [1, 14])
def test_streamed_decision_refuses_a_vector_action_value_at_every_width(
    *, width: int
) -> None:
    """A single's kernel publishing three values per action is refused at any width."""
    streamed = _StreamedArgmaxQOverA(
        Q_and_F=cast("QAndFFunction", _vector_valued_Q_and_F),
        action_names=("work", "consumption"),
        q_and_f_arg_names=frozenset(
            {"next_regime_to_V_arr", "consumption", "work", "wealth"}
        ),
        action_width_keyword="_lcm_action_block_width",
        members=SELF,
        pareto_weights=SELF_PARETO_WEIGHTS,
    )
    with pytest.raises(ValueError, match="one trailing member axis"):
        _decide(streamed=streamed, wealth=2.5, width=width)


@pytest.mark.parametrize(
    ("actions", "error", "match"),
    [
        ({"work": _TOY_ACTIONS["work"]}, TypeError, "Missing action-grid"),
        (
            {"work": jnp.asarray(1.0), "consumption": _TOY_ACTIONS["consumption"]},
            ValueError,
            "must be one-dimensional",
        ),
        (
            {"work": _TOY_ACTIONS["work"], "consumption": jnp.zeros(0)},
            ValueError,
            "must not be empty",
        ),
    ],
    ids=["missing-grid", "scalar-grid", "empty-grid"],
)
@pytest.mark.parametrize("width", [1, 14])
def test_streamed_decision_refuses_a_malformed_action_grid_at_every_width(
    *,
    actions: Mapping[str, jax.Array],
    error: type[Exception],
    match: str,
    width: int,
) -> None:
    """A missing, scalar or empty action grid is refused at any block width."""
    streamed = _streamed_reducer(Q_and_F=_toy_Q_and_F)
    with pytest.raises(error, match=match):
        _decide(streamed=streamed, wealth=2.5, width=width, actions=actions)


_LARGE_ACTIONS = {"first": jnp.arange(3.0), "second": jnp.arange(2000.0)}


def _large_feasible_nan_Q_and_F(
    *,
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
    first: FloatND,
    second: FloatND,
) -> tuple[FloatND, BoolND]:
    """Return the continuation everywhere but NaN at one interior action."""
    poisoned = (first == 1.0) & (second == 1500.0)
    value = jnp.where(poisoned, jnp.nan, next_regime_to_V_arr["target"])
    return value, jnp.ones((), dtype=bool)


def _large_tied_Q_and_F(
    *,
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
    first: FloatND,
    second: FloatND,
) -> tuple[FloatND, BoolND]:
    """Return the continuation at every action, feasible from `first = 1` up."""
    value = next_regime_to_V_arr["target"] + 0.0 * second
    return value, jnp.asarray(first >= 1.0)


@pytest.mark.parametrize(
    ("Q_and_F", "expected_index", "expected_value"),
    [(_large_feasible_nan_Q_and_F, 0, jnp.nan), (_large_tied_Q_and_F, 2000, 0.25)],
    ids=["feasible-nan", "tied"],
)
@pytest.mark.parametrize("width", [7, 6000])
def test_large_action_product_keeps_the_nan_and_tie_rules_at_every_width(
    *,
    Q_and_F: Callable[..., PytreeValue],
    expected_index: int,
    expected_value: float,
    width: int,
) -> None:
    """Over six thousand actions, the tie and NaN rules hold at any block width.

    A feasible NaN publishes NaN at identity zero, and a tie publishes the lowest
    feasible identity, both in one compiled program.
    """
    streamed = _StreamedArgmaxQOverA(
        Q_and_F=cast(
            "QAndFFunction",
            with_member_axis(cast("Callable[..., tuple[FloatND, BoolND]]", Q_and_F)),
        ),
        action_names=("first", "second"),
        q_and_f_arg_names=frozenset({"next_regime_to_V_arr", "first", "second"}),
        action_width_keyword="_lcm_action_block_width",
        members=SELF,
        pareto_weights=SELF_PARETO_WEIGHTS,
    )
    continuation = MappingProxyType({"target": jnp.asarray(0.25)})
    index, value = cast(
        "tuple[IntND, FloatND]",
        jax.jit(
            lambda first, second: streamed(
                next_regime_to_V_arr=continuation,
                first=first,
                second=second,
                _lcm_action_block_width=width,
            )
        )(**_LARGE_ACTIONS),
    )
    assert int(index) == expected_index
    assert_array_equal(value, jnp.asarray([expected_value], dtype=value.dtype))


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
    streamed_index, _ = cast(
        "tuple[IntND, FloatND]",
        streamed(
            next_regime_to_V_arr=continuation,
            wealth=jnp.asarray(wealth),
            _lcm_action_block_width=width,
            **_TOY_ACTIONS,
        ),
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
    _, streamed_value = cast(
        "tuple[IntND, FloatND]",
        streamed(
            next_regime_to_V_arr=continuation,
            wealth=jnp.asarray(wealth),
            _lcm_action_block_width=width,
            **_TOY_ACTIONS,
        ),
    )
    assert_values_agree(got=streamed_value, expected=dense_value, n_ulp=1)


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
    streamed_index, _ = cast(
        "tuple[IntND, FloatND]",
        streamed(
            next_regime_to_V_arr=continuation,
            wealth=infeasible,
            _lcm_action_block_width=width,
            **_TOY_ACTIONS,
        ),
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
