"""A gate reads names on its target's domain, never through the source's functions.

A gate and its projections run on the gated target's grid. A name the target
neither declares nor has the engine inject is a free parameter of the gate, at
its declaration path, whatever the source regime happens to call one of its own
functions. Only a direct read of a `next_` name stays refused.
"""

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    LinSpacedGrid,
    Model,
    ParticipationCheck,
    ParticipationRead,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    StochasticTransition,
    TimeVarying,
    Transition,
    categorical,
    fixed_transition,
    time_varying_params,
)
from lcm.exceptions import (
    InvalidNameError,
    InvalidParamsError,
)
from lcm.solver_api import ResultRetention, SolutionResult
from lcm.typing import (
    BoolND,
    ContinuousState,
    FloatND,
    ReferenceName,
    ScalarInt,
    StateName,
    UserFunction,
    UserParams,
)

type TemplateNode = str | Mapping[str, TemplateNode]


DISCOUNT_FACTOR = 0.5


@categorical(ordered=False)
class RegimeId:
    src: ScalarInt
    target: ScalarInt
    fallback: ScalarInt


def source_utility(z: ContinuousState) -> FloatND:
    return 1.0 + z


def target_utility() -> FloatND:
    return jnp.asarray(1.0)


def fallback_utility() -> FloatND:
    return jnp.asarray(0.5)


def fallback_utility_of_w(w: ContinuousState) -> FloatND:
    return 0.25 * (1.0 + w)


def next_y_value(next_y: ContinuousState) -> FloatND:
    return next_y


def through_inner_h(inner_h: FloatND) -> FloatND:
    return inner_h


def next_z_from_h(h: FloatND) -> FloatND:
    return h


def next_z_from_source_helper(source_helper: FloatND) -> FloatND:
    return source_helper


def always(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def always_through_h(*, age: FloatND, h: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float) + 0.0 * h


def gate_with_threshold_h(*, V_target: FloatND, h: float) -> BoolND:
    return V_target >= h


def gate_reading_next_y(*, V_target: FloatND, next_y: FloatND) -> BoolND:
    return V_target >= next_y


def gate_with_fixed_threshold(V_target: FloatND) -> BoolND:
    return V_target >= 0.75


def gate_against_reference(*, V_target: FloatND, V_fallback: FloatND) -> BoolND:
    return V_target >= V_fallback


def project_h(h: FloatND) -> FloatND:
    return jnp.asarray(h)


def project_next_y(next_y: FloatND) -> FloatND:
    return next_y


def build_model(
    *,
    helper_name: str,
    helper_depth: int,
    gate: Callable[..., BoolND] = gate_with_threshold_h,
    reference_projection: Callable[..., FloatND] | None = None,
    route_projection: Callable[..., FloatND] | None = None,
    law: Callable[..., FloatND] = always,
) -> Model:
    """Source `src` moves `z` through a helper reading `next_y`; age 1 is gated.

    `src@0 -> src@1 -> target@2`, with the gate's fallback `fallback@2`. Both
    endpoints are terminal; the target declares no state and no function `h`.
    A projection, if given, projects onto the fallback's state `w`: as the gate
    reference `V_fallback`, or as the route's fallback.
    """
    projects = reference_projection is not None or route_projection is not None
    functions = {
        "utility": source_utility,
        helper_name: next_y_value if helper_depth == 1 else through_inner_h,
    }
    if helper_depth == 2:
        functions["inner_h"] = next_y_value
    grid = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
    return Model(
        regimes={
            "src": Regime(
                states={"y": grid, "z": grid},
                state_transitions={
                    "y": fixed_transition("y"),
                    "z": (
                        next_z_from_h
                        if helper_name == "h"
                        else next_z_from_source_helper
                    ),
                },
                functions=functions,
            ),
            "target": Regime(functions={"utility": target_utility}),
            "fallback": (
                Regime(states={"w": grid}, functions={"utility": fallback_utility_of_w})
                if projects
                else Regime(functions={"utility": fallback_utility})
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "src"},
        edges={
            "src": Transition(
                law=ByAge(
                    cases={
                        0: "src",
                        1: {"target": StochasticTransition(func=law)},
                    }
                ),
                participation={
                    "target": ParticipationCheck(
                        predicate=gate,
                        reads={
                            "V_target": ParticipationRead(
                                kind="IntegratedFull", regime="target", projection={}
                            ),
                            **(
                                {}
                                if reference_projection is None
                                else {
                                    "V_fallback": ParticipationRead(
                                        kind="IntegratedFull",
                                        regime="fallback",
                                        projection={"w": reference_projection},
                                    )
                                }
                            ),
                        },
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback",
                                    projection=(
                                        {"w": route_projection or project_h}
                                        if projects
                                        else {}
                                    ),
                                )
                            )
                        },
                    )
                },
            ),
        },
    )


def leaf_paths(
    *, tree: TemplateNode, prefix: tuple[str, ...] = ()
) -> set[tuple[str, ...]]:
    if isinstance(tree, Mapping):
        return {
            path
            for key, value in tree.items()
            for path in leaf_paths(tree=value, prefix=(*prefix, key))
        }
    return {prefix}


HELPER_CASES = [
    pytest.param("h", 1, id="helper-named-like-the-gate-parameter"),
    pytest.param("h", 2, id="helper-named-like-the-gate-parameter-two-deep"),
    pytest.param("source_helper", 1, id="helper-named-apart"),
    pytest.param("source_helper", 2, id="helper-named-apart-two-deep"),
]


@pytest.mark.parametrize(("helper_name", "helper_depth"), HELPER_CASES)
def test_gate_argument_is_a_free_parameter_whatever_the_source_calls_a_helper(
    *, helper_name, helper_depth
):
    model = build_model(helper_name=helper_name, helper_depth=helper_depth)
    assert leaf_paths(tree=model.get_params_template()["edges"]) == {
        ("src", "target", "predicate", "h")
    }


@pytest.mark.parametrize(("helper_name", "helper_depth"), HELPER_CASES)
@pytest.mark.parametrize(
    ("threshold", "terminal_value"),
    [
        pytest.param(0.5, 1.0, id="gate-open"),
        pytest.param(1.5, 0.5, id="gate-closed"),
    ],
)
def test_gated_values_follow_the_gate_threshold(
    *, helper_name, helper_depth, threshold, terminal_value
):
    """Source values are `1+z` plus the discounted continuation, exactly.

    With discount 1/2 and terminal values 1 or 1/2 every value is a dyadic
    fraction, so the comparison is exact at either precision. Values are sorted
    so the check does not depend on the order of the `y` and `z` axes.
    """
    model = build_model(helper_name=helper_name, helper_depth=helper_depth)
    values = model.solve(
        params={
            "discount_factor": DISCOUNT_FACTOR,
            "edges": {"src": {"target": {"predicate": {"h": threshold}}}},
        },
        log_level="off",
    ).values
    at_age_1 = [
        1.0 + z + DISCOUNT_FACTOR * terminal_value for y in (0, 1) for z in (0, 1)
    ]
    at_age_0 = [
        1.0 + z + DISCOUNT_FACTOR * (1.0 + y + DISCOUNT_FACTOR * terminal_value)
        for y in (0, 1)
        for z in (0, 1)
    ]
    dtype = np.dtype("float64" if jax.config.jax_enable_x64 else "float32")
    got = [np.sort(np.asarray(values[age]["src"]).ravel()) for age in (1, 0)]
    np.testing.assert_array_equal(
        np.concatenate(got),
        np.asarray(sorted(at_age_1) + sorted(at_age_0), dtype=dtype),
        strict=True,
    )


def test_gate_reading_a_next_name_directly_is_refused():
    with pytest.raises(InvalidNameError, match="next_y"):
        build_model(helper_name="h", helper_depth=1, gate=gate_reading_next_y)


@pytest.mark.parametrize(("helper_name", "helper_depth"), HELPER_CASES)
@pytest.mark.parametrize(
    ("role", "leaf"),
    [
        pytest.param(
            "reference",
            {
                ("src", "target", "reads", "V_fallback", "w", "h"),
                ("src", "target", "routes", "only", "fallback", "w", "h"),
            },
            id="reference-projection",
        ),
        pytest.param(
            "route",
            {("src", "target", "routes", "only", "fallback", "w", "h")},
            id="route-projection",
        ),
    ],
)
def test_projection_argument_is_a_free_parameter_whatever_the_source_calls_a_helper(
    *, helper_name, helper_depth, role, leaf
):
    """A projection runs on the target too; its `h` is its own parameter."""
    model = build_model(
        helper_name=helper_name,
        helper_depth=helper_depth,
        gate=gate_against_reference
        if role == "reference"
        else gate_with_fixed_threshold,
        reference_projection=project_h if role == "reference" else None,
        route_projection=project_h,
    )
    assert leaf_paths(tree=model.get_params_template()["edges"]) == leaf


@pytest.mark.parametrize("role", ["reference", "route"])
def test_projection_reading_a_next_name_directly_is_refused(role):
    with pytest.raises(InvalidNameError, match="next_y"):
        build_model(
            helper_name="source_helper",
            helper_depth=1,
            gate=(
                gate_against_reference
                if role == "reference"
                else gate_with_fixed_threshold
            ),
            reference_projection=project_next_y if role == "reference" else None,
            route_projection=project_next_y if role == "route" else project_h,
        )


def test_edge_law_reading_a_next_name_through_a_source_helper_is_refused():
    """The law runs on the source, before any target law; it may not reach `next_y`."""
    with pytest.raises(InvalidNameError, match="next_y"):
        build_model(helper_name="h", helper_depth=1, law=always_through_h)


def test_gate_parameter_beside_an_unknown_leaf_is_refused():
    model = build_model(helper_name="source_helper", helper_depth=1)
    with pytest.raises(InvalidParamsError, match="extra"):
        model.solve(
            params={
                "discount_factor": DISCOUNT_FACTOR,
                "edges": {"src": {"target": {"predicate": {"h": 0.5, "extra": 1.0}}}},
            },
            log_level="off",
        )


@categorical(ordered=False)
class _ParticipationDomainRegimeId:
    source: ScalarInt
    target: ScalarInt
    fallback: ScalarInt
    done: ScalarInt


def _participation_domain_utility(*, x: ContinuousState, y: ContinuousState) -> FloatND:
    return 1.0 + x + 0.0 * y


def _participation_domain_utility_from_h(
    *, h: ContinuousState, y: ContinuousState
) -> FloatND:
    return 1.0 + h + 0.0 * y


def _participation_domain_identity_x(x: ContinuousState) -> ContinuousState:
    return x


def _participation_domain_identity_y(y: ContinuousState) -> ContinuousState:
    return y


def _participation_domain_next_y(next_y: ContinuousState) -> FloatND:
    return next_y


def _participation_domain_transition_from_h(h: FloatND) -> ContinuousState:
    return h


def _participation_domain_compare(*, V_target: FloatND, h: float) -> BoolND:
    return V_target >= h


def _participation_domain_compare_next(*, V_target: FloatND, next_y: FloatND) -> BoolND:
    return V_target >= next_y


def _participation_domain_open(V_target: FloatND) -> BoolND:
    return V_target >= 2.25


def _participation_domain_fallback_utility(
    *, x: ContinuousState, y: ContinuousState
) -> FloatND:
    return 0.25 + 2.0 * x + 0.0 * y


def _participation_domain_project_h(h: float) -> ContinuousState:
    return jnp.asarray(h)


def _participation_domain_project_next(next_y: FloatND) -> ContinuousState:
    return next_y


def _participation_domain_model(
    *,
    consumer: str,
    bad_name: str,
    with_participation: bool = True,
    temporal_h: bool = False,
    fixed_params: UserParams | None = None,
) -> Model:
    """Valid ordinary law helpers must not leak into a participation consumer.

    Source0 enters target1, which enters done2. The target helper h is either
    its current x or a legitimate next_y input to the target's x state law.
    A participation callback's h is its own independent edge parameter, even when
    a target or source helper uses that spelling. Exact aliasing must occur
    before target-function concatenation; a direct next_y remains phase-invalid.
    """
    grid = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
    target_functions = {"utility": _participation_domain_utility}
    target_laws = {"x": fixed_transition("x"), "y": fixed_transition("y")}
    source_functions = {"utility": _participation_domain_utility}
    source_laws = {"x": fixed_transition("x"), "y": fixed_transition("y")}
    if bad_name == "target_helper":
        target_functions["h"] = _participation_domain_identity_x
        target_functions["utility"] = _participation_domain_utility_from_h
    elif bad_name == "transitive_next":
        target_functions["h"] = _participation_domain_next_y
        target_laws["x"] = _participation_domain_transition_from_h
    elif bad_name == "source_helper":
        source_functions["h"] = _participation_domain_next_y
        source_laws["x"] = _participation_domain_transition_from_h
    project = (
        _participation_domain_project_next
        if bad_name == "direct_next"
        else _participation_domain_project_h
    )
    predicate = (
        _participation_domain_compare_next
        if consumer == "predicate" and bad_name == "direct_next"
        else _participation_domain_compare
        if consumer == "predicate"
        else _participation_domain_open
    )
    if temporal_h:
        predicate = time_varying_params("h")(predicate)
    return Model(
        regimes={
            "source": Regime(
                states={"x": grid, "y": grid},
                state_transitions=source_laws,
                functions=source_functions,
            ),
            "target": Regime(
                states={"x": grid, "y": grid},
                state_transitions=target_laws,
                functions=target_functions,
            ),
            "fallback": Regime(
                states={"x": grid, "y": grid},
                functions={"utility": _participation_domain_fallback_utility},
            ),
            "done": Regime(
                states={"x": grid, "y": grid},
                functions={"utility": _participation_domain_utility},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_ParticipationDomainRegimeId,
        initial_nodes={0: "source"},
        fixed_params=fixed_params or {},
        edges={
            "source": Transition(
                law=ByAge(cases={0: "target"}),
                participation={
                    "target": ParticipationCheck(
                        predicate=predicate,
                        reads={
                            "V_target": ParticipationRead(
                                kind="IntegratedFull",
                                regime="target",
                                projection={
                                    "x": project
                                    if consumer == "read_projection"
                                    else _participation_domain_identity_x,
                                    "y": _participation_domain_identity_y,
                                },
                            )
                        },
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback",
                                    projection={
                                        "x": project
                                        if consumer == "fallback_projection"
                                        else _participation_domain_identity_x,
                                        "y": _participation_domain_identity_y,
                                    },
                                )
                            )
                        },
                    )
                }
                if with_participation
                else {},
            ),
            "target": Transition(law=ByAge(cases={1: "done"})),
        },
    )


@pytest.mark.parametrize(
    "consumer", ["predicate", "read_projection", "fallback_projection"]
)
@pytest.mark.parametrize("bad_name", ["direct_next"])
def test_participation_consumer_refuses_reserved_next_names(
    *, consumer: str, bad_name: str
) -> None:
    """A transition-local value is unavailable to the landed participation check."""
    with pytest.raises(InvalidNameError, match="next_y"):
        _participation_domain_model(consumer=consumer, bad_name=bad_name)


@pytest.mark.parametrize(
    ("consumer", "bad_name"),
    [
        ("predicate", "source_helper"),
        ("predicate", "target_helper"),
        ("predicate", "transitive_next"),
        ("read_projection", "target_helper"),
        ("fallback_projection", "target_helper"),
    ],
)
def test_participation_own_edge_parameter_precedes_target_helper_concatenation(
    *, consumer: str, bad_name: str
) -> None:
    """The exact edge slot controls the Boolean rule or projected coordinate."""
    model = _participation_domain_model(consumer=consumer, bad_name=bad_name)
    path = (
        ("predicate",)
        if consumer == "predicate"
        else ("reads", "V_target", "x")
        if consumer == "read_projection"
        else ("routes", "only", "fallback", "x")
    )
    assert leaf_paths(tree=model.get_params_template()["edges"]) == {
        ("source", "target", *path, "h")
    }
    settings = (0.5, 5.0) if consumer == "predicate" else (0.0, 1.0)
    for h in settings:
        branch = {"h": h}
        for segment in reversed(path):
            branch = {segment: branch}
        solution = model.solve(
            params={"discount_factor": 0.5, "edges": {"source": {"target": branch}}},
            log_level="off",
        )
        expected = []
        for x in (0.0, 1.0):
            for y in (0.0, 1.0):
                landing_x = y if bad_name == "source_helper" else x
                future_x = y if bad_name == "transitive_next" else landing_x
                target = 1.0 + landing_x + 0.5 * (1.0 + future_x)
                comparison = 1.5 + 1.5 * h if consumer == "read_projection" else target
                threshold = h if consumer == "predicate" else 2.25
                fallback_x = h if consumer == "fallback_projection" else landing_x
                fallback = 0.25 + 2.0 * fallback_x
                continued = target if comparison >= threshold else fallback
                expected.append(1.0 + x + 0.5 * continued)
        np.testing.assert_array_equal(
            np.sort(np.asarray(solution.values[0]["source"]).ravel()),
            np.sort(np.asarray(expected)),
        )


@pytest.mark.parametrize("bad_name", ["target_helper", "transitive_next"])
def test_participation_admission_fixture_has_valid_ordinary_target_helpers(
    *, bad_name: str
) -> None:
    """The ordinary target payoff/transition owns these helpers legitimately."""
    model = _participation_domain_model(
        consumer="predicate", bad_name=bad_name, with_participation=False
    )
    assert model.get_params_template()["target"]["utility"] == {}


def test_participation_source_only_helper_stays_an_independent_edge_parameter() -> None:
    """A source's valid next_y helper cannot capture the check's free h."""
    model = _participation_domain_model(consumer="predicate", bad_name="source_helper")
    assert leaf_paths(tree=model.get_params_template()["edges"]) == {
        ("source", "target", "predicate", "h")
    }


@pytest.mark.parametrize("fixed", [False, True])
def test_participation_edge_parameter_uses_its_actual_date_in_solve_and_replay(
    *, fixed: bool
) -> None:
    """Source-date h rejects, but the true participation-date h accepts the target."""
    params = {
        "discount_factor": 0.5,
        "edges": {
            "source": {
                "target": {
                    "predicate": {
                        "h": TimeVarying(
                            values=jnp.asarray([5.0, 0.5, 5.0]), ages=(0, 1, 2)
                        )
                    }
                }
            }
        },
    }
    model = _participation_domain_model(
        consumer="predicate",
        bad_name="target_helper",
        temporal_h=True,
        fixed_params=params if fixed else None,
    )
    runtime = {} if fixed else params
    solution = model.solve(params=runtime, log_level="off")
    np.testing.assert_array_equal(
        np.sort(np.asarray(solution.values[0]["source"]).ravel()),
        [1.75, 1.75, 3.5, 3.5],
    )
    frame = model.simulate(
        params=runtime,
        solution=solution,
        initial_conditions={
            "age": jnp.asarray([0.0]),
            "regime_id": jnp.asarray([_ParticipationDomainRegimeId.source]),
            "x": jnp.asarray([0.0]),
            "y": jnp.asarray([0.0]),
        },
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    participation = frame.query("period == 1")
    np.testing.assert_array_equal(participation["regime_name"], ["target"])
    np.testing.assert_array_equal(participation["value"], [1.5])


@categorical(ordered=False)
class _ReadPriorityRegimeId:
    source: ScalarInt
    target: ScalarInt
    reference: ScalarInt
    fallback: ScalarInt


def _priority_zero(x: ContinuousState) -> FloatND:
    return 0.0 * x


def _priority_h(x: ContinuousState) -> FloatND:
    return 10.0 + x


def _priority_target(h: FloatND) -> FloatND:
    return 100.0 + h


def _priority_reference(z: ContinuousState) -> FloatND:
    return 2.0 + z


def _priority_fallback(w: ContinuousState) -> FloatND:
    return w


def _priority_original_x(x: ContinuousState) -> ContinuousState:
    return x


def _priority_fallback_state(x: ContinuousState) -> ContinuousState:
    return 10.0 + x


def _priority_read_h(h: FloatND) -> BoolND:
    return h < 5.0


def _priority_read_quote(quote: FloatND) -> BoolND:
    return quote < 5.0


def _priority_read_x(x: FloatND) -> BoolND:
    return x >= 2.5


def _priority_read_v_target(V_target: FloatND) -> BoolND:
    return V_target < 5.0


def _read_priority_replay(
    *, predicate: UserFunction, read_name: ReferenceName
) -> tuple[SolutionResult, pd.DataFrame]:
    """Replay a named comparison beside a genuinely live target helper h."""
    grid = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
    model = Model(
        regimes={
            "source": Regime(
                states={"x": grid},
                state_transitions={"x": fixed_transition("x")},
                functions={"utility": _priority_zero},
            ),
            "target": Regime(
                states={"x": grid},
                functions={"utility": _priority_target, "h": _priority_h},
            ),
            "reference": Regime(
                states={"z": grid}, functions={"utility": _priority_reference}
            ),
            "fallback": Regime(
                states={"w": LinSpacedGrid(start=10.0, stop=11.0, n_points=2)},
                functions={"utility": _priority_fallback},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_ReadPriorityRegimeId,
        initial_nodes={0: "source"},
        edges={
            "source": Transition(
                law=ByAge(cases={0: "target"}),
                participation={
                    "target": ParticipationCheck(
                        predicate=predicate,
                        reads={
                            read_name: ParticipationRead(
                                kind="Raw",
                                regime="reference",
                                projection={"z": _priority_original_x},
                            )
                        },
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback",
                                    projection={"w": _priority_fallback_state},
                                )
                            )
                        },
                    )
                },
            )
        },
    )
    params = {"discount_factor": 0.5}
    solution = model.solve(
        params=params, retention=ResultRetention.VALUES, log_level="off"
    )
    frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "age": jnp.asarray([0.0, 0.0]),
            "regime_id": jnp.full(2, _ReadPriorityRegimeId.source, dtype=jnp.int32),
            "x": jnp.asarray([0.0, 1.0]),
        },
        seed=41,
        log_level="off",
    ).to_dataframe()
    return solution, frame


@pytest.mark.parametrize(
    ("predicate", "read_name"),
    [(_priority_read_h, "h"), (_priority_read_quote, "quote")],
    ids=["live-target-helper", "disjoint-control"],
)
def test_participation_explicit_read_precedes_its_same_named_target_helper(
    *, predicate: UserFunction, read_name: ReferenceName
) -> None:
    """The declared reference value admits a target whose own helper would refuse."""
    solution, frame = _read_priority_replay(predicate=predicate, read_name=read_name)
    np.testing.assert_array_equal(solution.values[0]["source"], [55.0, 55.5])
    landed = frame.loc[frame["period"] == 1].sort_values("subject_id")
    np.testing.assert_array_equal(landed["subject_id"], [0, 1])
    np.testing.assert_array_equal(
        landed["regime_name"].to_numpy(), ["target", "target"]
    )
    np.testing.assert_array_equal(landed["x"], [0.0, 1.0])
    np.testing.assert_array_equal(landed["value"], [110.0, 111.0])


def test_participation_read_state_alias_preserves_original_projection_inputs() -> None:
    """Read x decides routing while fallback w uses the original landing x."""
    solution, frame = _read_priority_replay(predicate=_priority_read_x, read_name="x")
    np.testing.assert_array_equal(solution.values[0]["source"], [5.0, 55.5])
    landed = frame.loc[frame["period"] == 1].sort_values("subject_id")
    np.testing.assert_array_equal(landed["subject_id"], [0, 1])
    np.testing.assert_array_equal(
        landed["regime_name"].to_numpy(), ["fallback", "target"]
    )
    np.testing.assert_array_equal(landed["value"], [10.0, 111.0])
    assert landed.iloc[0]["w"] == 10.0
    assert landed.iloc[1]["x"] == 1.0


def test_participation_v_target_read_names_the_declared_reference_object() -> None:
    """Explicit V_target reads the reference, independently of the acting target."""
    solution, frame = _read_priority_replay(
        predicate=_priority_read_v_target, read_name="V_target"
    )
    np.testing.assert_array_equal(solution.values[0]["source"], [55.0, 55.5])
    landed = frame.loc[frame["period"] == 1].sort_values("subject_id")
    np.testing.assert_array_equal(landed["subject_id"], [0, 1])
    np.testing.assert_array_equal(
        landed["regime_name"].to_numpy(), ["target", "target"]
    )
    np.testing.assert_array_equal(landed["x"], [0.0, 1.0])
    np.testing.assert_array_equal(landed["value"], [110.0, 111.0])


def _collision_identity_x(*, x: FloatND) -> FloatND:
    return x


def _collision_identity_acting_target(*, acting_target: FloatND) -> FloatND:
    return acting_target


def _collision_source_x(*, x: FloatND) -> FloatND:
    return 0.0 * x


def _collision_source_acting_target(*, acting_target: FloatND) -> FloatND:
    return 0.0 * acting_target


def _collision_target_x(*, x: FloatND) -> FloatND:
    return 10.0 + x


def _collision_target_acting_target(*, acting_target: FloatND) -> FloatND:
    return 10.0 + acting_target


def _collision_fallback_x(*, x: FloatND) -> FloatND:
    return 1.0 + x


def _collision_fallback_acting_target(*, acting_target: FloatND) -> FloatND:
    return 1.0 + acting_target


def _collision_keep_x(*, x: FloatND) -> BoolND:
    return x > 1.5


def _collision_keep_acting_target(*, acting_target: FloatND) -> BoolND:
    return acting_target > 1.5


@pytest.mark.parametrize("state_name", ["x", "acting_target"])
def test_participation_internal_dependency_preserves_the_declared_physical_state(
    *, state_name: StateName
) -> None:
    """Renaming a physical state keeps participation pricing and the row routes."""
    if state_name == "x":
        source_flow = _collision_source_x
        target_flow = _collision_target_x
        fallback_flow = _collision_fallback_x
        predicate = _collision_keep_x
        projection = _collision_identity_x
    else:
        source_flow = _collision_source_acting_target
        target_flow = _collision_target_acting_target
        fallback_flow = _collision_fallback_acting_target
        predicate = _collision_keep_acting_target
        projection = _collision_identity_acting_target
    grid = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)
    model = Model(
        regimes={
            "src": Regime(
                states={state_name: grid},
                state_transitions={state_name: fixed_transition(state_name)},
                functions={"utility": source_flow},
            ),
            "target": Regime(
                states={state_name: grid}, functions={"utility": target_flow}
            ),
            "fallback": Regime(
                states={state_name: grid}, functions={"utility": fallback_flow}
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "src"},
        regime_id_class=RegimeId,
        edges={
            "src": Transition(
                law={"target": StochasticTransition(func=always)},
                participation={
                    "target": ParticipationCheck(
                        predicate=predicate,
                        routes={
                            "self": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback",
                                    projection={state_name: projection},
                                )
                            )
                        },
                    )
                },
            )
        },
    )
    params = {"discount_factor": 0.5}
    solution = model.solve(
        params=params, retention=ResultRetention.VALUES, log_level="off"
    )
    np.testing.assert_allclose(solution.values[0]["src"], [1.0, 6.0], rtol=1e-6)
    result = model.simulate(
        params=params,
        initial_conditions={
            state_name: jnp.asarray([1.0, 2.0]),
            "age": jnp.zeros(2),
            "regime_id": jnp.full(2, model.regime_names_to_ids["src"], dtype=jnp.int32),
        },
        solution=solution,
        seed=0,
        log_level="off",
    ).to_dataframe()
    landed = result.loc[result["period"] == 1].sort_values("subject_id")
    np.testing.assert_array_equal(landed["subject_id"], [0, 1])
    np.testing.assert_array_equal(landed["regime_name"], ["fallback", "target"])
    np.testing.assert_array_equal(landed[state_name], [1.0, 2.0])
    np.testing.assert_allclose(landed["value"], [2.0, 12.0], rtol=1e-6)
