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
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    Gate,
    LinSpacedGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    StochasticTransition,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import (
    InvalidNameError,
    InvalidParamsError,
    ModelInitializationError,
)
from lcm.typing import BoolND, ContinuousState, FloatND, ScalarInt

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


def target_h() -> FloatND:
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
    target_functions: Mapping[str, Callable[..., FloatND]] | None = None,
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
            "target": Regime(
                functions={"utility": target_utility, **(target_functions or {})}
            ),
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
                gates={
                    "target": Gate(
                        predicate=gate,
                        references=(
                            {}
                            if reference_projection is None
                            else {
                                "V_fallback": ProjectedRegimeValue(
                                    regime="fallback",
                                    projection={"w": reference_projection},
                                )
                            }
                        ),
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
    *, tree: Mapping | str, prefix: tuple[str, ...] = ()
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


def test_gate_reading_a_function_of_its_target_is_refused():
    """A gate argument named like a target function would be captured by it."""
    with pytest.raises(ModelInitializationError, match="TARGET regime's own function"):
        build_model(
            helper_name="source_helper",
            helper_depth=1,
            target_functions={"h": target_h},
        )


@pytest.mark.parametrize(("helper_name", "helper_depth"), HELPER_CASES)
@pytest.mark.parametrize(
    ("role", "leaf"),
    [
        pytest.param(
            "reference",
            {
                ("src", "target", "references", "V_fallback", "w", "h"),
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
