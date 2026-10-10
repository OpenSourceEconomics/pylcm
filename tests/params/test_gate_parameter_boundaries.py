"""A gate's parameters are read on its target's grid, and a gate is no law case.

A gate predicate runs on the gated target's grid, so a name only the source
declares is one of its free parameters, whether or not demand pruned that name
from the source. A law of any form may sit beside a separate gate: only the
law's own cases decide whether it reads parameters both over all targets and per
target.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    DeterministicTransition,
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
from lcm.typing import BoolND, ContinuousState, FloatND, ScalarInt

type TemplateNode = str | Mapping[str, TemplateNode]


@categorical(ordered=False)
class RegimeId:
    src: ScalarInt
    target: ScalarInt
    fallback: ScalarInt


def unit_utility() -> FloatND:
    return jnp.asarray(1.0)


def half_utility() -> FloatND:
    return jnp.asarray(0.5)


def certain(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def choose_target(choice_level: float) -> ScalarInt:
    return jnp.where(choice_level >= 0, RegimeId.target, RegimeId.fallback)


def target_probabilities(meeting: float) -> FloatND:
    return jnp.array([0.0, meeting, 1.0 - meeting])


def choose_target_without_parameter(age: FloatND) -> ScalarInt:
    return jnp.where(age >= 0, RegimeId.target, RegimeId.fallback)


def below_threshold(*, y: ContinuousState, threshold: float) -> BoolND:
    return y < threshold


def below_half(y: ContinuousState) -> BoolND:
    return y < 0.5


def gate_with_free_x(*, V_target: FloatND, x: float) -> BoolND:
    return V_target >= x


def routes() -> dict[str, StakeholderRoute]:
    return {
        "only": StakeholderRoute(
            fallback=ProjectedRegimeValue(regime="fallback", projection={})
        )
    }


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


def coarse_gate_model(
    *,
    kind: str,
    include_gate: bool = True,
    parameterized_gate: bool = True,
    parameterized_law: bool = True,
) -> Model:
    """Leave `src` for terminal `target` by a law over all targets, gated or not.

    Source age 0 is the only source age; `target` and `fallback` are terminal at
    age 1. The broadcast `y` is carried by identity and read by the gate.
    """
    law = (
        DeterministicTransition(func=choose_target)
        if kind == "deterministic"
        else StochasticTransition(func=target_probabilities)
    )
    if not parameterized_law:
        law = DeterministicTransition(func=choose_target_without_parameter)
    return Model(
        regimes={
            "src": Regime(functions={"utility": unit_utility}),
            "target": Regime(functions={"utility": unit_utility}),
            "fallback": Regime(functions={"utility": half_utility}),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "src"},
        states={"y": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={"y": fixed_transition("y")},
        edges={
            "src": Transition(
                targets={"target": 0, "fallback": 0},
                law=law,
                gates=(
                    {
                        "target": Gate(
                            predicate=(
                                below_threshold if parameterized_gate else below_half
                            ),
                            routes=routes(),
                        )
                    }
                    if include_gate
                    else {}
                ),
            )
        },
    )


@pytest.mark.parametrize("kind", ["deterministic", "vector"])
def test_parameterized_coarse_law_can_have_a_separate_gate(kind):
    """A law over all targets and a gate each keep their own parameter slot.

    With the law choosing `target` for sure, the gate opens at the grid's low
    `y` (value 1 + 0.9 * 1) and routes to `fallback` at its high `y`
    (1 + 0.9 * 0.5).
    """
    model = coarse_gate_model(kind=kind)
    name, value = ("choice_level", 1.0) if kind == "deterministic" else ("meeting", 1.0)
    assert leaf_paths(tree=model.get_params_template()["edges"]) == {
        ("src", name),
        ("src", "target", "predicate", "threshold"),
    }
    values = model.solve(
        params={
            "discount_factor": 0.9,
            "edges": {
                "src": {name: value, "target": {"predicate": {"threshold": 0.5}}}
            },
        },
        log_level="debug",
    ).values
    np.testing.assert_allclose(np.asarray(values[0]["src"]), [1.9, 1.45], rtol=1e-6)


@pytest.mark.parametrize("kind", ["deterministic", "vector"])
@pytest.mark.parametrize("include_gate", [False, True])
@pytest.mark.parametrize("parameterized_gate", [False, True])
def test_separate_gate_does_not_change_coarse_law_admission(
    *, kind, include_gate, parameterized_gate
):
    """A gate, with or without parameters, adds only its own slot."""
    instance = coarse_gate_model(
        kind=kind,
        include_gate=include_gate,
        parameterized_gate=parameterized_gate,
    )
    parameter = "choice_level" if kind == "deterministic" else "meeting"
    expected = {("src", parameter)}
    if include_gate and parameterized_gate:
        expected.add(("src", "target", "predicate", "threshold"))
    assert leaf_paths(tree=instance.get_params_template()["edges"]) == expected


def test_coarse_parameter_without_gate_control():
    """Without a gate, the law over all targets owns the only slot."""
    model = coarse_gate_model(kind="deterministic", include_gate=False)
    assert leaf_paths(tree=model.get_params_template()["edges"]) == {
        ("src", "choice_level"),
    }


def test_gate_with_parameterless_coarse_law_control():
    """A law without parameters leaves the gate's slot as the only one."""
    model = coarse_gate_model(kind="deterministic", parameterized_law=False)
    assert leaf_paths(tree=model.get_params_template()["edges"]) == {
        ("src", "target", "predicate", "threshold"),
    }


def test_parameterless_gate_is_not_a_per_target_law_case():
    """A gate without parameters beside a parameterized coarse law is admitted."""
    model = coarse_gate_model(kind="deterministic", parameterized_gate=False)
    assert leaf_paths(tree=model.get_params_template()["edges"]) == {
        ("src", "choice_level"),
    }


def source_name_gate_model(*, declare_pruned_source_x: bool) -> Model:
    """x is an edge free parameter; neither target regime declares that state.

    In the second declaration x is additionally an unused source-only broadcast
    state. Target/fallback masks keep their vocabulary identical across variants.
    Runtime pruning removes source x without changing the gate's parameter.
    """
    masks = {"x": None} if declare_pruned_source_x else {}
    return Model(
        regimes={
            "src": Regime(functions={"utility": unit_utility}),
            "target": Regime(states=masks, functions={"utility": unit_utility}),
            "fallback": Regime(states=masks, functions={"utility": half_utility}),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "src"},
        states=(
            {"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)}
            if declare_pruned_source_x
            else {}
        ),
        state_transitions=(
            {"x": fixed_transition("x")} if declare_pruned_source_x else {}
        ),
        edges={
            "src": Transition(
                law=ByAge(cases={0: {"target": StochasticTransition(func=certain)}}),
                gates={"target": Gate(predicate=gate_with_free_x, routes=routes())},
            )
        },
    )


@pytest.mark.parametrize("declare_pruned_source_x", [False, True])
@pytest.mark.parametrize(("threshold", "expected"), [(0.5, 1.9), (1.5, 1.45)])
def test_gate_free_param_is_not_a_source_only_declared_state(
    *, declare_pruned_source_x, threshold, expected
):
    """A name only the source declares stays the gate's own free parameter.

    The gate opens when the target's value 1 reaches `x`, giving 1 + 0.9 * 1;
    otherwise the fallback's 0.5 gives 1 + 0.9 * 0.5.
    """
    model = source_name_gate_model(declare_pruned_source_x=declare_pruned_source_x)
    assert "x" not in model.user_regimes["target"].states
    assert "x" not in model.user_regimes["src"].states
    assert leaf_paths(tree=model.get_params_template()["edges"]) == {
        ("src", "target", "predicate", "x")
    }
    values = model.solve(
        params={
            "discount_factor": 0.9,
            "edges": {"src": {"target": {"predicate": {"x": threshold}}}},
        },
        log_level="debug",
    ).values
    np.testing.assert_allclose(np.asarray(values[0]["src"]), expected, rtol=1e-6)
