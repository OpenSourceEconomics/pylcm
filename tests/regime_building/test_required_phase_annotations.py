"""Argument schemas must agree only across the regime-law phases a model demands.

A regime that is valued but never visited demands only its solve law; the
realized (simulate) side of a `Phased` law at such an age is dormant and must
not contribute arguments of the regime's own branch or annotation conflicts.
The declared law's parameters sit at `params["edges"][source]` whether or not
the regime is visited.
"""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    DeterministicTransition,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    Regime,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ModelInitializationError
from lcm.phased import Phased
from lcm.transition import AgeSelector
from lcm.typing import ContinuousState, FloatND, IntND, ScalarInt, UserFunction


@categorical(ordered=False)
class DemandId:
    source: ScalarInt
    perceived: ScalarInt
    realized: ScalarInt
    end: ScalarInt
    other_end: ScalarInt


def _wealth(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _bonus(*, wealth: ContinuousState, backward_bonus: float) -> FloatND:
    return wealth + backward_bonus


def _float_choice(*, realized_rate: float) -> IntND:
    return jnp.where(realized_rate >= 0.5, DemandId.end, DemandId.other_end)


def _int_choice(*, realized_rate: int) -> IntND:
    return jnp.where(realized_rate >= 1, DemandId.end, DemandId.other_end)


def _wealth_regime(*, terminal: bool, utility: UserFunction = _wealth) -> Regime:
    return Regime(
        states={"wealth": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={} if terminal else {"wealth": fixed_transition("wealth")},
        functions={"utility": utility},
    )


def _demand_model(
    *,
    promote: tuple[int, ...] = (),
    reverse_cases: bool = False,
    enable_jit: bool = True,
    compatible: bool = False,
) -> Model:
    """`perceived` is valued at ages 1 and 2 but visited only at promoted ages.

    Its realized choices declare `realized_rate` as float at age 1 and as int at
    age 2 (float at both when `compatible`).
    """
    choices = ((1, _float_choice), (2, _float_choice if compatible else _int_choice))
    if reverse_cases:
        choices = choices[::-1]
    perceived = ByAge(
        cases={age: DeterministicTransition(func=choice) for age, choice in choices}
    )
    roots: dict[AgeSelector, str] = {(0, 1): "source"}
    if promote:
        roots[promote] = "perceived"
    return Model(
        edges=Phased(
            solve={
                "source": {"perceived": (0, 1)},
                "perceived": {"end": (1, 2)},
                "realized": {"end": (1, 2)},
            },
            simulate={
                "source": {"realized": (0, 1)},
                "perceived": Transition(
                    targets={"end": (1, 2), "other_end": (1, 2)}, law=perceived
                ),
                "realized": {"end": (1, 2)},
            },
        ),
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=DemandId,
        initial_nodes=roots,
        enable_jit=enable_jit,
        execution_config=ExecutionConfig(device_memory_bytes=None),
        regimes={
            "source": _wealth_regime(terminal=False),
            "perceived": _wealth_regime(terminal=False, utility=_bonus),
            "realized": _wealth_regime(terminal=False),
            "end": _wealth_regime(terminal=True),
            "other_end": _wealth_regime(terminal=True),
        },
    )


def _leaves(*, tree: object) -> set[str]:
    if not isinstance(tree, Mapping):
        return set()
    return {
        name
        for key, value in tree.items()
        for name in (_leaves(tree=value) if isinstance(value, Mapping) else {key})
    }


def _edge_leaves(*, model: Model) -> set[str]:
    return _leaves(tree=model.get_params_template().get("edges", {}).get("perceived"))


def _params(*, promote: tuple[int, ...]) -> dict[str, Any]:
    return {
        "discount_factor": 0.5,
        "backward_bonus": 1.0,
        "realized_rate": 1 if promote == (2,) else 1.0,
    }


def test_dormant_realized_annotations_do_not_reject_a_backward_problem() -> None:
    """A valued-only regime needs only its backward parameters."""
    model = _demand_model(enable_jit=False)
    assert _leaves(tree=model.get_params_template()["perceived"]) == {
        "backward_bonus",
        "discount_factor",
    }


@pytest.mark.parametrize("reverse_cases", [False, True])
@pytest.mark.parametrize("promote", [(), (1,), (2,)])
def test_realized_parameter_is_an_edge_parameter_at_every_promotion(
    *, promote: tuple[int, ...], reverse_cases: bool
) -> None:
    """`perceived` owes its backward parameters; its edge holds `realized_rate`."""
    model = _demand_model(promote=promote, reverse_cases=reverse_cases)
    assert (
        _leaves(tree=model.get_params_template()["perceived"]),
        _edge_leaves(model=model),
    ) == ({"discount_factor", "backward_bonus"}, {"realized_rate"})


@pytest.mark.parametrize("promote", [(), (1,), (2,)])
def test_promoted_ages_are_exactly_the_extra_visited_nodes(
    *, promote: tuple[int, ...]
) -> None:
    """Physical visitation grows only by the promoted perceived ages."""
    model = _demand_model(promote=promote)
    physical = {
        (0, "source"),
        (1, "source"),
        (1, "realized"),
        (2, "realized"),
        (2, "end"),
        (3, "end"),
    }
    for period in promote:
        physical |= {(period, "perceived"), (period + 1, "other_end")}
    assert model.reachability.visited_nodes == frozenset(physical)


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("reverse_cases", [False, True])
@pytest.mark.parametrize("promote", [(), (1,), (2,)])
def test_source_values_follow_the_perceived_backward_law(
    *, promote: tuple[int, ...], reverse_cases: bool, enable_jit: bool
) -> None:
    """`V_source(w) = w + (w + 1 + w / 2) / 2` on wealth {0, 1}."""
    model = _demand_model(
        promote=promote, reverse_cases=reverse_cases, enable_jit=enable_jit
    )
    values = model.solve(params=_params(promote=promote), log_level="off").values
    np.testing.assert_array_equal(
        np.stack([np.asarray(values[period]["source"]) for period in (0, 1)]),
        [[0.5, 2.25], [0.5, 2.25]],
    )


@pytest.mark.parametrize("reverse_cases", [False, True])
def test_promoting_both_conflicting_schemas_raises(*, reverse_cases: bool) -> None:
    """Two demanded realized laws must agree on an argument's annotation."""
    with pytest.raises(ModelInitializationError, match=r"realized_rate.*annotated"):
        _demand_model(promote=(1, 2), reverse_cases=reverse_cases)


def test_promoting_both_compatible_schemas_is_valid() -> None:
    """Two demanded realized laws with one annotation share the parameter."""
    model = _demand_model(promote=(1, 2), compatible=True)
    assert _edge_leaves(model=model) == {"realized_rate"}
