"""Argument schemas must agree only across the regime-law phases a model demands.

A regime that is valued but never visited demands only its solve law; the
realized (simulate) side of a `Phased` law at such an age is dormant and must
not contribute arguments, parameters or annotation conflicts.
"""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    Choose,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ModelInitializationError
from lcm.phased import Phased
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


def _wealth_regime(*, law: Any, utility: UserFunction = _wealth) -> Regime:
    return Regime(
        regime_transitions=law,
        states={"wealth": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={} if law is None else {"wealth": fixed_transition("wealth")},
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
        cases={
            age: Phased(
                solve="end", simulate=Choose(func=choice, targets=("end", "other_end"))
            )
            for age, choice in choices
        }
    )
    roots: dict[object, str] = {(0, 1): "source"}
    if promote:
        roots[promote] = "perceived"
    return Model(
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=DemandId,
        initial_regimes=roots,
        enable_jit=enable_jit,
        execution_config=ExecutionConfig(device_memory_bytes=None),
        regimes={
            "source": _wealth_regime(
                law=ByAge(
                    cases={(0, 1): Phased(solve="perceived", simulate="realized")}
                )
            ),
            "perceived": _wealth_regime(law=perceived, utility=_bonus),
            "realized": _wealth_regime(law=ByAge(cases={(1, 2): "end"})),
            "end": _wealth_regime(law=None),
            "other_end": _wealth_regime(law=None),
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


def _params(*, promote: tuple[int, ...]) -> dict[str, Any]:
    params: dict[str, Any] = {"discount_factor": 0.5, "backward_bonus": 1.0}
    if promote:
        params["realized_rate"] = 1 if promote == (2,) else 1.0
    return params


def test_dormant_realized_annotations_do_not_reject_a_backward_problem() -> None:
    """A valued-only regime needs only its backward parameters."""
    model = _demand_model(enable_jit=False)
    assert _leaves(tree=model.get_params_template()["perceived"]) == {
        "backward_bonus",
        "discount_factor",
    }


@pytest.mark.parametrize("reverse_cases", [False, True])
@pytest.mark.parametrize("promote", [(), (1,), (2,)])
def test_promoted_ages_add_exactly_their_realized_parameter(
    *, promote: tuple[int, ...], reverse_cases: bool
) -> None:
    """Visiting one perceived age requires that age's realized schema alone."""
    model = _demand_model(promote=promote, reverse_cases=reverse_cases)
    expected = {"discount_factor", "backward_bonus"} | (
        {"realized_rate"} if promote else set()
    )
    assert _leaves(tree=model.get_params_template()["perceived"]) == expected


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
    assert "realized_rate" in _leaves(tree=model.get_params_template()["perceived"])
