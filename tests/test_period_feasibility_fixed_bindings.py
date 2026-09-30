"""Fixed and free parameter bindings of a feasibility input denote one model.

At age 1 the budget is `consumption <= spending_scale * wealth`, and the regime
law is defined only where `consumption <= wealth` (NaN elsewhere). Whether
`spending_scale` is a runtime parameter or `Model.fixed_params`, validating the
age-1 law masks the same economically feasible rows, so both forms solve to the
same values and an invalid law at a feasible row raises in both.
"""

from collections.abc import Mapping
from fractions import Fraction
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    AgeSpecializedFunction,
    ByAge,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import InvalidRegimeTransitionProbabilitiesError
from lcm.phased import Phased
from lcm.typing import BoolND, ContinuousState, FloatND, ScalarInt, UserFunction


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    left: ScalarInt
    right: ScalarInt


def _consumption(*, consumption: ContinuousState) -> FloatND:
    return consumption


def _ten() -> FloatND:
    return jnp.asarray(10.0)


def _zero() -> FloatND:
    return jnp.asarray(0.0)


def _half() -> FloatND:
    return jnp.asarray(0.5)


def _left(*, wealth: ContinuousState, consumption: ContinuousState) -> FloatND:
    # The raw Cartesian check must fail, but only economically infeasible rows
    # are undefined. Feasibility cannot be manufactured from this validity mask.
    return jnp.where(consumption <= wealth, 0.5, jnp.nan)


def _right(*, wealth: ContinuousState, consumption: ContinuousState) -> FloatND:
    return 1.0 - _left(wealth=wealth, consumption=consumption)


def _bad_left(*, wealth: ContinuousState, consumption: ContinuousState) -> FloatND:
    # This cell is feasible under every scale in this module.
    return jnp.where((wealth == 1.0) & (consumption == 0.0), 1.5, 0.5)


def _bad_right(*, wealth: ContinuousState, consumption: ContinuousState) -> FloatND:
    return 1.0 - _bad_left(wealth=wealth, consumption=consumption)


def _helper_factory(age: float) -> UserFunction:
    def limit(*, wealth: ContinuousState, spending_scale: float) -> ContinuousState:
        return spending_scale * (wealth if age >= 1.0 else jnp.ones_like(wealth))

    return limit


def _constraint_factory(age: float) -> UserFunction:
    def budget(
        *,
        wealth: ContinuousState,
        consumption: ContinuousState,
        spending_scale: float,
    ) -> BoolND:
        return consumption <= spending_scale * (
            wealth if age >= 1.0 else jnp.ones_like(wealth)
        )

    return budget


def _age_signature(age: float) -> float:
    return age


def _uses_limit(
    *,
    consumption: ContinuousState,
    spending_limit: ContinuousState,
) -> BoolND:
    return consumption <= spending_limit


def _leaf_names(*, tree: Any) -> set[str]:
    if not isinstance(tree, Mapping):
        return set()
    return {
        leaf
        for key, value in tree.items()
        for leaf in (_leaf_names(tree=value) if isinstance(value, Mapping) else {key})
    }


def _make_model(
    *,
    representation: Literal["helper", "constraint"],
    fixed: bool,
    spending_scale: float = 1.0,
    earlier_root: bool = False,
    n_points: int = 2,
    enable_jit: bool = False,
    law_phase: Literal["both", "solve", "simulate"] = "both",
    bad_feasible: bool = False,
) -> Model:
    grid = LinSpacedGrid(start=0.0, stop=1.0, n_points=n_points)
    functions: dict[str, Any] = {"utility": _consumption}
    if representation == "helper":
        functions["spending_limit"] = AgeSpecializedFunction(
            build=_helper_factory,
            signature=_age_signature,
        )
        constraints: dict[str, Any] = {"budget": _uses_limit}
    else:
        constraints = {
            "budget": AgeSpecializedFunction(
                build=_constraint_factory,
                signature=_age_signature,
            )
        }
    left, right = (_bad_left, _bad_right) if bad_feasible else (_left, _right)
    checked = {
        "left": MarkovTransition(func=left),
        "right": MarkovTransition(func=right),
    }
    constant = {
        "left": MarkovTransition(func=_half),
        "right": MarkovTransition(func=_half),
    }
    late_law = (
        checked
        if law_phase == "both"
        else Phased(
            solve=checked if law_phase == "solve" else constant,
            simulate=checked if law_phase == "simulate" else constant,
        )
    )
    return Model(
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RegimeId,
        initial_regimes={(0, 1): "working"} if earlier_root else {1: "working"},
        enable_jit=enable_jit,
        fixed_params={"spending_scale": spending_scale} if fixed else {},
        regimes={
            "working": Regime(
                regime_transitions=ByAge(cases={0: "left", 1: late_law}),
                states={"wealth": grid},
                actions={"consumption": grid},
                state_transitions={"wealth": fixed_transition("wealth")},
                functions=functions,
                constraints=constraints,
            ),
            "left": Regime(regime_transitions=None, functions={"utility": _ten}),
            "right": Regime(regime_transitions=None, functions={"utility": _zero}),
        },
    )


def _solve(
    *,
    model: Model,
    fixed: bool,
    spending_scale: float,
    log_level: Literal["off", "warning", "debug"],
) -> FloatND:
    params: dict[str, float] = {"discount_factor": 0.5}
    if not fixed:
        params["spending_scale"] = spending_scale
    return model.solve(params=params, log_level=log_level).values[1]["working"]


@pytest.mark.parametrize("representation", ["helper", "constraint"])
def test_fixed_feasibility_parameter_minimal_witness(
    *, representation: Literal["helper", "constraint"]
) -> None:
    """A fixed `spending_scale = 1` solves to `V_1 = [2.5, 3.5]`."""
    model = _make_model(representation=representation, fixed=True)
    assert "spending_scale" not in _leaf_names(tree=model.get_params_template())
    np.testing.assert_array_equal(
        np.asarray(
            _solve(model=model, fixed=True, spending_scale=1.0, log_level="off")
        ),
        [2.5, 3.5],
    )


@pytest.mark.parametrize("representation", ["helper", "constraint"])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("earlier_root", [False, True])
@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("n_points", [2, 3])
@pytest.mark.parametrize("law_phase", ["both", "solve", "simulate"])
def test_binding_partition_preserves_the_late_problem(
    *,
    representation: Literal["helper", "constraint"],
    fixed: bool,
    earlier_root: bool,
    enable_jit: bool,
    n_points: int,
    law_phase: Literal["both", "solve", "simulate"],
) -> None:
    """Fixed and free `spending_scale = 1` both solve to `V_1(w) = w + 2.5`."""
    model = _make_model(
        representation=representation,
        fixed=fixed,
        earlier_root=earlier_root,
        enable_jit=enable_jit,
        n_points=n_points,
        law_phase=law_phase,
    )
    leaves = _leaf_names(tree=model.get_params_template())
    assert ("spending_scale" in leaves) is (not fixed)
    expected = np.linspace(0.0, 1.0, n_points) + 2.5
    # Same model, same shapes, repeated public calls must remain numerically
    # identical. This asserts values, not a backend-compile count or a budget.
    for _ in range(2):
        np.testing.assert_array_equal(
            np.asarray(
                _solve(model=model, fixed=fixed, spending_scale=1.0, log_level="off")
            ),
            expected,
        )


@pytest.mark.parametrize("representation", ["helper", "constraint"])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("log_level", ["off", "warning", "debug"])
@pytest.mark.parametrize("law_phase", ["both", "solve", "simulate"])
def test_bad_feasible_probability_is_never_excused(
    *,
    representation: Literal["helper", "constraint"],
    fixed: bool,
    log_level: Literal["off", "warning", "debug"],
    law_phase: Literal["both", "solve", "simulate"],
) -> None:
    """An invalid law at a feasible row raises under either binding."""
    model = _make_model(
        representation=representation,
        fixed=fixed,
        law_phase=law_phase,
        earlier_root=True,
        bad_feasible=True,
    )
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError, match="outside "):
        _solve(model=model, fixed=fixed, spending_scale=1.0, log_level=log_level)


@pytest.mark.parametrize("representation", ["helper", "constraint"])
@pytest.mark.parametrize("spending_scale", [0.5, 1.0])
@pytest.mark.parametrize("log_level", ["off", "warning", "debug"])
def test_equal_shapes_with_different_fixed_bindings_keep_their_meaning(
    *,
    representation: Literal["helper", "constraint"],
    spending_scale: float,
    log_level: Literal["off", "warning", "debug"],
) -> None:
    """`V_1(w) = max_{c <= s w} c + 2.5` for fixed and free `s`."""
    # Dyadic values and grid; explicit literal maximization, no solver helpers.
    grid = np.array([0.0, 0.5, 1.0])
    expected = np.array(
        [max(c for c in grid if c <= spending_scale * w) + 2.5 for w in grid]
    )
    results = []
    for fixed in (False, True):
        model = _make_model(
            representation=representation,
            fixed=fixed,
            spending_scale=spending_scale,
            earlier_root=True,
            n_points=3,
        )
        results.append(
            np.asarray(
                _solve(
                    model=model,
                    fixed=fixed,
                    spending_scale=spending_scale,
                    log_level=log_level,
                )
            )
        )
        np.testing.assert_array_equal(results[-1], expected)
    np.testing.assert_array_equal(results[0], results[1])


@pytest.mark.parametrize("representation", ["helper", "constraint"])
@pytest.mark.parametrize("reverse", [False, True])
def test_fixed_binding_order_does_not_change_equal_shape_economics(
    *,
    representation: Literal["helper", "constraint"],
    reverse: bool,
) -> None:
    """Independent literal maxima; no production feasibility or value helper."""
    grid = tuple(Fraction(i, 4) for i in range(5))
    scales = (Fraction(1, 2), Fraction(1), Fraction(1, 2))
    if reverse:
        scales = (Fraction(1), Fraction(1, 2), Fraction(1))
    models = {
        (fixed, scale): _make_model(
            representation=representation,
            fixed=fixed,
            spending_scale=float(scale),
            n_points=5,
            earlier_root=True,
        )
        for fixed in (False, True)
        for scale in set(scales)
    }
    # Revisiting a previously used model checks binding independence, not a
    # compile-count/memory budget. The latter needs dedicated instrumentation.
    for scale in scales:
        expected = np.asarray(
            [
                float(max(c for c in grid if c <= scale * w) + Fraction(5, 2))
                for w in grid
            ]
        )
        for fixed in (False, True):
            actual = _solve(
                model=models[(fixed, scale)],
                fixed=fixed,
                spending_scale=float(scale),
                log_level="off",
            )
            np.testing.assert_array_equal(np.asarray(actual), expected)
