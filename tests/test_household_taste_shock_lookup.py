"""Policy lookup in a household with EV1 taste shocks and one decider.

The member with the positive Pareto weight decides. Per queried state row, the
lookup publishes for every discrete-action combination `d`, in C order:

- `Qc(d)`, the decider's maximum over the feasible continuous actions;
- the continuous actions `c*(d)` attaining it, which every member shares;
- the choice probability `softmax(Qc / scale)(d)`.

The value has one entry per member: the decider's smoothed maximum
`scale * logsumexp(Qc / scale)`, and every other member's mean action value
`sum_d p(d) Q(d, c*(d))` under those probabilities. No action is returned.

The terminal models are small enough for a literal enumeration in plain Python,
which is the reference here. The non-terminal toy is checked against its solved
values at the grid rows and against the single it is written from.
"""

import functools
import math
from collections.abc import Callable, Mapping
from itertools import product

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from lcm import (
    CollectiveUtility,
    DiscreteGrid,
    InitialNodes,
    LinSpacedGrid,
    Model,
    ParetoObjective,
    PolicyLookup,
    Regime,
    categorical,
)
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
    UserFunction,
)
from tests.conftest import DECIMAL_PRECISION, assert_agrees_to_ulp
from tests.test_models import taste_shocks_toy

_SCALE = 0.5
_ROWS = (0.25, 0.875, 1.0, 1.125)
_AMOUNTS = (0.0, 0.25, 0.5, 0.75, 1.0)


@categorical(ordered=False)
class RegimeId:
    agent: ScalarInt


@categorical(ordered=True)
class Binary:
    low: ScalarInt
    high: ScalarInt


def x_discrete(*, wealth: ContinuousState, choice: DiscreteAction) -> FloatND:
    return wealth + 2.0 * choice


def y_discrete(*, wealth: ContinuousState, choice: DiscreteAction) -> FloatND:
    return 3.0 * wealth - choice


def x_continuous(
    *, wealth: ContinuousState, choice: DiscreteAction, amount: ContinuousAction
) -> FloatND:
    return wealth + 2.0 * choice - (amount - (0.25 + 0.5 * choice)) ** 2


def y_continuous(
    *, wealth: ContinuousState, choice: DiscreteAction, amount: ContinuousAction
) -> FloatND:
    return 3.0 * wealth - choice + 0.75 * amount


def x_two_axes(
    *,
    wealth: ContinuousState,
    choice: DiscreteAction,
    other: DiscreteAction,
    amount: ContinuousAction,
) -> FloatND:
    return wealth + 2.0 * choice - 0.5 * other - (amount - (0.25 + 0.5 * choice)) ** 2


def y_two_axes(
    *,
    wealth: ContinuousState,
    choice: DiscreteAction,
    other: DiscreteAction,
    amount: ContinuousAction,
) -> FloatND:
    return 3.0 * wealth - choice + 1.5 * other + 0.75 * amount


def feasible(*, wealth: ContinuousState, amount: ContinuousAction) -> BoolND:
    return amount <= wealth


def weight_x(*, wealth: ContinuousState, threshold: float) -> FloatND:
    return jnp.where(wealth < threshold, 1.0, 0.0)


def weight_y(*, wealth: ContinuousState, threshold: float) -> FloatND:
    return jnp.where(wealth < threshold, 0.0, 1.0)


_UTILITIES: Mapping[str, Mapping[str, UserFunction]] = {
    "discrete": {"x": x_discrete, "y": y_discrete},
    "continuous": {"x": x_continuous, "y": y_continuous},
    "two_axes": {"x": x_two_axes, "y": y_two_axes},
}

# Case name to member names, constant weights or `None`, and the action layout. Without
# constant weights, `x` decides below the weights' `threshold` parameter and `y`
# from it on; the parameter is 1.
_CASES: Mapping[str, tuple[tuple[str, ...], Mapping[str, float] | None, str]] = {
    "x_decides": (("x", "y"), {"x": 1.0, "y": 0.0}, "discrete"),
    "y_decides": (("x", "y"), {"x": 0.0, "y": 1.0}, "discrete"),
    "members_reordered": (("y", "x"), {"y": 0.0, "x": 1.0}, "discrete"),
    "non_unit_weight": (("x", "y"), {"x": 8.0, "y": 0.0}, "discrete"),
    "infeasible_amounts": (("x", "y"), {"x": 1.0, "y": 0.0}, "continuous"),
    "partner_decides_amount": (("x", "y"), {"x": 0.0, "y": 1.0}, "continuous"),
    "weight_function": (("x", "y"), None, "continuous"),
    "two_discrete_axes": (("x", "y"), {"x": 1.0, "y": 0.0}, "two_axes"),
}
_CONTINUOUS_CASES = [case for case, spec in _CASES.items() if spec[2] != "discrete"]


def _terminal_model(
    *, member_names: tuple[str, ...], weights: Mapping[str, float] | None, layout: str
) -> Model:
    """Return a one-period household of the named members."""
    utilities = _UTILITIES[layout]
    declared: Mapping[str, UserFunction | float] = (
        {"x": weight_x, "y": weight_y} if weights is None else weights
    )
    actions: dict[str, DiscreteGrid | LinSpacedGrid] = {
        "choice": DiscreteGrid(category_class=Binary)
    }
    if layout == "two_axes":
        actions["other"] = DiscreteGrid(category_class=Binary)
    if layout != "discrete":
        actions["amount"] = LinSpacedGrid(start=0.0, stop=1.0, n_points=5)
    regime = Regime(
        states={"wealth": LinSpacedGrid(start=0.25, stop=2.0, n_points=8)},
        actions=actions,
        functions={
            "utility": CollectiveUtility(
                utilities={name: utilities[name] for name in member_names},
                objective=ParetoObjective(
                    weights={name: declared[name] for name in member_names}
                ),
            )
        },
        constraints={"feasible": feasible} if layout != "discrete" else {},
        taste_shocks=ExtremeValueTasteShocks(),
    )
    return Model(
        n_periods=1,
        regimes={"agent": regime},
        regime_id_class=RegimeId,
        initial_nodes=InitialNodes(by_period={0: "agent"}),
        edges={},
    )


def _member_values(
    *, layout: str, wealth: float, choice: int, other: int, amount: float
) -> dict[str, float]:
    """Return each member's utility at one action, written out by hand."""
    if layout == "discrete":
        return {"x": wealth + 2.0 * choice, "y": 3.0 * wealth - choice}
    x_target = 0.25 + 0.5 * choice
    return {
        "x": wealth + 2.0 * choice - 0.5 * other - (amount - x_target) ** 2,
        "y": 3.0 * wealth - choice + 1.5 * other + 0.75 * amount,
    }


def _reference(
    *,
    wealth: float,
    member_names: tuple[str, ...],
    weights: Mapping[str, float] | None,
    layout: str,
) -> dict[str, list[float]]:
    """Enumerate every action literally and read the lookup's outputs off it."""
    if weights is None:
        decider = "x" if wealth < 1.0 else "y"
    else:
        decider = next(name for name in member_names if weights[name] > 0)
    others = (0, 1) if layout == "two_axes" else (0,)
    amounts = (0.0,) if layout == "discrete" else _AMOUNTS
    conditional: list[tuple[float, float, dict[str, float]]] = []
    for choice, other in product((0, 1), others):
        best: tuple[float, float, dict[str, float]] | None = None
        for amount in amounts:
            if layout != "discrete" and amount > wealth:
                continue
            values = _member_values(
                layout=layout, wealth=wealth, choice=choice, other=other, amount=amount
            )
            # The first maximizer wins a tie.
            if best is None or values[decider] > best[0]:
                best = (values[decider], amount, values)
        assert best is not None
        conditional.append(best)
    ceiling = max(value for value, _, _ in conditional)
    terms = [math.exp((value - ceiling) / _SCALE) for value, _, _ in conditional]
    probabilities = [term / sum(terms) for term in terms]
    smoothed = ceiling + _SCALE * math.log(sum(terms))
    return {
        "value": [
            smoothed
            if name == decider
            else sum(
                p * values[name]
                for p, (_, _, values) in zip(probabilities, conditional, strict=True)
            )
            for name in member_names
        ],
        "probabilities": probabilities,
        "Qc": [value for value, _, _ in conditional],
        "amount": [amount for _, amount, _ in conditional],
    }


@functools.cache
def _terminal_lookup(case: str) -> PolicyLookup:
    """Solve one case and look it up at `_ROWS`, once per session."""
    member_names, weights, layout = _CASES[case]
    model = _terminal_model(member_names=member_names, weights=weights, layout=layout)
    params = {
        "agent": {
            "taste_shocks": {"scale": _SCALE},
            **({} if weights is not None else {"pareto_objective": {"threshold": 1.0}}),
        }
    }
    return model.lookup_policy(
        params=params,
        solution=model.solve(params=params, log_level="off"),
        period=0,
        regime_name="agent",
        states={"wealth": jnp.asarray(_ROWS)},
    )


def _terminal_expected(*, case: str, output: str) -> np.ndarray:
    member_names, weights, layout = _CASES[case]
    return np.array(
        [
            _reference(
                wealth=w, member_names=member_names, weights=weights, layout=layout
            )[output]
            for w in _ROWS
        ]
    )


def _flat_combinations(array: FloatND | None) -> np.ndarray:
    """Return a per-combination output with its discrete axes flattened in C order."""
    assert array is not None
    return np.asarray(array).reshape(len(_ROWS), -1)


def test_lookup_of_the_minimal_household_has_the_closed_form() -> None:
    """At wealth 1 and scale 1/2, `x = w + 2d` decides and `y = 3w - d` follows.

    `p(d=1) = e^4 / (1 + e^4)`; the decider's value is `1 + log(1 + e^4) / 2`,
    the partner's `3 - p(d=1)`.
    """
    p1 = math.exp(4.0) / (1.0 + math.exp(4.0))
    lookup = _terminal_lookup("x_decides")
    row = _ROWS.index(1.0)
    aaae(
        np.concatenate(
            [
                np.asarray(lookup.value)[row],
                _flat_combinations(lookup.choice_probabilities)[row],
            ]
        ),
        [1.0 + 0.5 * math.log(1.0 + math.exp(4.0)), 3.0 - p1, 1.0 - p1, p1],
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("case", list(_CASES))
def test_lookup_values_match_the_enumeration(case: str) -> None:
    """The decider's value is its logsum, every other member's its expected value."""
    aaae(
        np.asarray(_terminal_lookup(case).value),
        _terminal_expected(case=case, output="value"),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("case", list(_CASES))
def test_lookup_probabilities_match_the_enumeration(case: str) -> None:
    """The decider's choice probabilities, one per discrete combination in C order."""
    aaae(
        _flat_combinations(_terminal_lookup(case).choice_probabilities),
        _terminal_expected(case=case, output="probabilities"),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("case", list(_CASES))
def test_lookup_conditional_values_are_the_deciders(case: str) -> None:
    """`Qc` is the decider's own maximum over the feasible continuous actions."""
    aaae(
        _flat_combinations(_terminal_lookup(case).Qc),
        _terminal_expected(case=case, output="Qc"),
        decimal=DECIMAL_PRECISION,
    )


@pytest.mark.parametrize("case", _CONTINUOUS_CASES)
def test_lookup_conditional_actions_are_the_deciders_argmax(case: str) -> None:
    """The amount every member shares is the decider's first best feasible one."""
    conditional_actions = _terminal_lookup(case).conditional_actions
    assert conditional_actions is not None
    np.testing.assert_array_equal(
        _flat_combinations(conditional_actions["amount"]),
        _terminal_expected(case=case, output="amount"),
    )


def test_lookup_of_a_household_returns_no_actions() -> None:
    """The discrete choice is random, so the lookup returns no single action."""
    assert _terminal_lookup("infeasible_amounts").actions is None


_TOY_SCALE = 0.2
_TOY_ROWS = 6
_TOY_HOUSEHOLD_WEIGHTS: Mapping[str, Mapping[str, float]] = {
    "one": {"x": 1.0},
    "x_decides": {"x": 1.0, "y": 0.0},
    "y_decides": {"x": 0.0, "y": 1.0},
}


@functools.cache
def _toy_lookup(variant: str) -> tuple[PolicyLookup, np.ndarray]:
    """Return the toy's period-0 lookup at its grid rows and its solved values.

    `"single"` is the toy itself, every other variant a household of the toy.
    """
    if variant == "single":
        model = taste_shocks_toy.get_model()
        params = taste_shocks_toy.get_params(scale=_TOY_SCALE)
    else:
        model = taste_shocks_toy.get_household_model(
            weights=_TOY_HOUSEHOLD_WEIGHTS[variant]
        )
        params = taste_shocks_toy.get_household_params(scale=_TOY_SCALE)
    solution = model.solve(params=params, log_level="off")
    lookup = model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="alive",
        states={"wealth": jnp.linspace(1.0, 10.0, _TOY_ROWS)},
        keep_intermediates=("Q", "F", "U", "CE"),
    )
    return lookup, np.asarray(solution.values[0]["alive"])


def _folded(part: FloatND | BoolND | None) -> FloatND:
    """One row per state of an action part, a household's member axis folded in."""
    return jnp.reshape(jnp.asarray(part, dtype=float), (_TOY_ROWS, -1))


_TWIN_OUTPUTS: Mapping[str, Callable[[PolicyLookup], FloatND | None]] = {
    "value": lambda lookup: jnp.reshape(lookup.value, (_TOY_ROWS,)),
    "choice_probabilities": lambda lookup: lookup.choice_probabilities,
    "Qc": lambda lookup: lookup.Qc,
    "Q": lambda lookup: _folded(lookup.Q),
    "F": lambda lookup: _folded(lookup.F),
    "U": lambda lookup: _folded(lookup.U),
    "CE": lambda lookup: _folded(lookup.CE),
}


@pytest.mark.parametrize("output", list(_TWIN_OUTPUTS))
def test_lookup_of_a_household_of_one_agrees_with_the_single(output: str) -> None:
    """A household of one with weight 1 publishes the single's lookup.

    The toy is non-terminal, so the value prices the solved continuation.
    """
    read = _TWIN_OUTPUTS[output]
    assert_agrees_to_ulp(
        got=np.asarray(read(_toy_lookup("one")[0])),
        expected=np.asarray(read(_toy_lookup("single")[0])),
        n_ulp=16,
    )


def test_lookup_of_a_household_of_one_shares_the_singles_conditional_actions() -> None:
    """The conditional consumption of the household of one is the single's, exactly."""
    household = _toy_lookup("one")[0].conditional_actions
    single = _toy_lookup("single")[0].conditional_actions
    assert household is not None
    assert single is not None
    np.testing.assert_array_equal(
        np.asarray(household["consumption"]), np.asarray(single["consumption"])
    )


@pytest.mark.parametrize("variant", ["x_decides", "y_decides"])
def test_lookup_of_a_household_at_grid_rows_is_the_solved_value(variant: str) -> None:
    """At the grid rows of a non-terminal period, the lookup values are the solve's."""
    lookup, solved = _toy_lookup(variant)
    aaae(np.asarray(lookup.value), solved, decimal=DECIMAL_PRECISION)
