"""Lottery contraction at the edges of the value domain, through the public solve.

One IID binary draw `z` with a deterministic landing on a two-node wealth grid, so
every expectation is computable by hand:

- `finite`: rows `(M, 1)` and `(-M, 1)` with `M = 2**60` (`2**30` in float32). The
  exact expectation at wealth `c` is `c`. The contracted table is `(0, 1)` and returns
  it exactly; enumeration rounds each draw's read to `+-M/2` first and loses it in the
  interior, so the cancellation cases compare against the exact value only.
- `extended`: rows `(-inf, 0)` and `(0, +inf)`. Interpolation makes a read `-inf`
  whenever a corner under nonzero weight is `-inf`. Contraction applies that rule
  across draws as well: where a live draw holds `-inf`, the table is `-inf` and so is
  the read. Enumeration averages the draws' reads instead, and in the interior that
  is `-inf + inf = NaN`. A next-period value of `+inf` is outside the supported
  domain, so the contracted `-inf` is the pinned behavior and the enumerated NaN is
  recorded only to show where the routes part.
- `dead_poison`: a zero-probability draw holding NaN and `+inf` drops out of both.

The transposed cancellation goes the other way. With the large values on one draw's
row, `(M, -M, 1/4)` and `(1, 1, 1/4)` on a three-node grid, enumeration cancels the
large row exactly and keeps the small term, while the contracted table rounds
`(+-M + 1) / 2` to `+-M/2` and loses it. Neither summation order is the accurate one;
each is accurate only relative to the largest next-period value across the draws.
There an action choosing between landing at 1/2 and at 2 picks the midpoint
(exact 0.5) enumerated and the endpoint (0.25) contracted.
"""

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    StochasticTransition,
    categorical,
)
from lcm.regime import Regime
from lcm.solvers import GridSearch
from lcm.typing import (
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
)

type _Case = Literal["finite", "extended", "dead_poison"]


@categorical(ordered=False)
class _Binary:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _Regimes:
    alive: ScalarInt
    final: ScalarInt


def _utility(*, wealth: ContinuousState, z: DiscreteState) -> FloatND:
    # Reads both states so the regime declares no unused state; adds exactly zero.
    return jnp.zeros_like(wealth * 0.0 + z * 0.0)


def _finite_table_utility(*, wealth: ContinuousState, z: DiscreteState) -> FloatND:
    # At either precision, adding 0.5 to +-M/2 rounds it away.
    exponent = 60 if jax.config.jax_enable_x64 else 30
    magnitude = jnp.asarray(float(2**exponent))
    signed = jnp.where(z == 0, magnitude, -magnitude)
    return jnp.where(wealth == 0, signed, jnp.asarray(1.0))


def _extended_table_utility(*, wealth: ContinuousState, z: DiscreteState) -> FloatND:
    return jnp.where(
        z == 0,
        jnp.where(wealth == 0, -jnp.inf, 0.0),
        jnp.where(wealth == 0, 0.0, jnp.inf),
    )


def _poison_table_utility(*, wealth: ContinuousState, z: DiscreteState) -> FloatND:
    return jnp.where(
        z == 0, 2.0 + 2.0 * wealth, jnp.where(wealth == 0, jnp.nan, jnp.inf)
    )


_TERMINAL_UTILITIES: dict[_Case, Callable[..., FloatND]] = {
    "finite": _finite_table_utility,
    "extended": _extended_table_utility,
    "dead_poison": _poison_table_utility,
}


def _model(
    *,
    contract: bool,
    case: _Case,
    enable_jit: bool,
    coordinate: float,
    first_probability: float,
    reverse: bool,
) -> Model:
    states = {
        "wealth": LinSpacedGrid(start=0.0, stop=1.0, n_points=2),
        "z": DiscreteGrid(category_class=_Binary),
    }
    terminal = _TERMINAL_UTILITIES[case]

    def next_wealth() -> ContinuousState:
        return jnp.asarray(coordinate)

    def probabilities() -> FloatND:
        return jnp.asarray([first_probability, 1.0 - first_probability])

    def final_utility(*, wealth: ContinuousState, z: DiscreteState) -> FloatND:
        return terminal(wealth=wealth, z=1 - z if reverse else z)

    alive = Regime(
        states=states,
        state_transitions={
            "wealth": next_wealth,
            "z": StochasticTransition(func=probabilities),
        },
        functions={"utility": _utility},
        solver=GridSearch(lottery_contraction=contract),
    )
    final = Regime(states=states, functions={"utility": final_utility})
    return Model(
        regimes={"alive": alive, "final": final},
        regime_id_class=_Regimes,
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes=((0, "alive"),),
        edges={"alive": {"final": 0}},
        enable_jit=enable_jit,
        durable_identity=False,
    )


def _alive_values(
    *,
    contract: bool,
    case: _Case,
    enable_jit: bool,
    coordinate: float,
    first_probability: float,
    reverse: bool,
) -> np.ndarray:
    model = _model(
        contract=contract,
        case=case,
        enable_jit=enable_jit,
        coordinate=coordinate,
        first_probability=first_probability,
        reverse=reverse,
    )
    result = model.solve(
        params={
            "alive": {"koopmans_aggregator": {"discount_factor": 1.0}},
            "final": {},
        },
        log_level="off",
        # The extended cases' enumerated NaN would be rejected by the runtime check.
        runtime_checks=case == "finite",
    )
    return np.asarray(result.values[0]["alive"])


def _assert_all_equal(*, values: np.ndarray, expected: float) -> None:
    """Every alive state reads one expectation: the draw is IID, the landing fixed."""
    np.testing.assert_array_equal(values, np.full(values.shape, expected, values.dtype))


_CASES = [
    # case, coordinate, first probability, reverse, jit, contracted, enumerated
    pytest.param("finite", 0.5, 0.5, False, True, 0.5, None, id="finite-cancel-jit"),
    pytest.param("finite", 0.5, 0.5, False, False, 0.5, None, id="finite-cancel-eager"),
    pytest.param("finite", 0.5, 0.5, True, True, 0.5, None, id="finite-permutation"),
    pytest.param("finite", 0.0, 0.5, False, True, 0.0, 0.0, id="finite-left-end"),
    pytest.param("finite", 1.0, 0.5, False, True, 1.0, 1.0, id="finite-right-end"),
    pytest.param(
        "extended", 0.5, 0.5, False, True, -np.inf, np.nan, id="extended-interior-jit"
    ),
    pytest.param(
        "extended",
        0.5,
        0.5,
        False,
        False,
        -np.inf,
        np.nan,
        id="extended-interior-eager",
    ),
    pytest.param(
        "extended", 0.5, 0.25, False, True, -np.inf, np.nan, id="extended-asymmetric"
    ),
    pytest.param(
        "extended", 0.0, 0.5, False, True, -np.inf, -np.inf, id="extended-left-end"
    ),
    pytest.param(
        "extended", 1.0, 0.5, False, True, np.inf, np.inf, id="extended-right-end"
    ),
    pytest.param(
        "extended", 0.5, 0.0, False, True, np.inf, np.inf, id="extended-only-high"
    ),
    pytest.param(
        "extended", 0.5, 1.0, False, True, -np.inf, -np.inf, id="extended-only-low"
    ),
    pytest.param(
        "dead_poison", 0.5, 1.0, False, True, 3.0, 3.0, id="dead-nan-inf-draw"
    ),
]


@pytest.mark.parametrize(
    (
        "case",
        "coordinate",
        "first_probability",
        "reverse",
        "enable_jit",
        "contracted",
        "enumerated",
    ),
    _CASES,
)
def test_contraction_pins_edge_tables_of_the_value_domain(
    *,
    case: _Case,
    coordinate: float,
    first_probability: float,
    reverse: bool,
    enable_jit: bool,
    contracted: float,
    enumerated: float | None,
) -> None:
    """The contracted solve returns the hand-computed value; `None` skips enumeration.

    Where both are given and differ, the case shows where the routes part: only a
    `+inf` meeting a `-inf` across draws, which lies outside the supported domain.
    """
    arguments = {
        "case": case,
        "enable_jit": enable_jit,
        "coordinate": coordinate,
        "first_probability": first_probability,
        "reverse": reverse,
    }
    _assert_all_equal(
        values=_alive_values(contract=True, **arguments), expected=contracted
    )
    if enumerated is not None:
        _assert_all_equal(
            values=_alive_values(contract=False, **arguments), expected=enumerated
        )


def _transposed_terminal_utility(
    *, wealth: ContinuousState, z: DiscreteState
) -> FloatND:
    # Draw 0 holds `(M, -M, 1/4)` and draw 1 `(1, 1, 1/4)` at wealth 0, 1 and 2.
    exponent = 60 if jax.config.jax_enable_x64 else 30
    magnitude = jnp.asarray(float(2**exponent))
    large_row = jnp.where(wealth == 0, magnitude, -magnitude)
    small_row = jnp.asarray(1.0)
    return jnp.where(wealth == 2, 0.25, jnp.where(z == 0, large_row, small_row))


def _transposed_alive_utility(
    *, wealth: ContinuousState, z: DiscreteState, landing: DiscreteAction
) -> FloatND:
    # Reads every state and the action so none is unused; adds exactly zero.
    return jnp.zeros_like(wealth * 0.0 + z * 0.0 + landing * 0.0)


def _next_wealth_by_landing(*, landing: DiscreteAction) -> ContinuousState:
    return jnp.where(landing == 0, 0.5, 2.0)


def _equal_probabilities() -> FloatND:
    return jnp.asarray([0.5, 0.5])


def _transposed_model(*, contract: bool) -> Model:
    states = {
        "wealth": LinSpacedGrid(start=0.0, stop=2.0, n_points=3),
        "z": DiscreteGrid(category_class=_Binary),
    }
    alive = Regime(
        states=states,
        actions={"landing": DiscreteGrid(category_class=_Binary)},
        state_transitions={
            "wealth": _next_wealth_by_landing,
            "z": StochasticTransition(func=_equal_probabilities),
        },
        functions={"utility": _transposed_alive_utility},
        solver=GridSearch(lottery_contraction=contract),
    )
    final = Regime(states=states, functions={"utility": _transposed_terminal_utility})
    return Model(
        regimes={"alive": alive, "final": final},
        regime_id_class=_Regimes,
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes=((0, "alive"),),
        edges={"alive": {"final": 0}},
        durable_identity=False,
    )


@pytest.mark.parametrize(
    ("contract", "expected"),
    [
        pytest.param(False, 0.5, id="enumerated-keeps-the-small-term"),
        pytest.param(True, 0.25, id="contracted-loses-the-small-term"),
    ],
)
def test_transposed_cancellation_loses_the_small_term_when_contracted(
    *, contract: bool, expected: float
) -> None:
    """The exact value is 0.5 at the midpoint; contracted, the endpoint's 0.25 wins.

    Enumeration reads the midpoint exactly here, the transpose of the
    `finite-cancel-*` cases, where contraction does. The contracted 0.25 is
    recorded, so both orientations stay visible.
    """
    result = _transposed_model(contract=contract).solve(
        params={
            "alive": {"koopmans_aggregator": {"discount_factor": 1.0}},
            "final": {},
        },
        log_level="off",
    )
    _assert_all_equal(values=np.asarray(result.values[0]["alive"]), expected=expected)
