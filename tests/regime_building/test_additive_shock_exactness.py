"""An additive shock declaration reproduces the ordinary law's values and policies.

`lcm.AdditiveShockTransition(base=..., shock=...)` promises the same economic law
as an ordinary `base + shock` transition. Each fixture is one source period
followed by a terminal target, solved once through the ordinary law (`plain`) and
once through the declaration (`additive`). Both are compared with an independent
scalar oracle in exact rational arithmetic that imports no `lcm`, JAX or NumPy
helper, and the initial-period policy reported by `Model.simulate` must equal the
oracle's winner.

The geometry fixtures read a terminal value on linear, irregular and log-spaced
grids at stored points, inside cells, and beyond both ends, where the log grid
keeps extrapolating along its virtual geometric cells.
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AdditiveShockTransition,
    AgeGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    Model,
    Regime,
    Transition,
    categorical,
)
from lcm.regime import StateTransitionEntry
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt
from tests.conftest import X64_ENABLED

# Agreement owed to an analytically specified dyadic witness. The log-coordinate
# reader rounds through a logarithm, which the ordinary law measures at up to 24
# representable steps on these fixtures; the linear readers stay within a few.
_MAX_ULP = {"log": 32, "irregular": 8, "linear": 8}


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def _plain_law(*, base: FloatND, shock: FloatND) -> ContinuousState:
    return base + shock


def _law(route: str) -> StateTransitionEntry:
    if route == "additive":
        return AdditiveShockTransition(base="base", shock="shock")
    return _plain_law


def _identity(wealth: ContinuousState) -> FloatND:
    return wealth


def _zero_utility(wealth: ContinuousState) -> FloatND:
    return 0.0 * wealth


def _zero_shock() -> FloatND:
    return jnp.asarray(0.0)


def _one_step(*, alive: Regime, dead: Regime, enable_jit: bool) -> Model:
    return Model(
        regimes={"alive": alive, "dead": dead},
        edges={"alive": Transition(targets={"dead": (0,)}, law="dead")},
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "alive"},
        enable_jit=enable_jit,
    )


def _assert_values(*, actual: np.ndarray, expected: np.ndarray, max_ulp: int) -> None:
    dtype = np.dtype("float64" if X64_ENABLED else "float32")
    assert actual.dtype == dtype
    expected = np.asarray(expected, dtype=dtype)
    assert actual.shape == expected.shape
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_max_ulp(actual, expected, maxulp=max_ulp)


def _log_grid_value(value: Fraction) -> Fraction:
    """Read values `(1, 2, 4)` stored on the log-spaced grid `(1, 2, 4)`.

    The ordinary reader extrapolates along virtual geometric cells beyond either
    end, so the value at virtual node `2**k` is `k + 1` for `k <= 1` and `2 * k`
    for `k >= 1`, linear in the physical value between neighbouring nodes.
    """
    assert value > 0
    left = Fraction(1)
    rank = 0
    while value < left:
        left /= 2
        rank -= 1
    while value >= 2 * left:
        left *= 2
        rank += 1

    def node_value(k: int) -> Fraction:
        return Fraction(k + 1 if k <= 1 else 2 * k)

    share = (value - left) / left
    return node_value(rank) + share * (node_value(rank + 1) - node_value(rank))


@pytest.mark.parametrize("route", ["plain", "additive"])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("kind", ["log", "irregular", "linear"])
@pytest.mark.parametrize("scale", [1, 16])
@pytest.mark.parametrize("shock_offset", [0.0, 0.5])
def test_additive_shock_reads_the_target_grid_like_the_ordinary_law(
    *, route: str, enable_jit: bool, kind: str, scale: int, shock_offset: float
) -> None:
    """Landings at nodes, inside cells and beyond both ends read the exact value."""
    ratios = (0.25, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0)
    landings = tuple(scale * value for value in ratios)
    source_points = tuple(value - shock_offset for value in landings)
    grids = {
        "log": lambda: LogSpacedGrid(start=scale, stop=4 * scale, n_points=3),
        "irregular": lambda: IrregSpacedGrid(points=(scale, 2 * scale, 4 * scale)),
        "linear": lambda: LinSpacedGrid(start=scale, stop=4 * scale, n_points=3),
    }

    def shock() -> FloatND:
        return jnp.asarray(shock_offset)

    alive = Regime(
        states={"wealth": IrregSpacedGrid(points=source_points)},
        state_transitions={"wealth": _law(route)},
        functions={"utility": _zero_utility, "base": _identity, "shock": shock},
    )
    dead = Regime(states={"wealth": grids[kind]()}, functions={"utility": _identity})
    model = _one_step(alive=alive, dead=dead, enable_jit=enable_jit)
    solution = model.solve(params={"discount_factor": 1.0}, log_level="debug")
    expected = np.asarray(
        [
            float(scale * _log_grid_value(Fraction(value)))
            if kind == "log"
            else scale * value
            for value in ratios
        ]
    )
    _assert_values(
        actual=np.asarray(solution.values[0]["alive"]),
        expected=expected,
        max_ulp=_MAX_ULP[kind],
    )


def _log_choice_base(*, wealth: ContinuousState, choice: ContinuousAction) -> FloatND:
    return wealth + 4.0 + 4.0 * choice


def _log_choice_utility(choice: ContinuousAction) -> FloatND:
    return -1.5 * choice


def _log_choice_oracle() -> tuple[int, Fraction]:
    """Return the winner and value at source wealth 4: landings 8 and 12."""
    candidates = tuple(
        _log_grid_value(Fraction(8 + 4 * choice)) - Fraction(3, 2) * choice
        for choice in (0, 1)
    )
    assert candidates == (Fraction(6), Fraction(11, 2))
    winner = max(range(2), key=candidates.__getitem__)
    return winner, candidates[winner]


def _log_choice_model(*, route: str, enable_jit: bool) -> Model:
    alive = Regime(
        states={"wealth": LinSpacedGrid(start=4.0, stop=5.0, n_points=2)},
        actions={"choice": IrregSpacedGrid(points=(0.0, 1.0))},
        state_transitions={"wealth": _law(route)},
        functions={
            "utility": _log_choice_utility,
            "base": _log_choice_base,
            "shock": _zero_shock,
        },
    )
    dead = Regime(
        states={"wealth": LogSpacedGrid(start=1.0, stop=4.0, n_points=3)},
        functions={"utility": _identity},
    )
    return _one_step(alive=alive, dead=dead, enable_jit=enable_jit)


@pytest.mark.parametrize("route", ["plain", "additive"])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_additive_shock_on_a_log_grid_keeps_the_ordinary_value(
    *, route: str, enable_jit: bool
) -> None:
    """The source value at wealth 4 is the oracle's best candidate, 6."""
    model = _log_choice_model(route=route, enable_jit=enable_jit)
    solution = model.solve(params={"discount_factor": 1.0}, log_level="debug")
    _, value = _log_choice_oracle()
    _assert_values(
        actual=np.asarray(solution.values[0]["alive"])[:1],
        expected=np.asarray([float(value)]),
        max_ulp=_MAX_ULP["log"],
    )


@pytest.mark.parametrize("route", ["plain", "additive"])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_additive_shock_on_a_log_grid_keeps_the_ordinary_policy(
    *, route: str, enable_jit: bool
) -> None:
    """At wealth 4 the cheaper choice wins by half a unit, as under the plain law."""
    model = _log_choice_model(route=route, enable_jit=enable_jit)
    params = {"discount_factor": 1.0}
    solution = model.solve(params=params, log_level="debug")
    frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "wealth": jnp.array([4.0]),
            "age": jnp.array([0.0]),
            "regime_id": jnp.array([_RegimeId.alive]),
        },
        seed=0,
        log_level="debug",
    ).to_dataframe()
    winner, _ = _log_choice_oracle()
    actual = frame.loc[frame["age"] == 0, "choice"].to_numpy()
    np.testing.assert_array_equal(actual, [float(winner)])
