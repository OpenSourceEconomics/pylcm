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

The rare-event fixtures integrate two transition-local binary draws whose rare
marginal is a normal number but whose joint product lies below every
representable magnitude. A positive-probability landing on an infeasible node
keeps its action infeasible, and a rare landing on a large finite payoff keeps
its finite contribution, exactly as when the plain law enumerates the nodes.
"""

from fractions import Fraction
from itertools import product
from math import inf

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AdditiveShockTransition,
    AgeGrid,
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    Model,
    Regime,
    Transition,
    categorical,
)
from lcm.regime import StateTransitionEntry
from lcm.transition import StochasticTransition
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from tests.conftest import X64_ENABLED

# Agreement owed to an analytically specified dyadic witness. The log-coordinate
# reader rounds through a logarithm, which the ordinary law measures at up to 24
# representable steps on these fixtures; the linear readers stay within a few.
_MAX_ULP = {"log": 32, "irregular": 8, "linear": 8}


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class _Binary:
    zero: ScalarInt
    one: ScalarInt


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


# Exponents of the rare marginal `2**-e`. The joint product `2**-2e` is
# represented for "represented" and lies below the smallest subnormal for
# "vanishing" and "deeper".
_RARE_EXPONENT = (
    {"represented": 300, "vanishing": 600, "deeper": 700}
    if X64_ENABLED
    else {"represented": 40, "vanishing": 80, "deeper": 90}
)
_LARGE_EXPONENT = 1000 if X64_ENABLED else 100


def _initial_rare_conditions() -> dict[str, jnp.ndarray]:
    return {
        "wealth": jnp.array([1.0]),
        "u": jnp.array([0], dtype=jnp.int32),
        "v": jnp.array([0], dtype=jnp.int32),
        "age": jnp.array([0.0]),
        "regime_id": jnp.array([_RegimeId.alive]),
    }


def _rare_row(
    *, rare: float, reverse_codes: bool
) -> tuple[jnp.ndarray, Fraction, Fraction]:
    """Return the submitted marginal row and its common and rare masses."""
    row = jnp.asarray([1.0 - rare, rare])
    submitted = np.asarray(row)
    assert np.all(np.isfinite(submitted))
    assert np.sum(submitted, dtype=submitted.dtype) == 1
    if rare > 0:
        assert submitted[1] >= np.finfo(submitted.dtype).tiny
    common_mass, rare_mass = (Fraction(float(value)) for value in submitted)
    return (row[::-1] if reverse_codes else row), common_mass, rare_mass


def _positive_wealth(wealth: ContinuousState) -> BoolND:
    return wealth > 0.0


def _rare_base(*, wealth: ContinuousState, safe: ContinuousAction) -> FloatND:
    return 2.0 + safe + 0.0 * wealth


def _rare_utility(
    *,
    safe: ContinuousAction,
    wealth: ContinuousState,
    u: DiscreteState,
    v: DiscreteState,
) -> FloatND:
    return -2.0 * safe + 0.0 * wealth + 0.0 * u + 0.0 * v


def _feasible_terminal(value: Fraction) -> Fraction | float:
    """Read `V = (-inf, 1, 2)` on wealth `(0, 1, 2)` at a non-negative landing."""
    assert value >= 0
    return -inf if value < 1 else value


def _rare_event_oracle(
    *, common_mass: Fraction, rare_mass: Fraction, reverse_codes: bool
) -> tuple[int, Fraction | float]:
    """Enumerate the four joint draws; a positive-mass infeasible landing is -inf.

    `safe = 0` starts from 2 at no cost, `safe = 1` starts from 3 at cost 2, and
    every rare draw removes 1. An exactly zero-mass draw is omitted.
    """
    rows = (rare_mass, common_mass) if reverse_codes else (common_mass, rare_mass)
    denominator = sum(rows, start=Fraction(0)) ** 2
    candidates: list[Fraction | float] = []
    for safe in (0, 1):
        total = Fraction(0)
        infeasible = False
        for u, v in product((0, 1), repeat=2):
            probability = rows[u] * rows[v]
            if probability == 0:
                continue
            rare_count = (2 - u - v) if reverse_codes else (u + v)
            terminal = _feasible_terminal(Fraction(2 + safe - rare_count))
            if terminal == -inf:
                infeasible = True
                break
            total += probability * terminal
        candidates.append(-inf if infeasible else total / denominator - 2 * safe)
    winner = max(range(2), key=candidates.__getitem__)
    return winner, candidates[winner]


def _rare_event_model(
    *, route: str, enable_jit: bool, row: jnp.ndarray, reverse_codes: bool
) -> Model:
    def probabilities() -> FloatND:
        return row

    def shock(*, next_u: DiscreteState, next_v: DiscreteState) -> FloatND:
        return (next_u + next_v - 2.0) if reverse_codes else (0.0 - next_u - next_v)

    alive = Regime(
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=2),
            "u": DiscreteGrid(_Binary),
            "v": DiscreteGrid(_Binary),
        },
        actions={"safe": IrregSpacedGrid(points=(0.0, 1.0))},
        state_transitions={
            "wealth": _law(route),
            "u": StochasticTransition(func=probabilities),
            "v": StochasticTransition(func=probabilities),
        },
        functions={"utility": _rare_utility, "base": _rare_base, "shock": shock},
    )
    dead = Regime(
        states={"wealth": LinSpacedGrid(start=0.0, stop=2.0, n_points=3)},
        functions={"utility": _identity},
        constraints={"positive": _positive_wealth},
    )
    return _one_step(alive=alive, dead=dead, enable_jit=enable_jit)


_MASS_CASES = ("zero", "represented", "vanishing", "deeper")


@pytest.mark.parametrize("route", ["plain", "additive"])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("reverse_codes", [False, True])
@pytest.mark.parametrize("mass_case", _MASS_CASES)
def test_additive_shock_keeps_a_rare_infeasible_landing_infeasible(
    *, route: str, enable_jit: bool, reverse_codes: bool, mass_case: str
) -> None:
    """The source value is the oracle's: only an exactly null rare draw is omitted."""
    rare = 0.0 if mass_case == "zero" else 2.0 ** -_RARE_EXPONENT[mass_case]
    row, common_mass, rare_mass = _rare_row(rare=rare, reverse_codes=reverse_codes)
    model = _rare_event_model(
        route=route, enable_jit=enable_jit, row=row, reverse_codes=reverse_codes
    )
    solution = model.solve(params={"discount_factor": 1.0}, log_level="debug")
    _, value = _rare_event_oracle(
        common_mass=common_mass, rare_mass=rare_mass, reverse_codes=reverse_codes
    )
    actual = np.asarray(solution.values[0]["alive"])
    _assert_values(
        actual=actual,
        expected=np.full(actual.shape, float(value)),
        max_ulp=_MAX_ULP["linear"],
    )


@pytest.mark.parametrize("route", ["plain", "additive"])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("reverse_codes", [False, True])
@pytest.mark.parametrize("mass_case", _MASS_CASES)
def test_additive_shock_keeps_the_policy_under_a_rare_infeasible_landing(
    *, route: str, enable_jit: bool, reverse_codes: bool, mass_case: str
) -> None:
    """The unsafe action wins only when its infeasible draw has exactly zero mass."""
    rare = 0.0 if mass_case == "zero" else 2.0 ** -_RARE_EXPONENT[mass_case]
    row, common_mass, rare_mass = _rare_row(rare=rare, reverse_codes=reverse_codes)
    model = _rare_event_model(
        route=route, enable_jit=enable_jit, row=row, reverse_codes=reverse_codes
    )
    params = {"discount_factor": 1.0}
    solution = model.solve(params=params, log_level="debug")
    frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions=_initial_rare_conditions(),
        seed=0,
        log_level="debug",
    ).to_dataframe()
    winner, _ = _rare_event_oracle(
        common_mass=common_mass, rare_mass=rare_mass, reverse_codes=reverse_codes
    )
    assert winner == (0 if mass_case == "zero" else 1)
    actual = frame.loc[frame["age"] == 0, "safe"].to_numpy()
    np.testing.assert_array_equal(actual, [float(winner)])


def _rare_finite_oracle(
    *, common_mass: Fraction, rare_mass: Fraction, large: Fraction, payment: Fraction
) -> tuple[int, Fraction]:
    """Enumerate the four joint draws against `V = (large, 0, 0)` on `(0, 1, 2)`.

    `safe = 0` starts from 2, `safe = 1` starts from 4 and earns `payment`; every
    rare draw removes 1, and the last segment extrapolates at zero.
    """
    rows = (common_mass, rare_mass)
    denominator = sum(rows, start=Fraction(0)) ** 2
    payoff = {0: large, 1: Fraction(0), 2: Fraction(0), 3: Fraction(0), 4: Fraction(0)}
    candidates = tuple(
        sum(
            (
                rows[u] * rows[v] * payoff[2 + 2 * safe - u - v]
                for u, v in product((0, 1), repeat=2)
            ),
            start=Fraction(0),
        )
        / denominator
        + safe * payment
        for safe in (0, 1)
    )
    winner = max(range(2), key=candidates.__getitem__)
    return winner, candidates[winner]


def _rare_finite_model(
    *, route: str, enable_jit: bool, row: jnp.ndarray, large: float, payment: float
) -> Model:
    def probabilities() -> FloatND:
        return row

    def shock(*, next_u: DiscreteState, next_v: DiscreteState) -> FloatND:
        return 0.0 - next_u - next_v

    def base(*, wealth: ContinuousState, safe: ContinuousAction) -> FloatND:
        return 2.0 + 2.0 * safe + 0.0 * wealth

    def utility(
        *,
        safe: ContinuousAction,
        wealth: ContinuousState,
        u: DiscreteState,
        v: DiscreteState,
    ) -> FloatND:
        return payment * safe + 0.0 * wealth + 0.0 * u + 0.0 * v

    def terminal(wealth: ContinuousState) -> FloatND:
        return jnp.where(wealth == 0.0, large, 0.0)

    alive = Regime(
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=2),
            "u": DiscreteGrid(_Binary),
            "v": DiscreteGrid(_Binary),
        },
        actions={"safe": IrregSpacedGrid(points=(0.0, 1.0))},
        state_transitions={
            "wealth": _law(route),
            "u": StochasticTransition(func=probabilities),
            "v": StochasticTransition(func=probabilities),
        },
        functions={"utility": utility, "base": base, "shock": shock},
    )
    dead = Regime(
        states={"wealth": LinSpacedGrid(start=0.0, stop=2.0, n_points=3)},
        functions={"utility": terminal},
    )
    return _one_step(alive=alive, dead=dead, enable_jit=enable_jit)


def _rare_finite_setup(
    mass_case: str,
) -> tuple[jnp.ndarray, float, float, int, Fraction]:
    rare_exponent = _RARE_EXPONENT[mass_case]
    large = 2.0**_LARGE_EXPONENT
    payment = 2.0 ** (_LARGE_EXPONENT - 2 * rare_exponent - 1)
    row, common_mass, rare_mass = _rare_row(
        rare=2.0**-rare_exponent, reverse_codes=False
    )
    winner, value = _rare_finite_oracle(
        common_mass=common_mass,
        rare_mass=rare_mass,
        large=Fraction(large),
        payment=Fraction(payment),
    )
    assert winner == 0
    assert float(value) >= np.finfo(np.asarray(row).dtype).tiny
    return row, large, payment, winner, value


@pytest.mark.parametrize("route", ["plain", "additive"])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("mass_case", ["represented", "vanishing"])
def test_additive_shock_keeps_a_rare_finite_contribution(
    *, route: str, enable_jit: bool, mass_case: str
) -> None:
    """A rare draw's normal contribution survives although its probability does not."""
    row, large, payment, _, value = _rare_finite_setup(mass_case)
    model = _rare_finite_model(
        route=route, enable_jit=enable_jit, row=row, large=large, payment=payment
    )
    solution = model.solve(params={"discount_factor": 1.0}, log_level="debug")
    actual = np.asarray(solution.values[0]["alive"])
    _assert_values(
        actual=actual,
        expected=np.full(actual.shape, float(value)),
        max_ulp=_MAX_ULP["linear"],
    )


@pytest.mark.parametrize("route", ["plain", "additive"])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("mass_case", ["represented", "vanishing"])
def test_additive_shock_keeps_the_policy_under_a_rare_finite_contribution(
    *, route: str, enable_jit: bool, mass_case: str
) -> None:
    """The risky action's rare payoff outweighs the sure payment, as enumerated."""
    row, large, payment, winner, _ = _rare_finite_setup(mass_case)
    model = _rare_finite_model(
        route=route, enable_jit=enable_jit, row=row, large=large, payment=payment
    )
    params = {"discount_factor": 1.0}
    solution = model.solve(params=params, log_level="debug")
    frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions=_initial_rare_conditions(),
        seed=0,
        log_level="debug",
    ).to_dataframe()
    actual = frame.loc[frame["age"] == 0, "safe"].to_numpy()
    np.testing.assert_array_equal(actual, [float(winner)])
