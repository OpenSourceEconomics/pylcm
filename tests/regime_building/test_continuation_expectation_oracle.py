"""The continuation expectation agrees with a literal sum over every joint node.

The source regime `alive` carries wealth and four discrete states with their own
Markov laws: `income`, `bonus`, `health` and `pref`. It moves into a terminal
regime `final` that values all five states. Next-period wealth may read the
`income` and `bonus` draws, so those draws move the point at which the
continuation is interpolated; a draw nothing reads only selects which value
slice is read. A state that is neither read nor drawn transitions
deterministically.

The oracle is the textbook one-step Bellman operator written with explicit
loops over every joint node `(income', bonus', health', pref')`, weighting each
by the product of the four marginal rows and interpolating the terminal value
linearly in wealth. It shares nothing with the engine but the model's
primitives.
"""

import itertools
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinearExpectation,
    LinSpacedGrid,
    Model,
    PowerMean,
    StochasticTransition,
    Transition,
    categorical,
)
from lcm.certainty_equivalent import CertaintyEquivalent
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime, StateTransitionEntry
from lcm.solver_api import SolutionResult
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
    StateName,
    UserParams,
    UserParamsNode,
)


@categorical(ordered=False)
class _Income:
    low: ScalarInt
    lower_mid: ScalarInt
    upper_mid: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _Bonus:
    none: ScalarInt
    paid: ScalarInt


@categorical(ordered=False)
class _Health:
    h0: ScalarInt
    h1: ScalarInt
    h2: ScalarInt
    h3: ScalarInt
    h4: ScalarInt


@categorical(ordered=False)
class _Pref:
    p0: ScalarInt
    p1: ScalarInt
    p2: ScalarInt
    p3: ScalarInt
    p4: ScalarInt
    p5: ScalarInt
    p6: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    final: ScalarInt


_N_NODES = MappingProxyType({"income": 4, "bonus": 2, "health": 5, "pref": 7})
_WEALTH = (0.0, 4.0, 5)
_CONSUMPTION = (0.0, 1.5, 6)
_DISCOUNT_FACTOR = 0.9
_RISK_AVERSION = 2.0


def _markov_rows(*, n: int, decay: float, tilt: float) -> np.ndarray:
    """Return a non-uniform row-stochastic matrix with every entry positive."""
    distance = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    raw = np.exp(-decay * distance) * (1.0 + tilt * np.arange(n))
    return raw / raw.sum(axis=1, keepdims=True)


_TRANSITION_ROWS = MappingProxyType(
    {
        "income": np.array(
            [
                [0.5, 0.3, 0.15, 0.05],
                [0.2, 0.45, 0.25, 0.1],
                [0.1, 0.2, 0.5, 0.2],
                [0.05, 0.1, 0.25, 0.6],
            ]
        ),
        "bonus": np.array([[0.7, 0.3], [0.45, 0.55]]),
        "health": _markov_rows(n=5, decay=0.8, tilt=0.15),
        "pref": _markov_rows(n=7, decay=0.5, tilt=-0.05),
    }
)


def _income_probs(*, income: DiscreteState) -> FloatND:
    return jnp.asarray(_TRANSITION_ROWS["income"])[income]


def _bonus_probs(*, bonus: DiscreteState) -> FloatND:
    return jnp.asarray(_TRANSITION_ROWS["bonus"])[bonus]


def _health_probs(*, health: DiscreteState) -> FloatND:
    return jnp.asarray(_TRANSITION_ROWS["health"])[health]


def _pref_probs(*, pref: DiscreteState) -> FloatND:
    return jnp.asarray(_TRANSITION_ROWS["pref"])[pref]


def _health_probs_reading_the_income_draw(
    *, health: DiscreteState, next_income: DiscreteState
) -> FloatND:
    rows = jnp.asarray(_TRANSITION_ROWS["health"])
    return rows[(health + next_income) % _N_NODES["health"]]


def _same_income(*, income: DiscreteState) -> DiscreteState:
    return income


def _same_bonus(*, bonus: DiscreteState) -> DiscreteState:
    return bonus


def _same_health(*, health: DiscreteState) -> DiscreteState:
    return health


def _same_pref(*, pref: DiscreteState) -> DiscreteState:
    return pref


_PROBABILITY_LAWS = MappingProxyType(
    {
        "income": _income_probs,
        "bonus": _bonus_probs,
        "health": _health_probs,
        "pref": _pref_probs,
    }
)
_IDENTITY_LAWS = MappingProxyType(
    {
        "income": _same_income,
        "bonus": _same_bonus,
        "health": _same_health,
        "pref": _same_pref,
    }
)


def _next_wealth_reading_no_draw(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption + 0.5


def _next_wealth_reading_income(
    *,
    wealth: ContinuousState,
    consumption: ContinuousAction,
    next_income: DiscreteState,
) -> ContinuousState:
    return wealth - consumption + 0.5 + 0.5 * next_income


def _next_wealth_reading_income_and_bonus(
    *,
    wealth: ContinuousState,
    consumption: ContinuousAction,
    next_income: DiscreteState,
    next_bonus: DiscreteState,
) -> ContinuousState:
    return wealth - consumption + 0.5 + 0.5 * next_income + 0.25 * next_bonus


_WEALTH_LAWS = MappingProxyType(
    {
        (): _next_wealth_reading_no_draw,
        ("income",): _next_wealth_reading_income,
        ("income", "bonus"): _next_wealth_reading_income_and_bonus,
    }
)


def _alive_utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log1p(consumption)


def _final_utility(
    *,
    wealth: ContinuousState,
    income: DiscreteState,
    bonus: DiscreteState,
    health: DiscreteState,
    pref: DiscreteState,
) -> FloatND:
    return (
        jnp.sqrt(wealth + 2.0) * (1.0 + 0.3 * health + 0.2 * pref)
        + 0.1 * income
        + 0.05 * bonus
    )


def _final_utility_np(
    *,
    wealth: np.ndarray,
    income: float | npt.NDArray[np.number],
    bonus: float | npt.NDArray[np.number],
    health: float | npt.NDArray[np.number],
    pref: float | npt.NDArray[np.number],
) -> np.ndarray:
    return (
        np.sqrt(wealth + 2.0) * (1.0 + 0.3 * health + 0.2 * pref)
        + 0.1 * income
        + 0.05 * bonus
    )


type _Reads = tuple[str, ...]


def _drawn_states(*, reads: _Reads, slice_draws: bool) -> frozenset[str]:
    """States with a Markov law: every read draw, plus all four if `slice_draws`."""
    return frozenset(_N_NODES) if slice_draws else frozenset(reads)


def _model(
    *,
    reads: _Reads,
    slice_draws: bool,
    certainty_equivalent: CertaintyEquivalent | None = None,
    health_law: StateTransitionEntry = None,
) -> Model:
    drawn = _drawn_states(reads=reads, slice_draws=slice_draws)
    laws: dict[StateName, StateTransitionEntry] = {
        name: StochasticTransition(func=_PROBABILITY_LAWS[name])
        if name in drawn
        else _IDENTITY_LAWS[name]
        for name in _N_NODES
    }
    if health_law is not None:
        laws["health"] = health_law
    states = {
        "wealth": LinSpacedGrid(start=_WEALTH[0], stop=_WEALTH[1], n_points=_WEALTH[2]),
        "income": DiscreteGrid(category_class=_Income),
        "bonus": DiscreteGrid(category_class=_Bonus),
        "health": DiscreteGrid(category_class=_Health),
        "pref": DiscreteGrid(category_class=_Pref),
    }
    alive = Regime(
        states=states,
        state_transitions={"wealth": _WEALTH_LAWS[reads], **laws},
        actions={
            "consumption": LinSpacedGrid(
                start=_CONSUMPTION[0], stop=_CONSUMPTION[1], n_points=_CONSUMPTION[2]
            )
        },
        functions={"utility": _alive_utility},
        certainty_equivalent=certainty_equivalent,
    )
    final = Regime(
        states=states,
        functions={"utility": _final_utility},
    )
    return Model(
        regimes={"alive": alive, "final": final},
        regime_id_class=_RegimeId,
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={"alive": {"final": 0}},
        initial_nodes=((0, "alive"),),
    )


def _params(*, power_mean: bool = False) -> dict[str, UserParamsNode]:
    alive: dict[str, dict[str, float]] = {
        "koopmans_aggregator": {"discount_factor": _DISCOUNT_FACTOR}
    }
    if power_mean:
        alive["certainty_equivalent"] = {"risk_aversion": _RISK_AVERSION}
    return {"alive": alive, "final": {}}


_AXES = ("wealth", "income", "bonus", "health", "pref")


def _interpolate_in_wealth(*, values: np.ndarray, wealth: np.ndarray) -> np.ndarray:
    """Linearly interpolate `values` on the wealth grid, extrapolating linearly."""
    start, stop, n_points = _WEALTH
    coordinate = (wealth - start) / ((stop - start) / (n_points - 1))
    lower = np.clip(np.floor(coordinate), 0, n_points - 2).astype(int)
    upper_weight = coordinate - lower
    return (1.0 - upper_weight) * values[lower] + upper_weight * values[lower + 1]


def _along(*, axis: int, values: np.ndarray) -> np.ndarray:
    """Place `values` on `axis` of the six oracle axes, broadcasting the others."""
    shape = [1] * 6
    shape[axis] = values.size
    return values.reshape(shape)


def _oracle(
    *, reads: _Reads, slice_draws: bool, power_mean: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the value, the policy and the action-value gap of `alive` in period 0.

    All three are indexed `(wealth, income, bonus, health, pref)`. The gap is the
    distance between the best and second-best action value, so a comparison of
    policies can leave out cells where the two are numerically tied.
    """
    drawn = _drawn_states(reads=reads, slice_draws=slice_draws)
    rows = {
        name: _TRANSITION_ROWS[name] if name in drawn else np.eye(n)
        for name, n in _N_NODES.items()
    }
    wealth_grid = np.linspace(*_WEALTH)
    consumption_grid = np.linspace(*_CONSUMPTION)
    wealth = _along(axis=0, values=wealth_grid)
    income, bonus, health, pref = (
        _along(axis=axis, values=np.arange(n))
        for axis, n in enumerate(_N_NODES.values(), start=1)
    )
    consumption = _along(axis=5, values=consumption_grid)
    final_values = {
        node: _final_utility_np(
            wealth=wealth_grid,
            income=node[0],
            bonus=node[1],
            health=node[2],
            pref=node[3],
        )
        for node in itertools.product(*(range(n) for n in _N_NODES.values()))
    }

    expectation = np.zeros((_WEALTH[2], *_N_NODES.values(), _CONSUMPTION[2]))
    for node, values in final_values.items():
        next_income, next_bonus, next_health, next_pref = node
        probability = (
            rows["income"][income, next_income]
            * rows["bonus"][bonus, next_bonus]
            * rows["health"][health, next_health]
            * rows["pref"][pref, next_pref]
        )
        next_wealth = wealth - consumption + 0.5
        if "income" in reads:
            next_wealth = next_wealth + 0.5 * next_income
        if "bonus" in reads:
            next_wealth = next_wealth + 0.25 * next_bonus
        value = _interpolate_in_wealth(values=values, wealth=next_wealth)
        if power_mean:
            value = value ** (1.0 - _RISK_AVERSION)
        expectation += probability * value
    if power_mean:
        expectation = expectation ** (1.0 / (1.0 - _RISK_AVERSION))

    action_values = np.log1p(consumption) + _DISCOUNT_FACTOR * expectation
    ordered = np.sort(action_values, axis=-1)
    return (
        action_values.max(axis=-1),
        consumption_grid[action_values.argmax(axis=-1)],
        ordered[..., -1] - ordered[..., -2],
    )


def _solved_alive_values(*, model: Model, solution: SolutionResult) -> np.ndarray:
    names = model.state_names(regime_name="alive")
    return np.transpose(
        np.asarray(solution.values[0]["alive"]), [names.index(name) for name in _AXES]
    )


def _policy_on_the_grid(
    *, model: Model, solution: SolutionResult, params: UserParams
) -> np.ndarray:
    """Return the consumption the decision program picks at every grid state."""
    grids = np.meshgrid(
        np.linspace(*_WEALTH),
        *(np.arange(n) for n in _N_NODES.values()),
        indexing="ij",
    )
    states = {
        "wealth": jnp.asarray(grids[0].ravel()),
        **{
            name: jnp.asarray(grid.ravel(), dtype=jnp.int32)
            for name, grid in zip(_AXES[1:], grids[1:], strict=True)
        },
    }
    lookup = model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="alive",
        states=states,
    )
    return np.asarray(lookup.actions["consumption"]).reshape(grids[0].shape)


_DRAW_CASES = [
    pytest.param((), True, id="coordinate-none-slice-several"),
    pytest.param(("income",), True, id="coordinate-one-slice-several"),
    pytest.param(("income", "bonus"), True, id="coordinate-several-slice-several"),
    pytest.param(("income",), False, id="coordinate-one-slice-none"),
    pytest.param(("income", "bonus"), False, id="coordinate-several-slice-none"),
]
_PRECISIONS = [
    pytest.param("x64_enabled", 1e-10, 1e-12, id="fp64"),
    pytest.param("x64_disabled", 2e-5, 1e-4, id="fp32"),
]


@pytest.mark.parametrize(("fixture_name", "rtol", "tie_gap"), _PRECISIONS)
@pytest.mark.parametrize(("reads", "slice_draws"), _DRAW_CASES)
def test_solved_value_matches_the_full_node_product_oracle(
    *,
    reads: _Reads,
    slice_draws: bool,
    fixture_name: str,
    rtol: float,
    tie_gap: float,
    request: pytest.FixtureRequest,
) -> None:
    """The period-0 value equals the literal expectation over all joint nodes."""
    request.getfixturevalue(fixture_name)
    del tie_gap
    model = _model(reads=reads, slice_draws=slice_draws)
    solution = model.solve(params=_params(), log_level="off")
    expected, _, _ = _oracle(reads=reads, slice_draws=slice_draws)
    np.testing.assert_allclose(
        _solved_alive_values(model=model, solution=solution), expected, rtol=rtol
    )


@pytest.mark.parametrize(("fixture_name", "rtol", "tie_gap"), _PRECISIONS)
@pytest.mark.parametrize(("reads", "slice_draws"), _DRAW_CASES)
def test_policy_matches_the_full_node_product_oracle(
    *,
    reads: _Reads,
    slice_draws: bool,
    fixture_name: str,
    rtol: float,
    tie_gap: float,
    request: pytest.FixtureRequest,
) -> None:
    """Away from numerical ties, the chosen consumption is the oracle's argmax."""
    request.getfixturevalue(fixture_name)
    del rtol
    model = _model(reads=reads, slice_draws=slice_draws)
    params = _params()
    solution = model.solve(params=params, log_level="off")
    _, expected, gap = _oracle(reads=reads, slice_draws=slice_draws)
    decided = gap > tie_gap
    assert decided.mean() > 0.9
    np.testing.assert_allclose(
        _policy_on_the_grid(model=model, solution=solution, params=params)[decided],
        expected[decided],
        rtol=1e-6,
    )


class _ExplicitLinearExpectation(LinearExpectation):
    """The linear expectation, declared as a type of its own."""


@pytest.mark.parametrize(("fixture_name", "rtol"), [("x64_enabled", 1e-10)])
def test_linear_expectation_subclass_agrees_with_the_oracle(
    *, fixture_name: str, rtol: float, request: pytest.FixtureRequest
) -> None:
    """A subclass of the linear expectation states the same continuation."""
    request.getfixturevalue(fixture_name)
    model = _model(
        reads=("income",),
        slice_draws=True,
        certainty_equivalent=_ExplicitLinearExpectation(),
    )
    solution = model.solve(params=_params(), log_level="off")
    expected, _, _ = _oracle(reads=("income",), slice_draws=True)
    np.testing.assert_allclose(
        _solved_alive_values(model=model, solution=solution), expected, rtol=rtol
    )


@pytest.mark.parametrize(("fixture_name", "rtol"), [("x64_enabled", 1e-9)])
def test_power_mean_agrees_with_the_oracle(
    *, fixture_name: str, rtol: float, request: pytest.FixtureRequest
) -> None:
    """A power-mean continuation is the power mean over every joint node."""
    request.getfixturevalue(fixture_name)
    model = _model(
        reads=("income",), slice_draws=True, certainty_equivalent=PowerMean()
    )
    solution = model.solve(params=_params(power_mean=True), log_level="off")
    expected, _, _ = _oracle(reads=("income",), slice_draws=True, power_mean=True)
    np.testing.assert_allclose(
        _solved_alive_values(model=model, solution=solution), expected, rtol=rtol
    )


def test_a_draw_whose_probabilities_read_a_sibling_draw_is_rejected() -> None:
    """Draw probabilities conditioned on another draw are refused at build time.

    Every draw's weights are therefore a function of the current cell alone, which
    is what lets the expectation over one group of draws be taken inside the
    expectation over another.
    """
    with pytest.raises(ModelInitializationError, match="draws of the same regime"):
        _model(
            reads=("income",),
            slice_draws=True,
            health_law=StochasticTransition(func=_health_probs_reading_the_income_draw),
        )


@categorical(ordered=False)
class _Binary:
    zero: ScalarInt
    one: ScalarInt


@categorical(ordered=False)
class _Landing:
    risky_high: ScalarInt
    risky_zero: ScalarInt
    certain: ScalarInt


@categorical(ordered=False)
class _Slice:
    h00: ScalarInt
    h01: ScalarInt
    h02: ScalarInt
    h03: ScalarInt
    h04: ScalarInt
    h05: ScalarInt
    h06: ScalarInt
    h07: ScalarInt
    h08: ScalarInt
    h09: ScalarInt
    h10: ScalarInt
    h11: ScalarInt
    h12: ScalarInt
    h13: ScalarInt
    h14: ScalarInt
    h15: ScalarInt
    h16: ScalarInt
    h17: ScalarInt


# Eighteen strictly positive slice probabilities whose exact rational sum is one
# in the dtype that carries them: seventeen copies of the rounded `1/18` and the
# exactly representable residual. A row summing to one exactly leaves no
# rounding headroom for a sum of `p * max` terms to absorb.
_UNIT_SLICE_ROWS = MappingProxyType(
    {
        "float32": (
            *(float.fromhex("0x1.c71c72p-5"),) * 17,
            float.fromhex("0x1.c71c6ep-5"),
        ),
        "float64": (
            *(float.fromhex("0x1.c71c71c71c71cp-5"),) * 17,
            float.fromhex("0x1.c71c71c71c724p-5"),
        ),
    }
)


def _coordinate_probabilities() -> FloatND:
    return jnp.asarray([0.5, 0.5])


def _next_landing(*, next_z: DiscreteState) -> DiscreteState:
    return next_z


def _next_landing_with_choice(
    *, next_z: DiscreteState, choice: DiscreteAction
) -> DiscreteState:
    return jnp.where(choice == 0, next_z, jnp.int32(2))


def _zero_utility(*, landing: DiscreteState, h: DiscreteState) -> FloatND:
    # Reading both states makes them used in `alive`; the flow value stays zero.
    return jnp.where((landing >= 0) & (h >= 0), 0.0, 0.0)


def _peak_at_the_first_landing(
    *, landing: DiscreteState, h: DiscreteState, z: DiscreteState, peak: FloatND
) -> FloatND:
    # Reading `h` and `z` keeps both draws on the terminal value's axes; both
    # conditions hold at every code, so the value depends on `landing` alone.
    return jnp.where((landing == 0) & (h >= 0) & (z >= 0), peak, 0.0)


def _peak_or_three_quarters_for_certain(
    *, landing: DiscreteState, h: DiscreteState, z: DiscreteState, peak: FloatND
) -> FloatND:
    risky = jnp.where((landing == 0) & (h >= 0) & (z >= 0), peak, 0.0)
    return jnp.where(landing == 2, 0.75 * peak, risky)


def _finite_range_model(
    *, dtype: np.dtype, enable_jit: bool, reverse: bool, with_choice: bool
) -> Model:
    """A coordinate draw `z` (one half each) and an 18-node slice draw `h`.

    `landing` copies the `z` draw, so `z` moves the landing coordinate while
    `h` only selects the value slice read there. The flow utility of `alive`
    reads `landing` and `h` and is zero; the terminal utility of `final` reads
    all three states but varies only with `landing`.
    """
    row = _UNIT_SLICE_ROWS[dtype.name]
    if reverse:
        row = row[::-1]

    def slice_probabilities() -> FloatND:
        return jnp.asarray(row, dtype=dtype)

    states = {
        "z": DiscreteGrid(category_class=_Binary),
        "h": DiscreteGrid(category_class=_Slice),
        "landing": DiscreteGrid(category_class=_Landing if with_choice else _Binary),
    }
    alive = Regime(
        states=states,
        state_transitions={
            "z": StochasticTransition(func=_coordinate_probabilities),
            "h": StochasticTransition(func=slice_probabilities),
            "landing": _next_landing_with_choice if with_choice else _next_landing,
        },
        actions={"choice": DiscreteGrid(category_class=_Binary)} if with_choice else {},
        functions={"utility": _zero_utility},
    )
    final = Regime(
        states=states,
        functions={
            "utility": _peak_or_three_quarters_for_certain
            if with_choice
            else _peak_at_the_first_landing
        },
    )
    return Model(
        regimes={"alive": alive, "final": final},
        regime_id_class=_RegimeId,
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={"alive": {"final": 0}},
        initial_nodes=((0, "alive"),),
        enable_jit=enable_jit,
    )


def _finite_range_params(*, model: Model, peak: np.floating) -> UserParams:
    template = model.get_params_template()
    return {
        **{
            name: {function: {} for function in branch}
            for name, branch in template.items()
        },
        "alive": {
            **{function: {} for function in template["alive"]},
            "koopmans_aggregator": {"discount_factor": 1.0},
        },
        "final": {
            **{function: {} for function in template["final"]},
            "utility": {"peak": jnp.asarray(peak)},
        },
    }


_FINITE_RANGE_PRECISIONS = [
    pytest.param("x64_enabled", np.dtype("float64"), id="fp64"),
    pytest.param("x64_disabled", np.dtype("float32"), id="fp32"),
]


def _peak(*, dtype: np.dtype, level: str) -> np.floating:
    largest = np.finfo(dtype).max
    return {
        "ordinary": dtype.type(16.0),
        "max": largest,
        "one-step-below-max": np.nextafter(largest, dtype.type(0.0)),
        "negative-max": -largest,
    }[level]


@pytest.mark.parametrize(
    ("level", "reverse"),
    [
        pytest.param("max", False, id="max"),
        pytest.param("ordinary", False, id="ordinary"),
        pytest.param("one-step-below-max", False, id="one-step-below-max"),
        pytest.param("negative-max", False, id="negative-max"),
        pytest.param("max", True, id="max-reversed-row"),
    ],
)
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(("fixture_name", "dtype"), _FINITE_RANGE_PRECISIONS)
def test_expectation_of_finite_values_is_the_exact_finite_mean(
    *,
    level: str,
    reverse: bool,
    enable_jit: bool,
    fixture_name: str,
    dtype: np.dtype,
    request: pytest.FixtureRequest,
) -> None:
    """Value `peak` at one coordinate node and zero at the other averages to `peak/2`.

    This holds up to the largest finite `peak`, whose every slice term is finite
    once weighted by its full joint probability.
    """
    request.getfixturevalue(fixture_name)
    peak = _peak(dtype=dtype, level=level)
    model = _finite_range_model(
        dtype=dtype, enable_jit=enable_jit, reverse=reverse, with_choice=False
    )
    solution = model.solve(
        params=_finite_range_params(model=model, peak=peak), log_level="off"
    )
    np.testing.assert_allclose(
        np.asarray(solution.values[0]["alive"]),
        peak / dtype.type(2),
        rtol=32 * np.finfo(dtype).eps,
        atol=0.0,
    )


@pytest.mark.parametrize(("fixture_name", "dtype"), _FINITE_RANGE_PRECISIONS)
def test_certain_three_quarters_of_max_beats_a_lottery_worth_half_of_it(
    *, fixture_name: str, dtype: np.dtype, request: pytest.FixtureRequest
) -> None:
    """The certain `3M/4` is chosen over the risky lottery worth `M/2`."""
    request.getfixturevalue(fixture_name)
    model = _finite_range_model(
        dtype=dtype, enable_jit=True, reverse=False, with_choice=True
    )
    params = _finite_range_params(model=model, peak=np.finfo(dtype).max)
    solution = model.solve(params=params, log_level="off")
    lookup = model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="alive",
        states={
            "z": jnp.zeros(2, dtype=jnp.int32),
            "h": jnp.zeros(2, dtype=jnp.int32),
            "landing": jnp.zeros(2, dtype=jnp.int32),
        },
    )
    np.testing.assert_array_equal(
        np.asarray(lookup.actions["choice"]), np.ones(2, dtype=np.int32)
    )


@categorical(ordered=False)
class _OuterRegimeId:
    alive: ScalarInt
    high: ScalarInt
    nothing: ScalarInt


def _near_one(dtype: np.dtype) -> np.floating:
    return np.nextafter(dtype.type(1.0), dtype.type(0.0))


def _half() -> FloatND:
    return jnp.asarray(0.5)


def _high_probability(*, choice: DiscreteAction) -> FloatND:
    return jnp.where(choice == 0, 0.5, 1.0)


def _nothing_probability(*, choice: DiscreteAction) -> FloatND:
    return jnp.where(choice == 0, 0.5, 0.0)


def _nothing_value() -> FloatND:
    return jnp.asarray(0.0)


def _outer_weighted_model(
    *, dtype: np.dtype, enable_jit: bool, with_choice: bool
) -> Model:
    """The finite-range lottery behind a later regime probability.

    `z` lands on the peak with probability one step below one and on zero
    otherwise; the 18-node slice draw `h` sums to one exactly. The lottery's
    target `high` is reached with probability one half, the stateless target
    `nothing`, worth zero, with the other half. With a choice, action 1 moves
    to `high` for sure and lands on its certain node worth `3/4` of the peak.
    """
    q = _near_one(dtype)
    coordinate_row = (q, dtype.type(1.0) - q)
    slice_row = _UNIT_SLICE_ROWS[dtype.name]

    def coordinate_probabilities() -> FloatND:
        return jnp.asarray(coordinate_row, dtype=dtype)

    def slice_probabilities() -> FloatND:
        return jnp.asarray(slice_row, dtype=dtype)

    states = {
        "z": DiscreteGrid(category_class=_Binary),
        "h": DiscreteGrid(category_class=_Slice),
        "landing": DiscreteGrid(category_class=_Landing if with_choice else _Binary),
    }
    alive = Regime(
        states=states,
        state_transitions={
            "z": StochasticTransition(func=coordinate_probabilities),
            "h": StochasticTransition(func=slice_probabilities),
            "landing": _next_landing_with_choice if with_choice else _next_landing,
        },
        actions={"choice": DiscreteGrid(category_class=_Binary)} if with_choice else {},
        functions={"utility": _zero_utility},
    )
    high = Regime(
        states=states,
        functions={
            "utility": _peak_or_three_quarters_for_certain
            if with_choice
            else _peak_at_the_first_landing
        },
    )
    nothing = Regime(functions={"utility": _nothing_value})
    return Model(
        regimes={"alive": alive, "high": high, "nothing": nothing},
        regime_id_class=_OuterRegimeId,
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={
            "alive": Transition(
                targets={"high": 0, "nothing": 0},
                law={
                    "high": StochasticTransition(
                        func=_high_probability if with_choice else _half
                    ),
                    "nothing": StochasticTransition(
                        func=_nothing_probability if with_choice else _half
                    ),
                },
            )
        },
        initial_nodes=((0, "alive"),),
        enable_jit=enable_jit,
    )


def _outer_weighted_params(*, model: Model, peak: np.floating) -> UserParams:
    template = model.get_params_template()
    return {
        **{
            name: {function: {} for function in branch}
            for name, branch in template.items()
        },
        "alive": {
            **{function: {} for function in template["alive"]},
            "koopmans_aggregator": {"discount_factor": 1.0},
        },
        "high": {
            **{function: {} for function in template["high"]},
            "utility": {"peak": jnp.asarray(peak)},
        },
    }


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(("fixture_name", "dtype"), _FINITE_RANGE_PRECISIONS)
def test_certain_three_quarters_of_max_beats_a_half_weighted_near_max_target(
    *,
    enable_jit: bool,
    fixture_name: str,
    dtype: np.dtype,
    request: pytest.FixtureRequest,
) -> None:
    """The certain `3M/4` is chosen over the risky action worth about `M/2`."""
    request.getfixturevalue(fixture_name)
    model = _outer_weighted_model(dtype=dtype, enable_jit=enable_jit, with_choice=True)
    params = _outer_weighted_params(model=model, peak=np.finfo(dtype).max)
    solution = model.solve(params=params, log_level="off")
    lookup = model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="alive",
        states={
            "z": jnp.zeros(2, dtype=jnp.int32),
            "h": jnp.zeros(2, dtype=jnp.int32),
            "landing": jnp.zeros(2, dtype=jnp.int32),
        },
    )
    np.testing.assert_array_equal(
        np.asarray(lookup.actions["choice"]), np.ones(2, dtype=np.int32)
    )


@pytest.mark.parametrize(("fixture_name", "dtype"), _FINITE_RANGE_PRECISIONS)
def test_certain_action_value_is_three_quarters_of_max(
    *, fixture_name: str, dtype: np.dtype, request: pytest.FixtureRequest
) -> None:
    """The chosen certain action publishes `3M/4`, not an infinity."""
    request.getfixturevalue(fixture_name)
    model = _outer_weighted_model(dtype=dtype, enable_jit=True, with_choice=True)
    params = _outer_weighted_params(model=model, peak=np.finfo(dtype).max)
    solution = model.solve(params=params, log_level="off")
    np.testing.assert_allclose(
        np.asarray(solution.values[0]["alive"]),
        dtype.type(0.75) * np.finfo(dtype).max,
        rtol=32 * np.finfo(dtype).eps,
        atol=0.0,
    )
