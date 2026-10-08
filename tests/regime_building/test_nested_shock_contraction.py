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

import inspect
import itertools
import re
from fractions import Fraction
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building import Q_and_F as Q_and_F_module
from _lcm.regime_building.Q_and_F import _ExpectationOverSliceDraws, _slice_block_size
from _lcm.utils.dispatchers import productmap
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
from lcm.regime import Regime
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    IntND,
    ScalarInt,
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
    *, wealth: np.ndarray, income: Any, bonus: Any, health: Any, pref: Any
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
    health_law: Any = None,
) -> Model:
    drawn = _drawn_states(reads=reads, slice_draws=slice_draws)
    laws: dict[str, Any] = {
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


def _params(*, power_mean: bool = False) -> dict[str, Any]:
    alive: dict[str, Any] = {
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


def _solved_alive_values(*, model: Model, solution: Any) -> np.ndarray:
    names = model.state_names(regime_name="alive")
    return np.transpose(
        np.asarray(solution.values[0]["alive"]), [names.index(name) for name in _AXES]
    )


def _policy_on_the_grid(*, model: Model, solution: Any, params: Any) -> np.ndarray:
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


_N_SUBJECTS = 11
_SHAPE = re.compile(r"\b[a-z]+[0-9]*\[([0-9,]+)\]")


def _array_shapes_of_the_pointwise_Q(
    *, certainty_equivalent: CertaintyEquivalent | None, power_mean: bool
) -> list[tuple[int, ...]]:
    """Return the shape of every array in the traced pointwise `Q` of `alive`."""
    text = _traced_pointwise_Q_of_alive(
        reads=("income",),
        certainty_equivalent=certainty_equivalent,
        power_mean=power_mean,
    )
    return [
        tuple(int(dim) for dim in match.split(",")) for match in _SHAPE.findall(text)
    ]


def _traced_pointwise_Q_of_alive(
    *,
    reads: _Reads,
    certainty_equivalent: CertaintyEquivalent | None,
    power_mean: bool,
) -> str:
    """Return the jaxpr text of `alive`'s pointwise `Q` with all four states drawn."""
    model = _model(
        reads=reads,
        slice_draws=True,
        certainty_equivalent=certainty_equivalent,
    )
    params = _params(power_mean=power_mean)
    solution = model.solve(params=params, log_level="off")
    rng = np.random.default_rng(seed=0)
    subjects = {
        "wealth": jnp.asarray(rng.uniform(0.0, 4.0, _N_SUBJECTS)),
        "consumption": jnp.asarray(rng.uniform(0.0, 1.5, _N_SUBJECTS)),
        **{
            name: jnp.asarray(rng.integers(0, n, _N_SUBJECTS), dtype=jnp.int32)
            for name, n in _N_NODES.items()
        },
    }
    return _traced_pointwise_Q(
        model=model, params=params, solution=solution, subjects=subjects
    )


def _traced_pointwise_Q(
    *, model: Model, params: Any, solution: Any, subjects: dict[str, Any]
) -> str:
    """Return the jaxpr text of the period-0 pointwise `Q` of `alive`.

    Only the `subjects` entries that `Q` takes as arguments are passed to it.
    """
    regime = model._regimes["alive"]
    flat_params = model._process_params(params)["alive"]
    next_regime_to_V_arr = MappingProxyType(dict(solution.values[1]))
    assert model.ages is not None
    age = jnp.asarray(model.ages.period_to_age(0))
    Q_and_F = regime.simulation.Q_and_F[0]
    arguments = inspect.signature(Q_and_F).parameters

    def pointwise_Q() -> Any:
        return Q_and_F(
            **{name: value for name, value in subjects.items() if name in arguments},
            next_regime_to_V_arr=next_regime_to_V_arr,
            **flat_params,
            period=jnp.int32(0),
            age=age,
        )

    return str(jax.make_jaxpr(pointwise_Q)())


def _carries_the_joint_node_extent(shape: tuple[int, ...]) -> bool:
    """Whether a per-subject array spans every draw's nodes at once.

    With `income` read by the wealth law, the draws are `income` (4 nodes) and
    `bonus`, `health`, `pref` (2, 5, 7 nodes), so their joint extent is 280. The
    node counts are chosen so no other per-subject array — such as the three
    marginal factors stacked per slice node — has an extent divisible by it.
    """
    joint = int(np.prod(list(_N_NODES.values())))
    rest = int(np.prod(shape)) // _N_SUBJECTS
    return _N_SUBJECTS in shape and rest % joint == 0


def test_plain_expectation_never_forms_the_joint_node_array(
    x64_enabled: None,
) -> None:
    """Under the plain expectation no per-subject array spans all joint nodes."""
    del x64_enabled
    shapes = _array_shapes_of_the_pointwise_Q(
        certainty_equivalent=None, power_mean=False
    )
    assert [shape for shape in shapes if _carries_the_joint_node_extent(shape)] == []


@pytest.mark.parametrize(
    ("reads", "n_full_blocks", "loops"),
    [
        # 280 slice nodes would form 4 blocks of 70.
        pytest.param((), 4, False, id="coordinate-none"),
        # 70 slice nodes: 3 blocks of 18 in the loop, 16 in a remainder block.
        pytest.param(("income",), 3, True, id="coordinate-one"),
    ],
)
def test_slice_draws_are_looped_over_only_beside_a_coordinate_draw(
    *, reads: _Reads, n_full_blocks: int, loops: bool, x64_enabled: None
) -> None:
    """Slice draws are summed in a loop over node blocks only beside a coordinate draw.

    Without a coordinate-moving draw the joint node array is the slice nodes
    alone, so there is no larger array to avoid and the draws stay mapped.
    """
    del x64_enabled
    text = _traced_pointwise_Q_of_alive(
        reads=reads, certainty_equivalent=None, power_mean=False
    )
    assert (re.search(rf"\blength={n_full_blocks}\b", text) is not None) is loops


@pytest.mark.parametrize(
    ("certainty_equivalent", "power_mean"),
    [
        pytest.param(_ExplicitLinearExpectation(), False, id="linear-subclass"),
        pytest.param(PowerMean(), True, id="power-mean"),
    ],
)
def test_other_certainty_equivalents_aggregate_the_whole_joint_lottery(
    *,
    certainty_equivalent: CertaintyEquivalent,
    power_mean: bool,
    x64_enabled: None,
) -> None:
    """A certainty equivalent other than the plain expectation sees every node.

    It receives the joint lottery in one piece, so a per-subject array spanning
    all joint nodes is present; this also shows the shape check above can fire.
    """
    del x64_enabled
    shapes = _array_shapes_of_the_pointwise_Q(
        certainty_equivalent=certainty_equivalent, power_mean=power_mean
    )
    assert any(_carries_the_joint_node_extent(shape) for shape in shapes)


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


def _finite_range_params(*, model: Model, peak: np.floating) -> Any:
    params: Any = model.get_params_template()
    params["alive"]["koopmans_aggregator"]["discount_factor"] = 1.0
    params["final"]["utility"]["peak"] = jnp.asarray(peak)
    return params


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


@pytest.mark.parametrize("with_choice", [False, True], ids=["lottery", "choice"])
def test_finite_range_witness_sums_its_slice_draw_in_a_loop(
    *, with_choice: bool, x64_enabled: None
) -> None:
    """The pointwise `Q` loops over blocks of the 18 slice nodes of `h`.

    The 18 nodes form 3 looped blocks of 5 and a remainder block of 3. This is
    the nested route the finite-range witnesses below are about.
    """
    del x64_enabled
    model = _finite_range_model(
        dtype=np.dtype("float64"),
        enable_jit=True,
        reverse=False,
        with_choice=with_choice,
    )
    params = _finite_range_params(model=model, peak=np.float64(16.0))
    solution = model.solve(params=params, log_level="off")
    subjects = {
        "z": jnp.zeros(2, dtype=jnp.int32),
        "h": jnp.zeros(2, dtype=jnp.int32),
        "landing": jnp.zeros(2, dtype=jnp.int32),
        **({"choice": jnp.zeros(2, dtype=jnp.int32)} if with_choice else {}),
    }
    text = _traced_pointwise_Q(
        model=model, params=params, solution=solution, subjects=subjects
    )
    assert re.search(r"\blength=3\b", text) is not None


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
def test_nested_expectation_of_finite_values_is_the_exact_finite_mean(
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


def _outer_weighted_params(*, model: Model, peak: np.floating) -> Any:
    params: Any = model.get_params_template()
    params["alive"]["koopmans_aggregator"]["discount_factor"] = 1.0
    params["high"]["utility"]["peak"] = jnp.asarray(peak)
    return params


@pytest.mark.parametrize("sign", [1, -1], ids=["max", "negative-max"])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(("fixture_name", "dtype"), _FINITE_RANGE_PRECISIONS)
def test_regime_probability_weights_a_near_certain_maximum_to_half_of_it(
    *,
    sign: int,
    enable_jit: bool,
    fixture_name: str,
    dtype: np.dtype,
    request: pytest.FixtureRequest,
) -> None:
    """A target worth `q M`, reached half the time, contributes `q M / 2`.

    `q` is one step below one and `M` the largest finite value, so the target's
    expectation sits within a few roundings of the format's end while the
    published value is half of it.
    """
    request.getfixturevalue(fixture_name)
    q = _near_one(dtype)
    assert q < 1
    assert dtype.type(1.0) - q > 0
    assert sum(map(Fraction.from_float, _UNIT_SLICE_ROWS[dtype.name])) == 1
    peak = dtype.type(sign) * np.finfo(dtype).max
    model = _outer_weighted_model(dtype=dtype, enable_jit=enable_jit, with_choice=False)
    solution = model.solve(
        params=_outer_weighted_params(model=model, peak=peak), log_level="off"
    )
    np.testing.assert_allclose(
        np.asarray(solution.values[0]["alive"]),
        (peak / dtype.type(2)) * q,
        rtol=32 * np.finfo(dtype).eps,
        atol=0.0,
    )


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


@pytest.mark.parametrize(
    ("n_slice_nodes", "block_size"),
    [(1, 1), (2, 1), (4, 1), (5, 2), (18, 5), (36, 9), (70, 18), (280, 70)],
)
def test_slice_nodes_are_contracted_in_at_most_four_blocks(
    *, n_slice_nodes: int, block_size: int
) -> None:
    """A block holds a quarter of the slice nodes, rounded up."""
    assert _slice_block_size(n_slice_nodes=n_slice_nodes) == block_size


def _read_toy_value(
    *, next_c: IntND, next_a: IntND, next_b: IntND, toy_values: FloatND
) -> FloatND:
    return toy_values[next_c, next_a, next_b]


def _toy_expectation(*, block_size: int | None) -> _ExpectationOverSliceDraws:
    """Expect `toy_values[c, a, b]` over coordinate draw `c`, slice draws `a, b`."""
    return _ExpectationOverSliceDraws(
        interpolator=productmap(
            func=_read_toy_value, variables=("next_c",), batch_sizes={"next_c": 0}
        ),
        slice_draws=("next_a", "next_b"),
        weight_names=("weight_toy__next_a", "weight_toy__next_b"),
        coordinate_weight_names=("weight_toy__next_c",),
        block_size=block_size,
    )


def _toy_inputs(
    *, shape: tuple[int, int, int], dtype: np.dtype, seed: int
) -> dict[str, np.ndarray]:
    """Marginal probability rows for `c, a, b` and a value table on their nodes."""
    rng = np.random.default_rng(seed=seed)
    n_c, n_a, n_b = shape
    return {
        "weight_toy__next_c": rng.dirichlet(np.ones(n_c)).astype(dtype),
        "weight_toy__next_a": rng.dirichlet(np.ones(n_a)).astype(dtype),
        "weight_toy__next_b": rng.dirichlet(np.ones(n_b)).astype(dtype),
        "toy_values": rng.normal(size=shape).astype(dtype),
    }


def _toy_oracle(inputs: dict[str, Any]) -> float:
    """The literal mean over every joint node of positive weight, computed exactly."""
    p_c = inputs["weight_toy__next_c"]
    p_a = inputs["weight_toy__next_a"]
    p_b = inputs["weight_toy__next_b"]
    values = inputs["toy_values"]
    numerator = Fraction(0)
    mass = Fraction(0)
    for c, a, b in itertools.product(range(p_c.size), range(p_a.size), range(p_b.size)):
        weight = (
            Fraction(float(p_c[c])) * Fraction(float(p_a[a])) * Fraction(float(p_b[b]))
        )
        if weight > 0:
            numerator += weight * Fraction(float(values[c, a, b]))
            mass += weight
    return float(numerator / mass)


def _call_toy(
    *, expectation: _ExpectationOverSliceDraws, inputs: dict[str, Any]
) -> Any:
    n_c, n_a, n_b = inputs["toy_values"].shape
    return expectation(
        next_c=jnp.arange(n_c, dtype=jnp.int32),
        next_a=jnp.arange(n_a, dtype=jnp.int32),
        next_b=jnp.arange(n_b, dtype=jnp.int32),
        **{name: jnp.asarray(value) for name, value in inputs.items()},
    )


_TOY_PRECISIONS = [
    pytest.param("x64_enabled", np.dtype("float64"), 1e-10, id="fp64"),
    pytest.param("x64_disabled", np.dtype("float32"), 2e-5, id="fp32"),
]
_TOY_BLOCKINGS = [
    # (coordinate nodes, slice nodes of `a`, slice nodes of `b`), block size.
    pytest.param((3, 1, 1), None, id="single-slice-node"),
    pytest.param((3, 5, 1), None, id="five-nodes-two-blocks-and-a-remainder"),
    pytest.param((4, 3, 4), 5, id="twelve-nodes-blocks-of-five"),
    pytest.param((7, 6, 6), None, id="thirty-six-nodes-four-blocks"),
    pytest.param((7, 6, 6), 1, id="block-of-one-node"),
    pytest.param((7, 6, 6), 36, id="block-of-all-nodes"),
    pytest.param((7, 6, 6), 100, id="block-larger-than-all-nodes"),
]


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(("shape", "block_size"), _TOY_BLOCKINGS)
@pytest.mark.parametrize(("fixture_name", "dtype", "rtol"), _TOY_PRECISIONS)
def test_blocked_slice_expectation_is_the_literal_joint_node_mean(
    *,
    shape: tuple[int, int, int],
    block_size: int | None,
    enable_jit: bool,
    fixture_name: str,
    dtype: np.dtype,
    rtol: float,
    request: pytest.FixtureRequest,
) -> None:
    """Every block size, aligned with the slice-node count or not, gives the mean."""
    request.getfixturevalue(fixture_name)
    inputs = _toy_inputs(shape=shape, dtype=dtype, seed=sum(shape))
    expectation = _toy_expectation(block_size=block_size)
    call = (
        jax.jit(_call_toy, static_argnames="expectation") if enable_jit else _call_toy
    )
    np.testing.assert_allclose(
        float(call(expectation=expectation, inputs=inputs)),
        _toy_oracle(inputs),
        rtol=rtol,
    )


@pytest.mark.parametrize(("fixture_name", "dtype", "rtol"), _TOY_PRECISIONS)
def test_blocked_slice_expectation_maps_over_an_outer_batch(
    *, fixture_name: str, dtype: np.dtype, rtol: float, request: pytest.FixtureRequest
) -> None:
    """Mapped over rows of coordinate weights and values, each row gets its own mean."""
    request.getfixturevalue(fixture_name)
    rows = [_toy_inputs(shape=(7, 5, 2), dtype=dtype, seed=seed) for seed in range(6)]
    shared = {
        name: rows[0][name] for name in ("weight_toy__next_a", "weight_toy__next_b")
    }
    rows = [{**row, **shared} for row in rows]
    expectation = _toy_expectation(block_size=None)

    # keyword-only-exempt: library-callback=jax.vmap
    def expect_row(weights: FloatND, values: FloatND) -> FloatND:
        return _call_toy(
            expectation=expectation,
            inputs={**shared, "weight_toy__next_c": weights, "toy_values": values},
        )

    batched = jax.jit(jax.vmap(expect_row))(
        jnp.asarray(np.stack([row["weight_toy__next_c"] for row in rows])),
        jnp.asarray(np.stack([row["toy_values"] for row in rows])),
    )
    np.testing.assert_allclose(
        np.asarray(batched, dtype=np.float64),
        [_toy_oracle(row) for row in rows],
        rtol=rtol,
    )


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    "dead_nodes",
    [
        pytest.param((0,), id="first-block"),
        pytest.param((4,), id="remainder-block"),
        pytest.param((1, 4), id="both"),
    ],
)
@pytest.mark.parametrize(
    "unread", [np.inf, -np.inf, np.nan], ids=["inf", "-inf", "nan"]
)
@pytest.mark.parametrize(("fixture_name", "dtype", "rtol"), _TOY_PRECISIONS)
def test_zero_probability_slice_node_with_a_nonfinite_read_drops_out(
    *,
    dead_nodes: tuple[int, ...],
    unread: float,
    enable_jit: bool,
    fixture_name: str,
    dtype: np.dtype,
    rtol: float,
    request: pytest.FixtureRequest,
) -> None:
    """A slice node of probability zero leaves the mean of the live nodes finite.

    Five slice nodes form two blocks of two and a remainder block of one, so a
    dead node is placed both in a looped block and in the remainder.
    """
    request.getfixturevalue(fixture_name)
    inputs = _toy_inputs(shape=(4, 5, 1), dtype=dtype, seed=11)
    weights = inputs["weight_toy__next_a"].copy()
    weights[list(dead_nodes)] = 0.0
    inputs["weight_toy__next_a"] = weights / weights.sum(dtype=dtype)
    values = inputs["toy_values"].copy()
    values[:, list(dead_nodes), :] = unread
    inputs["toy_values"] = values
    expectation = _toy_expectation(block_size=None)
    call = (
        jax.jit(_call_toy, static_argnames="expectation") if enable_jit else _call_toy
    )
    np.testing.assert_allclose(
        float(call(expectation=expectation, inputs=inputs)),
        _toy_oracle(inputs),
        rtol=rtol,
    )


@pytest.mark.parametrize("block_size", [1, 8, 1000])
@pytest.mark.parametrize(("fixture_name", "rtol", "tie_gap"), _PRECISIONS)
def test_nested_route_at_every_block_size_matches_the_joint_route(
    *,
    block_size: int,
    fixture_name: str,
    rtol: float,
    tie_gap: float,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The solved value is the same whether the 70 slice nodes are blocked or joint."""
    request.getfixturevalue(fixture_name)
    del tie_gap
    with monkeypatch.context() as patch:
        patch.setattr(Q_and_F_module, "_slice_draws", lambda **_: ())
        joint_model = _model(reads=("income",), slice_draws=True)
        joint = joint_model.solve(params=_params(), log_level="off")
    monkeypatch.setattr(Q_and_F_module, "_slice_block_size", lambda **_: block_size)
    nested_model = _model(reads=("income",), slice_draws=True)
    nested = nested_model.solve(params=_params(), log_level="off")
    np.testing.assert_allclose(
        _solved_alive_values(model=nested_model, solution=nested),
        _solved_alive_values(model=joint_model, solution=joint),
        rtol=rtol,
    )
