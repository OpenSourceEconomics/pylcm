"""Spec for EV1 taste shocks in a household in which exactly one member decides.

The member with the positive Pareto weight decides. Its value is the smoothed
expected maximum of its own action values: the hard maximum over the
continuous actions, then `scale * logsumexp(Qc / scale)` over the discrete
ones. Every other member's value is the mean of its own action value at the
decider's continuous argmax, under the decider's choice probabilities over the
discrete actions.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from lcm import (
    AgeGrid,
    CollectiveUtility,
    DiscreteGrid,
    Exit,
    ExitTarget,
    InitialNodes,
    Model,
    ParetoObjective,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import InvalidParamsError, UnsupportedOperationError
from lcm.exits import Unilateral
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.typing import (
    BoolND,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
    UserFunction,
    UserParamsNode,
)
from tests.conftest import DECIMAL_PRECISION, assert_agrees_to_ulp
from tests.simulation.initial_conditions.test_own_member_required import (
    ConsentRegimeId,
    _make_consent_regimes,
    _u_married_f,
    _u_married_m,
)
from tests.test_models import taste_shocks_toy

_SCALE = 0.2
_DISCOUNT_FACTOR = 0.95


def _reference_household_V(
    *, decider: str, names: tuple[str, ...], scale: float, discount_factor: float
) -> np.ndarray:
    """Recompute the household's period-0 values with numpy on the same grids.

    Each member's action value is formed literally from its own utilities on
    the toy's grids; the decider's discrete choice probabilities then weight
    every other member's value at the decider's continuous argmax.
    """
    wealth = np.linspace(1.0, 10.0, 6)
    terminal_wealth = np.linspace(0.0, 12.0, 25)
    consumption = np.linspace(0.5, 5.0, 8)
    v_done = {
        "x": np.log(terminal_wealth + 1.0),
        "y": np.sqrt(terminal_wealth),
    }
    flow = {
        "x": lambda c, d: np.log(c) - taste_shocks_toy.KAPPA * d,
        "y": lambda c, d: np.sqrt(c) + taste_shocks_toy.PARTNER_WORK_TASTE * d,
    }
    feasible = consumption[None, :] <= wealth[:, None]
    q = {}
    for name in names:
        q[name] = np.empty((wealth.size, 2, consumption.size))
        for d in (0, 1):
            next_w = wealth[:, None] - consumption[None, :] + taste_shocks_toy.WAGE * d
            q[name][:, d, :] = flow[name](
                consumption[None, :], d
            ) + discount_factor * np.interp(next_w, terminal_wealth, v_done[name])
    q_decider = np.where(feasible[:, None, :], q[decider], -np.inf)
    best_c = np.argmax(q_decider, axis=2)
    qc = np.max(q_decider, axis=2)
    shifted = (qc - qc.max(axis=1, keepdims=True)) / scale
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    values = {
        decider: qc.max(axis=1) + scale * np.log(np.exp(shifted).sum(axis=1)),
    }
    for name in names:
        if name != decider:
            at_best = np.take_along_axis(q[name], best_c[:, :, None], axis=2)[..., 0]
            values[name] = np.sum(probabilities * at_best, axis=1)
    return np.stack([values[name] for name in names], axis=-1)


def test_household_of_one_decider_agrees_with_the_single() -> None:
    """A household of one member with weight 1 has the single's smoothed value.

    The same model written once with a plain utility and once as a household
    of one: the published values agree to within 16 units in the last place.
    """
    single = taste_shocks_toy.get_model().solve(
        params=taste_shocks_toy.get_params(
            scale=_SCALE, discount_factor=_DISCOUNT_FACTOR
        ),
        log_level="debug",
    )
    household = taste_shocks_toy.get_household_model(weights={"x": 1.0}).solve(
        params=taste_shocks_toy.get_household_params(
            scale=_SCALE, discount_factor=_DISCOUNT_FACTOR
        ),
        log_level="debug",
    )

    assert_agrees_to_ulp(
        got=np.asarray(household.values[0]["alive"])[..., 0],
        expected=np.asarray(single.values[0]["alive"]),
        n_ulp=16,
    )


@pytest.mark.parametrize(
    ("weights", "decider"),
    [({"x": 1.0, "y": 0.0}, "x"), ({"x": 0.0, "y": 1.0}, "y")],
)
def test_household_values_match_the_numpy_reference(
    *, weights: dict[str, float], decider: str
) -> None:
    """The decider's logsum and the other member's probability-weighted mean.

    Either member may be the decider; which one is read from the weights.
    """
    household = taste_shocks_toy.get_household_model(weights=weights).solve(
        params=taste_shocks_toy.get_household_params(
            scale=_SCALE, discount_factor=_DISCOUNT_FACTOR
        ),
        log_level="debug",
    )

    aaae(
        np.asarray(household.values[0]["alive"]),
        _reference_household_V(
            decider=decider,
            names=("x", "y"),
            scale=_SCALE,
            discount_factor=_DISCOUNT_FACTOR,
        ),
        decimal=DECIMAL_PRECISION,
    )


_AGES = AgeGrid(start=0, inclusive_stop=2, step="Y")


@categorical(ordered=False)
class Choice:
    a: ScalarInt
    b: ScalarInt


@categorical(ordered=False)
class Power:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class RegimeId:
    couple: ScalarInt
    couple_terminal: ScalarInt


def _utility_f_minus_inf(choice: DiscreteAction) -> FloatND:
    """Her value of `a` is an admissible `-inf`, not an infeasibility."""
    return jnp.where(choice == Choice.a, -jnp.inf, 0.0)


def _utility_m_prefers_a(choice: DiscreteAction) -> FloatND:
    return jnp.where(choice == Choice.a, 5.0, 1.0)


def _utility_f_one(choice: DiscreteAction) -> FloatND:
    return jnp.where(choice == Choice.a, 1.0, 3.0)


def _utility_m_minus_inf_at_b(choice: DiscreteAction) -> FloatND:
    """His value of `b` is `-inf`; `b` is infeasible for the household."""
    return jnp.where(choice == Choice.a, 2.0, -jnp.inf)


def _only_a(choice: DiscreteAction) -> BoolND:
    return choice == Choice.a


def _terminal_zero() -> FloatND:
    return jnp.asarray(0.0)


def _weight_f_by_power(power: DiscreteState) -> FloatND:
    return jnp.where(power == Power.high, 1.0, 0.0)


def _weight_m_by_power(power: DiscreteState) -> FloatND:
    """His weight is positive at every cell, hers only where she holds power."""
    return jnp.where(power == Power.high, 0.5, 1.0)


def _weight_m_complement(power: DiscreteState) -> FloatND:
    return jnp.where(power == Power.high, 0.0, 1.0)


def _couple(
    *,
    objective: ParetoObjective,
    utility_f: UserFunction,
    utility_m: UserFunction,
    constraints: dict[str, UserFunction] | None = None,
    with_power: bool = False,
    exit_declaration: Exit | None = None,
) -> Regime:
    return Regime(
        states={"power": DiscreteGrid(category_class=Power)} if with_power else {},
        state_transitions={"power": fixed_transition("power")} if with_power else {},
        actions={"choice": DiscreteGrid(category_class=Choice)},
        constraints=constraints or {},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": utility_f, "m": utility_m}, objective=objective
            )
        },
        taste_shocks=ExtremeValueTasteShocks(),
        exit=exit_declaration,
    )


def _couple_model(couple: Regime) -> Model:
    couple_terminal = Regime(
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _terminal_zero, "m": _terminal_zero}
            )
        },
    )
    return Model(
        edges={"couple": {"couple_terminal": 0}},
        regimes={"couple": couple, "couple_terminal": couple_terminal},
        ages=_AGES,
        regime_id_class=RegimeId,
        initial_nodes={0: "couple"},
    )


def _couple_params(*, scale: float) -> dict[str, UserParamsNode]:
    return {
        "couple": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            "taste_shocks": {"scale": scale},
        },
        "couple_terminal": {},
    }


@pytest.mark.parametrize(
    ("weights", "hard_max_values"),
    [({"f": 0.0, "m": 1.0}, [-np.inf, 5.0]), ({"f": 1.0, "m": 0.0}, [0.0, 1.0])],
)
def test_a_vanishing_scale_recovers_the_hard_max_household(
    *, weights: dict[str, float], hard_max_values: list[float]
) -> None:
    """With a tiny scale the household publishes the hard-max pair.

    These are the values the same household publishes without taste shocks:
    weighting only him picks `a` for his 5, where her own value is `-inf`;
    weighting only her picks `b`, where she is at 0 and he is at 1. At scale
    `1e-3` the losing action's probability is `exp(-4000)`, which is zero in
    either precision, so the pair is exact.
    """
    model = _couple_model(
        _couple(
            objective=ParetoObjective(weights=weights),
            utility_f=_utility_f_minus_inf,
            utility_m=_utility_m_prefers_a,
        )
    )

    solution = model.solve(params=_couple_params(scale=1e-3), log_level="debug")

    assert np.asarray(solution.values[0]["couple"]).tolist() == hard_max_values


def test_an_infeasible_action_leaves_the_other_members_value_finite() -> None:
    """An infeasible action carries probability zero and contributes nothing.

    `b` is infeasible, so she takes `a` with probability one and her value is
    her 1. His value at `b` is `-inf`; weighted by its zero probability it
    contributes exactly zero rather than `0 * -inf`, so his value is his 2.
    """
    model = _couple_model(
        _couple(
            objective=ParetoObjective(weights={"f": 1.0, "m": 0.0}),
            utility_f=_utility_f_one,
            utility_m=_utility_m_minus_inf_at_b,
            constraints={"only_a": _only_a},
        )
    )

    solution = model.solve(params=_couple_params(scale=0.5), log_level="debug")

    assert np.asarray(solution.values[0]["couple"]).tolist() == [1.0, 2.0]


def _utility_f_minus_inf_everywhere(choice: DiscreteAction) -> FloatND:
    return jnp.where(choice == Choice.a, -jnp.inf, -jnp.inf)


def _utility_m_prefers_b(choice: DiscreteAction) -> FloatND:
    return jnp.where(choice == Choice.a, 2.0, 7.0)


def test_a_decider_without_a_finite_action_takes_the_first_action() -> None:
    """With every action worth `-inf` to her, she takes `a` with probability one.

    Her value is `-inf`; his is his value of `a`, the action her simulated
    draw takes when no action is worth anything to her, not his 7 at `b`.
    """
    model = _couple_model(
        _couple(
            objective=ParetoObjective(weights={"f": 1.0, "m": 0.0}),
            utility_f=_utility_f_minus_inf_everywhere,
            utility_m=_utility_m_prefers_b,
        )
    )

    solution = model.solve(params=_couple_params(scale=0.5), log_level="debug")

    assert np.asarray(solution.values[0]["couple"]).tolist() == [-np.inf, 2.0]


@pytest.mark.parametrize(
    "objective",
    [
        ParetoObjective(weights={"f": 0.5, "m": 0.5}),
        ParetoObjective(weights={"f": 0.3, "m": 0.7}, normalization="none"),
    ],
)
def test_two_positive_constant_weights_with_taste_shocks_are_refused(
    objective: ParetoObjective,
) -> None:
    """Taste shocks need exactly one member with a positive weight."""
    with pytest.raises(NotImplementedError, match="exactly one member"):
        _couple(
            objective=objective,
            utility_f=_utility_f_one,
            utility_m=_utility_m_prefers_a,
        )


def test_a_weight_positive_for_two_members_at_some_cell_is_refused_at_solve() -> None:
    """A state-dependent weighting with two deciders in one cell stops the solve.

    Where she holds the power both weights are positive, so that cell has no
    single decider. The declaration is admissible; its values are not.
    """
    model = _couple_model(
        _couple(
            objective=ParetoObjective(
                weights={"f": _weight_f_by_power, "m": _weight_m_by_power}
            ),
            utility_f=_utility_f_one,
            utility_m=_utility_m_prefers_a,
            with_power=True,
        )
    )

    with pytest.raises(InvalidParamsError, match="exactly one member"):
        model.solve(params=_couple_params(scale=0.5), log_level="debug")


def test_a_weight_with_one_decider_at_every_cell_is_admitted() -> None:
    """She decides where she holds the power and he decides elsewhere.

    At `scale = 1e-3` each cell publishes the decider's hard-max pair: at his
    cell he picks `a` (5 against 1) and she is at 1; at hers she picks `b`
    (3 against 1) and he is at 1.
    """
    model = _couple_model(
        _couple(
            objective=ParetoObjective(
                weights={"f": _weight_f_by_power, "m": _weight_m_complement}
            ),
            utility_f=_utility_f_one,
            utility_m=_utility_m_prefers_a,
            with_power=True,
        )
    )

    solution = model.solve(params=_couple_params(scale=1e-3), log_level="debug")

    assert np.asarray(solution.values[0]["couple"]).tolist() == [
        [1.0, 5.0],
        [3.0, 1.0],
    ]


def test_a_household_with_taste_shocks_whose_decision_an_exit_reads_is_refused() -> (
    None
):
    """A single's consented exit into a taste-shock household is refused.

    Her consent reads the household's decision, which is not implemented under
    taste shocks; the message names the household.
    """
    regimes = _make_consent_regimes()
    married = regimes["married"]
    regimes["married"] = married.replace(
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _u_married_f, "m": _u_married_m},
                objective=ParetoObjective(weights={"f": 1.0, "m": 0.0}),
            )
        },
        taste_shocks=ExtremeValueTasteShocks(),
    )

    with pytest.raises(UnsupportedOperationError, match="'married' is a household"):
        Model(
            regimes=regimes,
            n_periods=1,
            regime_id_class=ConsentRegimeId,
            initial_nodes=InitialNodes(by_period={0: "single_f"}),
            edges={},
        )


def test_taste_shocks_in_a_household_with_an_exit_are_refused() -> None:
    """A household that may exit compares member values its smoothing does not own."""
    with pytest.raises(NotImplementedError, match="exit"):
        _couple(
            objective=ParetoObjective(weights={"f": 1.0, "m": 0.0}),
            utility_f=_utility_f_one,
            utility_m=_utility_m_prefers_a,
            exit_declaration=Exit(
                targets={
                    "f": ExitTarget(regime="single", projection={}),
                    "m": ExitTarget(regime="single", projection={}),
                },
                rule=Unilateral(),
            ),
        )


def test_taste_shocks_in_a_household_with_current_period_shocks_are_refused() -> None:
    """The smoothed household reduction under an observed shock is not covered."""
    couple = _couple(
        objective=ParetoObjective(weights={"f": 1.0, "m": 0.0}),
        utility_f=_utility_f_one,
        utility_m=_utility_m_prefers_a,
    )
    with pytest.raises(NotImplementedError, match="`shocks`"):
        couple.replace(shocks=("meeting",))
