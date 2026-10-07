"""A Markov draw read toward a target that does not carry the Markov state.

`alive` and `other` carry the Markov state `shock`; `dead` does not. The bequest
law into `dead`, `next_wealth = (wealth - consumption) * (1 + 0.1 * next_shock)`,
reads the shock's next-period draw, so toward `dead` the draw is taken from the
source's Markov law, with that law's parameters, inside the transition and then
discarded. `other` is reached only at age 1, when every edge leads to `dead`, so
no edge out of `other` carries the shock at all.

At age 1 every subject dies, and `dead` values wealth linearly, so the age-1
value of a living regime is the scalar maximum

```{math}
V_1(w, s) = \\max_{c \\le w} \\log c
    + \\beta \\sum_{s'} P(s' \\mid s)\\,(w - c)(1 + 0.1 s').
```

A target that carries `shock` at weight zero in its utility poses the same
problem, so its living values must coincide with those of the
transition-local draw in every period.
"""

from collections.abc import Callable, Mapping
from functools import partial
from types import MappingProxyType
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.grids import Grid
from lcm import (
    AgeGrid,
    AgeRange,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Transition,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime
from lcm.transition import StochasticTransition
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarFloat,
    ScalarInt,
    UserFunction,
)

DISCOUNT_FACTOR = 0.95
SHOCK_PROBS = np.array([[0.7, 0.3], [0.4, 0.6]])
WEALTH = LinSpacedGrid(start=1.0, stop=5.0, n_points=5)
CONSUMPTION = LinSpacedGrid(start=0.5, stop=1.0, n_points=3)
DEAD_WEALTH = LinSpacedGrid(start=0.0, stop=5.0, n_points=5)


@categorical(ordered=False)
class Shock:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    other: ScalarInt
    dead: ScalarInt


def _next_shock_probs(*, shock: DiscreteState, stay_low: float) -> FloatND:
    probs = jnp.array([[stay_low, 1.0 - stay_low], [0.4, 0.6]])
    return probs[shock]


def _wealth_alive(
    *, wealth: ContinuousState, consumption: ContinuousAction, next_shock: DiscreteState
) -> ContinuousState:
    return (wealth - consumption) * (1.0 + 0.1 * next_shock)


def _wealth_other(
    *, wealth: ContinuousState, consumption: ContinuousAction, next_shock: DiscreteState
) -> ContinuousState:
    return (wealth - consumption) * (1.0 + 0.2 * next_shock)


def _bequest(
    *, wealth: ContinuousState, consumption: ContinuousAction, next_shock: DiscreteState
) -> ContinuousState:
    return (wealth - consumption) * (1.0 + 0.1 * next_shock)


def _utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _p_alive(age: float) -> ScalarFloat:
    return jnp.where(age < 1, 0.5, 0.0)


def _p_other(age: float) -> ScalarFloat:
    return jnp.where(age < 1, 0.5, 0.0)


def _p_dead(age: float) -> ScalarFloat:
    return jnp.where(age < 1, 0.0, 1.0)


def _feasible(*, wealth: ContinuousState, consumption: ContinuousAction) -> BoolND:
    return consumption <= wealth


def _dead_utility(wealth: ContinuousState) -> FloatND:
    return wealth


def _dead_utility_carrying_shock(
    *, wealth: ContinuousState, shock: DiscreteState
) -> FloatND:
    return wealth + 0 * shock


_SHOCK_LAW = StochasticTransition(func=_next_shock_probs)


def _keep_level(level: DiscreteState) -> DiscreteState:
    return level


def _bequest_reading_level(
    *, wealth: ContinuousState, consumption: ContinuousAction, next_level: DiscreteState
) -> ContinuousState:
    return (wealth - consumption) * (1.0 + 0.1 * next_level)


def _living(
    *,
    shock_law: StochasticTransition | Mapping[str, StochasticTransition] = (_SHOCK_LAW),
    shock_at_regime: bool = True,
    extra_states: Mapping[str, Grid] = MappingProxyType({}),
    extra_transitions: Mapping[str, UserFunction] = MappingProxyType({}),
    bequest: Callable[..., ContinuousState] = _bequest,
) -> Regime:
    return Regime(
        states={
            "wealth": WEALTH,
            **({"shock": DiscreteGrid(Shock)} if shock_at_regime else {}),
            **extra_states,
        },
        actions={"consumption": CONSUMPTION},
        state_transitions={
            "wealth": {"alive": _wealth_alive, "other": _wealth_other, "dead": bequest},
            **({"shock": shock_law} if shock_at_regime else {}),
            **extra_transitions,
        },
        functions={"utility": _utility},
        constraints={"feasible": _feasible},
    )


def _model(
    *,
    dead_carries_shock: bool,
    living: Callable[[], Regime] = _living,
    declared_at: Literal["model", "regime"] = "regime",
) -> Model:
    dead = (
        Regime(
            states={"wealth": DEAD_WEALTH, "shock": DiscreteGrid(Shock)},
            functions={"utility": _dead_utility_carrying_shock},
        )
        if dead_carries_shock
        else Regime(
            states={"wealth": DEAD_WEALTH},
            functions={"utility": _dead_utility},
        )
    )
    return Model(
        regimes={"alive": living(), "other": living(), "dead": dead},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        edges={
            source: Transition(
                targets={
                    "alive": 0,
                    "other": 0,
                    "dead": AgeRange(start=0, exclusive_stop=2),
                },
                law={
                    "alive": StochasticTransition(func=_p_alive),
                    "other": StochasticTransition(func=_p_other),
                    "dead": StochasticTransition(func=_p_dead),
                },
            )
            for source in ("alive", "other")
        },
        initial_nodes=((0, "alive"),),
        **(
            {
                "states": {"shock": DiscreteGrid(Shock)},
                "state_transitions": {
                    "shock": StochasticTransition(func=_next_shock_probs)
                },
            }
            if declared_at == "model"
            else {}
        ),
    )


_PARAMS = {
    regime: {
        "koopmans_aggregator": {"discount_factor": DISCOUNT_FACTOR},
        "next_shock": {"stay_low": SHOCK_PROBS[0, 0]},
    }
    for regime in ("alive", "other")
}


def _living_values(model: Model) -> dict[tuple[int, str], np.ndarray]:
    """Living values per `(period, regime)`, indexed `(wealth, shock)`."""
    solution = model.solve(params=_PARAMS, log_level="off")
    values = {}
    for period, by_regime in solution.values.items():
        for regime in ("alive", "other"):
            if regime in by_regime:
                order = model.state_names(regime_name=regime)
                axes = [order.index(name) for name in ("wealth", "shock")]
                values[(period, regime)] = np.transpose(
                    np.asarray(by_regime[regime]), axes
                )
    return values


def _age_one_value_by_scalar_loop() -> np.ndarray:
    """`V_1(w, s)` from a literal loop over states, actions and next shocks."""
    wealth = np.asarray(WEALTH.to_jax(), dtype=np.float64)
    consumption = np.asarray(CONSUMPTION.to_jax(), dtype=np.float64)
    expected = np.empty((wealth.size, 2))
    for i, w in enumerate(wealth):
        for s in range(2):
            best = -np.inf
            for c in consumption:
                if c > w:
                    continue
                continuation = 0.0
                for s_next in range(2):
                    continuation += (
                        SHOCK_PROBS[s, s_next] * (w - c) * (1 + 0.1 * s_next)
                    )
                best = max(best, np.log(c) + DISCOUNT_FACTOR * continuation)
            expected[i, s] = best
    return expected


def test_the_non_carrying_target_keeps_only_its_own_states() -> None:
    model = _model(dead_carries_shock=False)

    assert model.state_names(regime_name="dead") == ("wealth",)


def test_age_one_value_matches_the_scalar_loop_reference() -> None:
    values = _living_values(_model(dead_carries_shock=False))
    expected = _age_one_value_by_scalar_loop()

    for regime in ("alive", "other"):
        np.testing.assert_allclose(
            values[(1, regime)], expected, rtol=0, atol=1e-5, err_msg=regime
        )


def test_the_reference_discriminates_between_lagged_shocks() -> None:
    """The lagged shock moves the age-one value well beyond the tolerance."""
    expected = _age_one_value_by_scalar_loop()

    assert np.ptp(expected[-1, :]) > 0.1


def test_values_match_a_target_that_carries_the_shock_at_weight_zero() -> None:
    local = _living_values(_model(dead_carries_shock=False))
    carried = _living_values(_model(dead_carries_shock=True))

    assert (
        sorted(local) == sorted(carried) == [(0, "alive"), (1, "alive"), (1, "other")]
    )
    for key, value in carried.items():
        np.testing.assert_allclose(local[key], value, rtol=0, atol=1e-6, err_msg=key)


def test_a_model_level_markov_state_is_kept_only_where_its_draw_is_read() -> None:
    model = _model(
        dead_carries_shock=False,
        living=partial(_living, shock_at_regime=False),
        declared_at="model",
    )

    assert {
        regime: set(model.state_names(regime_name=regime))
        for regime in ("alive", "other", "dead")
    } == {
        "alive": {"wealth", "shock"},
        "other": {"wealth", "shock"},
        "dead": {"wealth"},
    }


def test_a_model_level_markov_state_gives_the_regime_level_values() -> None:
    model_level = _living_values(
        _model(
            dead_carries_shock=False,
            living=partial(_living, shock_at_regime=False),
            declared_at="model",
        )
    )
    regime_level = _living_values(_model(dead_carries_shock=False))

    assert sorted(model_level) == sorted(regime_level)
    for key, value in regime_level.items():
        np.testing.assert_allclose(
            model_level[key], value, rtol=0, atol=1e-12, err_msg=key
        )


def test_a_law_reading_the_next_value_of_a_state_the_target_lacks_is_refused() -> None:
    """`dead` carries no `level` and no draw of a deterministic state is taken."""
    living = partial(
        _living,
        extra_states={"level": DiscreteGrid(Shock)},
        extra_transitions={"level": _keep_level},
        bequest=_bequest_reading_level,
    )

    with pytest.raises(
        ModelInitializationError,
        match=r"'next_wealth' from 'alive' to 'dead' reads 'next_level'",
    ):
        _model(dead_carries_shock=False, living=living)


def test_a_markov_draw_needs_a_law_declared_once_for_every_target() -> None:
    """A per-target Markov law defines no draw toward a target it does not name."""
    law = StochasticTransition(func=_next_shock_probs)
    living = partial(_living, shock_law={"alive": law, "other": law})

    with pytest.raises(
        ModelInitializationError,
        match=r"'next_wealth' from 'alive' to 'dead' reads 'next_shock'",
    ):
        _model(dead_carries_shock=False, living=living)
