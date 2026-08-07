"""A regime transition must be a distribution, not merely sum to one.

Unit mass is checked as arithmetic rather than as validation, so it holds at every
log level. It cannot on its own decide that a collection of weights is a
distribution: probabilities of `1.5` and `-0.5` sum to one. Non-negativity is
therefore checked the same way, and the two together give the full range, since
non-negative weights summing to one each lie in `[0, 1]`.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    PowerMean,
    Regime,
    categorical,
)
from lcm.exceptions import InvalidRegimeTransitionProbabilitiesError
from lcm.typing import ScalarFloat, ScalarInt

_WEALTH = LinSpacedGrid(start=1.0, stop=4.0, n_points=4)
_PARAMS = {"source": {"koopmans_aggregator": {"discount_factor": 1.0}}}
_POWER_MEAN_PARAMS = {
    "source": {
        "koopmans_aggregator": {"discount_factor": 1.0},
        "certainty_equivalent": {"risk_aversion": 2.0},
    }
}


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    a: ScalarInt
    b: ScalarInt


def _no_utility() -> ScalarFloat:
    return jnp.float32(0)


def _keep(wealth: ScalarFloat) -> ScalarFloat:
    return wealth


def _pays_wealth(wealth: ScalarFloat) -> ScalarFloat:
    return wealth


def _pays_ten_times(wealth: ScalarFloat) -> ScalarFloat:
    return 10.0 * wealth


def _build(probability_a, probability_b, certainty_equivalent=None) -> Model:
    def _to_a() -> ScalarFloat:
        return jnp.float32(probability_a)

    def _to_b() -> ScalarFloat:
        return jnp.float32(probability_b)

    return Model(
        regimes={
            "source": Regime(
                transition={"a": MarkovTransition(_to_a), "b": MarkovTransition(_to_b)},
                active=lambda age: age < 21,
                states={"wealth": _WEALTH},
                state_transitions={"wealth": _keep},
                functions={"utility": _no_utility},
                certainty_equivalent=certainty_equivalent,
            ),
            "a": Regime(
                transition=None,
                states={"wealth": _WEALTH},
                functions={"utility": _pays_wealth},
            ),
            "b": Regime(
                transition=None,
                states={"wealth": _WEALTH},
                functions={"utility": _pays_ten_times},
            ),
        },
        ages=AgeGrid(start=20, stop=21, step="Y"),
        regime_id_class=RegimeId,
    )


@pytest.mark.parametrize(
    "certainty_equivalent", [None, PowerMean()], ids=["linear", "power_mean"]
)
def test_a_negative_regime_probability_is_refused_even_at_unit_mass(
    certainty_equivalent,
) -> None:
    """`1.5` on one target and `-0.5` on another is not a distribution.

    The two sum to one, so the mass budget alone accepts them. Dropping the
    negative target instead would publish `1.5 * wealth` — precisely the value a
    well-posed model with all its mass on the first target would produce.
    """
    model = _build(1.5, -0.5, certainty_equivalent=certainty_equivalent)

    V = model.solve(
        params=_PARAMS if certainty_equivalent is None else _POWER_MEAN_PARAMS,
        log_level="off",
    )

    assert bool(jnp.all(jnp.isnan(jnp.asarray(V[0]["source"]))))


def test_a_well_formed_regime_transition_is_untouched() -> None:
    """`0.25` and `0.75` pay `0.25 * w + 0.75 * 10w = 7.75 * w`."""
    model = _build(0.25, 0.75)

    V = model.solve(params=_PARAMS, log_level="off")

    np.testing.assert_allclose(
        np.asarray(V[0]["source"]),
        np.asarray([7.75, 15.5, 23.25, 31.0]),
        rtol=1e-6,
    )


@categorical(ordered=False)
class RegimeIdWithInactiveTargets:
    source: ScalarInt
    live: ScalarInt
    gone_a: ScalarInt
    gone_b: ScalarInt


def test_a_signed_cell_on_a_target_that_drops_out_is_refused_by_validation() -> None:
    """The declared transition is checked as written, not as it survives pruning.

    A target that activity makes unreachable is a legal declaration — it simply
    needs no handoff — and its cell is dropped before any continuation is
    built. Nothing downstream can see it, so `+0.5` and `-0.5` on two such
    targets cancel and leave the live target's `1.0` looking well formed.
    Validation reads the transition as declared and refuses it.
    """

    def _all_mass_to_live() -> ScalarFloat:
        return jnp.float32(1.0)

    def _positive_on_a_dead_target() -> ScalarFloat:
        return jnp.float32(0.5)

    def _negative_on_a_dead_target() -> ScalarFloat:
        return jnp.float32(-0.5)

    def _terminal(active) -> Regime:
        return Regime(
            transition=None,
            active=active,
            states={"wealth": _WEALTH},
            functions={"utility": _pays_wealth},
        )

    model = Model(
        regimes={
            "source": Regime(
                transition={
                    "live": MarkovTransition(_all_mass_to_live),
                    "gone_a": MarkovTransition(_positive_on_a_dead_target),
                    "gone_b": MarkovTransition(_negative_on_a_dead_target),
                },
                active=lambda age: age < 21,
                states={"wealth": _WEALTH},
                state_transitions={"wealth": _keep},
                functions={"utility": _no_utility},
            ),
            "live": _terminal(lambda _age: True),
            "gone_a": _terminal(lambda age: age < 21),
            "gone_b": _terminal(lambda age: age < 21),
        },
        ages=AgeGrid(start=20, stop=21, step="Y"),
        regime_id_class=RegimeIdWithInactiveTargets,
    )

    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError, match=r"outside \[0, 1\]"
    ):
        model.solve(params=_PARAMS, log_level="debug")
