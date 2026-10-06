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
    AgeRange,
    ByAge,
    LinSpacedGrid,
    Model,
    PowerMean,
    Regime,
    StochasticTransition,
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


def _build(*, probability_a, probability_b, certainty_equivalent=None) -> Model:
    def _to_a() -> ScalarFloat:
        return jnp.float32(probability_a)

    def _to_b() -> ScalarFloat:
        return jnp.float32(probability_b)

    return Model(
        edges={"source": {"a": 20, "b": 20}},
        regimes={
            "source": Regime(
                regime_transitions={
                    "a": StochasticTransition(func=_to_a),
                    "b": StochasticTransition(func=_to_b),
                },
                states={"wealth": _WEALTH},
                state_transitions={"wealth": _keep},
                functions={"utility": _no_utility},
                certainty_equivalent=certainty_equivalent,
            ),
            "a": Regime(
                regime_transitions=None,
                states={"wealth": _WEALTH},
                functions={"utility": _pays_wealth},
            ),
            "b": Regime(
                regime_transitions=None,
                states={"wealth": _WEALTH},
                functions={"utility": _pays_ten_times},
            ),
        },
        ages=AgeGrid(start=20, inclusive_stop=21, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={20: "source"},
    )


@pytest.mark.parametrize(
    "certainty_equivalent", [None, PowerMean()], ids=["linear", "power_mean"]
)
def test_a_negative_regime_probability_is_refused_even_at_unit_mass(
    certainty_equivalent,
) -> None:
    """`1.5` on one target and `-0.5` on another is not a distribution.

    The two sum to one, so the mass budget alone accepts them; regime selection
    refuses the negative entry at every log level, `"off"` included.
    """
    model = _build(
        probability_a=1.5, probability_b=-0.5, certainty_equivalent=certainty_equivalent
    )

    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError, match=r"outside \[0, 1\]"
    ):
        model.solve(
            params=_PARAMS if certainty_equivalent is None else _POWER_MEAN_PARAMS,
            log_level="off",
        )


def test_a_well_formed_regime_transition_is_untouched() -> None:
    """`0.25` and `0.75` pay `0.25 * w + 0.75 * 10w = 7.75 * w`."""
    model = _build(probability_a=0.25, probability_b=0.75)

    V = model.solve(params=_PARAMS, log_level="off").values

    np.testing.assert_allclose(
        np.asarray(V[0]["source"]),
        np.asarray([7.75, 15.5, 23.25, 31.0]),
        rtol=1e-6,
    )


@categorical(ordered=False)
class RegimeIdWithSignedTargets:
    source: ScalarInt
    live: ScalarInt
    gone_a: ScalarInt
    gone_b: ScalarInt


def test_signed_cells_that_cancel_across_targets_are_refused_by_validation() -> None:
    """Every declared cell is checked, not only the row sum.

    `+0.5` and `-0.5` on two targets cancel and leave the row summing to one
    alongside the live target's `1.0`. Validation reads each cell as declared
    and refuses the negative one.
    """

    def _all_mass_to_live() -> ScalarFloat:
        return jnp.float32(1.0)

    def _positive_on_a_dead_target() -> ScalarFloat:
        return jnp.float32(0.5)

    def _negative_on_a_dead_target() -> ScalarFloat:
        return jnp.float32(-0.5)

    def _terminal() -> Regime:
        return Regime(
            regime_transitions=None,
            states={"wealth": _WEALTH},
            functions={"utility": _pays_wealth},
        )

    model = Model(
        edges={"source": {"live": 20, "gone_a": 20, "gone_b": 20}},
        regimes={
            "source": Regime(
                regime_transitions=ByAge(
                    cases={
                        AgeRange(exclusive_stop=21): {
                            "live": StochasticTransition(func=_all_mass_to_live),
                            "gone_a": StochasticTransition(
                                func=_positive_on_a_dead_target
                            ),
                            "gone_b": StochasticTransition(
                                func=_negative_on_a_dead_target
                            ),
                        }
                    }
                ),
                states={"wealth": _WEALTH},
                state_transitions={"wealth": _keep},
                functions={"utility": _no_utility},
            ),
            "live": _terminal(),
            "gone_a": _terminal(),
            "gone_b": _terminal(),
        },
        ages=AgeGrid(start=20, inclusive_stop=21, step="Y"),
        regime_id_class=RegimeIdWithSignedTargets,
        initial_nodes={20: "source"},
    )

    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError, match=r"outside \[0, 1\]"
    ):
        model.solve(params=_PARAMS, log_level="debug")
