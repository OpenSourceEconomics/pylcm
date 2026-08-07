"""A declared entry may depend on a draw taken on entry to the same regime.

An agent entering a regime may have its entry value determined by a shock drawn at
that moment — entry income that is a function of the entry shock. The entry names one
value per node of that draw, so the continuation is

```{math}
\\sum_j p_j \\, V\\!\\left(g(\\varepsilon_j),\\, \\varepsilon_j\\right),
```

taken on the nodes the draw already carries. The value is a value, not a lottery: the
coefficients placing it on the target's nodes are contracted at each node before the
draw is averaged over, which is what distinguishes it from a second source of risk.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    PowerMean,
    Regime,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.typing import ContinuousState, FloatND, ScalarFloat, ScalarInt

# Symmetric nodes on `(0, 1, 2)`, so the draw has mean one whatever weights the
# discretization assigns them.
_THREE_NODES = NormalIIDProcess(
    n_points=3, gauss_hermite=False, mu=1.0, sigma=0.5, n_std=2.0
)
_RUNTIME_NOISE = NormalIIDProcess(n_points=3, gauss_hermite=False)
_RUNTIME_PARAMS = {"mu": 0.0, "sigma": 0.5, "n_std": 2.0}
_WEALTH = LinSpacedGrid(start=0.0, stop=4.0, n_points=5)

_AGES = AgeGrid(start=20, stop=22, step="Y")
_DISCOUNT = {"source": {"koopmans_aggregator": {"discount_factor": 1.0}}}


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


def _to_target() -> ScalarFloat:
    return jnp.float32(1)


def _no_utility() -> ScalarFloat:
    return jnp.float32(0)


def _income_from_draw(next_shock: ContinuousState) -> ScalarFloat:
    return next_shock


def _scaled_draw(next_shock: ContinuousState) -> ScalarFloat:
    return next_shock


def _income_from_helper(scaled_draw: ScalarFloat) -> ScalarFloat:
    return scaled_draw


def _income_plus_shock(income: ScalarFloat, shock: ScalarFloat) -> ScalarFloat:
    return income + shock


@pytest.fixture(params=["direct", "through_a_helper"])
def model(request: pytest.FixtureRequest) -> Model:
    """Source entering the target's `income` process at the entry shock."""
    functions: dict = {"utility": _no_utility}
    if request.param == "direct":
        income_law = _income_from_draw
    else:
        functions["scaled_draw"] = _scaled_draw
        income_law = _income_from_helper

    return Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_to_target)},
                active=lambda age: age < 22,
                state_transitions={"income": {"target": income_law}},
                functions=functions,
            ),
            "target": Regime(
                transition=None,
                states={"income": _THREE_NODES, "shock": _THREE_NODES},
                functions={"utility": _income_plus_shock},
            ),
        },
        ages=_AGES,
        regime_id_class=RegimeId,
    )


def test_the_entry_is_evaluated_at_each_node_of_the_draw(model: Model) -> None:
    """`V = E[income + shock]` with `income = shock` gives `2 E[shock] = 2`.

    The entry names the drawn value itself, so the target is paid twice the draw at
    every node, and the expectation is two for any weights that make the symmetric
    node set average to one.
    """
    V = model.solve(params=_DISCOUNT, log_level="off")

    np.testing.assert_allclose(
        np.asarray(V[0]["source"]).ravel(), np.array([2.0]), atol=1e-5
    )


def test_the_draw_is_not_asked_of_the_user_as_a_parameter(model: Model) -> None:
    """No parameter slot appears for the draw the model resolves itself."""
    assert "next_shock" not in str(model.get_params_template()["source"])


def _wealth_from_fixed_draw(next_shock: ContinuousState) -> ScalarFloat:
    return 2.0 * next_shock


def _utility_reading_the_noise(noise: ScalarFloat) -> ScalarFloat:
    """Keeps the carried process alive without changing any value."""
    return 0.0 * noise


def _wealth_and_shock(
    wealth: ScalarFloat, shock: ScalarFloat, noise: ScalarFloat
) -> ScalarFloat:
    return wealth + shock + 0.0 * noise


def test_an_unread_runtime_process_does_not_block_a_fixed_draw() -> None:
    """Only the draws a dependent law reads need their nodes fixed at construction.

    `next_wealth` reads `next_shock`, whose nodes are fixed. The `noise` process is
    carried across and parameterized at solve time, but nothing reads `next_noise`,
    so it has no bearing on what the dependent law resolves against.
    """
    model = Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_to_target)},
                active=lambda age: age < 22,
                states={"noise": _RUNTIME_NOISE},
                state_transitions={"wealth": {"target": _wealth_from_fixed_draw}},
                functions={"utility": _utility_reading_the_noise},
            ),
            "target": Regime(
                transition=None,
                states={
                    "wealth": _WEALTH,
                    "shock": _THREE_NODES,
                    "noise": _RUNTIME_NOISE,
                },
                functions={"utility": _wealth_and_shock},
            ),
        },
        ages=_AGES,
        regime_id_class=RegimeId,
    )

    V = model.solve(
        params={
            "source": {
                "koopmans_aggregator": {"discount_factor": 1.0},
                "noise": _RUNTIME_PARAMS,
            },
            "target": {"noise": _RUNTIME_PARAMS},
        },
        log_level="off",
    )

    # `wealth = 2 eps` and the target pays `wealth + shock = 3 eps`, mean three —
    # at each of the source's own noise nodes, which change nothing.
    np.testing.assert_allclose(
        np.asarray(V[0]["source"]).ravel(), np.full(3, 3.0), atol=1e-5
    )


_WIDE_INCOME = NormalIIDProcess(
    n_points=5, gauss_hermite=False, mu=2.0, sigma=1.0, n_std=2.0
)
_OFFSET_SHOCK = NormalIIDProcess(
    n_points=3, gauss_hermite=False, mu=2.0, sigma=0.5, n_std=2.0
)


def _income_between_two_nodes(next_shock: ContinuousState) -> ScalarFloat:
    """Lands halfway between two income nodes at every node of the draw."""
    return next_shock + 0.5


def _income_only(income: ScalarFloat, shock: ScalarFloat) -> ScalarFloat:
    return income + 0.0 * shock


def test_a_dependent_entry_is_contracted_as_a_value_not_averaged_as_a_lottery() -> None:
    """The entry's node coefficients are contracted before the draw is averaged.

    The entry lands halfway between two income nodes at each node of the draw, so
    the two readings of its coefficients are distinguishable: as a value it pays
    the interpolated `shock + 0.5` there, and only the draw is risky. Reading the
    coefficients as probabilities would instead spread each node's payoff across
    two income nodes and hand that spread to the risk transform, which a
    power-mean certainty equivalent prices strictly lower.
    """
    model = Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_to_target)},
                active=lambda age: age < 22,
                state_transitions={"income": {"target": _income_between_two_nodes}},
                functions={"utility": _no_utility},
                certainty_equivalent=PowerMean(),
            ),
            "target": Regime(
                transition=None,
                states={"income": _WIDE_INCOME, "shock": _OFFSET_SHOCK},
                functions={"utility": _income_only},
            ),
        },
        ages=_AGES,
        regime_id_class=RegimeId,
    )

    risk_aversion = 3.0
    V = model.solve(
        params={
            "source": {
                "koopmans_aggregator": {"discount_factor": 1.0},
                "certainty_equivalent": {"risk_aversion": risk_aversion},
            }
        },
        log_level="off",
    )

    # Utility is linear in income, so linear interpolation is exact: the target
    # pays `eps_j + 0.5` at node j, and only the draw is averaged over.
    weights = np.asarray(_OFFSET_SHOCK.get_transition_probs()[0])
    payoffs = np.asarray(_OFFSET_SHOCK.to_jax()) + 0.5
    exponent = 1.0 - risk_aversion
    expected = (weights @ payoffs**exponent) ** (1.0 / exponent)

    np.testing.assert_allclose(
        np.asarray(V[0]["source"]).ravel(), np.array([expected]), atol=1e-5
    )


@categorical(ordered=False)
class Health:
    bad: ScalarInt
    good: ScalarInt


def _health_probs_from_a_draw(next_shock: ContinuousState) -> FloatND:
    """A discrete draw whose probabilities are conditioned on a sibling draw."""
    return jnp.stack([1.0 - next_shock / 4.0, next_shock / 4.0])


def _health_and_shock(health: ScalarInt, shock: ScalarFloat) -> ScalarFloat:
    return health + shock


def test_a_draw_conditioned_on_a_sibling_draw_is_rejected() -> None:
    """Two draws are combined as a product of marginals, which no joint kernel is.

    Each draw contributes its own axis and its own probabilities. A law whose
    probabilities depend on where a sibling landed describes a correlation that
    product cannot carry, so the model says so instead of pricing an independence
    it was not given.
    """
    conditioned = MarkovTransition(_health_probs_from_a_draw)
    with pytest.raises(ModelInitializationError, match="joint kernel"):
        Model(
            regimes={
                "source": Regime(
                    transition={"target": MarkovTransition(_to_target)},
                    active=lambda age: age < 22,
                    state_transitions={"health": {"target": conditioned}},
                    functions={"utility": _no_utility},
                ),
                "target": Regime(
                    transition=None,
                    states={"health": DiscreteGrid(Health), "shock": _THREE_NODES},
                    functions={"utility": _health_and_shock},
                ),
            },
            ages=_AGES,
            regime_id_class=RegimeId,
        )
