"""A law declared once for every target resolves separately on each target's nodes.

A deterministic law that reads a draw of the target it enters may be declared as a
single entry rather than per target. It then applies to every reachable target, and
each edge resolves it on the nodes that target's own process carries — so one
declaration yields as many integrands as there are targets,

```{math}
\\sum_{r'} \\pi_{r'} \\sum_j p^{r'}_j \\, V_{r'}\\!\\left(g(\\varepsilon^{r'}_j),\\,
\\varepsilon^{r'}_j\\right),
```

and the value differs from what any one target's nodes alone would give.
"""

import jax.numpy as jnp
import numpy as np

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    Regime,
    categorical,
)
from lcm.typing import ContinuousState, ScalarFloat, ScalarInt

# `sigma=0.5, n_std=2` at three points puts symmetric nodes on `mu + (-1, 0, 1)`, so
# each draw has mean `mu` whatever weights the discretization assigns its nodes.
_SHOCK_A = NormalIIDProcess(
    n_points=3, gauss_hermite=False, mu=1.0, sigma=0.5, n_std=2.0
)
_SHOCK_B = NormalIIDProcess(
    n_points=3, gauss_hermite=False, mu=3.0, sigma=0.5, n_std=2.0
)
# Holds `2 * eps` exactly on every node of both processes, so no interpolation
# error enters and the two targets stay distinguishable by value alone.
_WEALTH = LinSpacedGrid(start=0.0, stop=8.0, n_points=5)

_PARAMS = {"source": {"koopmans_aggregator": {"discount_factor": 1.0}}}


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    a: ScalarInt
    b: ScalarInt


def _half() -> ScalarFloat:
    return jnp.float32(0.5)


def _no_utility() -> ScalarFloat:
    return jnp.float32(0)


def _wealth_plus_shock(wealth: ScalarFloat, shock: ScalarFloat) -> ScalarFloat:
    return wealth + shock


def _next_wealth_from_draw(next_shock: ContinuousState) -> ScalarFloat:
    return 2.0 * next_shock


def _build(state_transitions) -> Model:
    """Two targets whose processes are centred three units apart."""
    return Model(
        regimes={
            "source": Regime(
                transition={t: MarkovTransition(_half) for t in ("a", "b")},
                active=lambda age: age < 22,
                state_transitions=state_transitions,
                functions={"utility": _no_utility},
            ),
            "a": Regime(
                transition=None,
                states={"wealth": _WEALTH, "shock": _SHOCK_A},
                functions={"utility": _wealth_plus_shock},
            ),
            "b": Regime(
                transition=None,
                states={"wealth": _WEALTH, "shock": _SHOCK_B},
                functions={"utility": _wealth_plus_shock},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
    )


def test_a_broadcast_law_resolves_on_each_target_s_own_nodes() -> None:
    """`V = 0.5 * 3 * E[eps_a] + 0.5 * 3 * E[eps_b] = 0.5 * 3 + 0.5 * 9 = 6`.

    Each target pays `wealth + shock` and wealth arrives as `2 * shock`, so the
    integrand is `3 * eps` on that target's own nodes. The draws have means one and
    three, so resolving both edges against either target alone would give three or
    nine instead.
    """
    model = _build({"wealth": _next_wealth_from_draw})

    V = model.solve(params=_PARAMS, log_level="off")

    np.testing.assert_allclose(
        np.asarray(V[0]["source"]).ravel(), np.array([6.0]), atol=1e-5
    )


def test_a_broadcast_law_and_the_same_law_written_per_target_agree() -> None:
    """Declaring one law for all targets is the model that names each target."""
    broadcast = _build({"wealth": _next_wealth_from_draw})
    per_target = _build({"wealth": dict.fromkeys(("a", "b"), _next_wealth_from_draw)})

    np.testing.assert_allclose(
        np.asarray(broadcast.solve(params=_PARAMS, log_level="off")[0]["source"]),
        np.asarray(per_target.solve(params=_PARAMS, log_level="off")[0]["source"]),
        atol=1e-6,
    )
