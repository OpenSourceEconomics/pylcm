"""Process laws supplied through `Model(fixed_params=...)`.

`fixed_params` is the public way to pin a parameter at model initialization, so
a process whose law arrives that way is fixed at construction in every sense
that matters: it is bound into the grid before the model is built, and every
consumer — the target's own nodes, a source's entry weights, diagnostics, and
simulation — reads one resolved process object.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    Regime,
    categorical,
)
from lcm.typing import ScalarFloat, ScalarInt

# `mu=1, sigma=0.5, n_std=2` at three points puts equidistant nodes on
# `(0, 1, 2)`, so the unconditional mean is `mu` and a dropped continuation
# would publish `0.0` instead.
_MU = 1.0
_PROCESS_LAW = {"mu": _MU, "sigma": 0.5, "n_std": 2.0}


def _process(*, at_construction: bool) -> NormalIIDProcess:
    """Return the shared process, with its law fixed at construction or not."""
    if at_construction:
        return NormalIIDProcess(
            n_points=3, gauss_hermite=False, mu=_MU, sigma=0.5, n_std=2.0
        )
    return NormalIIDProcess(n_points=3, gauss_hermite=False)


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


def _zero_utility() -> ScalarFloat:
    return jnp.float32(0)


def _shock_utility(shock: ScalarFloat) -> ScalarFloat:
    return shock


def _one_probability() -> ScalarFloat:
    return jnp.float32(1)


def _entered_process_model(*, at_construction: bool) -> Model:
    """Build a source whose only target carries a process the source lacks."""
    process = _process(at_construction=at_construction)
    fixed_params = {} if at_construction else {"target": {"shock": _PROCESS_LAW}}
    return Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_one_probability)},
                active=lambda age: age < 22,
                functions={"utility": _zero_utility},
            ),
            "target": Regime(
                transition=None,
                states={"shock": process},
                functions={"utility": _shock_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        fixed_params=fixed_params,
        enable_jit=False,
    )


_SOLVE_PARAMS = {"source": {"koopmans_aggregator": {"discount_factor": 1.0}}}


@pytest.mark.parametrize("at_construction", [True, False])
def test_entered_process_is_priced_at_its_own_law(*, at_construction: bool) -> None:
    """An entered process is priced at its mean whichever way its law is fixed.

    The entry distribution is the process's own, so with the target's payoff
    equal to the shock and no discounting the source's value is `mu`. Passing
    the law through `fixed_params` must reach the same value as passing it to
    the process constructor.
    """
    solution = _entered_process_model(at_construction=at_construction).solve(
        params=_SOLVE_PARAMS, log_level="debug"
    )
    np.testing.assert_allclose(np.asarray(solution[0]["source"]), _MU, atol=1e-6)


def test_fixed_process_law_leaves_the_params_template() -> None:
    """A law bound from `fixed_params` is no longer a runtime parameter."""
    template = _entered_process_model(at_construction=False).get_params_template()

    assert "shock" not in template["target"]


def test_carried_process_law_from_fixed_params_matches_construction() -> None:
    """Binding a carried process's law changes no value it already produced.

    The process here is carried by both regimes, so it never takes the entry
    path. Both ways of fixing its law must therefore agree node for node.
    """

    def _build(*, at_construction: bool) -> Model:
        process = _process(at_construction=at_construction)
        fixed_params = (
            {}
            if at_construction
            else {
                "source": {"shock": _PROCESS_LAW},
                "target": {"shock": _PROCESS_LAW},
            }
        )
        return Model(
            regimes={
                "source": Regime(
                    transition={"target": MarkovTransition(_one_probability)},
                    active=lambda age: age < 22,
                    states={"shock": process},
                    functions={"utility": _shock_utility},
                ),
                "target": Regime(
                    transition=None,
                    states={"shock": process},
                    functions={"utility": _shock_utility},
                ),
            },
            ages=AgeGrid(start=20, stop=22, step="Y"),
            regime_id_class=RegimeId,
            fixed_params=fixed_params,
            enable_jit=False,
        )

    from_construction = _build(at_construction=True).solve(
        params=_SOLVE_PARAMS, log_level="debug"
    )
    from_fixed_params = _build(at_construction=False).solve(
        params=_SOLVE_PARAMS, log_level="debug"
    )

    np.testing.assert_allclose(
        np.asarray(from_fixed_params[0]["source"]),
        np.asarray(from_construction[0]["source"]),
        atol=1e-6,
    )
