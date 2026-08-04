"""A target entered at an IID process is priced by that process's own law.

A regime whose only state is an IID process can be reached without the source
handing anything over: an IID draw does not depend on its previous value, so the
entry distribution is the process's own unconditional law. Pricing it therefore
means weighting the target's nodes by that law's probabilities.

Every case below uses a **nonuniform** law with a nonlinear or asymmetric payoff,
because that is the only shape that separates the two candidate answers. On a
uniform law, or on a symmetric law with a linear payoff, the weighted and
unweighted means coincide and a test cannot tell them apart -- the uniform case
is kept below as the negative control rather than as evidence.

Oracles come from `numpy.polynomial.hermite.hermgauss`, which shares no code with
the grid construction under test.
"""

import math
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.grids import Grid
from lcm import (
    AgeGrid,
    LinSpacedGrid,
    LogNormalIIDProcess,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    Regime,
    UniformIIDProcess,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.typing import FloatND, ScalarFloat, ScalarInt
from tests.conftest import DECIMAL_PRECISION


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


def _zero_utility() -> FloatND:
    return jnp.asarray(0.0)


def _shock_utility(shock: ScalarFloat) -> FloatND:
    return shock


def _squared_shock_utility(shock: ScalarFloat) -> FloatND:
    return shock**2


def _one_probability() -> FloatND:
    return jnp.asarray(1.0)


def _target_id() -> ScalarInt:
    return RegimeId.target


def _source_is_early(age: float) -> bool:
    return age < 22


def _gauss_hermite(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return standard-normal quadrature nodes and probabilities.

    `hermgauss` integrates against `exp(-x**2)`, so the nodes scale by `sqrt(2)`
    and the weights normalize by `sqrt(pi)` to become probabilities.
    """
    raw_nodes, raw_weights = np.polynomial.hermite.hermgauss(n_points)
    return math.sqrt(2.0) * raw_nodes, raw_weights / math.sqrt(math.pi)


def _build_model(
    *,
    process: Grid,
    target_utility: Callable[..., FloatND],
    coarse: bool,
    enable_jit: bool,
) -> Model:
    """Build a source whose declared target's only state is `process`."""
    transition = (
        _target_id if coarse else {"target": MarkovTransition(_one_probability)}
    )
    return Model(
        regimes={
            "source": Regime(
                transition=transition,
                active=_source_is_early,
                functions={"utility": _zero_utility},
            ),
            "target": Regime(
                transition=None,
                states={"shock": process},
                functions={"utility": target_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
    )


def _source_value(model: Model, params: dict) -> float:
    solution = model.solve(params=params, log_level="debug")
    last_living = max(period for period in solution if "source" in solution[period])
    return float(np.asarray(solution[last_living]["source"]).ravel()[0])


@pytest.mark.parametrize("coarse", [False, True], ids=["granular", "coarse"])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_normal_gauss_hermite_entry_weights_a_nonlinear_payoff(
    coarse: bool,  # noqa: FBT001
    enable_jit: bool,  # noqa: FBT001
) -> None:
    """`E[shock**2]` is the variance, not the unweighted mean of the squared nodes.

    On a 3-node Gauss-Hermite grid the probabilities are `(1/6, 2/3, 1/6)` and
    the nodes are `(-sqrt3, 0, sqrt3)`, so the weighted answer is `1.0` while
    averaging the nodes uniformly gives `2.0`.
    """
    nodes, weights = _gauss_hermite(3)
    expected = float(np.dot(nodes**2, weights))

    model = _build_model(
        process=NormalIIDProcess(n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0),
        target_utility=_squared_shock_utility,
        coarse=coarse,
        enable_jit=enable_jit,
    )
    got = _source_value(
        model,
        {
            "source": {"utility": {}, "H": {"discount_factor": 1.0}, "next_regime": {}},
            "target": {"utility": {}},
        },
    )
    np.testing.assert_almost_equal(got, expected, decimal=DECIMAL_PRECISION)


def test_lognormal_gauss_hermite_entry_weights_an_asymmetric_law() -> None:
    """A skewed law is priced at its own mean, not at the mean of its nodes.

    The payoff is linear here, so only the asymmetry of the law separates the
    two answers.
    """
    nodes, weights = _gauss_hermite(3)
    expected = float(np.dot(np.exp(nodes), weights))

    model = _build_model(
        process=LogNormalIIDProcess(n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0),
        target_utility=_shock_utility,
        coarse=False,
        enable_jit=False,
    )
    got = _source_value(
        model,
        {
            "source": {"utility": {}, "H": {"discount_factor": 1.0}, "next_regime": {}},
            "target": {"utility": {}},
        },
    )
    np.testing.assert_almost_equal(got, expected, decimal=DECIMAL_PRECISION)


def test_a_uniform_law_prices_identically_either_way() -> None:
    """The negative control: equal weights make both candidate answers agree.

    This case cannot distinguish a correct implementation from one that ignores
    the weights, which is precisely why it is not evidence on its own.
    """
    model = _build_model(
        process=UniformIIDProcess(n_points=4, start=0.0, stop=3.0),
        target_utility=_shock_utility,
        coarse=True,
        enable_jit=False,
    )
    got = _source_value(
        model,
        {
            "source": {"utility": {}, "H": {"discount_factor": 1.0}, "next_regime": {}},
            "target": {"utility": {}},
        },
    )
    np.testing.assert_almost_equal(got, 1.5, decimal=DECIMAL_PRECISION)


def test_the_entry_law_decides_the_action() -> None:
    """A competing action worth more than the law but less than the node mean.

    `E[shock**2] = 1.0` under the correct weights, so the competing payoff of
    `1.5` wins; under the unweighted node mean of `2.0` it loses. The published
    value therefore reports which law priced the target, and a tolerance cannot
    absorb the difference because the two answers are a finite distance apart.
    """

    def _stay_utility(wealth: ScalarFloat) -> FloatND:
        return 1.5 + 0.0 * wealth

    stay = Regime(
        transition=None,
        states={"wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=2)},
        functions={"utility": _stay_utility},
    )
    enter = Regime(
        transition=None,
        states={
            "shock": NormalIIDProcess(n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0)
        },
        functions={"utility": _squared_shock_utility},
    )

    @categorical(ordered=False)
    class _ThreeRegimeId:
        source: ScalarInt
        stay: ScalarInt
        enter: ScalarInt

    def _choose(go: ScalarInt) -> ScalarInt:
        return jnp.where(go == 1, _ThreeRegimeId.enter, _ThreeRegimeId.stay)

    source = Regime(
        transition=_choose,
        active=_source_is_early,
        actions={"go": LinSpacedGrid(start=0, stop=1, n_points=2)},
        state_transitions={"wealth": {"stay": lambda: jnp.asarray(1.0)}},
        functions={"utility": _zero_utility},
    )
    model = Model(
        regimes={"source": source, "stay": stay, "enter": enter},
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=_ThreeRegimeId,
        enable_jit=False,
    )
    got = _source_value(
        model,
        {
            "source": {
                "utility": {},
                "H": {"discount_factor": 1.0},
                "next_wealth": {},
                "next_regime": {},
            },
            "stay": {"utility": {}},
            "enter": {"utility": {}},
        },
    )
    np.testing.assert_almost_equal(got, 1.5, decimal=DECIMAL_PRECISION)


def test_a_law_supplied_at_runtime_cannot_be_entered() -> None:
    """Entering a process whose parameters arrive at runtime is a build error.

    The entry weights are evaluated inside the source's Bellman equation, which
    reads only the source's own parameters, so a law the target parameterizes at
    runtime has no value the source could read. The message names the state and
    the parameters that block it.
    """
    with pytest.raises(
        ModelInitializationError, match="passes 'mu', 'sigma' at runtime"
    ):
        _build_model(
            process=NormalIIDProcess(n_points=3, gauss_hermite=True),
            target_utility=_shock_utility,
            coarse=True,
            enable_jit=False,
        )


def test_entry_draws_come_from_the_process_law_not_from_its_solver_nodes() -> None:
    """Simulated entry values follow `U(0, 3)`, not a pick among the four nodes.

    The two candidates are separated by the second moment: the continuous law
    has `E[X**2] = 3`, while drawing uniformly among the solver nodes
    `(0, 1, 2, 3)` gives `3.5`.
    """
    model = _build_model(
        process=UniformIIDProcess(n_points=4, start=0.0, stop=3.0),
        target_utility=_shock_utility,
        coarse=True,
        enable_jit=False,
    )
    params = {
        "source": {"utility": {}, "H": {"discount_factor": 1.0}, "next_regime": {}},
        "target": {"utility": {}},
    }
    n_subjects = 20_000
    result = model.simulate(
        params=params,
        initial_conditions={
            "regime_id": jnp.full(n_subjects, RegimeId.source, dtype=jnp.int32),
            "age": jnp.full(n_subjects, 20.0),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )
    df = result.to_dataframe()
    entered = df.query("regime_name == 'target'")["shock"].to_numpy()

    assert len(entered) == n_subjects
    np.testing.assert_allclose(float(np.mean(entered**2)), 3.0, atol=0.1)
