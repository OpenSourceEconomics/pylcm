"""A participation reads a process landing as the node's value, not its position.

Source, target and fallback carry one process on nodes [-1, 0, 1]. Target utility
is `z**3 + 2z + 5` (2, 5, 8 at the nodes), fallback utility is `6 + z` (5, 6, 7).
Every participation check below keeps the target only at node 1, so the source value is
`p(-1) * 5 + p(0) * 6 + p(1) * 8`. A landing read at its position 0, 1, 2 instead
keeps the target at positions 1 and 2 and pays a cubic at 2.
"""

from math import erf, sqrt

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    Model,
    ParticipationCheck,
    ParticipationRead,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    TauchenAR1Process,
    Transition,
    UniformIIDProcess,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.transition import StochasticTransition
from lcm.typing import BoolND, ContinuousState, DiscreteState, FloatND, ScalarInt

NODES = (-1.0, 0.0, 1.0)


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    target: ScalarInt
    fallback: ScalarInt


def _zero(z: ContinuousState) -> FloatND:
    return 0.0 * z


def _cubic(z: ContinuousState) -> FloatND:
    return z * z * z + 2.0 * z + 5.0


def _line(z: ContinuousState) -> FloatND:
    return 6.0 + z


def _one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _identity(z: ContinuousState) -> ContinuousState:
    return z


def _node_at(z_code: DiscreteState) -> ContinuousState:
    return jnp.asarray(NODES)[z_code]


def _value_predicate(z: ContinuousState) -> BoolND:
    return z > 0.0


def _code_predicate(z_code: DiscreteState) -> BoolND:
    return z_code == 2


def _read_predicate(*, V_target: FloatND, V_alt: FloatND) -> BoolND:
    return V_target > V_alt


def _uniform():
    return UniformIIDProcess(n_points=3, start=-1.0, stop=1.0, simulate_draws="nodes")


def _tauchen():
    return TauchenAR1Process.on_nodes(
        nodes=np.array(NODES), rho=0.0, sigma=1.0, simulate_draws="nodes"
    )


def _model(*, process, case: str, enable_jit: bool = False) -> Model:
    fallback_projection = _node_at if case == "code-fallback" else _identity
    reads = {}
    predicate = _value_predicate
    if case == "code-predicate":
        predicate = _code_predicate
    elif case in {"code-read", "integrated-read"}:
        predicate = _read_predicate
        reads = {
            "V_target": ParticipationRead(
                kind="IntegratedFull" if case == "integrated-read" else "Raw",
                regime="target",
                projection={"z": _identity},
            ),
            "V_alt": ParticipationRead(
                kind="Raw", regime="fallback", projection={"z": _node_at}
            ),
        }
    return Model(
        regimes={
            "source": Regime(states={"z": process()}, functions={"utility": _zero}),
            "target": Regime(states={"z": process()}, functions={"utility": _cubic}),
            "fallback": Regime(states={"z": process()}, functions={"utility": _line}),
        },
        edges={
            "source": Transition(
                targets={"target": 0, "fallback": 0},
                law=ByAge(cases={0: {"target": StochasticTransition(func=_one)}}),
                participation={
                    "target": ParticipationCheck(
                        predicate=predicate,
                        reads=reads,
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback",
                                    projection={"z": fallback_projection},
                                )
                            )
                        },
                    )
                },
            )
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "source"},
        regime_id_class=_RegimeId,
        enable_jit=enable_jit,
    )


def _tauchen_weights() -> tuple[float, float, float]:
    """Standard normal masses of the bins cut at -0.5 and 0.5, from `math.erf`."""
    low = 0.5 * (1.0 + erf(-0.5 / sqrt(2.0)))
    return (low, 1.0 - 2.0 * low, low)


_PARAMS = {
    "source": {"koopmans_aggregator": {"discount_factor": 1.0}},
    "target": {},
    "fallback": {},
}


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    "case",
    [
        "value-predicate",
        "code-predicate",
        "code-read",
        "integrated-read",
        "code-fallback",
    ],
)
@pytest.mark.parametrize(
    ("process", "weights"),
    [(_uniform, (1 / 3, 1 / 3, 1 / 3)), (_tauchen, _tauchen_weights())],
    ids=["uniform-iid", "tauchen-ar1"],
)
def test_participation_reads_the_landing_process_node_value(
    *, process, weights: tuple[float, float, float], case: str, enable_jit: bool
) -> None:
    model = _model(process=process, case=case, enable_jit=enable_jit)
    values = model.solve(params=_PARAMS, log_level="off")
    expected = weights[0] * 5.0 + weights[1] * 6.0 + weights[2] * 8.0
    np.testing.assert_allclose(
        values.values[0]["source"], np.full(3, expected), rtol=0.0, atol=2e-6
    )


def test_participation_into_a_runtime_parameterized_process_is_refused() -> None:
    def runtime() -> UniformIIDProcess:
        return UniformIIDProcess(n_points=3, start=-1.0, simulate_draws="nodes")

    with pytest.raises(ModelInitializationError, match="parameterized at runtime"):
        _model(process=runtime, case="value-predicate")
