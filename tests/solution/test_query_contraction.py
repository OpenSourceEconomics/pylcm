"""Lottery contraction in exit queries and collective regimes, behind its switch.

A consent marriage: each single's health and each spouse's health is a Markov draw
whose weights read only its own current value, so it is summed into the target's
value array. `QUERY_CONTRACTION` decides whether the query variants and the
collective continuation do so; "off" is the shipped behaviour.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm import exit_queries
from _lcm.policy_queries import _PolicyQueryKernel
from _lcm.regime_building import lottery_contraction
from _lcm.regime_building.Q_and_F import _ComputeCE
from lcm import (
    CollectiveUtility,
    Consent,
    Counterparty,
    DiscreteGrid,
    Exit,
    ExitTarget,
    InitialNodes,
    Model,
    Periods,
    Regime,
    StochasticTransition,
    Unilateral,
)
from lcm.solvers import GridSearch
from lcm.typing import DiscreteAction, DiscreteState, FloatND
from tests.conftest import assert_agrees_to_ulp
from tests.solution.test_outside_value_truncation import _Marriage

_ROWS = np.array([[0.7, 0.3], [0.2, 0.8]])


def _probs(health: DiscreteState) -> FloatND:
    return jnp.asarray(_ROWS)[health]


def _probs_f(health_f: DiscreteState) -> FloatND:
    return jnp.asarray(_ROWS)[health_f]


def _probs_m(health_m: DiscreteState) -> FloatND:
    return jnp.asarray(_ROWS)[health_m]


def _single(*, health: DiscreteState, work: DiscreteAction) -> FloatND:
    return 1.0 + health - 0.4 * work


def _wife(*, health_f: DiscreteState, work: DiscreteAction) -> FloatND:
    return 1.5 + health_f + 0.3 * work


def _husband(*, health_m: DiscreteState, work: DiscreteAction) -> FloatND:
    return 1.2 + 0.5 * health_m - 0.2 * work


def _health(health: DiscreteState) -> DiscreteState:
    return health


def _health_f(health_f: DiscreteState) -> DiscreteState:
    return health_f


def _health_m(health_m: DiscreteState) -> DiscreteState:
    return health_m


def _model() -> Model:
    """Single_f marries by consent; the couple dissolves unilaterally; three periods."""
    health = DiscreteGrid(values=jnp.asarray([0, 1]))
    work = {"work": DiscreteGrid(values=jnp.asarray([0, 1]))}
    single = Regime(
        states={"health": health},
        actions=work,
        functions={"utility": _single},
        state_transitions={"health": StochasticTransition(func=_probs)},
        solver=GridSearch(lottery_contraction=True),
    )
    couple = Regime(
        states={"health_f": health, "health_m": health},
        actions=work,
        functions={"utility": CollectiveUtility(utilities={"f": _wife, "m": _husband})},
        state_transitions={
            "health_f": StochasticTransition(func=_probs_f),
            "health_m": StochasticTransition(func=_probs_m),
        },
        solver=GridSearch(lottery_contraction=True),
    )
    consent = Consent(
        counterparty=Counterparty(
            regime="single_m", role="m", projection={"health": _health}
        )
    )
    to_couple = ExitTarget(
        regime="couple", role="f", projection={"health_f": _health, "health_m": _health}
    )
    regimes = {
        "single_f": single.replace(
            exit=Exit(targets={"self": to_couple}, rule=consent)
        ),
        "single_m": single,
        "couple": couple.replace(
            exit=Exit(
                targets={
                    "f": ExitTarget(
                        regime="single_f", projection={"health": _health_f}
                    ),
                    "m": ExitTarget(
                        regime="single_m", projection={"health": _health_m}
                    ),
                },
                rule=Unilateral(),
            )
        ),
    }
    for name, regime in tuple(regimes.items()):
        regimes[f"{name}_terminal"] = regime.replace(exit=None, state_transitions={})
    return Model(
        regimes=regimes,
        regime_id_class=_Marriage,
        n_periods=3,
        edges={
            name: {name: Periods(values=(0,)), f"{name}_terminal": Periods(values=(1,))}
            for name in ("single_f", "single_m", "couple")
        },
        initial_nodes=InitialNodes(by_period={0: ("single_f", "single_m", "couple")}),
    )


def _contracted(model: Model) -> dict[tuple[str, str], frozenset[str]]:
    """Per (query target, continuation target): the states its query Q_and_F sums."""
    found = {}
    for regime in model._regimes.values():
        for kernel in regime.solution.period_kernels.values():
            if not isinstance(kernel, _PolicyQueryKernel):
                continue
            for node in kernel.definitions.values():
                function = node.context.Q_and_F_functions.get(node.read.period)
                compute_CE = getattr(function, "compute_CE", None)
                if isinstance(compute_CE, _ComputeCE):
                    for target, continuation in compute_CE.continuations.items():
                        summed = continuation.lottery_contraction
                        found[node.read.target, target.removesuffix("_terminal")] = (
                            frozenset(lot.state_name for lot in summed.lotteries)
                            if summed
                            else frozenset()
                        )
    return found


def test_query_and_collective_continuations_contract_only_when_switched_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hoisted, every health draw a query reads is summed; off, none is."""
    off = _contracted(_model())
    monkeypatch.setattr(lottery_contraction, "QUERY_CONTRACTION", "hoisted")
    hoisted = _contracted(_model())
    expected = {
        ("couple", "couple"): {"health_f", "health_m"},
        ("single_f", "single_f"): {"health"},
        ("single_m", "single_m"): {"health"},
    }
    assert (off, hoisted) == (dict.fromkeys(expected, frozenset()), expected)


def _values(*, switch: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, np.ndarray]:
    monkeypatch.setattr(lottery_contraction, "QUERY_CONTRACTION", switch)
    solution = _model().solve(
        params={"discount_factor": 0.9}, log_level="off", runtime_checks=False
    )
    return {
        f"{period}/{regime}": np.asarray(value)
        for period, by_regime in solution.values.items()
        for regime, value in by_regime.items()
    }


@pytest.mark.parametrize("stored", [True, False])
def test_hoisted_contraction_agrees_with_enumeration_to_16_ulp(
    *, monkeypatch: pytest.MonkeyPatch, stored: bool
) -> None:
    """Values with summed draws agree with enumerated ones at the array's scale."""
    monkeypatch.setattr(exit_queries, "_READ_STORED_OUTSIDE_VALUES", stored)
    off = _values(switch="off", monkeypatch=monkeypatch)
    hoisted = _values(switch="hoisted", monkeypatch=monkeypatch)
    assert hoisted.keys() == off.keys()
    for key, expected in off.items():
        np.testing.assert_array_equal(np.isfinite(hoisted[key]), np.isfinite(expected))
        assert_agrees_to_ulp(
            got=hoisted[key],
            expected=expected,
            n_ulp=16,
            operand_magnitude=float(np.max(np.abs(expected[np.isfinite(expected)]))),
            err_msg=key,
        )
