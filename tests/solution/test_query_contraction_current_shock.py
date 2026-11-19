"""Lottery contraction of a draw whose probabilities read a current shock.

A query builds its contracted tables before its loops and binds each information
row's current shocks inside them, so a draw whose weights read one, directly, through a
helper or alone, stays enumerated in a query, and in the regime's own kernel, which its
exit's query reads. The ordinary solve of a regime without an exit binds the shock per
cell, so it still contracts such a draw. Weights that read only the drawn state or the
period are contracted in the query too.

The oracle is the uncontracted solve and the exact values of the two-period model: the
single never works, its outside option is worse everywhere, and its terminal value is
`1 + health`.
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm import exit_queries
from _lcm.execution.core_program import CoreProgramGraphAware
from _lcm.regime_building.Q_and_F import _ComputeCE
from lcm import (
    DiscreteGrid,
    ExecutionConfig,
    Exit,
    ExitTarget,
    InitialNodes,
    Model,
    Periods,
    Regime,
    StochasticTransition,
    Unilateral,
    categorical,
)
from lcm.shocks import Discrete, ShockGroup
from lcm.solvers import GridSearch
from lcm.typing import DiscreteAction, DiscreteState, FloatND, IntND, ScalarInt
from tests.conftest import assert_agrees_to_ulp
from tests.regime_building.test_lottery_contraction_activation import _owned_objects
from tests.solution.test_query_contraction import _query_functions


@categorical(ordered=False)
class RegimeId:
    single: ScalarInt
    outside: ScalarInt
    single_terminal: ScalarInt
    outside_terminal: ScalarInt


@categorical(ordered=False)
class _Signal:
    low: ScalarInt
    high: ScalarInt


def _single(*, health: DiscreteState, work: DiscreteAction, eps: IntND) -> FloatND:
    return 1.0 + health - 0.4 * work + 0.2 * eps


def _terminal(*, health: DiscreteState, work: DiscreteAction) -> FloatND:
    return 1.0 + health - 0.4 * work


def _outside(*, health: DiscreteState, work: DiscreteAction, eps: IntND) -> FloatND:
    return -10.0 + 0.01 * health - 0.4 * work + 0.01 * eps


def _outside_terminal(*, health: DiscreteState, work: DiscreteAction) -> FloatND:
    return -10.0 + 0.01 * health - 0.4 * work


def _health(health: DiscreteState) -> DiscreteState:
    return health


def _risk(*, health: DiscreteState, eps: IntND) -> FloatND:
    return (1.0 + 2.0 * health + 4.0 * eps) / 10.0


def _direct(*, health: DiscreteState, eps: IntND) -> FloatND:
    high = _risk(health=health, eps=eps)
    return jnp.stack((1.0 - high, high))


def _through_a_helper(risk: FloatND) -> FloatND:
    return jnp.stack((1.0 - risk, risk))


def _shock_alone(eps: IntND) -> FloatND:
    high = (1.0 + 4.0 * eps) / 10.0
    return jnp.stack((1.0 - high, high))


def _state_alone(health: DiscreteState) -> FloatND:
    high = (1.0 + 2.0 * health) / 10.0
    return jnp.stack((1.0 - high, high))


def _clock(*, health: DiscreteState, period: IntND) -> FloatND:
    high = (1.0 + 2.0 * health + 0.5 * period) / 10.0
    return jnp.stack((1.0 - high, high))


_WEIGHTS = {
    "direct": _direct,
    "through-a-helper": _through_a_helper,
    "shock-alone": _shock_alone,
    "state-alone": _state_alone,
    "clock": _clock,
}
_READS_THE_SHOCK = frozenset({"direct", "through-a-helper", "shock-alone"})

# Period-0 value of the single at health 0 and 1: E[flow] + 0.9 * E[1 + next health].
_EXACT = {
    "direct": (Fraction(241, 100), Fraction(359, 100)),
    "through-a-helper": (Fraction(241, 100), Fraction(359, 100)),
    "shock-alone": (Fraction(241, 100), Fraction(341, 100)),
    "state-alone": (Fraction(214, 100), Fraction(332, 100)),
    "clock": (Fraction(214, 100), Fraction(332, 100)),
}


def _model(*, weights: str, contract: bool) -> Model:
    """The single's health draw reads `weights`; the outside's reads the shock."""
    health = DiscreteGrid(values=jnp.asarray([0, 1]))
    work = {"work": DiscreteGrid(values=jnp.asarray([0, 1]))}
    single = Regime(
        states={"health": health},
        actions=work,
        functions={
            "utility": _single,
            **({"risk": _risk} if weights == "through-a-helper" else {}),
        },
        shocks=("signal",),
        state_transitions={"health": StochasticTransition(func=_WEIGHTS[weights])},
        solver=GridSearch(lottery_contraction=contract),
        exit=Exit(
            targets={
                "self": ExitTarget(regime="outside", projection={"health": _health})
            },
            rule=Unilateral(),
        ),
    )
    outside = Regime(
        states={"health": health},
        actions=work,
        functions={"utility": _outside},
        shocks=("signal",),
        state_transitions={"health": StochasticTransition(func=_direct)},
        solver=GridSearch(lottery_contraction=contract),
    )
    return Model(
        regimes={
            "single": single,
            "outside": outside,
            "single_terminal": single.replace(
                exit=None,
                shocks=(),
                state_transitions={},
                functions={"utility": _terminal},
            ),
            "outside_terminal": outside.replace(
                shocks=(),
                state_transitions={},
                functions={"utility": _outside_terminal},
            ),
        },
        shocks={
            "signal": ShockGroup(
                shocks={"eps": Discrete(categories=_Signal, probs=(0.25, 0.75))}
            )
        },
        regime_id_class=RegimeId,
        n_periods=2,
        edges={
            name: {f"{name}_terminal": Periods(values=(0,))}
            for name in ("single", "outside")
        },
        initial_nodes=InitialNodes(by_period={0: ("single", "outside")}),
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def _values(model: Model) -> dict[str, np.ndarray]:
    solution = model.solve(
        params={"discount_factor": 0.9}, log_level="off", runtime_checks=False
    )
    return {
        f"{period}/{regime}": np.asarray(value)
        for period, by_regime in solution.values.items()
        for regime, value in by_regime.items()
    }


def _single_query_contractions(model: Model) -> list[frozenset[str]]:
    """Per query of the single: the states its terminal continuation sums."""
    found = []
    for target, compute_CE in _query_functions(model):
        continuation = compute_CE.continuations.get("single_terminal")
        if target == "single" and continuation is not None:
            contraction = continuation.lottery_contraction
            found.append(
                frozenset(lottery.state_name for lottery in contraction.lotteries)
                if contraction
                else frozenset()
            )
    return found


@pytest.mark.parametrize("stored", [False, True], ids=["computed", "stored"])
@pytest.mark.parametrize("weights", tuple(_WEIGHTS))
def test_a_query_keeps_a_draw_whose_weights_read_a_current_shock(
    *, monkeypatch: pytest.MonkeyPatch, stored: bool, weights: str
) -> None:
    """Values agree with enumeration and the exact values; only shock readers stay."""
    monkeypatch.setattr(exit_queries, "_READ_STORED_OUTSIDE_VALUES", stored)
    off = _values(_model(weights=weights, contract=False))
    exact = np.asarray([float(value) for value in _EXACT[weights]])
    assert_agrees_to_ulp(
        got=off["0/single"],
        expected=exact.astype(off["0/single"].dtype),
        n_ulp=16,
        operand_magnitude=float(np.max(exact)),
        err_msg=f"exact: {weights}",
    )
    model = _model(weights=weights, contract=True)
    on = _values(model)
    assert on.keys() == off.keys()
    for key, expected in off.items():
        np.testing.assert_array_equal(np.isfinite(on[key]), np.isfinite(expected))
        assert_agrees_to_ulp(
            got=on[key],
            expected=expected,
            n_ulp=16,
            operand_magnitude=float(np.max(np.abs(expected[np.isfinite(expected)]))),
            err_msg=f"{weights}: {key}",
        )
    contractions = _single_query_contractions(model)
    assert contractions
    expected_states = (
        frozenset() if weights in _READS_THE_SHOCK else frozenset({"health"})
    )
    assert set(contractions) == {expected_states}


def test_the_ordinary_solve_of_a_regime_without_an_exit_contracts_a_shock_reader() -> (
    None
):
    """The outside's draw reads the shock; its solve binds it per cell and sums it."""
    model = _model(weights="direct", contract=True)
    kernel = model._regimes["outside"].solution.period_kernels[0]
    assert isinstance(kernel, CoreProgramGraphAware)
    contracted = {
        frozenset(lottery.state_name for lottery in contraction.lotteries)
        if (contraction := owned.continuations["outside_terminal"].lottery_contraction)
        else frozenset()
        for program in kernel.core_programs().values()
        for owned in _owned_objects(root=program.function)
        if isinstance(owned, _ComputeCE) and "outside_terminal" in owned.continuations
    }
    assert contracted == {frozenset({"health"})}
