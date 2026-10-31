"""Keep counterfactual participation prices apart from physical participation actors."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    CollectiveUtility,
    Consent,
    Counterparty,
    ExecutionConfig,
    Exit,
    ExitTarget,
    InitialNodes,
    Model,
    ParticipationCheck,
    ParticipationRead,
    Periods,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    categorical,
)
from lcm.typing import BoolND, FloatND, RegimeName, ScalarInt
from tests.conftest import DECIMAL_PRECISION

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


class _HouseholdAnswer(NamedTuple):
    source_value: float
    counterparty_value: float
    path: tuple[RegimeName, RegimeName]


class _PhysicalAnswer(NamedTuple):
    path: tuple[RegimeName, RegimeName]
    values: tuple[float, float]


def household_answer(*, target_payoff: int) -> _HouseholdAnswer:
    """Evaluate the authored participation check and strict bilateral gains directly."""
    continuation = target_payoff if target_payoff > 2 else 40.0
    source_raw = 1.0 + 5.0
    counterparty_raw = 3.0 + continuation
    couple_values = (8.0 + 10.0, 7.0 + 20.0)
    forms = couple_values[0] > source_raw and couple_values[1] > counterparty_raw
    return _HouseholdAnswer(
        source_value=couple_values[0] if forms else source_raw,
        counterparty_value=counterparty_raw,
        path=("formation", "couple_done") if forms else ("source", "done"),
    )


def physical_answer(*, target_payoff: int) -> _PhysicalAnswer:
    """Evaluate one actor followed by its authored accept-or-fallback rule."""
    target = "target" if target_payoff > 2 else "fallback"
    continuation = target_payoff if target_payoff > 2 else 40.0
    return _PhysicalAnswer(
        path=("counterparty", target), values=(3.0 + continuation, continuation)
    )


@categorical(ordered=False)
class HouseholdOwners:
    source: ScalarInt
    formation: ScalarInt
    counterparty: ScalarInt
    done: ScalarInt
    couple_done: ScalarInt
    target: ScalarInt
    fallback: ScalarInt


@categorical(ordered=False)
class PhysicalOwners:
    source: ScalarInt
    counterparty: ScalarInt
    target: ScalarInt
    fallback: ScalarInt
    done: ScalarInt


def zero() -> float:
    return 0.0


def one() -> float:
    return 1.0


def three() -> float:
    return 3.0


def five() -> float:
    return 5.0


def seven() -> float:
    return 7.0


def eight() -> float:
    return 8.0


def ten() -> float:
    return 10.0


def twenty() -> float:
    return 20.0


def thirty() -> float:
    return 30.0


def forty() -> float:
    return 40.0


def accept_participation(*, V_target: FloatND) -> BoolND:
    return V_target > 2.0


def counterparty_transition() -> Transition:
    return Transition(
        targets={"target": Periods(values=(0,)), "fallback": Periods(values=(0,))},
        law="target",
        participation={
            "target": ParticipationCheck(
                predicate=accept_participation,
                reads={
                    "V_target": ParticipationRead(
                        regime="target", kind="Raw", projection={}
                    )
                },
                routes={
                    "self": StakeholderRoute(
                        fallback=ProjectedRegimeValue(regime="fallback", projection={})
                    )
                },
            )
        },
    )


def household_model(*, target_payoff: int, budget: int | None) -> Model:
    target_utility = {1: one, 5: five, 30: thirty}[target_payoff]
    return Model(
        regimes={
            "source": Regime(
                functions={"utility": one},
                exits={
                    "marry": Exit(
                        targets={
                            "self": ExitTarget(
                                regime="formation", role="f", projection={}
                            )
                        },
                        rule=Consent(
                            counterparty=Counterparty(
                                regime="counterparty", role="m", projection={}
                            )
                        ),
                    )
                },
            ),
            "formation": Regime(
                functions={
                    "utility": CollectiveUtility(utilities={"f": eight, "m": seven})
                }
            ),
            "counterparty": Regime(functions={"utility": three}),
            "done": Regime(functions={"utility": five}),
            "couple_done": Regime(
                functions={
                    "utility": CollectiveUtility(utilities={"f": ten, "m": twenty})
                }
            ),
            "target": Regime(functions={"utility": target_utility}),
            "fallback": Regime(functions={"utility": forty}),
        },
        n_periods=2,
        regime_id_class=HouseholdOwners,
        initial_nodes=InitialNodes(by_period={0: "source"}),
        edges={
            "source": {"done": Periods(values=(0,))},
            "formation": {"couple_done": Periods(values=(0,))},
            "counterparty": counterparty_transition(),
        },
        execution_config=ExecutionConfig(device_memory_bytes=budget),
    )


@pytest.mark.parametrize("target_payoff", [1, 5, 30])
@pytest.mark.parametrize("budget", [None, 2**28])
def test_counterfactual_participation_is_priced_without_physical_replay(
    *, target_payoff: int, budget: int | None
) -> None:
    """A never-occupied counterparty needs its continuation but no entry actor."""
    model = household_model(target_payoff=target_payoff, budget=budget)
    assert not model._regimes["counterparty"].simulation.programs.decision
    expected = household_answer(target_payoff=target_payoff)
    solution = model.solve(params={"discount_factor": 1.0}, log_level="off")
    np.testing.assert_array_almost_equal(
        solution.values[0]["source"], expected.source_value, decimal=DECIMAL_PRECISION
    )
    frame = model.simulate(
        params={"discount_factor": 1.0},
        solution=solution,
        initial_conditions={
            "period": jnp.asarray([0], dtype=jnp.int32),
            "regime_id": jnp.asarray([0], dtype=jnp.int32),
        },
        seed=17,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    assert tuple(frame.sort_values("period")["regime_name"]) == expected.path


def physical_model(*, incoming: bool, target_payoff: int, budget: int | None) -> Model:
    return Model(
        regimes={
            "source": Regime(
                functions={"utility": zero},
                exits={
                    "move": Exit(
                        targets={
                            "self": ExitTarget(regime="counterparty", projection={})
                        },
                        rule="unilateral",
                    )
                },
            ),
            "counterparty": Regime(functions={"utility": three}),
            "target": Regime(functions={"utility": {1: one, 5: five}[target_payoff]}),
            "fallback": Regime(functions={"utility": forty}),
            "done": Regime(functions={"utility": zero}),
        },
        n_periods=2,
        regime_id_class=PhysicalOwners,
        initial_nodes=InitialNodes(
            by_period={0: "source" if incoming else "counterparty", 1: "target"}
        ),
        edges={
            "source": {"done": Periods(values=(0,))},
            "counterparty": counterparty_transition(),
        },
        execution_config=ExecutionConfig(device_memory_bytes=budget),
    )


@pytest.mark.parametrize("incoming", [False, True])
@pytest.mark.parametrize("target_payoff", [1, 5])
def test_real_participation_from_original_or_participating_actor_is_retained(
    *, incoming: bool, target_payoff: int
) -> None:
    """Incoming-only actor has no Full but retains its next-period participation check.

    A second subject starts at the target at date one and must not borrow the
    counterparty's participation rule or fallback.
    """
    model = physical_model(incoming=incoming, target_payoff=target_payoff, budget=None)
    expected = physical_answer(target_payoff=target_payoff)
    frame = model.simulate(
        params={"discount_factor": 1.0},
        initial_conditions={
            "period": jnp.asarray([0, 1], dtype=jnp.int32),
            "regime_id": jnp.asarray([0 if incoming else 1, 2], dtype=jnp.int32),
        },
        seed=17,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    primary = frame.loc[frame["subject_id"] == 0].sort_values("period")
    assert tuple(primary["regime_name"]) == expected.path
    np.testing.assert_array_almost_equal(
        primary["value"], expected.values, decimal=DECIMAL_PRECISION
    )
    initial_target = frame.loc[frame["subject_id"] == 1]
    assert tuple(initial_target["regime_name"]) == ("target",)
    np.testing.assert_array_almost_equal(
        initial_target["value"], target_payoff, decimal=DECIMAL_PRECISION
    )
