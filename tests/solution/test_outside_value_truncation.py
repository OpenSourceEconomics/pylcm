"""An exit's outside value is the destination's problem value, without its exit.

A subject moves at most once per period, so the value of leaving is the value of the
destination's action problem at the projected state: the destination's own exit does
not constrain it. A consent target is the exception: the formed couple applies its own
rule for staying, and the outside values that rule compares are again problem values.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.policy_queries import QueryKind
from lcm import (
    CollectiveUtility,
    Consent,
    Counterparty,
    DiscreteGrid,
    ExecutionConfig,
    Exit,
    ExitTarget,
    InitialNodes,
    LinSpacedGrid,
    Model,
    Regime,
    Unilateral,
    categorical,
)
from lcm.exceptions import RegimeInitializationError
from lcm.typing import FloatND, IntND, ScalarInt
from tests.conftest import DECIMAL_PRECISION


@categorical(ordered=False)
class _Chain:
    source: ScalarInt
    destination: ScalarInt
    onward: ScalarInt


@categorical(ordered=False)
class _Formation:
    single_f: ScalarInt
    single_m: ScalarInt
    couple: ScalarInt
    divorced_f: ScalarInt
    onward: ScalarInt


@categorical(ordered=False)
class _TwoEntries:
    single_f: ScalarInt
    widow: ScalarInt
    single_m: ScalarInt
    couple: ScalarInt


def _identity(wealth: FloatND) -> FloatND:
    return wealth


def _one(wealth: FloatND) -> FloatND:
    return 1.0 + 0 * wealth


def _three_per_choice(*, choice: IntND, wealth: FloatND) -> FloatND:
    return 3.0 * choice + 0 * wealth


def _two_plus_two_wealth(wealth: FloatND) -> FloatND:
    return 2.0 + 2.0 * wealth


def _chain_model() -> Model:
    """Source -> destination -> onward, every move unilateral, one period.

    Closed form at wealth 0 and 1:

    - onward is worth 2 + 2 * wealth;
    - the destination's problem is worth max(0, 3) = 3; its own exit rejects every
      action at wealth 1, where onward is worth 4, and keeps choice 1 at wealth 0;
    - the source stays at 1, so it leaves for the destination's problem, worth 3,
      at both cells. Were the destination's exit applied, the destination would offer
      nothing at wealth 1 and the source would stay at 1 there.
    """
    wealth = LinSpacedGrid(start=0, stop=1, n_points=2)
    return Model(
        regimes={
            "source": Regime(
                states={"wealth": wealth},
                functions={"utility": _one},
                exit=Exit(
                    targets={
                        "self": ExitTarget(
                            regime="destination", projection={"wealth": _identity}
                        )
                    },
                    rule=Unilateral(),
                ),
            ),
            "destination": Regime(
                states={"wealth": wealth},
                actions={"choice": DiscreteGrid(values=jnp.asarray([0, 1]))},
                functions={"utility": _three_per_choice},
                exit=Exit(
                    targets={
                        "self": ExitTarget(
                            regime="onward", projection={"wealth": _identity}
                        )
                    },
                    rule=Unilateral(),
                ),
            ),
            "onward": Regime(
                states={"wealth": wealth},
                functions={"utility": _two_plus_two_wealth},
            ),
        },
        regime_id_class=_Chain,
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("source",)}),
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def test_an_exit_reads_its_destination_s_problem_without_the_destination_s_exit() -> (
    None
):
    """The source leaves for the destination's problem value 3 at both cells."""
    problem_read = np.asarray([3.0, 3.0])
    nested_read = np.asarray([3.0, 1.0])
    assert not np.allclose(problem_read, nested_read)
    solution = _chain_model().solve(params={}, log_level="off", runtime_checks=False)
    np.testing.assert_array_almost_equal(
        solution.values[0]["source"], problem_read, decimal=DECIMAL_PRECISION
    )


def _five() -> float:
    return 5.0


def _eight() -> float:
    return 8.0


def _female(*, action: IntND) -> FloatND:
    return 20.0 - 14.0 * action


def _male(*, action: IntND) -> FloatND:
    return 4.0 + 2.0 * action


def _seven_per_choice(*, choice: IntND) -> FloatND:
    return 7.0 * choice


def _formation_model() -> Model:
    """A single forms a couple by consent; the couple dissolves unilaterally.

    Closed form, one period:

    - the couple's actions pay (f, m) = (20, 4) and (6, 6);
    - a dissolving husband becomes single, worth 5;
    - a dissolving wife becomes divorced: that problem is worth max(0, 7) = 7,
      while its own exit to onward (worth 8) would reject both of its actions;
    - the couple keeps an action only if both members weakly prefer it to leaving,
      so with the outside values (7, 5) it keeps nothing and cannot form;
    - the single therefore stays at 5. With the divorced exit applied, the wife's
      outside value would be empty, the couple would keep (6, 6), and the single
      would marry at 6.
    """
    return Model(
        regimes={
            "single_f": Regime(
                functions={"utility": _five},
                exit=Exit(
                    targets={
                        "self": ExitTarget(regime="couple", role="f", projection={})
                    },
                    rule=Consent(
                        counterparty=Counterparty(
                            regime="single_m", role="m", projection={}
                        )
                    ),
                ),
            ),
            "single_m": Regime(functions={"utility": _five}),
            "couple": Regime(
                actions={"action": DiscreteGrid(values=jnp.asarray([0, 1]))},
                functions={
                    "utility": CollectiveUtility(utilities={"f": _female, "m": _male})
                },
                exit=Exit(
                    targets={
                        "f": ExitTarget(regime="divorced_f", projection={}),
                        "m": ExitTarget(regime="single_m", projection={}),
                    },
                    rule=Unilateral(),
                ),
            ),
            "divorced_f": Regime(
                actions={"choice": DiscreteGrid(values=jnp.asarray([0, 1]))},
                functions={"utility": _seven_per_choice},
                exit=Exit(
                    targets={"self": ExitTarget(regime="onward", projection={})},
                    rule=Unilateral(),
                ),
            ),
            "onward": Regime(functions={"utility": _eight}),
        },
        regime_id_class=_Formation,
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("single_f",)}),
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def test_a_formation_applies_the_couple_s_stay_rule_with_problem_outside_values() -> (
    None
):
    """The divorced problem's value 7 rejects (6, 6), so the single stays at 5."""
    solution = _formation_model().solve(
        params={}, log_level="off", runtime_checks=False
    )
    np.testing.assert_array_almost_equal(
        solution.values[0]["single_f"], 5.0, decimal=DECIMAL_PRECISION
    )


def test_a_mover_acts_in_the_destination_problem_it_was_priced_at() -> None:
    """Movers take the problem's best choice 1 and report the priced value 3."""
    model = _chain_model()
    solution = model.solve(params={}, log_level="off", runtime_checks=False)
    frame = model.simulate(
        params={},
        solution=solution,
        initial_conditions={
            "period": jnp.zeros(2, dtype=jnp.int32),
            "regime_id": jnp.zeros(2, dtype=jnp.int32),
            "wealth": jnp.asarray([0.0, 1.0]),
        },
        seed=17,
        log_level="off",
        runtime_checks=False,
    ).to_dataframe(use_categorical_codes=True)
    observed = {
        "regime": frame["regime_name"].astype(str).tolist(),
        "choice": frame["choice"].tolist(),
        "value": np.round(frame["value"].to_numpy(), DECIMAL_PRECISION).tolist(),
    }
    priced = np.round(np.asarray(solution.values[0]["source"]), DECIMAL_PRECISION)
    assert observed == {
        "regime": ["destination", "destination"],
        "choice": [1, 1],
        "value": priced.tolist(),
    }


def _widow_utility() -> float:
    return 0.0


def _two_entries_model(*, couple_exit: bool) -> Model:
    """A couple entered by a single's consent and by a widow's unilateral move."""
    return Model(
        regimes={
            "single_f": Regime(
                functions={"utility": _five},
                exit=Exit(
                    targets={
                        "self": ExitTarget(regime="couple", role="f", projection={})
                    },
                    rule=Consent(
                        counterparty=Counterparty(
                            regime="single_m", role="m", projection={}
                        )
                    ),
                ),
            ),
            "widow": Regime(
                functions={"utility": _widow_utility},
                exit=Exit(
                    targets={
                        "self": ExitTarget(regime="couple", role="f", projection={})
                    },
                    rule=Unilateral(),
                ),
            ),
            "single_m": Regime(functions={"utility": _five}),
            "couple": Regime(
                actions={"action": DiscreteGrid(values=jnp.asarray([0, 1]))},
                functions={
                    "utility": CollectiveUtility(utilities={"f": _female, "m": _male})
                },
                exit=Exit(
                    targets={
                        "f": ExitTarget(regime="single_f", projection={}),
                        "m": ExitTarget(regime="single_m", projection={}),
                    },
                    rule=Unilateral(),
                )
                if couple_exit
                else None,
            ),
        },
        regime_id_class=_TwoEntries,
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: ("single_f", "widow")}),
        execution_config=ExecutionConfig(device_memory_bytes=None),
    )


def test_a_regime_entered_by_consent_and_by_dissolution_is_refused() -> None:
    """Consent movers act with the couple's own exit, unilateral movers without it."""
    with pytest.raises(
        RegimeInitializationError, match=r"'couple'.*'single_f'.*'widow'"
    ):
        _two_entries_model(couple_exit=True)


def test_a_regime_without_an_exit_may_be_entered_by_consent_and_by_dissolution() -> (
    None
):
    """Without an exit of its own the couple is one problem for both movers."""
    model = _two_entries_model(couple_exit=False)
    acting = model._regimes["couple"].simulation.exit_queries[0].participating["couple"]
    assert acting.definitions[acting.root].read.kind is QueryKind.PARTICIPATING
