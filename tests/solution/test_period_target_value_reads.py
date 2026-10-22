"""A period program receives the values of its period targets, and only those.

A value is either a target's next-period value array or its next-period carry,
depending on what the regime's solver reads.

The grid-search model has three regimes with different active periods: `a` is
active at periods 0 to 3, `b` at periods 1 to 3, and the terminal `c` only at
period 4. At periods 0 to 2 each non-terminal regime's only period target is
itself; at period 3 both enter `c`. Every regime carries wealth and a fixed
two-category health state, so its value array has two axes.

The DC-EGM model is the four-period retirement model: working life leads to work,
retirement or death and retirement to retirement or death, except at period 2,
where both lead only to death.

The NB-EGM and nested models have one living regime that stays alive or dies,
except at its last active period, where it only dies.

The gated model's `src` stays in `src` at periods 0 and 1 and at period 2 enters
`tgt` through a gate whose closed branch pays the fallback regime `fb`; both are
terminal at period 3.
"""

from types import MappingProxyType
from typing import Unpack

import jax.numpy as jnp
import pytest

from _lcm.execution.output_layout import PlannedCore
from _lcm.solution import backward_induction
from _lcm.typing import PytreeValue, RegimeName
from lcm import (
    AgeGrid,
    AgeRange,
    DeterministicTransition,
    DiscreteGrid,
    Gate,
    LinSpacedGrid,
    Model,
    ProjectedRegimeValue,
    StakeholderRoute,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.regime import Regime as UserRegime
from lcm.solvers import NBEGM
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from lcm_examples import iskhakov_et_al_2017
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.solution import test_nbegm_epstein_zin_multi_regime as nbegm_model
from tests.solution._callback_types import (
    RunPeriodKernelKwargs,
    RunPeriodKernelResult,
)
from tests.test_models import n_nbegm_toy

type _Dispatch = tuple[RegimeName, int]


@categorical(ordered=False)
class _RegimeId:
    a: ScalarInt
    b: ScalarInt
    c: ScalarInt


@categorical(ordered=False)
class _Health:
    bad: ScalarInt
    good: ScalarInt


def _utility(*, consumption: ContinuousAction, health: DiscreteState) -> FloatND:
    return jnp.log(consumption) + 0.1 * health


def _terminal_utility(*, wealth: ContinuousState, health: DiscreteState) -> FloatND:
    return jnp.log(wealth) + 0.1 * health


def _next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption + 1.0


def _borrowing_constraint(
    *, consumption: ContinuousAction, wealth: ContinuousState
) -> FloatND:
    return consumption <= wealth


def _states() -> dict[str, LinSpacedGrid | DiscreteGrid]:
    return {
        "wealth": LinSpacedGrid(start=1.0, stop=5.0, n_points=4),
        "health": DiscreteGrid(_Health),
    }


def _working_regime() -> UserRegime:
    return UserRegime(
        functions={"utility": _utility},
        states=_states(),
        state_transitions={
            "wealth": _next_wealth,
            "health": fixed_transition("health"),
        },
        actions={"consumption": LinSpacedGrid(start=0.5, stop=4.0, n_points=5)},
        constraints={"borrowing_constraint": _borrowing_constraint},
    )


def _three_regime_model() -> Model:
    return Model(
        regimes={
            "a": _working_regime(),
            "b": _working_regime(),
            "c": UserRegime(
                functions={"utility": _terminal_utility},
                states=_states(),
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=4, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "a", 1: "b"},
        edges={
            "a": {"a": AgeRange(exclusive_stop=3), "c": 3},
            "b": {"b": AgeRange(start=1, exclusive_stop=3), "c": 3},
        },
    )


@categorical(ordered=False)
class _GatedRegimeId:
    src: ScalarInt
    tgt: ScalarInt
    fb: ScalarInt


def _next_gated_regime(age: float) -> ScalarInt:
    return jnp.where(age < 2, _GatedRegimeId.src, _GatedRegimeId.tgt)


def _open_when_rich(V_target: FloatND) -> BoolND:
    return V_target > 0.5


def _same_wealth(wealth: ContinuousState) -> ContinuousState:
    return wealth


def _same_health(health: DiscreteState) -> DiscreteState:
    return health


def _gated_model() -> Model:
    terminal = UserRegime(functions={"utility": _terminal_utility}, states=_states())
    return Model(
        regimes={"src": _working_regime(), "tgt": terminal, "fb": terminal},
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=_GatedRegimeId,
        initial_nodes={0: "src"},
        edges={
            "src": Transition(
                targets={"src": AgeRange(exclusive_stop=2), "tgt": 2, "fb": 2},
                law=DeterministicTransition(func=_next_gated_regime),
                gates={
                    "tgt": Gate(
                        predicate=_open_when_rich,
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fb",
                                    projection={
                                        "wealth": _same_wealth,
                                        "health": _same_health,
                                    },
                                )
                            )
                        },
                    )
                },
            )
        },
    )


def _record_value_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[_Dispatch, frozenset[RegimeName]]:
    """Record the target keys of the values each regime-period's programs get."""
    recorded: dict[_Dispatch, frozenset[RegimeName]] = {}
    current: list[_Dispatch] = []
    run_period_kernel = backward_induction._run_period_kernel
    planned_call = PlannedCore.__call__

    def _run(**kwargs: Unpack[RunPeriodKernelKwargs]) -> RunPeriodKernelResult:
        current.append((kwargs["regime_name"], kwargs["period"]))
        try:
            return run_period_kernel(**kwargs)
        finally:
            current.pop()

    def _call(
        self: PlannedCore, *args: PytreeValue, **kwargs: PytreeValue
    ) -> PytreeValue:
        keys = recorded.get(current[-1], frozenset())
        for name in ("next_regime_to_V_arr", "next_regime_to_continuation"):
            values = kwargs.get(name, MappingProxyType({}))
            assert isinstance(values, MappingProxyType)
            keys |= frozenset(values)
        recorded[current[-1]] = keys
        return planned_call(self, *args, **kwargs)

    monkeypatch.setattr(backward_induction, "_run_period_kernel", _run)
    monkeypatch.setattr(PlannedCore, "__call__", _call)
    return recorded


def test_grid_search_program_receives_only_its_period_targets_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each grid-search call reads exactly the next-period values of its targets."""
    recorded = _record_value_keys(monkeypatch)
    _three_regime_model().solve(params={"discount_factor": 0.95}, log_level="off")
    assert recorded == {
        ("a", 0): frozenset({"a"}),
        ("a", 1): frozenset({"a"}),
        ("a", 2): frozenset({"a"}),
        ("a", 3): frozenset({"c"}),
        ("b", 1): frozenset({"b"}),
        ("b", 2): frozenset({"b"}),
        ("b", 3): frozenset({"c"}),
        ("c", 4): frozenset(),
    }


def test_dcegm_program_receives_only_its_period_targets_carries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each DC-EGM call reads exactly the next-period carries of its targets."""
    n_periods = 4
    params = iskhakov_et_al_2017.get_params(
        n_periods=n_periods,
        discount_factor=0.98,
        disutility_of_work=1.0,
        interest_rate=0.0,
        wage=20.0,
    )
    recorded = _record_value_keys(monkeypatch)
    iskhakov_et_al_2017.get_dcegm_model(n_periods=n_periods).solve(
        params=params, log_level="off"
    )
    assert recorded == {
        ("working_life", 0): frozenset({"working_life", "retirement", "dead"}),
        ("working_life", 1): frozenset({"working_life", "retirement", "dead"}),
        ("working_life", 2): frozenset({"dead"}),
        ("retirement", 1): frozenset({"retirement", "dead"}),
        ("retirement", 2): frozenset({"dead"}),
        ("dead", 1): frozenset(),
        ("dead", 2): frozenset(),
        ("dead", 3): frozenset(),
    }


def test_nbegm_program_receives_only_its_period_targets_carries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each NB-EGM call reads exactly the next-period carries of its targets."""
    model = nbegm_model._build_model(
        solver=NBEGM(
            savings_grid=nbegm_model._SAVINGS_GRID, envelope_arithmetic="ordinary"
        )
    )
    recorded = _record_value_keys(monkeypatch)
    model.solve(params=nbegm_model._PARAMS, log_level="off")
    assert recorded == {
        ("alive", 0): frozenset({"alive", "dead"}),
        ("alive", 1): frozenset({"dead"}),
        ("dead", 1): frozenset(),
        ("dead", 2): frozenset(),
    }


@pytest.mark.parametrize(
    "variant",
    [
        "negm",
        pytest.param(
            "n_nbegm",
            marks=pytest.mark.requires_exact_affine_kernel(
                reason=EXACT_KERNEL_SKIP_REASON
            ),
        ),
    ],
)
def test_nested_programs_receive_only_their_period_targets_carries(
    *, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    """Every keeper, adjuster and outer-sweep call reads only its targets' carries."""
    model = n_nbegm_toy.build_model(variant=variant, n_periods=4)
    recorded = _record_value_keys(monkeypatch)
    model.solve(params={"discount_factor": 0.95}, log_level="off")
    assert recorded == {
        ("alive", 0): frozenset({"alive", "dead"}),
        ("alive", 1): frozenset({"alive", "dead"}),
        ("alive", 2): frozenset({"dead"}),
        ("dead", 0): frozenset(),
        ("dead", 1): frozenset(),
        ("dead", 2): frozenset(),
        ("dead", 3): frozenset(),
    }
