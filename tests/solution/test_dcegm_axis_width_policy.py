"""What width the plan gives each DC-EGM axis, observed through a period capture.

An unbudgeted solve streams every declared axis at its bootstrap width — the
largest power of two strictly below the extent — so a DC-EGM regime's loops are
bounded without the user saying anything. The whole axis is reached two ways:
by naming its extent in `ExecutionConfig.axis_widths`, or by declaring a
device-memory budget the widest candidate fits in. The capture records exactly
what the solve dispatched, so these are facts about the run rather than about
the planner in isolation.
"""

import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import cloudpickle
import jax.numpy as jnp
import pytest

from _lcm.solution import backward_induction
from _lcm.solution.period_capture import _PAYLOAD_NAME
from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.regime import Regime as UserRegime
from lcm.solvers import (
    CELL_AXIS,
    DCEGM,
    EULER_POINT_AXIS,
    SAVINGS_POINT_AXIS,
    STOCHASTIC_NODE_AXIS,
)
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from tests.conftest import EXACT_KERNEL_SKIP_REASON

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

_CAPTURE_TARGET = "working@0"
# Synthetic compiler peak per unit of width product, so the selection rule is
# checked on any backend rather than on this machine's compiler report.
_BYTES_PER_CELL = 1000

N_HEALTH = 3
N_WEALTH = 12
N_SAVINGS = 30
# Extents of the three tiled axes, and the bootstrap width each takes when no
# budget and no explicit width is declared.
# The Markov `health` state is both an output state cell and the child
# stochastic mesh the continuation folds, so this one regime declares all four.
_EXTENTS = MappingProxyType(
    {
        CELL_AXIS: N_HEALTH,
        SAVINGS_POINT_AXIS: N_SAVINGS,
        EULER_POINT_AXIS: N_WEALTH,
        STOCHASTIC_NODE_AXIS: N_HEALTH,
    }
)
_BOOTSTRAP = MappingProxyType(
    {
        CELL_AXIS: 2,
        SAVINGS_POINT_AXIS: 16,
        EULER_POINT_AXIS: 8,
        STOCHASTIC_NODE_AXIS: 2,
    }
)


@categorical(ordered=False)
class Health:
    bad: ScalarInt
    fair: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    dead: ScalarInt


def survival(wealth: ContinuousState) -> FloatND:
    # Reading the Euler state here puts the kernel in asset-row mode, which is
    # what makes `euler_point` a loop the regime actually runs.
    return 0.5 + 0.45 * jnp.clip(wealth / 100.0, 0.0, 1.0)


def stay_prob(*, wealth: ContinuousState, age: int, final_age_alive: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 0.0, survival(wealth))


def death_prob(*, wealth: ContinuousState, age: int, final_age_alive: float) -> FloatND:
    return 1.0 - stay_prob(wealth=wealth, age=age, final_age_alive=final_age_alive)


def health_transition(health: DiscreteState) -> FloatND:
    stay = jnp.where(health == Health.good, 0.7, 0.5)
    others = (1.0 - stay) / 2.0
    return jnp.stack([others, others, stay])


def utility(*, consumption: ContinuousAction, health: DiscreteState) -> FloatND:
    return jnp.log(consumption) - jnp.where(health == Health.bad, 0.2, 0.0)


def savings(*, wealth: FloatND, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def next_wealth(savings: FloatND) -> ContinuousState:
    return savings + 3.0


def bequest(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth + 1.0)


def _ages() -> AgeGrid:
    return AgeGrid(start=40, stop=60, step="10Y")


def _model(*, execution_config: ExecutionConfig) -> Model:
    """Asset-row DC-EGM toy declaring all three tiled axes at once."""
    ages = _ages()
    last_age = ages.exact_values[-1]

    working = ConsumptionSavingsRegime(
        transition={
            "working": MarkovTransition(stay_prob),
            "dead": MarkovTransition(death_prob),
        },
        active=lambda age, la=last_age: age < la,
        actions={"consumption": LinSpacedGrid(start=0.25, stop=100.0, n_points=20)},
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=N_WEALTH),
            "health": DiscreteGrid(category_class=Health),
        },
        state_transitions={
            "wealth": next_wealth,
            "health": MarkovTransition(health_transition),
        },
        functions={
            "utility": utility,
            "savings": savings,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=DCEGM(
            savings_grid=LinSpacedGrid(start=0.0, stop=110.0, n_points=N_SAVINGS),
            n_constrained_points=8,
        ),
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="wealth",
            post_decision_state="savings",
        ),
    )
    dead = UserRegime(
        transition=None,
        states={"wealth": LinSpacedGrid(start=1.0, stop=120.0, n_points=12)},
        functions={"utility": bequest},
    )
    return Model(
        regimes={"working": working, "dead": dead},
        ages=ages,
        regime_id_class=RegimeId,
        execution_config=execution_config,
    )


@pytest.fixture
def synthetic_peaks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report a peak of `_BYTES_PER_CELL` per unit of width product."""

    def peak(*, compiled: object, widths: Mapping[str, int]) -> int:
        del compiled
        return _BYTES_PER_CELL * math.prod(widths.values())

    monkeypatch.setattr(backward_induction, "compiler_peak_bytes", peak)


def _captured_widths(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    execution_config: ExecutionConfig,
) -> dict[str, int]:
    """Solve while capturing the first working period; return its main widths."""
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", _CAPTURE_TARGET)
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    model = _model(execution_config=execution_config)
    model.solve(
        params={
            "working": {
                "koopmans_aggregator": {"discount_factor": 0.95},
                "working": {"next_regime": {"final_age_alive": 50.0}},
                "dead": {"next_regime": {"final_age_alive": 50.0}},
            },
            "dead": {},
        },
        log_level="off",
    )
    with (tmp_path / _CAPTURE_TARGET / _PAYLOAD_NAME).open("rb") as stream:
        return cloudpickle.load(stream)["core_tile_widths"]["main"]


def test_an_unbudgeted_solve_streams_every_axis_at_its_bootstrap_width(
    *, monkeypatch, tmp_path
) -> None:
    """With nothing declared, each loop runs bounded rather than fused."""
    assert _captured_widths(
        monkeypatch=monkeypatch, tmp_path=tmp_path, execution_config=ExecutionConfig()
    ) == dict(_BOOTSTRAP)


@pytest.mark.parametrize("axis", sorted(_EXTENTS))
def test_naming_the_extent_plans_the_whole_axis(
    *, axis: str, monkeypatch, tmp_path
) -> None:
    """`axis_widths={axis: extent}` dispatches the whole axis, i.e. one tile."""
    widths = _captured_widths(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        execution_config=ExecutionConfig(axis_widths={axis: _EXTENTS[axis]}),
    )

    assert widths[axis] == _EXTENTS[axis]


@pytest.mark.usefixtures("synthetic_peaks")
def test_a_budget_the_widest_candidate_fits_reaches_every_extent(
    *, monkeypatch, tmp_path
) -> None:
    """A declared device-memory budget large enough plans the full extents."""
    budget = _BYTES_PER_CELL * math.prod(_EXTENTS.values()) * 2

    assert _captured_widths(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        execution_config=ExecutionConfig(device_memory_bytes=budget),
    ) == dict(_EXTENTS)
