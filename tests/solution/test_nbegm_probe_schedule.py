"""How often NB-EGM re-runs its parameter-dependent preconditions.

The affine-budget, interval-constancy, and single-power-flow preconditions
differentiate the model's DAG against real parameter values, so they cannot run
at model build and are charged per solve instead — which an estimation loop pays
on every criterion evaluation. `probe_schedule` decides how often they run:

- `"every_solve"` — check every draw; the safe default because parameters can
  invalidate a declaration after an earlier draw passed.
- `"first_solve"` — check once per model, then trust the author's assertion that
  validity is parameter-invariant over the supported domain.
- `"never"` — skip them entirely; the model author asserts all three preconditions.
"""

from collections.abc import Mapping
from typing import Any, Literal

import jax.numpy as jnp
import pytest

from _lcm.solution import nbegm as nbegm_module
from _lcm.solution.preconditions import check_solver_params
from lcm import (
    AgeGrid,
    CESAggregator,
    DiscreteGrid,
    LinSpacedGrid,
    PowerMean,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.exceptions import RegimeInitializationError
from lcm.model import Model
from lcm.solvers import NBEGM
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from tests.test_models import nbegm_ride_discrete_toy as ride_toy


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class _Kind:
    low: ScalarInt
    high: ScalarInt


type ProbeSchedule = Literal["first_solve", "every_solve", "never"]


def _run_probes(*, model: Model, params: dict | None = None) -> None:
    """Run the solver's parameter-dependent preconditions, and nothing else."""
    check_solver_params(
        regimes=model._regimes,
        flat_params=model._process_params(
            ride_toy.build_params() if params is None else params
        ),
    )


def _single_power_model(*, probe_schedule: ProbeSchedule) -> Model:
    """Build an Epstein-Zin ride-along model with flow `q(c) = c + B`."""

    def _flow(
        *, consumption: ContinuousAction, kind: DiscreteState, flow_offset: float
    ) -> FloatND:
        return consumption + flow_offset + 0.0 * kind

    def _resources(*, wealth: ContinuousState, kind: DiscreteState) -> FloatND:
        return wealth + 0.5 * kind

    def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
        return resources - consumption

    def _next_wealth(savings: FloatND) -> ContinuousState:
        return savings

    def _next_regime() -> ScalarInt:
        return _RegimeId.dead

    def _bequest(*, wealth: ContinuousState, kind: DiscreteState) -> FloatND:
        return jnp.sqrt(wealth) + 0.0 * kind

    wealth = LinSpacedGrid(start=1.0, stop=10.0, n_points=5)
    kind = DiscreteGrid(category_class=_Kind)
    alive = ConsumptionSavingsRegime(
        transition=_next_regime,
        states={"wealth": wealth, "kind": kind},
        state_transitions={
            "wealth": _next_wealth,
            "kind": fixed_transition("kind"),
        },
        actions={"consumption": LinSpacedGrid(start=0.5, stop=5.0, n_points=5)},
        functions={
            "utility": _flow,
            "resources": _resources,
            "savings": _savings,
        },
        koopmans_aggregator=CESAggregator(),
        certainty_equivalent=PowerMean(),
        solver=NBEGM(
            savings_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
            probe_schedule=probe_schedule,
        ),
        active=lambda age: age < 41,
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        ),
    )
    dead = Regime(
        transition=None,
        states={"wealth": wealth, "kind": kind},
        functions={"utility": _bequest},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=40, stop=41, step="Y"),
        regime_id_class=_RegimeId,
    )


def _single_power_params(*, model: Model, flow_offset: float) -> dict[str, Any]:
    """Fill the model's parameter template and set the flow offset."""

    def _fill(node: object) -> object:
        if isinstance(node, Mapping):
            return {key: _fill(value) for key, value in node.items()}
        return 1.0

    params = _fill(model.get_params_template())
    assert isinstance(params, dict)
    params["alive"]["utility"]["flow_offset"] = flow_offset
    return params


# Every count in this module is relative to the first solve's tally, so a probe
# that never ran leaves each comparison reading 0 against 0 — and the schedule
# under test would look honoured whichever value it held. Each relative
# assertion is preceded by this one.
_NO_PROBE_RAN = (
    "no precondition probe ran at all, so the counts below compare zero with "
    "zero and cannot distinguish one schedule from another"
)


def _counting_model(
    monkeypatch: pytest.MonkeyPatch, **solver_kwargs: Any
) -> tuple[Model, list[int]]:
    """Build a probe-clean NB-EGM toy whose constancy probe counts its calls."""
    calls: list[int] = []
    original = nbegm_module._fail_if_liquid_reading_next_state_varies_within_interval

    def _counted(**kwargs: Any) -> None:
        calls.append(1)
        return original(**kwargs)

    monkeypatch.setattr(
        nbegm_module,
        "_fail_if_liquid_reading_next_state_varies_within_interval",
        _counted,
    )
    model = ride_toy.build_model(variant="nbegm", **solver_kwargs)
    return model, calls


def test_probe_schedule_defaults_to_every_solve() -> None:
    """A model that does not opt out revalidates every parameter draw."""
    assert NBEGM.__dataclass_fields__["probe_schedule"].default == "every_solve"


def test_default_rejects_a_later_draw_that_invalidates_budget_affinity() -> None:
    """A valid first draw must not license a curved second draw on the same model."""
    model = ride_toy.build_model(
        variant="nbegm",
        nonlinear_budget_above_ten=True,
    )
    _run_probes(
        model=model,
        params=ride_toy.build_params(
            nonlinear_budget_above_ten=True,
            curvature=0.0,
        ),
    )

    with pytest.raises(RegimeInitializationError, match="must be affine"):
        _run_probes(
            model=model,
            params=ride_toy.build_params(
                nonlinear_budget_above_ten=True,
                curvature=0.05,
            ),
        )


@pytest.mark.parametrize(
    ("probe_schedule", "rejects_later_draw"),
    [
        ("every_solve", True),
        ("first_solve", False),
        ("never", False),
    ],
)
def test_probe_schedule_controls_single_power_flow_revalidation(
    *, probe_schedule: ProbeSchedule, rejects_later_draw: bool
) -> None:
    """The schedule controls whether a later non-power flow is rejected."""
    model = _single_power_model(probe_schedule=probe_schedule)

    _run_probes(
        model=model,
        params=_single_power_params(model=model, flow_offset=0.0),
    )
    later_params = _single_power_params(model=model, flow_offset=0.25)

    if rejects_later_draw:
        with pytest.raises(RegimeInitializationError, match="single power"):
            _run_probes(model=model, params=later_params)
    else:
        _run_probes(model=model, params=later_params)


def test_first_solve_runs_the_preconditions_once_across_two_solves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second solve reuses the first solve's verdict."""
    model, calls = _counting_model(monkeypatch, probe_schedule="first_solve")

    _run_probes(model=model)
    after_one_solve = len(calls)
    _run_probes(model=model)

    assert after_one_solve > 0, _NO_PROBE_RAN
    assert len(calls) == after_one_solve


def test_every_solve_runs_the_preconditions_on_each_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each draw is re-checked when the model asks for it."""
    model, calls = _counting_model(monkeypatch, probe_schedule="every_solve")

    _run_probes(model=model)
    after_one_solve = len(calls)
    _run_probes(model=model)

    assert after_one_solve > 0, _NO_PROBE_RAN
    assert len(calls) == 2 * after_one_solve


def test_never_runs_no_precondition_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skipping the preconditions costs nothing and checks nothing."""
    model, calls = _counting_model(monkeypatch, probe_schedule="never")

    _run_probes(model=model)

    assert calls == []


def test_never_solves_a_model_the_preconditions_would_refuse() -> None:
    """A smoothly liquid-varying co-state law is admitted when checks are off."""
    model = ride_toy.build_model(
        variant="nbegm",
        costate_reads_liquid=True,
        costate_smooth=True,
        probe_schedule="never",
    )

    _run_probes(model=model)


def test_first_solve_still_refuses_a_model_the_preconditions_reject() -> None:
    """Checking once is still checking: a violation raises out of the first solve."""
    model = ride_toy.build_model(
        variant="nbegm",
        costate_reads_liquid=True,
        costate_smooth=True,
        probe_schedule="first_solve",
    )

    with pytest.raises(RegimeInitializationError, match="piecewise-constant"):
        _run_probes(model=model)
