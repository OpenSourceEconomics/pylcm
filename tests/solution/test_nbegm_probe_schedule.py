"""How often NB-EGM re-runs its parameter-dependent preconditions.

The affine-budget and interval-constancy preconditions differentiate the model's
DAG against real parameter values, so they cannot run at model build and are
charged per solve instead — which an estimation loop pays on every criterion
evaluation. `probe_schedule` decides how often they run:

- `"every_solve"` — check every draw; the safe default because parameters can
  invalidate a declaration after an earlier draw passed.
- `"first_solve"` — check once per model, then trust the author's assertion that
  validity is parameter-invariant over the supported domain.
- `"never"` — skip them entirely; the model author asserts both preconditions.
"""

from typing import Any

import pytest

from _lcm.solution import nbegm as nbegm_module
from _lcm.solution.preconditions import check_solver_params
from lcm.exceptions import RegimeInitializationError
from lcm.model import Model
from lcm.solvers import NBEGM
from tests.test_models import nbegm_ride_discrete_toy as ride_toy


def _run_probes(*, model: Model, params: dict | None = None) -> None:
    """Run the solver's parameter-dependent preconditions, and nothing else."""
    check_solver_params(
        regimes=model._regimes,
        flat_params=model._process_params(
            ride_toy.build_params() if params is None else params
        ),
    )


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
