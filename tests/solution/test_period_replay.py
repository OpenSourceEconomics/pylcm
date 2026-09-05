"""One regime-period can be captured during a solve and re-run on its own.

Diagnosing a kernel that is slow, or whose allocation is refused, otherwise costs a
full backward induction: every period above the one in question has to be solved
before the interesting one is reached. Capturing the inputs of a single regime-period
turns that into one kernel invocation.

The capture is written from the funnel every regime-period passes through, so what is
replayed is what ran — not a reconstruction that might differ from it.
"""

import logging
import math
import re

import cloudpickle
import numpy as np
import pytest

from _lcm.solution import backward_induction, period_replay
from lcm import AgeGrid, Model
from lcm.persistence import replay_period
from lcm.solver_api import ResultRetention
from tests.regime_building.test_gated_edges_collective_solve import (
    EKLRegimeId,
    _make_full_topology_regimes,
)
from tests.test_models.deterministic.discrete import (
    get_model,
    get_params,
)

_N_PERIODS = 3


def _solve_capturing(*, monkeypatch, tmp_path, target: str | None):
    """Solve the discrete toy, optionally capturing one regime-period."""
    if target is None:
        monkeypatch.delenv("LCM_CAPTURE_PERIOD", raising=False)
    else:
        monkeypatch.setenv("LCM_CAPTURE_PERIOD", target)
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    model = get_model(n_periods=_N_PERIODS)
    params = get_params(n_periods=_N_PERIODS)
    solution = model.solve(params=params, log_level="off").values
    return model, solution


def test_no_capture_is_written_without_the_target(*, monkeypatch, tmp_path):
    """The instrument costs nothing until a regime-period is named."""
    _solve_capturing(monkeypatch=monkeypatch, tmp_path=tmp_path, target=None)
    assert list(tmp_path.iterdir()) == []


def test_exactly_the_named_regime_period_is_captured(*, monkeypatch, tmp_path):
    """Naming one regime-period writes one capture, not one per period."""
    _solve_capturing(
        monkeypatch=monkeypatch, tmp_path=tmp_path, target="working_life@1"
    )
    captures = sorted(path.name for path in tmp_path.iterdir())
    assert captures == ["working_life@1"]


def test_a_capture_names_the_regime_period_and_age_it_holds(*, monkeypatch, tmp_path):
    """The capture is self-describing, so a directory of them can be read."""
    model, _ = _solve_capturing(
        monkeypatch=monkeypatch, tmp_path=tmp_path, target="working_life@1"
    )
    replay = replay_period(directory=tmp_path / "working_life@1")
    assert (replay.regime_name, replay.period, replay.age) == (
        "working_life",
        1,
        float(model.ages.values[1]),
    )


def test_replay_reproduces_the_value_function_of_the_full_solve(
    *, monkeypatch, tmp_path
):
    """Re-running the captured kernel gives back the array the solve produced.

    This is what makes the harness usable as a stand-in for the full run: the
    replayed kernel is the same computation on the same inputs, so a measurement
    taken on it transfers.
    """
    _, solution = _solve_capturing(
        monkeypatch=monkeypatch, tmp_path=tmp_path, target="working_life@1"
    )
    replay = replay_period(directory=tmp_path / "working_life@1")

    np.testing.assert_array_equal(
        np.asarray(replay.output.value),
        np.asarray(solution[1]["working_life"]),
    )


def test_replay_does_not_recapture_when_the_selector_remains_set(
    *, monkeypatch, tmp_path
):
    """Replay reads a selected capture without writing into its directory."""
    _, solution = _solve_capturing(
        monkeypatch=monkeypatch, tmp_path=tmp_path, target="working_life@1"
    )
    capture = tmp_path / "working_life@1"
    capture.chmod(0o555)
    try:
        replay = replay_period(directory=capture)
    finally:
        capture.chmod(0o755)

    np.testing.assert_array_equal(
        np.asarray(replay.output.value),
        np.asarray(solution[1]["working_life"]),
    )


def test_replay_runs_one_kernel_and_no_others(*, monkeypatch, tmp_path, caplog):
    """Replay does not re-solve the periods above the captured one."""
    _solve_capturing(
        monkeypatch=monkeypatch, tmp_path=tmp_path, target="working_life@1"
    )
    monkeypatch.setenv("LCM_LOG_KERNEL_ATTRIBUTION", "1")
    with caplog.at_level(logging.NOTSET, logger="lcm"):
        replay_period(directory=tmp_path / "working_life@1")

    executed = [
        record.getMessage()
        for record in caplog.records
        if re.search(r"\[attr\] \S+ age", record.getMessage())
    ]
    assert len(executed) == 1


def test_an_unknown_regime_period_is_refused(*, monkeypatch, tmp_path):
    """A target that never runs produces no capture rather than an empty one."""
    _solve_capturing(
        monkeypatch=monkeypatch, tmp_path=tmp_path, target="working_life@99"
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("target", ["working_life", "working_life@", "@1", "x@y"])
def test_a_malformed_target_is_rejected_loudly(*, monkeypatch, tmp_path, target):
    """A typo in the target must not read as "nothing to capture"."""
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", target)
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    model = get_model(n_periods=_N_PERIODS)
    params = get_params(n_periods=_N_PERIODS)
    with pytest.raises(ValueError, match="LCM_CAPTURE_PERIOD"):
        model.solve(params=params, log_level="off")


def test_a_malformed_target_is_rejected_before_kernel_compilation(
    *, monkeypatch, tmp_path
):
    """Capture selection is validated before solve kernels are compiled."""
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", "working_life")
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))

    def fail_if_compilation_starts(*_args: object, **_kwargs: object) -> None:
        msg = "kernel compilation started"
        raise AssertionError(msg)

    monkeypatch.setattr(
        backward_induction, "_compile_all_functions", fail_if_compilation_starts
    )
    model = get_model(n_periods=_N_PERIODS)
    params = get_params(n_periods=_N_PERIODS)

    with pytest.raises(ValueError, match="LCM_CAPTURE_PERIOD"):
        model.solve(params=params, log_level="off")


def test_a_gated_edge_source_replays_to_the_value_the_solve_published(
    *, monkeypatch, tmp_path
):
    """A regime whose continuation reads a gated edge replays to the solved `V_arr`.

    The source's kernel does not read its target's raw value function: it reads the
    gated continuation folded onto the target's grid. Replay must hand the kernel
    that same object, so a captured gated-edge period returns the array the solve
    published rather than refusing at lowering.
    """
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", "single_f@0")
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    model = Model(
        regimes=_make_full_topology_regimes(),
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=EKLRegimeId,
    )
    params = {"discount_factor": 0.95, "delta_f": 0.5, "delta_m": 0.2}
    solution = model.solve(params=params, log_level="off").values

    replay = replay_period(directory=tmp_path / "single_f@0")

    np.testing.assert_array_equal(
        np.asarray(replay.output.value),
        np.asarray(solution[0]["single_f"]),
    )


@pytest.mark.parametrize(
    ("retention", "retain_replay"),
    [
        pytest.param(ResultRetention.VALUES, False, id="values"),
        pytest.param(ResultRetention.VALUES_AND_REPLAY, True, id="values-and-replay"),
    ],
)
def test_replay_lowers_the_scope_the_solve_dispatched(
    *, monkeypatch, tmp_path, retention, retain_replay
):
    """A replay selects the programs the captured solve dispatched for the regime.

    The solve dispatches a regime's replay-scoped programs only when the retention
    keeps replay artifacts and the regime declares a route that consumes them; the
    capture records that decision and the replay lowers exactly that scope.
    """
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", "working_life@1")
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    model = get_model(n_periods=_N_PERIODS)
    params = get_params(n_periods=_N_PERIODS)
    model.solve(params=params, log_level="off", retention=retention)
    regime_retains_replay = (
        retain_replay
        and model._regimes["working_life"].simulation.egm_policy_read is not None
    )

    observed: list[tuple[bool, frozenset[object]]] = []
    real_select = period_replay.select_programs

    def record_select(**kwargs):
        observed.append((kwargs["retain_replay"], kwargs["selected_artifact_keys"]))
        return real_select(**kwargs)

    monkeypatch.setattr(period_replay, "select_programs", record_select)
    replay_period(directory=tmp_path / "working_life@1")

    assert observed == [(regime_retains_replay, frozenset())]


def _rewrite_capture(*, directory, mutate):
    """Load, mutate, and rewrite one capture payload."""
    path = directory / period_replay._PAYLOAD_NAME
    with path.open("rb") as stream:
        payload = cloudpickle.load(stream)
    mutate(payload)
    with path.open("wb") as stream:
        cloudpickle.dump(payload, stream)


def test_a_capture_records_the_tile_widths_the_solve_dispatched(
    *, monkeypatch, tmp_path
):
    """The captured widths are the full action product of the unbudgeted solve."""
    model, _ = _solve_capturing(
        monkeypatch=monkeypatch, tmp_path=tmp_path, target="working_life@1"
    )
    path = tmp_path / "working_life@1" / period_replay._PAYLOAD_NAME
    with path.open("rb") as stream:
        payload = cloudpickle.load(stream)

    regime = model._regimes["working_life"]
    space = regime.solution.state_action_space(
        regime_params=model._process_params(get_params(n_periods=_N_PERIODS))[
            "working_life"
        ]
    )
    extent = math.prod(space.actions_grid_shapes)
    assert payload["core_tile_widths"] == {"main": {"action": extent}}


def test_a_capture_without_tile_widths_is_refused(*, monkeypatch, tmp_path):
    """Replay never plans widths itself, so a capture must state them."""
    _solve_capturing(
        monkeypatch=monkeypatch, tmp_path=tmp_path, target="working_life@1"
    )
    _rewrite_capture(
        directory=tmp_path / "working_life@1",
        mutate=lambda payload: payload.pop("core_tile_widths"),
    )

    with pytest.raises(ValueError, match="core_tile_widths"):
        replay_period(directory=tmp_path / "working_life@1")


@pytest.mark.parametrize(
    ("widths", "error"),
    [
        ({"main": {"action": 0}}, ValueError),
        ({"main": {"action": 2.0}}, TypeError),
        ({"other": {"action": 2}}, ValueError),
    ],
    ids=["nonpositive-width", "noninteger-width", "wrong-core-names"],
)
def test_malformed_captured_tile_widths_are_refused(
    *, monkeypatch, tmp_path, widths, error
):
    _solve_capturing(
        monkeypatch=monkeypatch, tmp_path=tmp_path, target="working_life@1"
    )
    _rewrite_capture(
        directory=tmp_path / "working_life@1",
        mutate=lambda payload: payload.__setitem__("core_tile_widths", widths),
    )

    with pytest.raises(error):
        replay_period(directory=tmp_path / "working_life@1")
