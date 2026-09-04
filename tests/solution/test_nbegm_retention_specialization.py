"""A solve dispatches only the programs its result retention consumes.

The ride-along NB-EGM graph publishes `main` (values only) and `replay` (values
plus the simulation policy). A values-only solve compiles and runs `main` alone,
everywhere. A replay-retaining solve runs `replay` only where a declared replay
route consumes the policy: inside NNBEGM's keeper and adjuster. A standalone
NB-EGM regime has no such route, so it runs `main` under every retention and its
policy is omitted as not applicable rather than assembled and discarded.

The two programs are one body, so the values and carries they publish agree to
the working format's spacing. Under a values-only retention the nested finite
solve never assembles its candidate banks. Persistence-oriented retention builds
the finite bank but skips the non-persistable adaptive policy and generated
authority. Solver diagnostics keep following `log_level` alone.
"""

from collections.abc import Callable
from typing import Any, cast

import pytest

from _lcm.egm.published_policy import NBEGMGridPolicy
from _lcm.solution import backward_induction
from _lcm.solution import nbegm as nbegm_module
from _lcm.solution import nnbegm as nnbegm_module
from _lcm.solution.contract import GENERATED_REPLAY_AUTHORITY
from _lcm.solution.solver_diagnostics import SolverDiagnostics
from lcm.solver_api import (
    EGM_CONTINUATION,
    SIMULATION_POLICY,
    SOLVER_DIAGNOSTICS,
    OmissionReason,
)
from lcm.solver_api import ResultRetention as Retention
from tests.conftest import assert_agrees_to_ulp
from tests.simulation.test_nnbegm_split_workflow_parity import _MESH, _PARAMS
from tests.test_models import n_nbegm_toy, nbegm_ride_along_toy

_ROUTES = {"finite": None, "adaptive": _MESH}


def _standalone() -> tuple[Any, Any]:
    model = nbegm_ride_along_toy.build_model(
        variant="nbegm", n_periods=3, n_liquid=12, n_savings=16
    )
    return model, nbegm_ride_along_toy.build_params()


def _nested(route: str) -> tuple[Any, Any]:
    model = n_nbegm_toy.build_model(
        variant="n_nbegm", n_periods=3, outer_search=_ROUTES[route]
    )
    return model, _PARAMS


_MODELS: dict[str, Callable[[], tuple[Any, Any]]] = {
    "standalone": _standalone,
    "nested-finite": lambda: _nested("finite"),
    "nested-adaptive": lambda: _nested("adaptive"),
}


def _record_dispatched_programs(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    recorded: list[tuple] = []
    original = nbegm_module._RideAlongNBEGMPeriodKernel.__call__

    def recording_call(self: Any, **kwargs: Any) -> Any:
        recorded.append(tuple(kwargs["compiled_cores"]))
        return original(self, **kwargs)

    monkeypatch.setattr(
        nbegm_module._RideAlongNBEGMPeriodKernel, "__call__", recording_call
    )
    return recorded


@pytest.mark.parametrize("route", list(_MODELS))
def test_a_values_only_solve_runs_main_alone(
    *, route: str, monkeypatch: pytest.MonkeyPatch
):
    recorded = _record_dispatched_programs(monkeypatch)
    model, params = _MODELS[route]()

    model.solve(params=params, log_level="off", retention=Retention.VALUES)

    assert recorded
    assert set(recorded) == {("main",)}


def test_a_standalone_regime_runs_main_under_every_retention(
    monkeypatch: pytest.MonkeyPatch,
):
    recorded = _record_dispatched_programs(monkeypatch)
    model, params = _standalone()

    solution = model.solve(
        params=params, log_level="off", retention=Retention.VALUES_AND_REPLAY
    )

    assert recorded
    assert set(recorded) == {("main",)}
    policy_refs = {ref for ref in solution.omissions if ref.key == SIMULATION_POLICY}
    assert policy_refs
    assert not any(ref.key == SIMULATION_POLICY for ref in solution.replay_artifacts)
    assert {solution.omissions[ref] for ref in policy_refs} == {
        OmissionReason.NOT_APPLICABLE
    }
    expected_type_id = f"{NBEGMGridPolicy.__module__}.{NBEGMGridPolicy.__qualname__}"
    assert {
        solution.metadata.artifact_descriptors[ref].payload_type_id
        for ref in policy_refs
    } == {expected_type_id}


@pytest.mark.parametrize("route", ["finite", "adaptive"])
def test_a_nested_replay_solve_runs_replay_alone(
    *, route: str, monkeypatch: pytest.MonkeyPatch
):
    recorded = _record_dispatched_programs(monkeypatch)
    model, params = _nested(route)

    model.solve(params=params, log_level="off", retention=Retention.VALUES_AND_REPLAY)

    assert recorded
    assert set(recorded) == {("replay",)}


@pytest.mark.parametrize(
    ("route", "expected_program"), [("finite", "replay"), ("adaptive", "main")]
)
def test_all_persistable_dispatches_only_model_verifiable_replay(
    *,
    route: str,
    expected_program: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = _record_dispatched_programs(monkeypatch)
    model, params = _nested(route)

    model.solve(
        params=params,
        log_level="off",
        retention=Retention.ALL_PERSISTABLE_ARTIFACTS,
    )

    assert recorded
    assert set(recorded) == {(expected_program,)}


@pytest.mark.parametrize("route", list(_MODELS))
def test_values_only_and_replay_solves_publish_the_same_values(*, route: str):
    model, params = _MODELS[route]()

    values_only = model.solve(
        params=params, log_level="off", retention=Retention.VALUES
    )
    with_replay = model.solve(
        params=params, log_level="off", retention=Retention.VALUES_AND_REPLAY
    )

    assert values_only.values.keys() == with_replay.values.keys()
    for period, regime_to_value in values_only.values.items():
        assert regime_to_value.keys() == with_replay.values[period].keys()
        for regime_name, value in regime_to_value.items():
            assert_agrees_to_ulp(
                got=value,
                expected=with_replay.values[period][regime_name],
                n_ulp=4,
                err_msg=f"({period}, {regime_name})",
            )


def test_a_values_only_finite_nested_solve_never_assembles_candidate_banks(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[int] = []
    original = nnbegm_module._conditional_nnbegm_banks

    def counting(**kwargs: Any) -> Any:
        calls.append(1)
        return original(**kwargs)

    monkeypatch.setattr(nnbegm_module, "_conditional_nnbegm_banks", counting)
    model, params = _nested("finite")

    solution = model.solve(params=params, log_level="off", retention=Retention.VALUES)
    assert not calls
    policy_refs = {ref for ref in solution.omissions if ref.key == SIMULATION_POLICY}
    assert policy_refs
    assert {solution.omissions[ref] for ref in policy_refs} == {
        OmissionReason.NOT_REQUESTED
    }

    model.solve(params=params, log_level="off", retention=Retention.VALUES_AND_REPLAY)
    assert calls


def test_a_values_only_adaptive_nested_solve_publishes_no_policy_or_authority(
    monkeypatch: pytest.MonkeyPatch,
):
    results: list[Any] = []
    original = backward_induction._run_period_kernel

    def recording(**kwargs: Any) -> Any:
        result = original(**kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(backward_induction, "_run_period_kernel", recording)
    model, params = _nested("adaptive")

    model.solve(params=params, log_level="off", retention=Retention.VALUES)

    assert results
    assert all(SIMULATION_POLICY not in result.replay for result in results)
    assert all(GENERATED_REPLAY_AUTHORITY not in result.auxiliary for result in results)
    assert all(EGM_CONTINUATION in result.continuations for result in results[:-1])


@pytest.mark.parametrize(
    ("log_level", "expects_diagnostics"), [("warning", True), ("off", False)]
)
def test_diagnostics_follow_log_level_under_a_values_only_retention(
    *, log_level: str, expects_diagnostics: bool
):
    model, params = _nested("adaptive")

    solution = model.solve(
        params=params, log_level=log_level, retention=Retention.VALUES
    )

    retained = solution.diagnostics.project(SOLVER_DIAGNOSTICS)
    assert bool(retained) is expects_diagnostics
    for regime_to_diagnostics in retained.values():
        for diagnostics in regime_to_diagnostics.values():
            error = cast("SolverDiagnostics", diagnostics).max_outer_interpolation_error
            assert all(device.platform == "cpu" for device in error.devices())
