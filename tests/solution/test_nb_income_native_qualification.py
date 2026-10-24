"""Qualify native capture without a second export invocation."""

import hashlib
import json
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from _lcm.solution import nbegm
from lcm import Model
from lcm.typing import ScalarInt
from tests.solution import _nb_scientific_capture as observer
from tests.solution import test_nb_income_scientific_contract as scientific
from tests.solution import test_state_dependent_borrowing_limit as fixture


@fixture._EXACT
def test_scientific_capture_has_native_parity_and_lossless_roundtrip(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Share one scientific capture and compare fresh unhooked public results."""
    original_hooks = _hook_ids()
    original_capture = scientific.capture_original_nb_income
    assert original_capture is observer.capture_original_nb_income
    original_context = observer.drain_owned_callbacks
    captures: list[observer.NativeIncomeCapture] = []
    drains: list[str] = []

    def retain_capture(
        *,
        root: Path,
        source_map: Path | None,
        source_root: Path | None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> observer.NativeIncomeCapture:
        result = original_capture(
            root=root,
            source_map=source_map,
            source_root=source_root,
            monkeypatch=monkeypatch,
        )
        captures.append(result)
        return result

    @contextmanager
    def observe_drain(drain: Callable[[], None]) -> Generator[None]:
        def actual_drain() -> None:
            drain()
            drains.append("completed")

        with original_context(actual_drain):
            yield

    with monkeypatch.context() as hooks:
        hooks.setattr(scientific, "capture_original_nb_income", retain_capture)
        hooks.setattr(observer, "drain_owned_callbacks", observe_drain)
        scientific.test_original_nb_income_meets_independent_scientific_contract(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
        )
    assert _hook_ids() == original_hooks
    assert drains == ["completed"]
    assert len(captures) == 1
    capture = captures[0]
    assert len(capture.manifest["rows"]) == 4
    root = tmp_path / "capture"
    for solver in ("nbegm", "grid"):
        _assert_unhooked_parity(capture=capture, root=root, solver=solver)
    kinds = _assert_npz_roundtrip(capture=capture, root=root)
    assert {"numeric", "bool", "string"}.issubset(kinds)
    assert _hook_ids() == original_hooks
    (root / "native-qualification.json").write_text(
        json.dumps(
            {
                "precision": capture.manifest["precision"],
                "source_map_sha256": capture.manifest["source_map_sha256"],
                "scientific_capture_invocations": len(captures),
                "public_parity_methods": ["nbegm", "grid"],
                "fresh_unhooked_models": True,
                "success_drains": drains,
                "roundtrip_kinds": sorted(kinds),
                "four_hooks_restored": True,
            },
            indent=2,
        )
        + "\n"
    )


def test_capture_restores_hooks_when_primary_and_owned_drain_fail(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Preserve the primary exception after callback completion and drain failure."""
    original_hooks = _hook_ids()
    original_context = observer.drain_owned_callbacks
    primary = RuntimeError("primary fixture qualification failure")
    callbacks: list[int] = []
    drains: list[str] = []
    source_map = os.environ.get("NB_COMPLETE_OBSERVER_SOURCE_MAP")
    source_root = os.environ.get("NB_COMPLETE_OBSERVER_SOURCE_ROOT")

    def fail_invocation(
        *,
        solver: str,
        limit: str,
        lower_kind: str,
        taxed: bool,
    ) -> NDArray[np.generic]:
        assert (solver, limit, lower_kind, taxed) == (
            "nbegm",
            "income",
            "function",
            True,
        )
        assert all(
            actual != original
            for actual, original in zip(_hook_ids(), original_hooks, strict=True)
        )
        raise primary

    @contextmanager
    def fail_after_real_drain(drain: Callable[[], None]) -> Generator[None]:
        def actual_drain() -> None:
            drain()
            drains.append("completed-before-native-callback")
            _complete_native_callback(seen=callbacks)
            drains.append("native-callback-completed")
            raise RuntimeError("injected synchronous owned-drain failure")

        with original_context(actual_drain):
            yield

    with monkeypatch.context() as hooks:
        hooks.setattr(fixture, "_period0_consumption", fail_invocation)
        hooks.setattr(observer, "drain_owned_callbacks", fail_after_real_drain)
        with pytest.raises(
            RuntimeError, match="primary fixture qualification"
        ) as error:
            observer.capture_original_nb_income(
                root=tmp_path / "failure-capture",
                source_map=None if source_map is None else Path(source_map),
                source_root=None if source_root is None else Path(source_root),
                monkeypatch=monkeypatch,
            )
    assert error.value is primary
    assert callbacks == [7]
    assert drains == ["completed-before-native-callback", "native-callback-completed"]
    assert len(primary.__notes__) == 1
    assert "Owned callback drain also failed" in primary.__notes__[0]
    assert "injected synchronous owned-drain failure" in primary.__notes__[0]
    assert _hook_ids() == original_hooks
    jax.effects_barrier()
    (tmp_path / "failure-qualification.json").write_text(
        json.dumps(
            {
                "primary_preserved": True,
                "secondary_notes": primary.__notes__,
                "callback_arguments": callbacks,
                "drains": drains,
                "four_hooks_restored": True,
                "scope": (
                    "Real JAX callback completed; synchronous drain failure; no solve"
                ),
            },
            indent=2,
        )
        + "\n"
    )


def _hook_ids() -> tuple[int, int, int, int]:
    """Identify the exact four public and row-observer hook owners."""
    return (
        id(Model._process_params),
        id(nbegm._solve_ride_along_cell_step),
        id(Model.solve),
        id(Model.simulate),
    )


def _assert_unhooked_parity(
    *,
    capture: observer.NativeIncomeCapture,
    root: Path,
    solver: str,
) -> None:
    """Construct a fresh model after restoration and compare complete public data."""
    original_hooks = _hook_ids()
    model = fixture._model(
        solver=solver,
        limit="income",
        lower=fixture.income_lower,
        taxed=True,
    )
    assert model is not capture.models[solver]
    params = fixture._params(
        limit_param=None if solver == "grid" else "limit_share",
        taxed=True,
    )
    with np.load(root / f"{solver}-initial.npz", allow_pickle=False) as saved:
        initial = {name: jnp.asarray(saved[name]) for name in saved.files}
    solution = model.solve(params=params, log_level="off")
    assert set(solution.values) == set(capture.solutions[solver].values)
    for period, regimes in solution.values.items():
        assert set(regimes) == set(capture.solutions[solver].values[period])
        for regime, value in regimes.items():
            _assert_array_bits(
                actual=np.asarray(value),
                expected=np.asarray(capture.solutions[solver].values[period][regime]),
            )
    simulated = model.simulate(
        params=params,
        initial_conditions=initial,
        solution=solution,
        seed=528,
        log_level="off",
    )
    jax.effects_barrier()
    actual = simulated.to_dataframe(use_categorical_codes=True)
    expected = capture.simulations[solver].to_dataframe(use_categorical_codes=True)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    actions = actual.query("period == 0").sort_values("subject_id")["consumption"]
    _assert_array_bits(actual=actions.to_numpy(), expected=capture.actions[solver])
    _assert_saved_panel(capture=capture, root=root, solver=solver)
    assert _hook_ids() == original_hooks


def _assert_saved_panel(
    *,
    capture: observer.NativeIncomeCapture,
    root: Path,
    solver: str,
) -> None:
    """Decode actual saved public panel strings and compare original columns."""
    public = capture.simulations[solver].to_dataframe(use_categorical_codes=True)
    entry = next(
        entry
        for entry in capture.manifest["saved"]
        if entry["file"] == f"{solver}-simulation.npz"
    )
    with np.load(root / entry["file"], allow_pickle=False) as saved:
        assert set(saved.files) == set(public.columns)
        for name in saved.files:
            array = saved[name]
            encoding = entry["arrays"][name]["encoding"]
            if encoding == "json-string-ascii-in-fixed-unicode":
                decoded = [json.loads(value) for value in array.flat]
                assert decoded == public[name].to_list()
                assert all(isinstance(value, str) for value in decoded)
            else:
                assert encoding == "identity"
                _assert_array_bits(actual=array, expected=public[name].to_numpy())


def _assert_npz_roundtrip(
    *,
    capture: observer.NativeIncomeCapture,
    root: Path,
) -> set[str]:
    """Roundtrip actual numeric, boolean and lossless string NPZ arrays."""
    destination = root.parent / "roundtrip"
    destination.mkdir()
    kinds = set()
    for entry in capture.manifest["saved"]:
        source = root / entry["file"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry["sha256"]
        with np.load(source, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        np.savez(destination / entry["file"], **arrays)
        with np.load(destination / entry["file"], allow_pickle=False) as restored:
            assert set(restored.files) == set(arrays)
            for name, expected in arrays.items():
                _assert_array_bits(actual=restored[name], expected=expected)
                if expected.dtype.kind in "fiu":
                    kinds.add("numeric")
                elif expected.dtype.kind == "b":
                    kinds.add("bool")
                elif expected.dtype.kind == "U":
                    kinds.add("string")
                else:
                    pytest.fail(f"Unexpected captured array dtype {expected.dtype}")
    return kinds


def _assert_array_bits(
    *,
    actual: NDArray[np.generic],
    expected: NDArray[np.generic],
) -> None:
    """Require unchanged dtype, shape and complete native element bits."""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes(order="C") == expected.tobytes(order="C")


def _complete_native_callback(*, seen: list[int]) -> None:
    """Complete a real ordered callback and drain its native effects."""

    def record(value: NDArray[np.generic]) -> None:
        seen.append(int(value[()]))

    @jax.jit
    def dispatch(value: ScalarInt) -> ScalarInt:
        jax.debug.callback(record, value, ordered=True)
        return value + 1

    try:
        result = dispatch(jnp.asarray(7, dtype=jnp.int32)).block_until_ready()
        np.testing.assert_array_equal(result, 8)
    finally:
        jax.effects_barrier()


def test_capture_drains_and_restores_hooks_without_audit_environment(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ordinary test runs preserve the same primary, drain and hook contracts."""
    monkeypatch.delenv("NB_COMPLETE_OBSERVER_SOURCE_MAP", raising=False)
    monkeypatch.delenv("NB_COMPLETE_OBSERVER_SOURCE_ROOT", raising=False)
    test_capture_restores_hooks_when_primary_and_owned_drain_fail(
        monkeypatch=monkeypatch, tmp_path=tmp_path
    )
