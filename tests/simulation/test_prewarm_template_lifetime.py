"""AOT gate preparation does not retain a bank of concrete value templates."""

import dataclasses
import logging
import threading
import weakref
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import jax
import pytest

import _lcm.simulation.compile as compile_module
from _lcm.regime_building.Q_and_F import SAME_PERIOD_V_ARG
from lcm import Model, categorical
from lcm.typing import ScalarInt
from lcm_examples.collective_regimes import get_dissolution_model, get_params


@categorical(ordered=False)
class _ThreeSourceRegimeId:
    """Three genuine gates share the same collective destination."""

    source_a: ScalarInt
    source_b: ScalarInt
    source_c: ScalarInt
    married_with_participation: ScalarInt
    married_terminal: ScalarInt
    single_f: ScalarInt
    single_m: ScalarInt


def _three_gate_model() -> Model:
    """Reuse the economic declarations with three independently addressed gates."""
    original = get_dissolution_model()
    regimes = dict(original.user_regimes)
    source = regimes.pop("married")
    return Model(
        regimes={
            "source_a": source,
            "source_b": source,
            "source_c": source,
            **regimes,
        },
        ages=original.ages,
        regime_id_class=_ThreeSourceRegimeId,
        n_subjects=3,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ObservedGateTemplates:
    """Track the actual target-V arrays returned to gate preparation."""

    original: Callable[..., tuple[dict[str, object], dict[str, object]]]
    owners: list[weakref.ReferenceType[jax.Array]]
    peak: list[int]
    third_live: threading.Event

    def __call__(self, **kwargs: Any) -> tuple[dict[str, object], dict[str, object]]:
        arguments = self.original(**kwargs)
        values = arguments[1][SAME_PERIOD_V_ARG]
        assert isinstance(values, Mapping)
        target_value = values[kwargs["target_name"]]
        assert isinstance(target_value, jax.Array)
        self.owners.append(weakref.ref(target_value))
        live = sum(owner() is not None for owner in self.owners)
        self.peak[0] = max(self.peak[0], live)
        if live > 2:
            self.third_live.set()
        return arguments


@dataclasses.dataclass(frozen=True, kw_only=True)
class _BlockedGateCompiler:
    """Hold external compiler work in flight while the producer advances."""

    started: threading.Semaphore
    release: threading.Event

    def __call__(self, **kwargs: object) -> None:
        self.started.release()
        if not self.release.wait(timeout=15):
            raise RuntimeError("Controlled gate compiler was never released.")
        tuple(kwargs)


def _skip_core_prewarming(**kwargs: object) -> None:
    """Keep this owner-lifetime control focused on gate template preparation."""
    del kwargs


def test_slow_gate_compilers_bound_live_template_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two workers cannot keep every gate's concrete target value alive."""
    model = _three_gate_model()
    owners: list[weakref.ReferenceType[jax.Array]] = []
    peak = [0]
    started = threading.Semaphore(0)
    release = threading.Event()
    third_live = threading.Event()
    monkeypatch.setattr(
        compile_module,
        "_build_gate_evaluator_args",
        _ObservedGateTemplates(
            original=compile_module._build_gate_evaluator_args,
            owners=owners,
            peak=peak,
            third_live=third_live,
        ),
    )
    monkeypatch.setattr(compile_module, "_prepare_and_log", _skip_core_prewarming)
    monkeypatch.setattr(
        compile_module,
        "_compile_and_install_gate",
        _BlockedGateCompiler(started=started, release=release),
    )
    with ThreadPoolExecutor(max_workers=1) as runner:
        future = runner.submit(
            model._ensure_simulate_compiled,
            compile_batch_size=3,
            flat_params=model._process_params(get_params()),
            max_compilation_workers=2,
            log=logging.getLogger("gate-template-lifetime"),
        )
        try:
            for _ in range(2):
                if not started.acquire(timeout=10):
                    raise RuntimeError("Controlled gate compilers did not start.")
            third_live.wait(timeout=0.5)
        finally:
            release.set()
        future.result(timeout=15)

    assert len(owners) == 3
    assert peak[0] <= 2
    assert all(owner() is None for owner in owners)
