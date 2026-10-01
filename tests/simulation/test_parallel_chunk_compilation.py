"""Chunk planning lowers forward programs on the caller and compiles them on a pool."""

import re
import threading
from collections.abc import Hashable
from typing import Any

import jax
import jax.numpy as jnp
import pandas as pd
import pytest

import lcm.model as model_module
from _lcm.simulation.chunk_planning import SimulationChunkPlan
from tests.simulation.test_budget_lifecycle import (
    _LifecycleRegimeId,
    _stateful_target_model,
)

_PARAMS = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}
_DEBUG_SECTIONS = frozenset(
    {"FileNames", "FunctionNames", "FileLocations", "StackFrames"}
)
_WORKERS = (1, 2)


def _program_text(*, hlo: str) -> str:
    """Return compiled HLO without source-location metadata.

    The metadata names source files and wrapper addresses, which differ between
    two otherwise identical compilations.
    """
    blocks = (
        block
        for block in hlo.split("\n\n")
        if block.strip().split("\n", 1)[0] not in _DEBUG_SECTIONS
    )
    return re.sub(r", metadata=\{[^}]*\}", "", "\n\n".join(blocks))


def _readable_key(key: Hashable) -> tuple[object, ...]:
    """Project a runtime lowering key onto its argument names and specialization.

    The program identity is an object address, so it is dropped; the argument
    names, subject extent and axis widths identify each forward program of the
    small model.
    """
    _, arguments, specialization, *_ = key  # ty: ignore[not-iterable]
    return (tuple(name for name, _ in arguments), *specialization[:2])


def _simulate_and_observe(
    *, monkeypatch: pytest.MonkeyPatch, max_compilation_workers: int
) -> dict[str, Any]:
    """Simulate a fresh budgeted model, recording chunk planning's lowerings,
    compiles, published forward executables and plan."""
    model = _stateful_target_model()
    initial = {
        "wealth": jnp.asarray([1.0, 2.0, 3.0]),
        "age": jnp.zeros(3),
        "regime_id": jnp.full(3, _LifecycleRegimeId.alive, dtype=jnp.int32),
    }
    solution = model.solve(params=_PARAMS, log_level="off")
    caller = threading.get_ident()
    planning = [False]
    lock = threading.Lock()
    lowerings: list[bool] = []
    compiles: list[tuple[bool, str]] = []
    published: dict[Hashable, str] = {}
    plans: list[SimulationChunkPlan] = []
    lower_body = jax.stages.Traced.lower
    compile_body = jax.stages.Lowered.compile
    prepare = model_module.prepare_simulation_chunks

    def observe_lower(self: jax.stages.Traced, *args: Any, **kwargs: Any) -> Any:
        lowered = lower_body(self, *args, **kwargs)
        if planning[0]:
            with lock:
                lowerings.append(threading.get_ident() == caller)
        return lowered

    def observe_compile(self: jax.stages.Lowered, *args: Any, **kwargs: Any) -> Any:
        compiled = compile_body(self, *args, **kwargs)
        if planning[0]:
            with lock:
                compiles.append(
                    (
                        threading.get_ident() == caller,
                        _program_text(hlo=compiled.as_text() or ""),
                    )
                )
        return compiled

    def observe_prepare(**call: Any) -> Any:
        runtime = next(iter(call["regimes"].values())).simulation.programs.executor
        before = set(runtime.cache)
        planning[0] = True
        try:
            prepared = prepare(**call)
        finally:
            planning[0] = False
        published.update(
            {
                key: _program_text(hlo=entry.executable.as_text() or "")
                for key, entry in runtime.cache.items()
                if key not in before
            }
        )
        plans.append(prepared.plan)
        return prepared

    with monkeypatch.context() as patch:
        patch.setattr(jax.stages.Traced, "lower", observe_lower)
        patch.setattr(jax.stages.Lowered, "compile", observe_compile)
        patch.setattr(model_module, "prepare_simulation_chunks", observe_prepare)
        result = model.simulate(
            params=_PARAMS,
            initial_conditions=initial,
            solution=solution,
            seed=17,
            log_level="off",
            max_compilation_workers=max_compilation_workers,
        )
    (plan,) = plans
    profile = plan.profile
    contract = (
        profile.n_subjects,
        profile.padded_population,
        tuple(
            (stage.name, stage.peak_bytes, stage.reservation_bytes)
            for stage in profile.stages
        ),
        dict(profile.fixed_reservation),
        dict(profile.output_reservation),
        dict(profile.setup_reservation),
        dict(plan.required_bytes),
    )
    return {
        "lowerings": lowerings,
        "compiles": compiles,
        "published": published,
        "contract": contract,
        "panel": result.to_dataframe(),
    }


@pytest.fixture(scope="module")
def observed() -> dict[int, dict[str, Any]]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        return {
            workers: _simulate_and_observe(
                monkeypatch=monkeypatch, max_compilation_workers=workers
            )
            for workers in _WORKERS
        }


@pytest.mark.parametrize("workers", _WORKERS)
def test_chunk_planning_lowers_every_program_on_the_calling_thread(
    *, observed: dict[int, dict[str, Any]], workers: int
) -> None:
    lowerings = observed[workers]["lowerings"]
    assert (len(lowerings) > 0, all(lowerings)) == (True, True)


@pytest.mark.parametrize("workers", _WORKERS)
def test_chunk_planning_compiles_every_forward_program_off_the_calling_thread(
    *, observed: dict[int, dict[str, Any]], workers: int
) -> None:
    pooled = {
        text for on_caller, text in observed[workers]["compiles"] if not on_caller
    }
    assert set(observed[workers]["published"].values()) - pooled == set()


@pytest.mark.parametrize("workers", _WORKERS)
def test_chunk_planning_publishes_one_executable_per_forward_program(
    *, observed: dict[int, dict[str, Any]], workers: int
) -> None:
    """The small model's four forward programs, each at three subjects."""
    step = ("age", "koopmans_aggregator__discount_factor", "period", "saving", "wealth")
    keys = sorted(map(_readable_key, observed[workers]["published"]), key=repr)
    assert keys == [
        # The alive decision, over its two-point action product.
        (
            (
                "age",
                "koopmans_aggregator__discount_factor",
                "next_regime_to_V_arr",
                "period",
                "saving",
                "wealth",
            ),
            3,
            (("action_product", 2), ("subject", 3)),
        ),
        # The alive transition and route, which read the same operands.
        (step, 3, (("subject", 3),)),
        (step, 3, (("subject", 3),)),
        # The terminal done decision.
        (("age", "next_regime_to_V_arr", "period", "wealth"), 3, (("subject", 3),)),
    ]


def test_chunk_planning_compiles_the_same_forward_programs_for_every_worker_count(
    observed: dict[int, dict[str, Any]],
) -> None:
    def programs(workers: int) -> list[str]:
        return sorted(observed[workers]["published"].values())

    assert (all("ENTRY" in text for text in programs(1)), programs(2)) == (
        True,
        programs(1),
    )


def test_chunk_planning_admits_the_same_chunk_contract_for_every_worker_count(
    observed: dict[int, dict[str, Any]],
) -> None:
    assert observed[2]["contract"] == observed[1]["contract"]


def test_chunk_planning_simulates_the_same_panel_for_every_worker_count(
    observed: dict[int, dict[str, Any]],
) -> None:
    pd.testing.assert_frame_equal(
        observed[2]["panel"], observed[1]["panel"], check_exact=True
    )
