"""A compilation wave hands each lowered program to the compile pool at once."""

import logging
import threading
from collections.abc import Hashable
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    ResolvedCoreProgram,
)
from _lcm.execution.output_layout import (
    VALUE,
    ResolvedOutputLayout,
    resolve_output_layout,
)
from _lcm.solution import backward_induction
from lcm import AgeGrid

_KEYS = ("first", "second", "third")


def _program(*, wealth: jax.Array, scale: int) -> jax.Array:
    """A distinct small value program per scale."""
    return jnp.cumsum(wealth * scale)


def _layout(*, core_key: str, wealth: jax.Array) -> ResolvedOutputLayout:
    """The value-only output layout the wave lowers each program against."""
    return resolve_output_layout(
        core_key=core_key,
        value_template=wealth,
        state_order=("wealth",),
        output_roles=VALUE,
    )


def _run_wave(*, n_workers: int) -> dict[Hashable, jax.stages.Compiled]:
    """Lower and compile one wave of three distinct programs."""
    wealth = jnp.linspace(0.0, 1.0, 8)
    candidates: dict[Hashable, backward_induction._CoreCandidate] = {
        key: ((key, 0, key), ()) for key in _KEYS
    }
    resolved = {
        candidate: ResolvedCoreProgram(
            name=candidate[0][2],
            function=_program,
            arguments={"wealth": wealth},
            static_kwargs={"scale": i + 1},
            requirements=CoreExecutionRequirements(),
            output_roles=VALUE,
            disposition=CoreExecutionDisposition.PLANNED,
            donation_candidates=(),
            tile_widths={},
            specialization_key=(),
            input_transfer_plan=(),
        )
        for i, (key, candidate) in enumerate(candidates.items())
    }
    compiled: dict[Hashable, jax.stages.Compiled] = {}
    backward_induction._lower_and_compile_wave(
        new_lowerings=candidates,
        resolved_programs=resolved,
        all_layouts={
            triple: _layout(core_key=triple[2], wealth=wealth)
            for triple, _ in candidates.values()
        },
        internal_templates={candidate: {} for candidate in candidates.values()},
        donations=dict.fromkeys(candidates.values(), ()),
        ages=AgeGrid(start=0, stop=1, step="Y"),
        n_triples_per_lowering=dict.fromkeys(_KEYS, 1),
        log_kernel_memory=False,
        n_workers=n_workers,
        logger=logging.getLogger("pipelined-lowering-test"),
        compiled=compiled,
        labels={},
    )
    return compiled


def test_lower_and_compile_wave_compiles_a_program_before_the_next_is_lowered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compilation of the first program starts while the second is being lowered.

    The second lowering waits for the first compile to begin; a wave that lowers
    every program before compiling any records the second lowering first.
    """
    events: list[str] = []
    first_compile_started = threading.Event()
    compile_and_log = backward_induction._compile_and_log
    assert_roles = backward_induction._assert_lowered_output_roles

    def observed_compile(**kwargs: Any) -> Any:
        events.append(f"compile {kwargs['lowering_key']}")
        first_compile_started.set()
        return compile_and_log(**kwargs)

    def observed_roles(**kwargs: Any) -> None:
        assert_roles(**kwargs)
        n_lowered = sum(event.startswith("lowered") for event in events)
        if n_lowered == 1:
            first_compile_started.wait(timeout=5.0)
        events.append(f"lowered {_KEYS[n_lowered]}")

    monkeypatch.setattr(backward_induction, "_compile_and_log", observed_compile)
    monkeypatch.setattr(
        backward_induction, "_assert_lowered_output_roles", observed_roles
    )
    _run_wave(n_workers=2)
    assert events.index("compile first") < events.index("lowered second")


def test_lower_and_compile_wave_executables_match_direct_compilation() -> None:
    """Each executable's compiled text equals a direct `jit.lower.compile`."""
    wealth = jnp.linspace(0.0, 1.0, 8)
    compiled = _run_wave(n_workers=2)
    direct = {
        key: jax.jit(
            _program,
            static_argnames=("scale",),
            out_shardings=_layout(core_key=key, wealth=wealth).out_shardings,
        )
        .lower(wealth=wealth, scale=i + 1)
        .compile()
        .as_text()
        for i, key in enumerate(_KEYS)
    }
    assert {key: exe.as_text() for key, exe in compiled.items()} == direct


def test_lower_and_compile_wave_raises_a_lowering_error_after_earlier_compiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lowering failure propagates even while earlier programs are compiling."""
    assert_roles = backward_induction._assert_lowered_output_roles
    n_calls = 0

    def failing_roles(**kwargs: Any) -> None:
        nonlocal n_calls
        n_calls += 1
        if n_calls == 3:
            raise RuntimeError("third lowering failed")
        assert_roles(**kwargs)

    monkeypatch.setattr(
        backward_induction, "_assert_lowered_output_roles", failing_roles
    )
    with pytest.raises(RuntimeError, match="third lowering failed"):
        _run_wave(n_workers=2)
