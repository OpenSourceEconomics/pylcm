"""Warm GridSearch solves reuse an immutable structural blueprint.

A model keeps, for each validated abstract schema of a solve's inputs, the
blueprint its programs were resolved into: abstract argument trees, value-read
plans, output layouts and ranked width frontiers. A warm solve with the same
schema binds that blueprint to the current parameters instead of materializing,
preparing and resolving every program again. The liveness ledger, the donation
decisions, the lowering keys, the candidate frontier and admission are built
afresh on every call, so a changed budget changes the admitted widths and every
result equals the one a fresh model returns.
"""

import copy
import dataclasses
import gc
import logging
import math
import weakref
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType
from typing import NotRequired, TypedDict, Unpack

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.workspace_planning import CompilerMemoryReservation
from _lcm.simulation.entry_allocations import SimulationEntryAllocations
from _lcm.solution import backward_induction as bi
from _lcm.typing import ArgumentTree, FlatParams
from lcm import ExecutionConfig, Model
from lcm.solver_api import SolutionResult
from lcm.typing import UserParams
from tests.execution.test_compiler_allocation_reservation import synthetic_memory
from tests.solution._candidate_census import _CompilationKwargs
from tests.solution.test_invariant_blocking import _workload

_WORKLOADS = ("independent_types", "sector_typed_terminal", "sector_type_free_terminal")
_EXPENSIVE_BUILDERS = (
    "materialize_core_program",
    "_prepare_abstract_program",
    "resolve_core_program_candidates",
)


type _PlanSummary = tuple[
    dict[str, int],
    tuple[bi.ResolvedValueTransfer, ...],
    tuple[str, ...],
    bi.ResolvedOutputLayout,
    dict[bi.ReferenceName, bi.ShapeDtypePytree],
]


class _ProcessKwargs(TypedDict):
    array_writer: NotRequired[SimulationEntryAllocations | None]


def _config(*, blocked: bool, budget: int | None = 2**30) -> ExecutionConfig:
    return ExecutionConfig(
        devices=(0,),
        device_memory_bytes=budget,
        invariant_block_widths={"pref_type": 1} if blocked else {},
    )


def _solve(*, model: Model, params: UserParams) -> SolutionResult:
    result = model.solve(params=params, log_level="off")
    for values in result.values.values():
        jax.block_until_ready(tuple(values.values()))
    return result


def _count_builder[**Parameters, Result](
    *, original: Callable[Parameters, Result], name: str, calls: dict[str, int]
) -> Callable[Parameters, Result]:
    def observe(*args: Parameters.args, **kwargs: Parameters.kwargs) -> Result:
        calls[name] += 1
        return original(*args, **kwargs)

    return observe


@contextmanager
def _counted_builders(monkeypatch: pytest.MonkeyPatch) -> Generator[dict[str, int]]:
    """Count calls of the builders a blueprint hit skips."""
    calls = dict.fromkeys(_EXPENSIVE_BUILDERS, 0)
    with monkeypatch.context() as patch:
        observers = {
            "materialize_core_program": _count_builder(
                original=bi.materialize_core_program,
                name="materialize_core_program",
                calls=calls,
            ),
            "_prepare_abstract_program": _count_builder(
                original=bi._prepare_abstract_program,
                name="_prepare_abstract_program",
                calls=calls,
            ),
            "resolve_core_program_candidates": _count_builder(
                original=bi.resolve_core_program_candidates,
                name="resolve_core_program_candidates",
                calls=calls,
            ),
        }
        for name, observer in observers.items():
            patch.setattr(bi, name, observer)
        yield calls


@contextmanager
def _captured_plans(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[list[bi._CompiledPrograms]]:
    """Capture the compiled-program plan of every solve inside the block."""
    plans: list[bi._CompiledPrograms] = []
    original = bi._compile_all_functions
    with monkeypatch.context() as patch:

        def capture(**kwargs: Unpack[_CompilationKwargs]) -> bi._CompiledPrograms:
            compiled = original(**kwargs)
            plans.append(compiled)
            return compiled

        patch.setattr(bi, "_compile_all_functions", capture)
        yield plans


def _plan_summary(compiled: bi._CompiledPrograms) -> dict[bi._CoreTriple, _PlanSummary]:
    """Describe every selected core by what its executable was planned with."""
    return {
        (regime, period, name): (
            dict(core.tile_widths),
            core.input_transfer_plan,
            core.donated_arguments,
            core.layout,
            dict(core.internal_input_templates),
        )
        for (regime, period), cores in compiled.executables.items()
        for name, core in cores.items()
    }


def _admitted_widths(
    compiled: bi._CompiledPrograms,
) -> dict[bi._CoreTriple, dict[str, int]]:
    return {
        (regime, period, name): dict(core.tile_widths)
        for (regime, period), cores in compiled.executables.items()
        for name, core in cores.items()
    }


def _assert_values_identical(*, first: SolutionResult, second: SolutionResult) -> None:
    assert set(first.values) == set(second.values)
    for period, by_regime in first.values.items():
        assert set(by_regime) == set(second.values[period])
        for regime, value in by_regime.items():
            left = np.asarray(value)
            right = np.asarray(second.values[period][regime])
            assert (left.shape, left.dtype) == (right.shape, right.dtype)
            assert left.tobytes() == right.tobytes()


def _warm(*, model: Model, params: UserParams, n: int = 2) -> None:
    for _ in range(n):
        warmed = _solve(model=model, params=params)
        del warmed


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("workload", _WORKLOADS)
def test_same_schema_warm_calls_rebind_values_without_rebuilding_structure(
    *, blocked: bool, workload: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same-schema warm calls skip the builders and still equal a fresh model."""
    config = _config(blocked=blocked)
    model, params = _workload(name=workload, execution_config=config)
    _warm(model=model, params=params, n=3)
    changed = copy.deepcopy(params)
    working = changed["working"]
    assert isinstance(working, dict)
    utility = working["utility"]
    assert isinstance(utility, dict)
    weight = utility["weight"]
    assert isinstance(weight, jax.Array)
    utility["weight"] = weight * 1.125
    with _counted_builders(monkeypatch) as calls:
        same = _solve(model=model, params=params)
        different = _solve(model=model, params=changed)
    assert calls == dict.fromkeys(calls, 0), calls
    fresh_same, _ = _workload(name=workload, execution_config=config)
    fresh_changed, _ = _workload(name=workload, execution_config=config)
    _assert_values_identical(first=same, second=_solve(model=fresh_same, params=params))
    _assert_values_identical(
        first=different, second=_solve(model=fresh_changed, params=changed)
    )
    assert any(
        np.asarray(value).tobytes() != np.asarray(different.values[t][r]).tobytes()
        for t, row in same.values.items()
        for r, value in row.items()
    ), "The changed-parameter control must actually change at least one value."


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("workload", _WORKLOADS)
def test_a_blueprint_hit_plans_every_core_as_a_fresh_model_does(
    *, blocked: bool, workload: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Widths, read plans, codes, donations and layouts equal a fresh model's."""
    config = _config(blocked=blocked)
    model, params = _workload(name=workload, execution_config=config)
    _warm(model=model, params=params)
    fresh, _ = _workload(name=workload, execution_config=config)
    with _captured_plans(monkeypatch) as plans:
        _solve(model=model, params=params)
        _solve(model=fresh, params=params)
    hit, cold = (_plan_summary(compiled) for compiled in plans)
    assert hit == cold


@pytest.mark.parametrize("blocked", [False, True])
def test_user_typing_and_placement_normalise_to_the_cached_schema(
    *, blocked: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Typing and placement that parameter processing erases reuse the blueprint.

    Every float parameter is cast to a strongly typed, uncommitted array of the
    canonical float dtype before the structural key is taken, so a strongly
    typed scalar, a narrower float or a committed array solves with the cached
    blueprint and equals a fresh model.
    """
    config = _config(blocked=blocked)
    model, params = _workload(name="independent_types", execution_config=config)
    _warm(model=model, params=params)
    variant = copy.deepcopy(params)
    discount_factor = params["discount_factor"]
    assert isinstance(discount_factor, float)
    working = variant["working"]
    assert isinstance(working, dict)
    utility = working["utility"]
    assert isinstance(utility, dict)
    weight = utility["weight"]
    assert isinstance(weight, jax.Array)
    variant["discount_factor"] = jnp.asarray(discount_factor, dtype=jnp.float16)
    utility["weight"] = jax.device_put(weight, jax.devices()[0])
    with _counted_builders(monkeypatch) as calls:
        got = _solve(model=model, params=variant)
    assert calls == dict.fromkeys(calls, 0), calls
    fresh, _ = _workload(name="independent_types", execution_config=config)
    _assert_values_identical(first=got, second=_solve(model=fresh, params=variant))


@pytest.mark.parametrize("blocked", [False, True])
def test_a_weak_typing_change_rebuilds_the_structure(
    *, blocked: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A weakly typed processed scalar where a strong one was cached is a miss."""
    config = _config(blocked=blocked)
    model, params = _workload(name="independent_types", execution_config=config)
    _warm(model=model, params=params)
    fresh, _ = _workload(name="independent_types", execution_config=config)
    for target in (model, fresh):
        _respecify_processed_params(
            model=target, monkeypatch=monkeypatch, weak=True, committed=False
        )
    with _counted_builders(monkeypatch) as calls:
        got = _solve(model=model, params=params)
    assert calls["materialize_core_program"] > 0, calls
    _assert_values_identical(first=got, second=_solve(model=fresh, params=params))


@pytest.mark.parametrize("blocked", [False, True])
def test_a_layout_change_rebuilds_the_structure(
    *, blocked: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Processed arrays committed to a device where uncommitted ones were is a miss."""
    config = _config(blocked=blocked)
    model, params = _workload(name="independent_types", execution_config=config)
    _warm(model=model, params=params)
    fresh, _ = _workload(name="independent_types", execution_config=config)
    for target in (model, fresh):
        _respecify_processed_params(
            model=target, monkeypatch=monkeypatch, weak=False, committed=True
        )
    with _counted_builders(monkeypatch) as calls:
        got = _solve(model=model, params=params)
    assert calls["materialize_core_program"] > 0, calls
    _assert_values_identical(first=got, second=_solve(model=fresh, params=params))


def _respecify_processed_params(
    *, model: Model, monkeypatch: pytest.MonkeyPatch, weak: bool, committed: bool
) -> None:
    """Change the abstract schema of `model`'s processed parameters, not their values.

    With `weak`, every scalar leaf becomes a weakly typed array of the same
    dtype; with `committed`, every array leaf is committed to the first device.
    Leaves shared between slots stay shared.
    """
    process = model._process_params

    def respecify(leaf: ArgumentTree) -> jax.Array:
        assert isinstance(leaf, jax.Array)
        if weak and leaf.ndim == 0:
            leaf = jnp.asarray(leaf.item())
        if committed and leaf.ndim > 0:
            leaf = jax.device_put(leaf, jax.devices()[0])
        return leaf

    def processed(params: UserParams, **kwargs: Unpack[_ProcessKwargs]) -> FlatParams:
        flat_params = process(params, **kwargs)
        memo: dict[int, jax.Array] = {}
        respecified = MappingProxyType(
            {
                regime: MappingProxyType(
                    {
                        name: memo.setdefault(id(leaf), respecify(leaf))
                        for name, leaf in leaves.items()
                    }
                )
                for regime, leaves in flat_params.items()
            }
        )
        leaves = jax.tree.leaves(respecified)
        assert any(leaf.weak_type for leaf in leaves) is weak
        assert any(leaf.committed for leaf in leaves) is committed
        return respecified

    monkeypatch.setattr(model, "_process_params", processed)


@pytest.mark.parametrize("blocked", [False, True])
def test_a_budget_change_rebinds_and_readmits_without_rebuilding(
    *, blocked: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A narrower budget between warm calls changes the admitted widths."""
    monkeypatch.setattr(bi, "compiler_memory_reservation", _width_proportional_peak)
    model, params = _workload(
        name="independent_types", execution_config=_config(blocked=blocked)
    )
    _warm(model=model, params=params)
    narrow = 25_000
    with _captured_plans(monkeypatch) as plans, _counted_builders(monkeypatch) as calls:
        _solve(model=model, params=params)
        model._execution = dataclasses.replace(
            model._execution, device_memory_bytes=narrow
        )
        narrowed = _solve(model=model, params=params)
    assert calls["materialize_core_program"] == 0, calls
    assert calls["_prepare_abstract_program"] == 0, calls
    wide_widths, narrow_widths = (_admitted_widths(compiled) for compiled in plans)
    assert wide_widths != narrow_widths
    fresh, _ = _workload(
        name="independent_types",
        execution_config=_config(blocked=blocked, budget=narrow),
    )
    with _captured_plans(monkeypatch) as fresh_plans:
        fresh_narrowed = _solve(model=fresh, params=params)
    assert narrow_widths == _admitted_widths(fresh_plans[0])
    _assert_values_identical(first=narrowed, second=fresh_narrowed)


def _width_proportional_peak(
    *, compiled: jax.stages.Compiled, widths: Mapping[str, int]
) -> CompilerMemoryReservation:
    """Report a compiler peak of one kilobyte per streamed width-product point."""
    del compiled
    return synthetic_memory(1_000 * math.prod(widths.values()))


def test_removing_the_budget_rebuilds_the_structure(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unbudgeted solve ranks one width per core, so it is a different recipe."""
    model, params = _workload(
        name="independent_types", execution_config=_config(blocked=True)
    )
    _warm(model=model, params=params)
    model._execution = dataclasses.replace(model._execution, device_memory_bytes=None)
    with _counted_builders(monkeypatch) as calls:
        _solve(model=model, params=params)
    assert calls["materialize_core_program"] > 0, calls


def test_the_cache_is_bounded_and_released_with_its_model(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Distinct schemas never grow the cache past its bound or past the model."""
    model, params = _workload(
        name="independent_types", execution_config=_config(blocked=True)
    )
    cache = model._structural_blueprints
    working = params["working"]
    assert isinstance(working, dict)
    utility = working["utility"]
    assert isinstance(utility, dict)
    weight = utility["weight"]
    assert isinstance(weight, jax.Array)
    for dtype in (None, jnp.float32, jnp.float16):
        for committed in (False, True):
            variant = copy.deepcopy(params)
            if dtype is not None:
                variant["discount_factor"] = jnp.asarray(0.9, dtype=dtype)
            if committed:
                variant_working = variant["working"]
                assert isinstance(variant_working, dict)
                variant_utility = variant_working["utility"]
                assert isinstance(variant_utility, dict)
                variant_utility["weight"] = jax.device_put(weight, jax.devices()[0])
            result = _solve(model=model, params=variant)
            del result
    assert (len(cache), cache.misses) == (1, 1)
    for budget in (2**30, None):
        model._execution = dataclasses.replace(
            model._execution, device_memory_bytes=budget
        )
        for weak in (False, True):
            for committed in (False, True):
                with monkeypatch.context() as patch:
                    _respecify_processed_params(
                        model=model, monkeypatch=patch, weak=weak, committed=committed
                    )
                    result = _solve(model=model, params=params)
                    del result
    assert cache.misses == 8
    assert len(cache) == cache.max_entries
    reference = weakref.ref(cache)
    del cache, model
    gc.collect()
    assert reference() is None


def test_cached_blueprints_retain_no_concrete_array() -> None:
    """A blueprint holds abstract trees only, never a caller's or a solve's array."""
    model, params = _workload(
        name="independent_types", execution_config=_config(blocked=True)
    )
    _warm(model=model, params=params)
    entries = list(model._structural_blueprints.values())
    assert entries
    concrete = [leaf for leaf in _walk(value=entries) if isinstance(leaf, jax.Array)]
    assert concrete == []


def _walk(
    *,
    value: object,  # noqa: PAN001 - Object graph inspection includes arbitrary dataclass fields.
    seen: set[int] | None = None,
) -> Iterator[object]:  # noqa: PAN001 - Yields arbitrary reachable graph members for array detection.
    """Yield every object reachable through containers and dataclass fields."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    yield value
    if isinstance(value, Mapping | MappingProxyType):
        for key, child in value.items():
            yield from _walk(value=key, seen=seen)
            yield from _walk(value=child, seen=seen)
    elif isinstance(value, tuple | list | set | frozenset):
        for child in value:
            yield from _walk(value=child, seen=seen)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _walk(value=getattr(value, field.name), seen=seen)


def test_pickling_a_model_drops_its_blueprints() -> None:
    """The cache is runtime state: a copy starts empty, solves, and fills its own."""
    model, params = _workload(
        name="independent_types", execution_config=_config(blocked=False)
    )
    first = _solve(model=model, params=params)
    assert len(model._structural_blueprints) > 0
    assert "_structural_blueprints" not in model.__getstate__()
    restored = cloudpickle.loads(cloudpickle.dumps(model))
    assert len(restored._structural_blueprints) == 0
    _assert_values_identical(first=first, second=_solve(model=restored, params=params))
    assert len(restored._structural_blueprints) == 1


def test_a_blueprint_holding_other_program_objects_is_rebuilt(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stored blueprint whose programs are not the keyed objects is not bound.

    The key names programs by identity, so an entry whose `programs` are equal
    copies rather than the very objects of the current graph is treated as a
    miss: the structure is resolved again and the entry is replaced.
    """
    model, params = _workload(
        name="independent_types", execution_config=_config(blocked=True)
    )
    _warm(model=model, params=params)
    cache = model._structural_blueprints
    ((key, stored),) = cache._entries.items()
    copies = tuple(copy.copy(program) for program in stored.programs)
    cache.put(key=key, blueprint=dataclasses.replace(stored, programs=copies))
    with _counted_builders(monkeypatch) as calls:
        _solve(model=model, params=params)
    assert calls["materialize_core_program"] > 0, calls
    assert not any(
        new is old
        for new, old in zip(cache._entries[key].programs, copies, strict=True)
    )


def _logged_outcomes(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the lookup outcome of every structural-blueprint debug line."""
    prefix = "structural blueprint: "
    return [
        record.getMessage().removeprefix(prefix).split(" ")[0]
        for record in caplog.records
        if record.getMessage().startswith(prefix)
    ]


def test_the_debug_log_reports_a_miss_then_a_hit(
    *, caplog: pytest.LogCaptureFixture
) -> None:
    """A cold solve logs `miss` and the same-schema warm solve logs `hit`."""
    model, params = _workload(
        name="independent_types", execution_config=_config(blocked=True)
    )
    with caplog.at_level(logging.DEBUG, logger="lcm"):
        for _ in range(2):
            result = model.solve(params=params, log_level="debug")
            del result
    assert _logged_outcomes(caplog) == ["miss", "hit"]


def test_the_debug_log_reports_an_uncertified_graph_as_uncached(
    *, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A solve whose programs yield no structural key logs `uncached`."""
    monkeypatch.setattr(bi, "_structural_key", lambda **_: None)
    model, params = _workload(
        name="independent_types", execution_config=_config(blocked=True)
    )
    with caplog.at_level(logging.DEBUG, logger="lcm"):
        result = model.solve(params=params, log_level="debug")
        del result
    assert _logged_outcomes(caplog) == ["uncached"]
