"""One immutable canonical program generation, retained by its real consumers."""

import dataclasses
import functools
import threading
import types
import weakref
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax

from _lcm.grids import DiscreteGrid
from _lcm.materialization import DatedGenerationMaterializer
from _lcm.params.edges import flat_namespaces
from _lcm.processes.grid_resolution import ProcessGridResolver
from _lcm.solution.fingerprint import (
    _param_shape_signature,
    _Reachable,
    _semantic_fingerprint,
)
from _lcm.typing import FlatParams, PytreeValue
from lcm._solver_api.authority import ArtifactAuthority, _ArrayCopier
from lcm.exceptions import UnsupportedOperationError
from lcm.typing import RegimeName

if TYPE_CHECKING:
    from _lcm.continuation import ProjectedContinuationSpec
    from _lcm.current_iid import CurrentIIDPlan
    from _lcm.egm.step import _EGMStep
    from _lcm.solution.contract import PeriodKernel, SolverBuildContext
    from _lcm.solution.dcegm import EGMStepBuild, _BoundDCEGM
    from _lcm.solution.model_authority import SolutionAuthority
    from lcm.model import Model
else:
    # Model imports this private record. Runtime validation checks the actual
    # producer through its seal rather than importing the public class in a cycle.
    type Model = object
    type SolutionAuthority = object
    type ProjectedContinuationSpec = object
    type CurrentIIDPlan = object
    type _EGMStep = object
    type SolverBuildContext = object
    type PeriodKernel = object
    type EGMStepBuild = object
    type _BoundDCEGM = object


def canonical_parameter_schema(*, flat_params: FlatParams) -> str:
    """Walk real namespaces before applying the existing leaf-only signature."""
    return _semantic_fingerprint(
        (
            "pylcm-canonical-parameter-schema-v1",
            tuple(
                (
                    path,
                    tuple(
                        (name, type(value).__qualname__, _param_shape_signature(value))
                        for name, value in sorted(parameters.items())
                    ),
                )
                for path, parameters in sorted(flat_namespaces(flat_params))
            ),
        )
    )


def _binding_children(value: _Reachable) -> tuple[_Reachable, ...]:
    """Read only actual canonical record and Python callable binding containers."""
    if type(value) is ArtifactAuthority:
        from _lcm.solution.retained_buffers import (  # noqa: PLC0415 - ownership imports this generation record
            retained_artifact_authority_buffers,
        )

        value = retained_artifact_authority_buffers(authority=value)
    if isinstance(value, DiscreteGrid):
        value = value.materialized_array_roots
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return tuple(getattr(value, item.name) for item in dataclasses.fields(value))
    if isinstance(value, Mapping):
        return tuple(value.values())
    if isinstance(value, tuple | list):
        return tuple(value)
    if isinstance(value, functools.partial):
        return (value.func, value.args, value.keywords)
    if isinstance(value, types.FunctionType):
        children: list[_Reachable] = [value.__defaults__, value.__kwdefaults__]
        for cell in value.__closure__ or ():
            try:
                children.append(cell.cell_contents)
            except ValueError:
                continue
        return tuple(children)
    return ()


def canonical_generation_roots(
    *, declarations: tuple[_Reachable, ...]
) -> tuple[jax.Array, ...]:
    """Expose actual buffers in canonical fields and Python binding containers.

    Executable workspace remains separately admitted by the compiler path.
    No Model/cache/lock is an input. Identity visitation terminates shared
    callable/date/program references without copying a buffer.
    """
    seen: set[int] = set()
    arrays: list[jax.Array] = []
    pending: list[_Reachable] = [declarations]
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, jax.Array):
            arrays.append(value)
        else:
            pending.extend(_binding_children(value))
    return tuple(arrays)


@dataclass(frozen=True, kw_only=True, eq=False)
class GenerationSolverPreparation:
    """Original numerical pieces retained by one real installed Model generation."""

    generation: CanonicalModelGeneration
    regime_name: RegimeName
    period: int
    context: SolverBuildContext
    build: EGMStepBuild
    step: _EGMStep
    plan: CurrentIIDPlan | None
    declarations: Mapping[RegimeName, ProjectedContinuationSpec]
    installed_kernel: PeriodKernel
    solver: _BoundDCEGM

    def validate_owner(self) -> None:
        """Refuse a substituted context/step/plan instead of accepting a token."""
        actual = self.generation.solver_preparation(
            regime_name=self.regime_name, period=self.period
        )
        for name in (
            "context",
            "build",
            "step",
            "plan",
            "declarations",
            "installed_kernel",
            "solver",
        ):
            if getattr(self, name) is not getattr(actual, name):
                raise TypeError(
                    f"Canonical generation solver preparation substituted {name!r}."
                )

    @property
    def code_roots(self) -> tuple[jax.Array, ...]:
        """Borrow actual source/future grids and plans, without claiming admission."""
        self.validate_owner()
        return canonical_generation_roots(
            declarations=(
                self.context,
                self.build,
                self.step,
                self.plan,
                self.declarations,
            )
        )


type _GenerationState = tuple[
    str, str, Model, tuple[PytreeValue, ...], list[SolutionAuthority]
]


@dataclass(frozen=True, kw_only=True, eq=False, weakref_slot=True, slots=True)
class CanonicalModelGeneration:
    """The same genuine execution Model owns programs, seal and authority."""

    key: str
    declaration_fingerprint: str
    execution: Model
    roots: tuple[PytreeValue, ...]
    _authority: list[SolutionAuthority] = field(default_factory=list, repr=False)
    _authority_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __getstate__(self) -> _GenerationState:
        """Pickle every field but the lock, which belongs to one process."""
        return (
            self.key,
            self.declaration_fingerprint,
            self.execution,
            self.roots,
            self._authority,
        )

    def __setstate__(self, state: _GenerationState) -> None:
        """Restore the fields and give the restored generation its own lock."""
        key, declaration_fingerprint, execution, roots, authority = state
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "declaration_fingerprint", declaration_fingerprint)
        object.__setattr__(self, "execution", execution)
        object.__setattr__(self, "roots", roots)
        object.__setattr__(self, "_authority", authority)
        object.__setattr__(self, "_authority_lock", threading.Lock())

    def live_roots(self) -> tuple[PytreeValue, ...]:
        """Borrow original roots and already-published authority template storage."""
        from _lcm.solution.retained_buffers import (  # noqa: PLC0415 - ownership imports this generation record
            retained_artifact_authority_buffers,
        )

        # Completed authorities are append-only. Do not take the authority lock:
        # its producer may observe live roots while already holding that lock.
        authorities = tuple(self._authority)
        return (
            *self.roots,
            *(
                retained_artifact_authority_buffers(authority=artifact)
                for authority in authorities
                for artifact in authority.artifacts.values()
            ),
        )

    def solver_preparation(
        self, *, regime_name: RegimeName, period: int
    ) -> GenerationSolverPreparation:
        """Acquire genuine dated builder objects, never archive-supplied programs."""
        from _lcm.egm.step import _EGMStep  # noqa: PLC0415 - Model import closure

        self.execution._check_identity_runtime()  # noqa: SLF001 - actual canonical generation handoff
        prepared = self.execution._dated_generation  # noqa: SLF001 - actual canonical generation handoff
        if prepared is None:
            raise UnsupportedOperationError(
                "This generation has no genuine retained dated solver preparation."
            )
        numerical = prepared.preparation.numerical
        if regime_name not in numerical.steps:
            raise UnsupportedOperationError(
                f"Generation source {regime_name!r} has no genuine DCEGM preparation."
            )
        build = numerical.steps[regime_name]
        if type(period) is not int or period not in build.steps:
            raise UnsupportedOperationError(
                f"Generation source {regime_name!r} has no original date {period!r}."
            )
        context = numerical.contexts[regime_name]
        step = build.steps[period]
        if not isinstance(step, _EGMStep):
            raise UnsupportedOperationError(
                "Generation lacks original scalar Euler pieces."
            )
        kernel = self.execution._regimes[regime_name].solution.period_kernels[period]  # noqa: SLF001 - actual canonical generation handoff
        original = prepared.regimes[regime_name].solution.period_kernels[period]
        if kernel is not original or context.regime_name != regime_name:
            raise TypeError(
                "Installed generation substituted its original dated kernel."
            )
        return GenerationSolverPreparation(
            generation=self,
            regime_name=regime_name,
            period=period,
            context=context,
            build=build,
            step=step,
            plan=context.current_iid_plans.get(period),
            declarations=context.projected_continuation_declarations,
            installed_kernel=kernel,
            solver=numerical.solvers[regime_name],
        )

    def declared_authority(
        self,
        *,
        flat_params: FlatParams,
        process_grid_resolver: ProcessGridResolver | None,
        materializer: DatedGenerationMaterializer | None = None,
        array_copier: _ArrayCopier | None = None,
    ) -> SolutionAuthority:
        """Derive authority at its original validation frontier, single-flight."""
        from _lcm.solution.model_authority import (  # noqa: PLC0415 - authority imports artifacts
            build_solution_authority,
        )

        with self._authority_lock:
            if not self._authority:
                self._authority.append(
                    build_solution_authority(
                        regimes=self.execution._regimes,  # noqa: SLF001 - canonical generation owns this private lifecycle handoff
                        flat_params=flat_params,
                        ages=self.execution._time,  # noqa: SLF001 - canonical generation owns this private lifecycle handoff
                        process_grid_resolver=process_grid_resolver,
                        materializer=materializer,
                        array_copier=array_copier,
                    )
                )
            return self._authority[0]


@dataclass(kw_only=True)
class CanonicalGenerationCache:
    """Single-flight publication plus weak custody of evicted retained roots.

    The lock covers the factory, not just insertion. A result retaining its
    producing generation keeps it in live even after LRU eviction.
    """

    capacity: int
    _cached: OrderedDict[str, CanonicalModelGeneration] = field(
        default_factory=OrderedDict, init=False
    )
    _live: weakref.WeakValueDictionary[str, CanonicalModelGeneration] = field(
        default_factory=weakref.WeakValueDictionary, init=False
    )
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def acquire(
        self, *, key: str, factory: Callable[[], CanonicalModelGeneration]
    ) -> CanonicalModelGeneration:
        """Publish only a completely constructed generation; failures stay absent."""
        with self._lock:
            generation = self._cached.get(key) or self._live.get(key)
            if generation is None:
                generation = factory()
                if generation.key != key:
                    raise RuntimeError("Canonical generation factory changed its key.")
                self._live[key] = generation
            self._cached[key] = generation
            self._cached.move_to_end(key)
            while len(self._cached) > self.capacity:
                self._cached.popitem(last=False)
            # The source seal is checked after acquisition, including cache hits.
            generation.execution._check_identity_runtime()  # noqa: SLF001 - canonical generation owns this private lifecycle handoff
            return generation

    def roots(self) -> tuple[PytreeValue, ...]:
        """Return actual owners, leaving alias deduplication to buffer accounting."""
        with self._lock:
            generations = tuple(self._live.values())
        return tuple(generation.live_roots() for generation in generations)
