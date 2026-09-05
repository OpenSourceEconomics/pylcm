"""Durable semantic fingerprints for labelled solution compatibility."""

import annotationlib
import builtins
import dataclasses
import dis
import functools
import hashlib
import inspect
import json
import sys
import types
import typing
from collections.abc import Callable, Iterable, Mapping
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, cast

import dags.exceptions as dags_exceptions
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxlib
import numpy as np
from beartype import beartype as _beartype_decorator
from jax import Array

import _lcm.certainty_equivalent as certainty_equivalent_declarations
import _lcm.constraints.ir as constraint_ir
import _lcm.egm.interp as interp_declarations
import _lcm.grids as grid_declarations
import _lcm.optimization.golden_section as golden_section_declarations
import _lcm.optimization.implicit_outer_derivative as implicit_outer_declarations
import _lcm.power_mean as power_mean_declarations
import _lcm.probability as probability_declarations
import _lcm.utils.functools as functools_declarations
import _lcm.zero_safe as zero_safe_declarations
import lcm.exceptions as lcm_exceptions
import lcm.koopmans_aggregation as koopmans_declarations
import lcm.processes as process_declarations
from _lcm.certainty_equivalent import CertaintyEquivalent
from _lcm.egm.outer_search import AdaptiveOuterMesh, FiniteOuterGrid
from _lcm.engine import Regime
from _lcm.grids import DiscreteGrid, Grid
from _lcm.optimization.golden_section import GoldenSectionResult
from _lcm.solution.dcegm import DCEGM, ExactEnvelope, FUESEnvelope
from _lcm.solution.grid_search import GridSearch
from _lcm.solution.nbegm import NBEGM
from _lcm.solution.negm import NEGM
from _lcm.typing import FlatParams, RegimeName, RegimeNamesToIds
from lcm.ages import AgeGrid
from lcm.case_piece import (
    AffineBreakpoint,
    CaseBoundary,
    PieceMeta,
    PiecewiseAffineMeta,
)
from lcm.phased import Phased
from lcm.solver_api import (
    ArtifactChannel,
    AxisRole,
    PersistencePolicy,
    SolverIdentity,
)

_BEARTYPE_CLAW_STATE = vars(certainty_equivalent_declarations).get(
    "__claw_state_beartype__"
)
_INSPECT_SIGNATURE_FUNCTION = inspect.signature
_INSPECT_SIGNATURE_CODE = inspect.signature.__code__
_DATACLASSES_MISSING = dataclasses.MISSING
_DATACLASSES_FIELD_MARKERS: tuple[tuple[str, object], ...] = tuple(
    (name, vars(dataclasses)[name])
    for name in ("_FIELD", "_FIELD_CLASSVAR", "_FIELD_INITVAR")
)
_PYTHON_IMPLEMENTATION_SEAL = (
    sys.implementation.name,
    tuple(sys.implementation.version),
    sys.implementation.cache_tag,
)
# These names are execution policy only for pylcm's own implementations. The
# predicate below deliberately scopes them by owner type: a plugin or user
# callable is free to give a mathematically meaningful field the same name.
_GRID_EXECUTION_FIELDS = frozenset({"batch_size", "distributed"})
_BUILTIN_EXECUTION_FIELDS_BY_TYPE: tuple[tuple[type[object], frozenset[str]], ...] = (
    (AdaptiveOuterMesh, frozenset({"batch_size"})),
    (FiniteOuterGrid, frozenset({"batch_size"})),
    (DCEGM, frozenset({"stochastic_node_batch_size"})),
    (ExactEnvelope, frozenset({"cell_batch_size"})),
    (FUESEnvelope, frozenset({"scan_unroll"})),
    (GridSearch, frozenset({"action_block_width"})),
    (
        NBEGM,
        frozenset(
            {
                "branch_batch_size",
                "cell_block_size",
                "envelope_segment_block_size",
                "interval_batch_size",
                "stochastic_node_batch_size",
            }
        ),
    ),
    (NEGM, frozenset({"outer_batch_size"})),
)
_BUILTIN_TYPE_OBJECTS = frozenset(
    value for value in vars(builtins).values() if isinstance(value, type)
)
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))
_TRUSTED_DECLARATIVE_ENUM_TYPES = frozenset(
    {
        ArtifactChannel,
        AxisRole,
        PersistencePolicy,
        type(annotationlib.Format.VALUE),
        type(inspect.Parameter.POSITIONAL_ONLY),
    }
)
_TRUSTED_GRID_TYPE_OBJECTS = frozenset(
    value
    for name in grid_declarations.__all__
    if isinstance(value := getattr(grid_declarations, name), type)
    and issubclass(value, Grid)
)
_TRUSTED_GRID_EXECUTION_TYPE_OBJECTS = _TRUSTED_GRID_TYPE_OBJECTS | frozenset(
    value
    for name in process_declarations.__all__
    if isinstance(value := getattr(process_declarations, name), type)
    and issubclass(value, Grid)
)
_TRUSTED_CONSTRAINT_TYPE_OBJECTS = frozenset(
    value
    for value in vars(constraint_ir).values()
    if isinstance(value, type) and value.__module__ == constraint_ir.__name__
)
_TRUSTED_CERTAINTY_EQUIVALENT_TYPE_OBJECTS = frozenset(
    value
    for value in vars(certainty_equivalent_declarations).values()
    if isinstance(value, type)
    and value.__module__ == certainty_equivalent_declarations.__name__
)
_TRUSTED_CASE_DECLARATION_TYPES = frozenset(
    {AffineBreakpoint, CaseBoundary, PieceMeta, PiecewiseAffineMeta}
)
_TRUSTED_DAGS_EXCEPTION_TYPES = frozenset(
    value
    for value in vars(dags_exceptions).values()
    if isinstance(value, type)
    and issubclass(value, Exception)
    and value.__module__ == dags_exceptions.__name__
)
_TRUSTED_PYLCM_EXCEPTION_TYPES = frozenset(
    value
    for value in vars(lcm_exceptions).values()
    if isinstance(value, type)
    and issubclass(value, Exception)
    and value.__module__ == lcm_exceptions.__name__
)
_TRUSTED_DIRECT_TYPE_OBJECTS = (
    frozenset(
        {functools.partial, inspect.Signature.empty, np.dtype, GoldenSectionResult}
    )
    | _TRUSTED_GRID_TYPE_OBJECTS
    | _TRUSTED_CONSTRAINT_TYPE_OBJECTS
    | _TRUSTED_CERTAINTY_EQUIVALENT_TYPE_OBJECTS
    | _TRUSTED_DAGS_EXCEPTION_TYPES
    | _TRUSTED_PYLCM_EXCEPTION_TYPES
)
_TRUSTED_TERMINAL_OBJECT_TYPES = (
    _TRUSTED_GRID_TYPE_OBJECTS
    | _TRUSTED_CONSTRAINT_TYPE_OBJECTS
    | _TRUSTED_CERTAINTY_EQUIVALENT_TYPE_OBJECTS
    | _TRUSTED_CASE_DECLARATION_TYPES
)


def _anchored_module_closure(
    *roots: types.ModuleType,
) -> tuple[types.ModuleType, ...]:
    """Capture public submodules reachable from genuine package-module roots."""
    modules: list[types.ModuleType] = []
    seen: set[int] = set()
    queue = [(root, root.__name__) for root in roots]
    while queue:
        module, prefix = queue.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        modules.append(module)
        queue.extend(
            (child, prefix)
            for name, child in vars(module).items()
            if not name.startswith("_")
            if isinstance(child, types.ModuleType)
            and child.__name__.startswith(f"{prefix}.")
        )
    return tuple(modules)


# JAX wraps many public numerical functions in callable extension objects, NumPy
# exposes ufuncs, and both libraries expose scalar constructors as class objects.
# Capture the genuine public objects by identity while the libraries are imported:
# neither a mutable ``__module__`` string nor assignment to a module namespace can
# make an arbitrary Python class/callable enter these sets later.
_JAX_PJIT_FUNCTION_TYPE = type(jnp.exp)
_JAX_UFUNC_TYPE = type(jnp.maximum)
_JAX_CUSTOM_JVP_TYPE = jax.custom_jvp
_JAX_PUBLIC_NUMERIC_MODULES = (
    jax,
    *_anchored_module_closure(
        jnp,
        jax.lax,
        jax.nn,
        jax.ops,
        jax.random,
        jsp,
    ),
)
_JAX_PUBLIC_NUMERIC_VALUES = tuple(
    value
    for module in _JAX_PUBLIC_NUMERIC_MODULES
    for name, value in vars(module).items()
    if not name.startswith("_")
)
_JAX_NUMPY_UFUNCS = tuple(
    value for value in _JAX_PUBLIC_NUMERIC_VALUES if type(value) is _JAX_UFUNC_TYPE
)
_JAX_NUMERIC_PYTHON_FUNCTIONS = tuple(
    (value, value.__code__)
    for value in _JAX_PUBLIC_NUMERIC_VALUES
    if isinstance(value, types.FunctionType)
) + tuple(
    (value, value.__code__)
    for ufunc in _JAX_NUMPY_UFUNCS
    for value in cast(
        "Mapping[str, object]", vars(ufunc)["_ufunc__static_props"]
    ).values()
    if isinstance(value, types.FunctionType)
)
_JAX_NUMERIC_PJIT_FUNCTIONS = tuple(
    (value, vars(value).get("_fun"), getattr(vars(value).get("_fun"), "__code__", None))
    for value in _JAX_PUBLIC_NUMERIC_VALUES
    if type(value) is _JAX_PJIT_FUNCTION_TYPE
) + tuple(
    (value, vars(value).get("_fun"), getattr(vars(value).get("_fun"), "__code__", None))
    for ufunc in _JAX_NUMPY_UFUNCS
    for value in cast(
        "Mapping[str, object]", vars(ufunc)["_ufunc__static_props"]
    ).values()
    if type(value) is _JAX_PJIT_FUNCTION_TYPE
)
_JAX_NUMERIC_SCALAR_TYPES = tuple(
    value
    for value in _JAX_PUBLIC_NUMERIC_VALUES
    if isinstance(value, type) and type(value) is type(jnp.int32)
)
_JAX_PUBLIC_NUMERIC_TYPE_OBJECTS = tuple(
    value for value in _JAX_PUBLIC_NUMERIC_VALUES if isinstance(value, type)
)
_NUMPY_UFUNCS = tuple(value for value in vars(np).values() if type(value) is np.ufunc)
_NUMPY_SCALAR_TYPES = tuple(
    value
    for value in vars(np).values()
    if isinstance(value, type) and issubclass(value, np.generic)
)
_NUMPY_PUBLIC_TYPE_OBJECTS = tuple(
    value
    for name, value in vars(np).items()
    if not name.startswith("_") and isinstance(value, type)
)

type _SemanticCollection = (
    tuple[object, ...] | list[object] | frozenset[object] | set[object]
)

if TYPE_CHECKING:
    type _ProjectionRegime = Regime
    type _ProjectionRegimes = Mapping[RegimeName, Regime]

    class _SolverDeclaration(Protocol):
        @property
        def identity(self) -> SolverIdentity: ...

    class _UserRegimeDeclaration(Protocol):
        @property
        def solver(self) -> _SolverDeclaration: ...

    type _FingerprintUserRegimes = Mapping[RegimeName, _UserRegimeDeclaration]
else:
    # Runtime structural tests and extension boundaries reach the function's own
    # conservative attribute inspection instead of decorator nominal checking.
    type _ProjectionRegime = object
    type _ProjectionRegimes = object
    type _FingerprintUserRegimes = object


def project_solution_params(
    *, flat_params: FlatParams, regimes: _ProjectionRegimes
) -> FlatParams:
    """Drop canonical parameters proved to be realized-transition-only.

    A stored policy is priced against the solve-phase transition laws. A
    ``Phased`` simulate variant governs the realized path after the action has
    been chosen, so a parameter used exclusively by that realized transition
    may legitimately differ when the same solution is replayed.

    The proof is deliberately conservative. A key is removed only when its
    *canonical qualified name* occurs in a simulate transition and nowhere in
    the complete solve-side semantic callable pool. This catches transitive DAG
    parameters (the compiled signatures carry their qualified names) while
    preserving a name shared with utility, feasibility, Pareto weights, a
    continuation helper, or a solve transition. Unknown/generic signatures bind
    more parameters; they never weaken compatibility.
    """
    projected: dict[RegimeName, MappingProxyType[str, object]] = {}
    for regime_name, regime_params in flat_params.items():
        regime = regimes[regime_name]
        solve_names, solve_accepts_unknown = _solution_parameter_usage(regime)
        simulate_names = _nested_callable_parameter_names(
            (
                regime.simulation.transitions,
                regime.simulation.compute_regime_transition_probs,
            )
        )
        # Only exact engine-qualified arguments prove ownership. A raw suffix
        # match could remove ``utility__rho`` merely because a transition has a
        # distinct ``next_state__rho`` parameter.
        realized_only = {
            name
            for name in regime_params
            if not solve_accepts_unknown
            and name in simulate_names
            and name not in solve_names
        }
        projected[regime_name] = MappingProxyType(
            {
                name: value
                for name, value in regime_params.items()
                if name not in realized_only
            }
        )
    return cast("FlatParams", MappingProxyType(projected))


def _solution_parameter_usage(
    regime: _ProjectionRegime,
) -> tuple[frozenset[str], bool]:
    """Return named solve params and whether a callable can accept unknown keys."""
    solution = regime.solution
    names, accepts_unknown = _nested_callable_parameter_usage(
        (
            solution.functions,
            solution.continuation_functions,
            solution.constraints,
            solution.transitions,
            solution.compute_regime_transition_probs,
            solution.pareto_weights,
        )
    )
    pareto_weights = solution.pareto_weights
    if pareto_weights is not None:
        names |= frozenset(pareto_weights.param_names)
    return names, accepts_unknown


def _nested_callable_parameter_names(value: object) -> frozenset[str]:
    """Collect explicit argument names from nested semantic callables."""
    names, _accepts_unknown = _nested_callable_parameter_usage(value)
    return names


def _nested_callable_parameter_usage(value: object) -> tuple[frozenset[str], bool]:
    """Collect explicit names and conservatively flag generic/opaque callables."""
    seen: set[int] = set()

    def walk(current: object) -> tuple[frozenset[str], bool]:  # noqa: PLR0911
        if current is None or _has_exact_type(
            value=current, candidates=(bool, int, float, str, bytes)
        ):
            return frozenset(), False
        identity = id(current)
        if identity in seen:
            return frozenset(), False
        seen.add(identity)
        try:
            if isinstance(current, Mapping):
                return _union_parameter_usage(walk(child) for child in current.values())
            if _has_exact_type(value=current, candidates=(tuple, list, frozenset, set)):
                children = cast("Iterable[object]", current)
                return _union_parameter_usage(walk(child) for child in children)
            if dataclasses.is_dataclass(current) and not isinstance(current, type):
                field_names, field_unknown = _union_parameter_usage(
                    walk(getattr(current, declaration.name))
                    for declaration in dataclasses.fields(current)
                    if not _exclude_field(owner=current, field_name=declaration.name)
                )
                if not callable(current):
                    return field_names, field_unknown
                callable_names, callable_unknown = _callable_parameter_usage(current)
                return (
                    field_names | callable_names,
                    field_unknown or callable_unknown,
                )
            if not callable(current):
                return frozenset(), False
            return _callable_parameter_usage(current)
        finally:
            seen.remove(identity)

    return walk(value)


def _callable_parameter_usage(value: object) -> tuple[frozenset[str], bool]:
    """Return explicit parameters and whether arbitrary keywords may be read."""
    try:
        parameters = inspect.signature(cast("Callable[..., object]", value)).parameters
    except TypeError, ValueError:
        return frozenset(), True
    return frozenset(parameters), any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _union_parameter_usage(
    usages: Iterable[tuple[frozenset[str], bool]],
) -> tuple[frozenset[str], bool]:
    """Union a one-shot iterable of parameter-usage pairs."""
    names: set[str] = set()
    accepts_unknown = False
    for child_names, child_accepts_unknown in usages:
        names.update(child_names)
        accepts_unknown |= child_accepts_unknown
    return frozenset(names), accepts_unknown


def fingerprint_model(
    *,
    ages: AgeGrid,
    regimes: Mapping[RegimeName, Regime],
    user_regimes: _FingerprintUserRegimes,
    regime_names_to_ids: RegimeNamesToIds,
    flat_params: FlatParams,
) -> str:
    """Hash the model facts that determine stored mathematical interpretation.

    The record includes period/regime topology, state and action names, concrete
    grid support, solver and replay identities, solve-side user callable
    semantics, fixed numerical conventions, and canonical solution parameters.
    Hardware placement, compiler, sharding, tiling controls, prose descriptions,
    and ``Phased.simulate`` truth are intentionally excluded.
    """
    projected_fixed_params = project_solution_params(
        flat_params=MappingProxyType(
            {name: regime.resolved_fixed_params for name, regime in regimes.items()}
        ),
        regimes=regimes,
    )
    record = (
        ("pylcm-model-fingerprint", 5),
        tuple(ages.exact_values),
        {name: int(regime_id) for name, regime_id in regime_names_to_ids.items()},
        {
            name: {
                "active_periods": regime.active_periods,
                "state_names": regime.solution.state_names,
                "action_names": regime.solution.action_names,
                "fold_state_names": regime.fold_state_names,
                "stakeholders": regime.stakeholders,
                "solver_identity": user_regimes[name].solver.identity,
                "replay_identity": (
                    None
                    if regime.simulation.external_replay_route is None
                    else regime.simulation.external_replay_route.identity
                ),
                "artifact_descriptors": {
                    key: authority.descriptor
                    for key, authority in regime.solution.artifact_authorities.items()
                },
                "grid_support": {
                    "states": regime.solution.state_action_space(
                        regime_params=flat_params[name]
                    ).states,
                    "discrete_actions": regime.solution.state_action_space(
                        regime_params=flat_params[name]
                    ).discrete_actions,
                    "continuous_actions": regime.solution.state_action_space(
                        regime_params=flat_params[name]
                    ).continuous_actions,
                    "period_state_axes": regime.solution.period_state_axes,
                },
                # ``Phased`` is projected to its solve member by the semantic
                # hasher. The declaration retains user-level function bodies,
                # defaults, closures and globals instead of relying on compiled
                # wrapper identity.
                "declaration": _project_user_regime_declaration(user_regimes[name]),
                "fixed_params": projected_fixed_params[name],
            }
            for name, regime in regimes.items()
        },
        project_solution_params(flat_params=flat_params, regimes=regimes),
    )
    return _semantic_fingerprint(record)


def _project_user_regime_declaration(regime: object) -> MappingProxyType[str, object]:
    """Return the semantic dataclass fields without importing declaration topology."""
    if type(regime) is types.SimpleNamespace:
        fields = vars(regime).items()
    elif dataclasses.is_dataclass(regime) and not isinstance(regime, type):
        fields = (
            (declaration.name, getattr(regime, declaration.name))
            for declaration in dataclasses.fields(cast("Any", regime))
        )
    else:
        msg = "A model fingerprint requires a dataclass user-regime declaration."
        raise TypeError(msg)
    declaration_type = type(regime)
    return MappingProxyType(
        {
            "type": f"{declaration_type.__module__}.{declaration_type.__qualname__}",
            "fields": MappingProxyType(
                {name: value for name, value in fields if name != "description"}
            ),
        }
    )


def _semantic_fingerprint(value: object) -> str:
    """Return a durable digest for one nested semantic value.

    This private entry point also keeps the collision-focused tests small: they
    can exercise the serializer without constructing or solving a model.
    """
    hasher = _SemanticHasher()
    hasher.visit(value=value)
    return hasher.hexdigest()


class _SemanticHasher:
    """Length-framed serializer feeding a SHA-256 digest."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self._active: dict[int, int] = {}
        self._active_bound_methods: dict[tuple[int, int], int] = {}

    def hexdigest(self) -> str:
        return self._digest.hexdigest()

    def frame(self, *, label: str, payload: bytes = b"") -> None:
        for part in (label.encode(), payload):
            self._digest.update(len(part).to_bytes(8, byteorder="big"))
            self._digest.update(part)

    def visit(self, *, value: object, _ignore_beartype_guards: bool = False) -> None:  # noqa: C901, PLR0911, PLR0912, PLR0915
        if value is Ellipsis:
            self.frame(label="ellipsis")
            return
        if value is _DATACLASSES_MISSING:
            self.frame(label="dataclasses-missing")
            return
        if (marker_name := _dataclasses_field_marker_name(value)) is not None:
            self.frame(label="dataclasses-field-marker", payload=marker_name.encode())
            return
        if value is None or _has_exact_type(
            value=value, candidates=(bool, int, float, str, bytes)
        ):
            self.frame(label=type(value).__name__, payload=repr(value).encode())
            return
        if value is _beartype_decorator:
            self.frame(label="transparent-beartype-decorator")
            return
        if type(value) is slice:
            semantic_slice = cast("slice", value)
            self.frame(label="slice-start")
            self.visit(
                value=(semantic_slice.start, semantic_slice.stop, semantic_slice.step)
            )
            self.frame(label="slice-end")
            return
        if type(value) is typing.TypeAliasType:
            self.frame(label="type-alias")
            self._visit_annotation(value)
            return
        if isinstance(value, Fraction):
            if type(value) is not Fraction:
                msg = (
                    "Cannot durably fingerprint Fraction subclass semantic value "
                    f"{type(value).__module__}.{type(value).__qualname__}."
                )
                raise TypeError(msg)
            self.frame(
                label="Fraction",
                payload=f"{value.numerator}/{value.denominator}".encode(),
            )
            return
        if isinstance(value, Enum):
            if not _has_exact_type(
                value=value, candidates=_TRUSTED_DECLARATIVE_ENUM_TYPES
            ):
                msg = (
                    "Cannot durably fingerprint custom Enum semantic value "
                    f"{type(value).__module__}.{type(value).__qualname__}."
                )
                raise TypeError(msg)
            self.frame(
                label="Enum",
                payload=(
                    f"{type(value).__module__}.{type(value).__qualname__}:"
                    f"{value.value!r}"
                ).encode(),
            )
            return
        if isinstance(value, np.dtype):
            self.frame(label="dtype", payload=value.str.encode())
            return
        is_exact_numpy_value = type(value) is np.ndarray or _has_exact_type(
            value=value, candidates=_NUMPY_SCALAR_TYPES
        )
        if isinstance(value, Array) or is_exact_numpy_value:
            # The shape frame records the rank and shape the declaration actually
            # holds: a 0-d array is a different mathematical object from a
            # length-one vector even though their bytes agree. Memory order is
            # storage, not identity, and ``tobytes(order="C")`` serializes any
            # layout, so no contiguity normalization happens before framing.
            array = np.asarray(value)
            if array.dtype.hasobject:
                msg = "Object arrays cannot enter a durable model fingerprint."
                raise TypeError(msg)
            self.frame(
                label="array-shape", payload=json.dumps(list(array.shape)).encode()
            )
            self.frame(label="array-dtype", payload=array.dtype.str.encode())
            self.frame(label="array-bytes", payload=array.tobytes(order="C"))
            return
        if isinstance(value, np.ndarray | np.generic):
            msg = (
                "Cannot durably fingerprint NumPy protocol subclass semantic value "
                f"{type(value).__module__}.{type(value).__qualname__}."
            )
            raise TypeError(msg)
        if value is _INSPECT_SIGNATURE_FUNCTION:
            if _INSPECT_SIGNATURE_FUNCTION.__code__ is not _INSPECT_SIGNATURE_CODE:
                msg = (
                    "Cannot durably fingerprint inspect.signature after its "
                    "captured code identity changed."
                )
                raise TypeError(msg)
            self.frame(label="inspect-signature-python-seal")
            self.visit(value=_PYTHON_IMPLEMENTATION_SEAL)
            return
        if self._visit_native_numeric_callable(value):
            return
        if _is_native_numeric_type(value):
            self._visit_native_numeric_type(cast("type[object]", value))
            return

        identity = id(value)
        if identity in self._active:
            self.frame(
                label="cycle-backreference",
                payload=(
                    f"{self._active[identity]}:"
                    f"{type(value).__module__}.{type(value).__qualname__}"
                ).encode(),
            )
            return
        # The anchor is its deterministic depth in the current traversal, never
        # the process-local object id. Tracking functions and classes as well as
        # containers is necessary because a helper method may refer back to the
        # class attribute through which it was reached.
        self._active[identity] = len(self._active)
        try:
            if isinstance(value, types.ModuleType):
                msg = (
                    "Cannot durably fingerprint direct module dependency "
                    f"{value.__name__}; reference a statically inspectable module "
                    "attribute instead."
                )
                raise TypeError(msg)
            if isinstance(value, DiscreteGrid):
                if type(value) is not DiscreteGrid:
                    msg = (
                        "Cannot durably fingerprint DiscreteGrid subclass semantic "
                        f"value {type(value).__module__}.{type(value).__qualname__}."
                    )
                    raise TypeError(msg)
                self.frame(
                    label="DiscreteGrid-start",
                    payload=f"{type(value).__module__}.{type(value).__qualname__}".encode(),
                )
                self.visit(value=value.categories)
                self.visit(value=value.codes)
                self.visit(value=value.ordered)
                self.frame(label="DiscreteGrid-end")
                return
            if isinstance(value, Phased):
                if type(value) is not Phased:
                    msg = (
                        "Cannot durably fingerprint Phased subclass semantic value "
                        f"{type(value).__module__}.{type(value).__qualname__}."
                    )
                    raise TypeError(msg)
                self.frame(label="Phased-solve-start")
                self.visit(value=value.solve)
                self.frame(label="Phased-solve-end")
                return
            if isinstance(value, types.CodeType):
                self._visit_code(value)
                return
            if isinstance(value, type):
                if (
                    not _contains_identity(
                        value=value, candidates=_BUILTIN_TYPE_OBJECTS
                    )
                    and not _contains_identity(
                        value=value, candidates=_TRUSTED_DIRECT_TYPE_OBJECTS
                    )
                    and not _is_versioned_numeric_library_type(value)
                ):
                    msg = (
                        "Cannot durably fingerprint direct class dependency "
                        f"{value.__module__}.{value.__qualname__}."
                    )
                    raise TypeError(msg)
                self._visit_type(value)
                return
            if isinstance(value, Mapping):
                if not _has_exact_type(
                    value=value, candidates=(dict, _MAPPING_PROXY_TYPE)
                ):
                    msg = (
                        "Cannot durably fingerprint custom Mapping semantic value "
                        f"{type(value).__module__}.{type(value).__qualname__}."
                    )
                    raise TypeError(msg)
                mapping = cast("Mapping[object, object]", value)
                self.frame(label="mapping-start", payload=str(len(mapping)).encode())
                for key in sorted(mapping, key=_semantic_sort_key):
                    self.visit(value=key)
                    self.visit(value=mapping[key])
                self.frame(label="mapping-end")
                return
            if _has_exact_type(value=value, candidates=(tuple, list, frozenset, set)):
                collection = cast("_SemanticCollection", value)
                children = (
                    sorted(collection, key=_semantic_sort_key)
                    if isinstance(collection, frozenset | set)
                    else collection
                )
                self.frame(
                    label=type(value).__name__ + "-start",
                    payload=str(len(collection)).encode(),
                )
                for child in children:
                    self.visit(value=child)
                self.frame(label=type(value).__name__ + "-end")
                return
            if isinstance(value, functools.partial):
                self._validate_partial_arguments(value)
                self.frame(label="partial-start")
                self.visit(value=value.func)
                self.visit(value=value.args)
                self.visit(value=value.keywords or {})
                self._visit_named_state(owner=value, state=value.__dict__)
                self.frame(label="partial-end")
                return
            if inspect.ismethod(value):
                self._visit_referenced_bound_method(value)
                return
            if inspect.isfunction(value):
                self._visit_function(
                    function=value, ignore_beartype_guards=_ignore_beartype_guards
                )
                return
            if inspect.isbuiltin(value):
                self.frame(
                    label="builtin",
                    payload=(
                        f"{getattr(value, '__module__', '')}."
                        f"{getattr(value, '__qualname__', value.__name__)}"
                    ).encode(),
                )
                owner = getattr(value, "__self__", None)
                if owner is not None and not isinstance(owner, types.ModuleType):
                    self.visit(value=owner)
                return
            if _is_solver_instance(value):
                self._visit_solver(value)
                return
            if self._visit_certainty_equivalent(value):
                return
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                self._visit_dataclass(value)
                return
            if callable(value):
                self._visit_callable_object(value)
                return
            state = getattr(value, "__dict__", None)
            slots = _slot_state(value)
            if state or slots:
                self.frame(
                    label="object-state-start",
                    payload=f"{type(value).__module__}.{type(value).__qualname__}".encode(),
                )
                self._visit_named_state(owner=value, state=state or {})
                self._visit_named_state(owner=value, state=slots)
                self.frame(label="object-state-end")
                return
            msg = (
                "Cannot durably fingerprint opaque semantic value of type "
                f"{type(value).__module__}.{type(value).__qualname__}."
            )
            raise TypeError(msg)
        finally:
            del self._active[identity]

    def _visit_code(self, code: types.CodeType) -> None:
        """Hash executable code without source paths or line-table noise."""
        self.frame(label="code-start")
        self.visit(
            value=(
                code.co_argcount,
                code.co_posonlyargcount,
                code.co_kwonlyargcount,
                code.co_nlocals,
                code.co_stacksize,
                code.co_flags,
            )
        )
        self.frame(label="code-bytes", payload=code.co_code)
        self.frame(
            label="exception-table", payload=getattr(code, "co_exceptiontable", b"")
        )
        constants = code.co_consts
        # A function docstring is the first code constant but is not executable
        # mathematics. Comments and source locations are absent for the same reason.
        if constants and isinstance(constants[0], str):
            constants = constants[1:]
        self.visit(value=constants)
        self.visit(value=code.co_names)
        self.frame(label="code-end")

    def _visit_type(self, value: type) -> None:
        self.frame(
            label="type-start",
            payload=f"{value.__module__}.{value.__qualname__}".encode(),
        )
        if _is_versioned_numeric_library_type(value):
            self.frame(label="numeric-library-type-version-seal")
            self.visit(value=_native_numeric_versions())
        # Category classes are dataclasses whose ordered flag and class-level
        # ScalarInt values carry meaning. Most types are identities only.
        if dataclasses.is_dataclass(value) and hasattr(value, "_ordered"):
            self.visit(value=bool(inspect.getattr_static(value, "_ordered")))
            for declaration in dataclasses.fields(value):
                self.frame(label="category-field", payload=declaration.name.encode())
                self.visit(value=getattr(value, declaration.name))
        self.frame(label="type-end")

    def _visit_native_numeric_callable(self, value: object) -> bool:
        """Hash supported native numerical callables without opaque object trust."""
        kind = _native_numeric_callable_kind(value)
        if kind is None:
            return False

        self.frame(label="native-numeric-callable-start", payload=kind.encode())
        self.visit(value=_native_numeric_versions())
        if kind == "jax-custom-jvp":
            self._visit_custom_jvp(value)
        elif kind == "jax-function":
            function = cast("types.FunctionType", value)
            self.frame(label="canonical-jax-function")
            self._visit_native_python_function_seal(function)
        elif kind == "jax-pjit":
            state = vars(value)
            function = state.get("_fun")
            if not isinstance(function, types.FunctionType):
                raise TypeError(
                    "A JAX PjitFunction has no statically inspectable Python function."
                )
            self.frame(
                label="native-callable-name",
                payload=(
                    f"{getattr(value, '__module__', '')}."
                    f"{getattr(value, '__qualname__', getattr(value, '__name__', ''))}"
                ).encode(),
            )
            if _is_captured_jax_pjit(value):
                # Installed JAX code is sealed by package versions, public identity,
                # and executable bytes. Walking all of its private global helpers
                # would turn implementation details into model dependencies.
                self.frame(label="canonical-jax-pjit")
                self._visit_native_python_function_seal(function)
            else:
                # A user-created jitted function has the same extension type. Hash
                # its complete Python semantics rather than trusting that type alone.
                self.frame(label="user-jax-pjit")
                self.visit(value=function)
        elif kind == "jax-ufunc":
            state = vars(value)
            properties = state.get("_ufunc__static_props")
            if type(properties) is not dict:
                raise TypeError("A JAX ufunc has no exact static-properties mapping.")
            self.frame(
                label="native-callable-name",
                payload=str(state.get("__name__", "")).encode(),
            )
            self.visit(value=properties)
        else:
            ufunc = cast("np.ufunc", value)
            self.visit(
                value=(
                    ufunc.__name__,
                    ufunc.nin,
                    ufunc.nout,
                    ufunc.nargs,
                    ufunc.identity,
                    tuple(ufunc.types),
                    getattr(ufunc, "signature", None),
                )
            )
        self.frame(label="native-numeric-callable-end")
        return True

    def _visit_custom_jvp(self, value: object) -> None:
        """Hash an exact JAX custom-JVP wrapper through its semantic callables."""
        identity = id(value)
        if identity in self._active:
            self.frame(
                label="cycle-backreference",
                payload=(
                    f"{self._active[identity]}:"
                    f"{type(value).__module__}.{type(value).__qualname__}"
                ).encode(),
            )
            return
        self._active[identity] = len(self._active)
        try:
            state = vars(value)
            metadata_fields = {
                "__annotate__",
                "__beartype_annotations",
                "__beartype_args_lens",
                "__beartype_wrapper",
                "__doc__",
                "__module__",
                "__name__",
                "__qualname__",
                "__type_params__",
                "__wrapped__",
            }
            semantic_fields = {"fun", "jvp", "nondiff_argnums", "symbolic_zeros"}
            if unexpected := set(state) - metadata_fields - semantic_fields:
                raise TypeError(
                    "Cannot durably fingerprint JAX custom_jvp with unknown state "
                    f"field(s): {sorted(unexpected)}."
                )
            function = state.get("fun")
            jvp = state.get("jvp")
            nondiff_argnums = state.get("nondiff_argnums")
            symbolic_zeros = state.get("symbolic_zeros")
            if not callable(function) or (jvp is not None and not callable(jvp)):
                raise TypeError("A JAX custom_jvp has an uninspectable callable state.")
            if type(nondiff_argnums) is not tuple or any(
                type(index) is not int for index in nondiff_argnums
            ):
                raise TypeError("A JAX custom_jvp has invalid nondiff_argnums state.")
            if symbolic_zeros is not None and type(symbolic_zeros) is not bool:
                raise TypeError("A JAX custom_jvp has invalid symbolic_zeros state.")
            self.visit(value=(function, jvp, nondiff_argnums, symbolic_zeros))
        finally:
            del self._active[identity]

    def _visit_native_python_function_seal(self, function: types.FunctionType) -> None:
        """Hash one captured library function without following private globals."""
        self.frame(
            label="native-python-function-start",
            payload=f"{function.__module__}.{function.__qualname__}".encode(),
        )
        self._visit_signature(function)
        self.visit(value=function.__code__)
        self.visit(value=function.__defaults__)
        self.visit(value=function.__kwdefaults__)
        self.frame(label="native-python-function-end")

    def _visit_native_numeric_type(self, value: type[object]) -> None:
        """Hash a genuine NumPy/JAX scalar constructor by dtype and runtime seal."""
        self.frame(
            label="native-numeric-type-start",
            payload=f"{value.__module__}.{value.__qualname__}".encode(),
        )
        self.visit(value=_native_numeric_versions())
        self.frame(label="native-numeric-dtype", payload=np.dtype(value).str.encode())
        self.frame(label="native-numeric-type-end")

    def _visit_signature(self, function: object) -> None:
        try:
            signature = inspect.signature(cast("Callable[..., object]", function))
        except TypeError, ValueError:
            self.frame(label="signature-unavailable")
            return
        self.frame(label="signature-start")
        for parameter in signature.parameters.values():
            self.frame(label="parameter", payload=parameter.name.encode())
            self.visit(value=parameter.kind.value)
            if parameter.default is inspect.Signature.empty:
                self.frame(label="default-empty")
            else:
                self.visit(value=parameter.default)
            self._visit_annotation(parameter.annotation)
        self.frame(label="return-annotation")
        self._visit_annotation(signature.return_annotation)
        self.frame(label="signature-end")

    def _visit_annotation(  # noqa: C901, PLR0911, PLR0912
        self, annotation: object
    ) -> None:
        """Hash type metadata without treating an annotation as executable input."""
        if annotation is inspect.Signature.empty:
            self.frame(label="annotation-empty")
            return
        if annotation is None:
            self.frame(label="annotation-none")
            return
        if _has_exact_type(value=annotation, candidates=(bool, int, float, str, bytes)):
            self.frame(label="annotation-literal", payload=repr(annotation).encode())
            return
        if annotation is Ellipsis:
            self.frame(label="annotation-ellipsis")
            return
        if _has_exact_type(value=annotation, candidates=(list, tuple)):
            arguments = cast("list[object] | tuple[object, ...]", annotation)
            self.frame(
                label="annotation-arguments-start",
                payload=f"{type(annotation).__name__}:{len(arguments)}".encode(),
            )
            for argument in arguments:
                self._visit_annotation(argument)
            self.frame(label="annotation-arguments-end")
            return
        if isinstance(annotation, type):
            self.frame(
                label="annotation-type",
                payload=f"{annotation.__module__}.{annotation.__qualname__}".encode(),
            )
            return

        origin = typing.get_origin(annotation)
        if origin is not None:
            self.frame(label="annotation-generic-start")
            self._visit_annotation(origin)
            for argument in typing.get_args(annotation):
                self._visit_annotation(argument)
            self.frame(label="annotation-generic-end")
            return

        if isinstance(annotation, typing.TypeVar):
            self.frame(
                label="annotation-typevar-start", payload=annotation.__name__.encode()
            )
            self.visit(value=annotation.__covariant__)
            self.visit(value=annotation.__contravariant__)
            self._visit_annotation(annotation.__bound__)
            for constraint in annotation.__constraints__:
                self._visit_annotation(constraint)
            self.frame(label="annotation-typevar-end")
            return

        module = getattr(annotation, "__module__", None)
        qualname = getattr(annotation, "__qualname__", None)
        if isinstance(module, str) and isinstance(qualname, str):
            self.frame(
                label="annotation-identity", payload=f"{module}.{qualname}".encode()
            )
            return
        name = getattr(annotation, "__name__", None)
        if isinstance(module, str) and isinstance(name, str):
            self.frame(label="annotation-identity", payload=f"{module}.{name}".encode())
            return
        msg = (
            "Cannot durably fingerprint unsupported function annotation of type "
            f"{type(annotation).__module__}.{type(annotation).__qualname__}."
        )
        raise TypeError(msg)

    def _visit_function_annotations(self, annotations: Mapping[str, object]) -> None:
        """Hash a function's raw annotation mapping via metadata-only traversal."""
        self.frame(label="function-annotations-start")
        for name in sorted(annotations):
            self.frame(label="function-annotation", payload=name.encode())
            self._visit_annotation(annotations[name])
        self.frame(label="function-annotations-end")

    def _visit_function(  # noqa: C901, PLR0912
        self, *, function: types.FunctionType, ignore_beartype_guards: bool = False
    ) -> None:
        wrapped = _unwrap_exact_beartype_wrapper(function)
        if wrapped is not None:
            self.frame(
                label="transparent-beartype-start",
                payload=f"{function.__module__}.{function.__qualname__}".encode(),
            )
            self._visit_signature(function)
            self.visit(value=wrapped, _ignore_beartype_guards=True)
            self.frame(label="transparent-beartype-end")
            return
        self._validate_function_defaults(function)
        self.frame(
            label="function-start",
            payload=f"{function.__module__}.{function.__qualname__}".encode(),
        )
        self._visit_signature(function)
        self.visit(value=function.__code__)
        self.visit(value=function.__defaults__)
        self.visit(value=function.__kwdefaults__)
        self._visit_function_annotations(function.__annotations__)
        self.frame(label="function-type-params-start")
        for type_parameter in getattr(function, "__type_params__", ()):
            self._visit_annotation(type_parameter)
        self.frame(label="function-type-params-end")
        if function.__closure__:
            self.frame(label="closure-start")
            closure_references = _referenced_closure_attribute_paths(
                code=function.__code__,
                ignore_beartype_guards=ignore_beartype_guards,
            )
            for name, cell in zip(
                function.__code__.co_freevars,
                function.__closure__,
                strict=True,
            ):
                self.frame(label="closure", payload=name.encode())
                try:
                    value = cell.cell_contents
                except ValueError:
                    self.frame(label="empty-cell")
                    continue
                paths = closure_references.get(name, frozenset({()}))
                if any(paths):
                    self._visit_object_reference(
                        value=value,
                        attribute_paths=paths,
                    )
                else:
                    self._visit_direct_global_reference(value)
            self.frame(label="closure-end")
        self.frame(label="globals-start")
        references = _referenced_global_attribute_paths(
            code=function.__code__,
            ignore_beartype_guards=ignore_beartype_guards,
        )
        for name in sorted(references):
            if name in function.__globals__:
                self.frame(label="global", payload=name.encode())
                value = function.__globals__[name]
                if value is _BEARTYPE_CLAW_STATE:
                    self.frame(label="transparent-beartype-claw-state")
                    continue
                if isinstance(value, types.ModuleType):
                    self._visit_module_reference(
                        module=value,
                        attribute_paths=references[name],
                    )
                elif any(references[name]):
                    self._visit_object_reference(
                        value=value,
                        attribute_paths=references[name],
                    )
                else:
                    self._visit_direct_global_reference(value)
        self.frame(label="globals-end")
        # The import-time beartype claw annotates the original function with
        # cached signature metadata. The signature and annotations above already
        # carry that information; the caches are not executable user state.
        redundant_metadata = {
            "__annotate__",
            "__annotations__",
            "__beartype_annotations",
            "__beartype_args_lens",
            "__signature__",
            "__type_params__",
        }
        function_state = {
            name: member
            for name, member in function.__dict__.items()
            if name not in redundant_metadata
        }
        self._visit_named_state(owner=function, state=function_state)
        self.frame(label="function-end")

    @staticmethod
    def _validate_function_defaults(function: types.FunctionType) -> None:
        """Fail closed when a Python default's executable semantics are not sealed."""
        positional_names = function.__code__.co_varnames[
            : function.__code__.co_argcount
        ]
        positional_defaults = function.__defaults__ or ()
        positional_default_names = positional_names[
            len(positional_names) - len(positional_defaults) :
        ]
        defaults = (*zip(positional_default_names, positional_defaults, strict=True),)
        defaults += tuple((function.__kwdefaults__ or {}).items())
        for parameter_name, default in defaults:
            if _is_closed_terminal_reference(value=default):
                continue
            msg = (
                "Cannot durably fingerprint default dependency for parameter "
                f"{parameter_name!r} of function "
                f"{function.__module__}.{function.__qualname__}."
            )
            raise TypeError(msg)

    @staticmethod
    def _validate_partial_arguments(value: functools.partial[object]) -> None:
        """Fail closed when a partial binds an argument with unsealed semantics."""
        for index, argument in enumerate(value.args):
            if not _is_closed_terminal_reference(value=argument):
                raise TypeError(
                    "Cannot durably fingerprint partial bound positional argument "
                    f"{index}."
                )
        for name, argument in (value.keywords or {}).items():
            if not _is_closed_terminal_reference(value=argument):
                raise TypeError(
                    "Cannot durably fingerprint partial bound keyword argument "
                    f"{name!r}."
                )

    def _visit_object_reference(
        self,
        *,
        value: object,
        attribute_paths: frozenset[tuple[str, ...]],
    ) -> None:
        """Hash only the class/object attributes that bytecode actually reads."""
        owner_type = value if isinstance(value, type) else type(value)
        self.frame(
            label="object-reference-start",
            payload=f"{owner_type.__module__}.{owner_type.__qualname__}".encode(),
        )
        for path in sorted(attribute_paths):
            self.frame(label="object-attribute-path", payload=".".join(path).encode())
            if not path:
                # If the object itself is consumed, no narrower semantic
                # projection is justified.
                self._visit_direct_global_reference(value)
                continue
            current = value
            for attribute in path:
                current = self._resolve_referenced_attribute(
                    value=current,
                    attribute=attribute,
                    path=path,
                )
            if inspect.ismethod(current):
                self._visit_referenced_bound_method(current)
            else:
                self._visit_terminal_reference(current)
        self.frame(label="object-reference-end")

    def _visit_direct_global_reference(self, value: object) -> None:
        """Hash a direct global value only when its complete semantics are closed."""
        self._visit_terminal_reference(value)

    def _visit_terminal_reference(
        self,
        value: object,
    ) -> None:
        """Hash one consumed reference or reject mutable protocol dispatch."""
        if inspect.ismethod(value):
            self._visit_referenced_bound_method(value)
            return
        if _is_closed_terminal_reference(value=value):
            self.visit(value=value)
            return
        value_type = value if isinstance(value, type) else type(value)
        kind = "class" if isinstance(value, type) else "object"
        type_name = f"{value_type.__module__}.{value_type.__qualname__}"
        msg = (
            f"Cannot durably fingerprint direct {kind} dependency {type_name}; "
            "reference a statically inspectable attribute instead."
        )
        raise TypeError(msg)

    def _visit_referenced_bound_method(self, method: types.MethodType) -> None:
        """Hash a method and exactly the receiver attributes its code reads."""
        raw_function = method.__func__
        if not isinstance(raw_function, types.FunctionType):
            callable_type = (
                f"{type(raw_function).__module__}.{type(raw_function).__qualname__}"
            )
            msg = (
                "Cannot durably fingerprint non-Python bound method callable "
                f"{callable_type}."
            )
            raise TypeError(msg)
        wrapped = _unwrap_exact_beartype_wrapper(raw_function)
        ignore_beartype_guards = wrapped is not None
        function = raw_function if wrapped is None else wrapped
        receiver = method.__self__
        identity = (id(function), id(receiver))
        if identity in self._active_bound_methods:
            receiver_type = receiver if isinstance(receiver, type) else type(receiver)
            self.frame(
                label="bound-method-cycle-backreference",
                payload=(
                    f"{self._active_bound_methods[identity]}:"
                    f"{function.__module__}.{function.__qualname__}:"
                    f"{receiver_type.__module__}.{receiver_type.__qualname__}"
                ).encode(),
            )
            return

        self._active_bound_methods[identity] = len(self._active_bound_methods)
        try:
            self.frame(
                label="referenced-bound-method-start",
                payload=f"{function.__module__}.{function.__qualname__}".encode(),
            )
            self.visit(
                value=function,
                _ignore_beartype_guards=ignore_beartype_guards,
            )
            receiver_name = _bound_method_receiver_name(function)
            paths, has_opaque_use = _referenced_receiver_attribute_paths(
                code=function.__code__,
                receiver_name=receiver_name,
                ignore_beartype_guards=ignore_beartype_guards,
            )
            if has_opaque_use:
                self.frame(label="bound-receiver-full")
                self.visit(value=receiver)
            receiver_type = receiver if isinstance(receiver, type) else type(receiver)
            self.frame(
                label="bound-receiver-type",
                payload=f"{receiver_type.__module__}.{receiver_type.__qualname__}".encode(),
            )
            if paths:
                self._visit_object_reference(
                    value=receiver,
                    attribute_paths=paths,
                )
            self.frame(label="referenced-bound-method-end")
        finally:
            del self._active_bound_methods[identity]

    def _resolve_referenced_attribute(
        self,
        *,
        value: object,
        attribute: str,
        path: tuple[str, ...],
    ) -> object:
        """Resolve a statically inspectable attribute without executing user code."""
        _validate_static_attribute_access(value=value, path=path)
        try:
            member = inspect.getattr_static(value, attribute)
        except AttributeError as error:
            dotted = ".".join(path)
            msg = f"Cannot durably fingerprint missing referenced attribute {dotted!r}."
            raise TypeError(msg) from error
        return _bind_referenced_member(
            value=value,
            attribute=attribute,
            member=member,
            path=path,
        )

    def _visit_module_reference(
        self,
        *,
        module: types.ModuleType,
        attribute_paths: frozenset[tuple[str, ...]],
    ) -> None:
        """Hash exact module-qualified values referenced by callable bytecode."""
        self.frame(label="module-reference-start", payload=module.__name__.encode())
        version = vars(module).get("__version__")
        if _has_exact_type(value=version, candidates=(str, int, float)):
            self.frame(label="module-version")
            self.visit(value=version)
        for path in sorted(attribute_paths):
            if not path:
                msg = (
                    "Cannot durably fingerprint direct module dependency "
                    f"{module.__name__}; reference a statically inspectable module "
                    "attribute instead."
                )
                raise TypeError(msg)
            self.frame(label="module-attribute-path", payload=".".join(path).encode())
            current: object = module
            for attribute in path:
                if isinstance(current, types.ModuleType):
                    _validate_static_attribute_access(value=current, path=path)
                    try:
                        current = inspect.getattr_static(current, attribute)
                    except AttributeError as error:
                        dotted = ".".join(path)
                        msg = (
                            "Cannot durably fingerprint dynamic or missing module "
                            f"attribute {module.__name__}.{dotted}."
                        )
                        raise TypeError(msg) from error
                else:
                    current = self._resolve_referenced_attribute(
                        value=current,
                        attribute=attribute,
                        path=path,
                    )
            self._visit_terminal_reference(current)
        self.frame(label="module-reference-end")

    def _visit_dataclass(self, value: object) -> None:
        self.frame(
            label="dataclass-start",
            payload=f"{type(value).__module__}.{type(value).__qualname__}".encode(),
        )
        for declaration in dataclasses.fields(cast("Any", value)):
            if _exclude_field(owner=value, field_name=declaration.name):
                continue
            self.frame(label="field", payload=declaration.name.encode())
            self.visit(value=getattr(value, declaration.name))
        # Exact pylcm declarations have their implementation sealed by the
        # separately checked pylcm version; their fully traversed fields carry
        # instance semantics. Extension dataclasses still bind their own call code.
        if callable(value) and not _has_exact_type(
            value=value, candidates=_TRUSTED_TERMINAL_OBJECT_TYPES
        ):
            call = inspect.getattr_static(type(value), "__call__", None)
            if not inspect.isfunction(call):
                self._raise_uninspectable_callable(value)
            self.frame(label="dataclass-call")
            self._visit_referenced_bound_method(types.MethodType(call, value))
        self.frame(label="dataclass-end")

    def _visit_callable_object(self, value: object) -> None:
        self.frame(
            label="callable-object-start",
            payload=f"{type(value).__module__}.{type(value).__qualname__}".encode(),
        )
        self._visit_signature(value)
        self.frame(
            label="callable-name",
            payload=str(
                getattr(value, "__qualname__", getattr(value, "__name__", ""))
            ).encode(),
        )
        # Signature and annotation metadata are already serialized by
        # ``_visit_signature``. Treating those introspection objects as executable
        # instance state both double-counts them and sends Python 3.14
        # ``TypeAliasType`` annotations through the opaque-value policy.
        dictionary_state = {
            name: member
            for name, member in (getattr(value, "__dict__", {}) or {}).items()
            if name not in {"__annotations__", "__signature__"}
        }
        self._visit_named_state(owner=value, state=dictionary_state)
        self._visit_named_state(owner=value, state=_slot_state(value))
        call = inspect.getattr_static(type(value), "__call__", None)
        if not inspect.isfunction(call):
            self._raise_uninspectable_callable(value)
        self._visit_referenced_bound_method(types.MethodType(call, value))
        self.frame(label="callable-object-end")

    @staticmethod
    def _raise_uninspectable_callable(value: object) -> None:
        callable_type = f"{type(value).__module__}.{type(value).__qualname__}"
        msg = (
            "Cannot durably fingerprint callable object with non-Python __call__: "
            f"{callable_type}."
        )
        raise TypeError(msg)

    def _visit_solver(self, value: object) -> None:
        """Hash a Solver's public compatibility identity and instance state.

        A stateless solver's class implementation and class attributes are covered by
        its exact versioned ``SolverIdentity`` contract. Instance configuration is
        still traversed, so an opaque state token cannot hide a semantic branch.
        """
        _validate_static_attribute_access(value=value, path=("identity",))
        identity = getattr(value, "identity", None)
        if type(identity) is not SolverIdentity:
            msg = (
                "Cannot durably fingerprint Solver with a non-exact SolverIdentity: "
                f"{type(value).__module__}.{type(value).__qualname__}."
            )
            raise TypeError(msg)
        self.frame(
            label="solver-start",
            payload=f"{type(value).__module__}.{type(value).__qualname__}".encode(),
        )
        self.visit(value=identity)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for declaration in dataclasses.fields(cast("Any", value)):
                if _exclude_field(owner=value, field_name=declaration.name):
                    continue
                self.frame(label="solver-field", payload=declaration.name.encode())
                self.visit(value=getattr(value, declaration.name))
        else:
            self._visit_named_state(
                owner=value, state=getattr(value, "__dict__", {}) or {}
            )
            self._visit_named_state(owner=value, state=_slot_state(value))
        self.frame(label="solver-end")

    def _visit_certainty_equivalent(self, value: object) -> bool:
        """Hash a CE extension through state and its three protocol operations."""
        if not isinstance(value, CertaintyEquivalent):
            return False
        for name in ("param_names", "aggregate", "aggregate_scaled"):
            _validate_static_attribute_access(value=value, path=(name,))

        self.frame(
            label="certainty-equivalent-start",
            payload=f"{type(value).__module__}.{type(value).__qualname__}".encode(),
        )
        self._visit_named_state(owner=value, state=getattr(value, "__dict__", {}) or {})
        self._visit_named_state(owner=value, state=_slot_state(value))

        # Shipped implementations are sealed by the exact pylcm version checked
        # alongside every durable solution. User implementations have no package
        # seal, so bind their complete protocol implementation here.
        if not _has_exact_type(
            value=value, candidates=_TRUSTED_CERTAINTY_EQUIVALENT_TYPE_OBJECTS
        ):
            for name in ("param_names", "aggregate", "aggregate_scaled"):
                self.frame(
                    label="certainty-equivalent-operation", payload=name.encode()
                )
                if name == "param_names":
                    descriptor = inspect.getattr_static(type(value), name, None)
                    if type(descriptor) is not property or not inspect.isfunction(
                        descriptor.fget
                    ):
                        msg = (
                            "Cannot durably fingerprint CertaintyEquivalent property "
                            f"{type(value).__module__}.{type(value).__qualname__}."
                            "param_names."
                        )
                        raise TypeError(msg)
                    operation = types.MethodType(descriptor.fget, value)
                else:
                    operation = self._resolve_referenced_attribute(
                        value=value,
                        attribute=name,
                        path=(name,),
                    )
                    if not inspect.ismethod(operation):
                        msg = (
                            "Cannot durably fingerprint CertaintyEquivalent operation "
                            f"{type(value).__module__}.{type(value).__qualname__}."
                            f"{name}."
                        )
                        raise TypeError(msg)
                self._visit_referenced_bound_method(operation)
        self.frame(label="certainty-equivalent-end")
        return True

    def _visit_named_state(self, *, owner: object, state: Mapping[str, object]) -> None:
        entries = {
            name: member
            for name, member in state.items()
            if not _exclude_field(owner=owner, field_name=name)
        }
        self.frame(label="state-start", payload=str(len(entries)).encode())
        for name in sorted(entries):
            self.frame(label="state-field", payload=name.encode())
            self.visit(value=entries[name])
        self.frame(label="state-end")


def _exclude_field(*, owner: object, field_name: str) -> bool:
    """Whether one field is non-semantic for this precise owner type."""
    owner_type = type(owner)
    if (
        _has_exact_type(value=owner, candidates=_TRUSTED_GRID_EXECUTION_TYPE_OBJECTS)
        and field_name in _GRID_EXECUTION_FIELDS
    ):
        return True
    for registered_type, fields in _BUILTIN_EXECUTION_FIELDS_BY_TYPE:
        if owner_type is registered_type:
            return field_name in fields
    return False


def _is_shipped_pylcm_module_name(name: str) -> bool:
    return name in {"_lcm", "lcm"} or name.startswith(("_lcm.", "lcm."))


def _nested_code_names(code: types.CodeType) -> frozenset[str]:
    names = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names.update(_nested_code_names(constant))
    return frozenset(names)


def _capture_shipped_beartype_wrappers() -> tuple[  # noqa: C901, PLR0912
    tuple[types.FunctionType, types.CodeType, types.FunctionType], ...
]:
    """Capture exact wrapper/code/wrappee identities from shipped dependency roots."""
    explicit_roots = (
        certainty_equivalent_declarations,
        constraint_ir,
        functools_declarations,
        golden_section_declarations,
        grid_declarations,
        implicit_outer_declarations,
        interp_declarations,
        koopmans_declarations,
        power_mean_declarations,
        probability_declarations,
        zero_safe_declarations,
    )
    loaded_roots = tuple(
        module
        for name, module in tuple(sys.modules.items())
        if isinstance(module, types.ModuleType) and _is_shipped_pylcm_module_name(name)
    )
    queue: list[object] = [*explicit_roots, *loaded_roots]
    seen: set[int] = set()
    captures: list[tuple[types.FunctionType, types.CodeType, types.FunctionType]] = []

    while queue:
        value = queue.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)

        if isinstance(value, types.ModuleType):
            if _is_shipped_pylcm_module_name(value.__name__):
                queue.extend(tuple(vars(value).values()))
            continue

        if type(value) is _JAX_CUSTOM_JVP_TYPE:
            state = vars(value)
            queue.extend((state.get("fun"), state.get("jvp")))
            continue

        if isinstance(value, types.FunctionType):
            if not _is_shipped_pylcm_module_name(value.__module__):
                continue
            state = value.__dict__
            wrapped = state.get("__wrapped__")
            keyword_defaults = value.__kwdefaults__
            if (
                state.get("__beartype_wrapper") is True
                and isinstance(wrapped, types.FunctionType)
                and type(keyword_defaults) is dict
                and keyword_defaults.get("__beartype_func") is wrapped
                and value.__code__.co_filename.startswith("<@beartype(")
            ):
                captures.append((value, value.__code__, wrapped))
                queue.append(wrapped)
            queue.extend(
                value.__globals__[name]
                for name in _nested_code_names(value.__code__)
                if name in value.__globals__
            )
            if value.__closure__:
                for cell in value.__closure__:
                    try:
                        queue.append(cell.cell_contents)
                    except ValueError:
                        continue
            continue

        if isinstance(value, type) and _is_shipped_pylcm_module_name(value.__module__):
            for member in tuple(vars(value).values()):
                if isinstance(member, staticmethod | classmethod):
                    queue.append(member.__func__)
                elif isinstance(member, property):
                    queue.extend((member.fget, member.fset, member.fdel))
                else:
                    queue.append(member)

    return tuple(captures)


_TRUSTED_BEARTYPE_WRAPPER_CAPTURES = _capture_shipped_beartype_wrappers()


def _unwrap_exact_beartype_wrapper(
    function: types.FunctionType,
) -> types.FunctionType | None:
    """Return a wrapper's exact beartype-bound function or fail closed."""
    state = function.__dict__
    if state.get("__beartype_wrapper") is not True:
        return None
    capture = next(
        (
            candidate
            for candidate in _TRUSTED_BEARTYPE_WRAPPER_CAPTURES
            if function is candidate[0]
        ),
        None,
    )
    if capture is None:
        msg = (
            "Cannot durably fingerprint an uncaptured transparent beartype wrapper "
            f"{function.__module__}.{function.__qualname__}."
        )
        raise TypeError(msg)

    _, captured_code, captured_wrapped = capture
    wrapped = state.get("__wrapped__")
    keyword_defaults = function.__kwdefaults__
    is_exact_wrapper = (
        function.__code__ is captured_code
        and wrapped is captured_wrapped
        and type(keyword_defaults) is dict
        and keyword_defaults.get("__beartype_func") is captured_wrapped
        and function.__code__.co_filename.startswith("<@beartype(")
    )
    if not is_exact_wrapper:
        msg = (
            "Cannot durably fingerprint an inexact transparent beartype wrapper "
            f"{function.__module__}.{function.__qualname__}."
        )
        raise TypeError(msg)
    return captured_wrapped


def _is_solver_instance(value: object) -> bool:
    """Recognize the one stateless extension object sealed by public identity."""
    from _lcm.solution.contract import Solver  # noqa: PLC0415

    return isinstance(value, Solver)


def _slot_state(value: object) -> dict[str, object]:
    """Read inherited slot state without invoking arbitrary properties."""
    result: dict[str, object] = {}
    for owner in type(value).__mro__:
        declared = owner.__dict__.get("__slots__", ())
        names = (declared,) if isinstance(declared, str) else declared
        for raw_name in names:
            if raw_name in {"__dict__", "__weakref__"}:
                continue
            name = (
                f"_{owner.__name__.lstrip('_')}{raw_name}"
                if raw_name.startswith("__") and not raw_name.endswith("__")
                else raw_name
            )
            try:
                result[raw_name] = object.__getattribute__(value, name)
            except AttributeError:
                continue
    return result


def _contains_identity(*, value: object, candidates: Iterable[object]) -> bool:
    """Check a captured runtime allowlist without invoking overloaded equality."""
    return any(value is candidate for candidate in candidates)


def _has_exact_type(*, value: object, candidates: Iterable[object]) -> bool:
    """Classify a value by type identity without invoking metaclass equality."""
    return _contains_identity(value=type(value), candidates=candidates)


def _is_captured_jax_python_function(value: object) -> bool:
    """Whether this is an unchanged function captured from a public JAX module."""
    return isinstance(value, types.FunctionType) and any(
        value is candidate and value.__code__ is code
        for candidate, code in _JAX_NUMERIC_PYTHON_FUNCTIONS
    )


def _is_captured_jax_pjit(value: object) -> bool:
    """Whether this is an unchanged PjitFunction captured from public JAX APIs."""
    if type(value) is not _JAX_PJIT_FUNCTION_TYPE:
        return False
    function = vars(value).get("_fun")
    code = getattr(function, "__code__", None)
    return any(
        value is candidate and function is captured_function and code is captured_code
        for candidate, captured_function, captured_code in _JAX_NUMERIC_PJIT_FUNCTIONS
    )


def _native_numeric_versions() -> tuple[tuple[str, str], ...]:
    """Return the implementation seals for supported native numerical objects."""
    return (
        ("jax", jax.__version__),
        ("jaxlib", jaxlib.__version__),
        ("numpy", np.__version__),
    )


def _native_numeric_callable_kind(value: object) -> str | None:
    """Classify only native callables whose executable semantics can be sealed."""
    if type(value) is _JAX_CUSTOM_JVP_TYPE:
        return "jax-custom-jvp"
    if _is_captured_jax_python_function(value):
        return "jax-function"
    if type(value) is _JAX_PJIT_FUNCTION_TYPE:
        return "jax-pjit"
    if type(value) is _JAX_UFUNC_TYPE and _contains_identity(
        value=value, candidates=_JAX_NUMPY_UFUNCS
    ):
        return "jax-ufunc"
    if type(value) is np.ufunc and _contains_identity(
        value=value, candidates=_NUMPY_UFUNCS
    ):
        return "numpy-ufunc"
    return None


def _is_native_numeric_type(value: object) -> bool:
    """Whether a class is a captured genuine NumPy/JAX scalar constructor."""
    return _contains_identity(
        value=value,
        candidates=cast(
            "tuple[object, ...]", _JAX_NUMERIC_SCALAR_TYPES + _NUMPY_SCALAR_TYPES
        ),
    )


def _is_versioned_numeric_library_type(value: object) -> bool:
    """Whether a class is a captured genuine public NumPy/JAX class."""
    return _contains_identity(
        value=value,
        candidates=cast(
            "tuple[object, ...]",
            _JAX_PUBLIC_NUMERIC_TYPE_OBJECTS + _NUMPY_PUBLIC_TYPE_OBJECTS,
        ),
    )


def _is_closed_terminal_reference(  # noqa: C901, PLR0911, PLR0912
    *,
    value: object,
    _active: set[int] | None = None,
) -> bool:
    """Whether the semantic serializer closes direct use of this exact value."""
    if value is Ellipsis or value is _DATACLASSES_MISSING:
        return True
    if _dataclasses_field_marker_name(value) is not None:
        return True
    if value is None or _has_exact_type(
        value=value, candidates=(bool, int, float, str, bytes)
    ):
        return True
    if value is _beartype_decorator:
        return True
    if _has_exact_type(value=value, candidates=(slice, typing.TypeAliasType)):
        return True
    if _has_exact_type(
        value=value, candidates=_TRUSTED_TERMINAL_OBJECT_TYPES
    ) or isinstance(value, CertaintyEquivalent):
        return True
    if isinstance(value, Fraction | Enum | DiscreteGrid | Phased):
        return True
    if isinstance(value, Array | np.dtype):
        return True
    if isinstance(value, np.ndarray | np.generic):
        return True
    if _native_numeric_callable_kind(value) is not None or _is_native_numeric_type(
        value
    ):
        return True
    if inspect.isfunction(value) or inspect.isbuiltin(value):
        return True
    if isinstance(value, type):
        return (
            _contains_identity(value=value, candidates=_BUILTIN_TYPE_OBJECTS)
            or _contains_identity(value=value, candidates=_TRUSTED_DIRECT_TYPE_OBJECTS)
            or _is_versioned_numeric_library_type(value)
        )

    if not isinstance(value, functools.partial) and not _has_exact_type(
        value=value,
        candidates=(tuple, list, frozenset, set, dict, _MAPPING_PROXY_TYPE),
    ):
        return False

    active = set() if _active is None else _active
    identity = id(value)
    if identity in active:
        return False
    active.add(identity)
    try:
        if isinstance(value, functools.partial):
            return _is_closed_terminal_reference(
                value=value.func, _active=active
            ) and all(
                _is_closed_terminal_reference(value=item, _active=active)
                for item in (*value.args, *(value.keywords or {}).values())
            )
        if _has_exact_type(value=value, candidates=(tuple, list, frozenset, set)):
            return all(
                _is_closed_terminal_reference(value=item, _active=active)
                for item in cast("Iterable[object]", value)
            )
        mapping = cast("Mapping[object, object]", value)
        return all(
            _is_closed_terminal_reference(value=item, _active=active)
            for pair in mapping.items()
            for item in pair
        )
    finally:
        active.remove(identity)


def _dataclasses_field_marker_name(value: object) -> str | None:
    """Return the stable name of one exact stdlib dataclass field marker."""
    return next(
        (name for name, marker in _DATACLASSES_FIELD_MARKERS if value is marker),
        None,
    )


def _validate_static_attribute_access(*, value: object, path: tuple[str, ...]) -> None:
    """Reject lookup hooks whose runtime value static inspection cannot reproduce."""
    if isinstance(value, types.ModuleType):
        accessor = inspect.getattr_static(type(value), "__getattribute__", None)
        if accessor is types.ModuleType.__getattribute__:
            return
        dotted = ".".join(path)
        msg = (
            "Cannot durably fingerprint dynamic module attribute lookup on "
            f"{type(value).__module__}.{type(value).__qualname__} at referenced "
            f"attribute {dotted!r}."
        )
        raise TypeError(msg)
    value_type = type(value)
    accessor = inspect.getattr_static(value_type, "__getattribute__", None)
    standard_accessor = (
        type.__getattribute__ if isinstance(value, type) else object.__getattribute__
    )
    if accessor is not standard_accessor:
        dotted = ".".join(path)
        msg = (
            "Cannot durably fingerprint dynamic attribute lookup on "
            f"{value_type.__module__}.{value_type.__qualname__} at referenced "
            f"attribute {dotted!r}."
        )
        raise TypeError(msg)


def _bind_referenced_member(
    *, value: object, attribute: str, member: object, path: tuple[str, ...]
) -> object:
    """Apply only descriptor bindings whose semantics can be inspected statically."""
    binding_context = _descriptor_binding_context(
        value=value,
        attribute=attribute,
        member=member,
    )
    if binding_context is None:
        # A descriptor stored in an instance dictionary is an ordinary value:
        # Python does not invoke its ``__get__`` protocol there.
        return member

    binding_instance, binding_owner = binding_context
    resolved = member
    if type(member) is staticmethod:
        resolved = member.__func__
    elif type(member) is classmethod and inspect.isfunction(member.__func__):
        resolved = types.MethodType(member.__func__, binding_owner)
    elif inspect.isfunction(member):
        resolved = (
            member
            if binding_instance is None
            else types.MethodType(member, binding_instance)
        )
    elif (
        type(member) is types.MethodDescriptorType
        and binding_instance is not None
        and _has_exact_type(value=binding_instance, candidates=_BUILTIN_TYPE_OBJECTS)
    ):
        # Exact built-in descriptors have interpreter-sealed binding semantics.
        resolved = member.__get__(binding_instance, binding_owner)
    elif (
        isinstance(member, types.MemberDescriptorType) and binding_instance is not None
    ):
        try:
            resolved = member.__get__(binding_instance, binding_owner)
        except AttributeError as error:
            dotted = ".".join(path)
            msg = (
                "Cannot durably fingerprint unset slot at referenced attribute "
                f"{dotted!r}."
            )
            raise TypeError(msg) from error
    elif inspect.getattr_static(type(member), "__get__", None) is not None:
        dotted = ".".join(path)
        descriptor_type = f"{type(member).__module__}.{type(member).__qualname__}"
        msg = (
            "Cannot durably fingerprint dynamic descriptor "
            f"{descriptor_type} at referenced attribute {dotted!r}."
        )
        raise TypeError(msg)
    return resolved


def _descriptor_binding_context(
    *, value: object, attribute: str, member: object
) -> tuple[object | None, type] | None:
    """Return the standard descriptor arguments when ``member`` comes from a type."""
    sentinel = object()
    if not isinstance(value, type):
        try:
            instance_state = object.__getattribute__(value, "__dict__")
        except AttributeError:
            instance_state = None
        if isinstance(instance_state, Mapping) and (
            instance_state.get(attribute, sentinel) is member
        ):
            return None
        for owner in type(value).__mro__:
            if owner.__dict__.get(attribute, sentinel) is member:
                return value, type(value)
        return None

    for owner in value.__mro__:
        if owner.__dict__.get(attribute, sentinel) is member:
            return None, value
    for owner in type(value).__mro__:
        if owner.__dict__.get(attribute, sentinel) is member:
            return value, type(value)
    return None


def _bound_method_receiver_name(function: types.FunctionType) -> str:
    """Return the positional receiver name for one Python bound method."""
    if function.__code__.co_argcount < 1:
        msg = (
            "Cannot durably fingerprint bound method "
            f"{function.__module__}.{function.__qualname__}: no receiver argument."
        )
        raise TypeError(msg)
    return function.__code__.co_varnames[0]


def _is_annotationlib_thunk(code: types.CodeType) -> bool:
    """Whether code is Python 3.14's generated lazy-annotation evaluator."""
    return code.co_name == "__annotate__"


def _beartype_local_annotation_guard_offsets(
    instructions: tuple[dis.Instruction, ...],
) -> frozenset[int]:
    """Locate exact straight-line local-annotation guards injected by beartype."""
    guarded: set[int] = set()
    branch_opcodes = frozenset((*dis.hasjabs, *dis.hasjrel))
    terminators = {
        "POP_TOP",
        "RAISE_VARARGS",
        "RERAISE",
        "RETURN_CONST",
        "RETURN_VALUE",
        "YIELD_VALUE",
    }
    for start, instruction in enumerate(instructions):
        if (
            instruction.opname != "LOAD_GLOBAL"
            or instruction.argval != "__die_if_unbearable_beartype__"
            or instruction.is_jump_target
        ):
            continue

        for end in range(start + 1, len(instructions)):
            following = instructions[end]
            if (
                following.is_jump_target
                or following.opcode in branch_opcodes
                or following.opname in terminators
            ):
                break
            if following.opname in {"CALL", "CALL_KW"}:
                pop = end + 1
                if (
                    pop < len(instructions)
                    and instructions[pop].opname == "POP_TOP"
                    and not instructions[pop].is_jump_target
                ):
                    guarded.update(
                        item.offset for item in instructions[start : pop + 1]
                    )
                break
            if following.opname.startswith("CALL"):
                break
    return frozenset(guarded)


def _referenced_receiver_attribute_paths(
    *,
    code: types.CodeType,
    receiver_name: str,
    ignore_beartype_guards: bool = False,
) -> tuple[frozenset[tuple[str, ...]], bool]:
    """Collect static receiver paths and flag any opaque receiver consumption."""
    instructions = tuple(dis.get_instructions(code))
    ignored_offsets = (
        _beartype_local_annotation_guard_offsets(instructions)
        if ignore_beartype_guards
        else frozenset()
    )
    collected, has_opaque_use = _receiver_paths_from_instructions(
        instructions=instructions,
        receiver_name=receiver_name,
        ignored_offsets=ignored_offsets,
    )

    for constant in code.co_consts:
        if not isinstance(constant, types.CodeType):
            continue
        if _is_annotationlib_thunk(constant):
            continue
        if receiver_name not in constant.co_freevars:
            continue
        nested_paths, nested_opaque = _referenced_receiver_attribute_paths(
            code=constant,
            receiver_name=receiver_name,
            ignore_beartype_guards=ignore_beartype_guards,
        )
        collected.update(nested_paths)
        has_opaque_use |= nested_opaque
    return frozenset(collected), has_opaque_use


def _referenced_closure_attribute_paths(  # noqa: C901
    *,
    code: types.CodeType,
    ignore_beartype_guards: bool = False,
) -> dict[str, frozenset[tuple[str, ...]]]:
    """Collect static attribute chains rooted at closure-cell loads."""
    collected: dict[str, set[tuple[str, ...]]] = {}
    instructions = tuple(dis.get_instructions(code))
    ignored_offsets = (
        _beartype_local_annotation_guard_offsets(instructions)
        if ignore_beartype_guards
        else frozenset()
    )
    for index, instruction in enumerate(instructions):
        if instruction.offset in ignored_offsets:
            continue
        if instruction.opname not in {
            "LOAD_DEREF",
            "LOAD_CLASSDEREF",
            "LOAD_FROM_DICT_OR_DEREF",
        }:
            continue
        name = str(instruction.argval)
        attributes: list[str] = []
        for following in instructions[index + 1 :]:
            if following.offset in ignored_offsets:
                break
            if following.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
                break
            attributes.append(str(following.argval))
        collected.setdefault(name, set()).add(tuple(attributes))

    for constant in code.co_consts:
        if not isinstance(constant, types.CodeType):
            continue
        if _is_annotationlib_thunk(constant):
            continue
        nested = _referenced_closure_attribute_paths(
            code=constant,
            ignore_beartype_guards=ignore_beartype_guards,
        )
        for name, paths in nested.items():
            collected.setdefault(name, set()).update(paths)
    return {name: frozenset(paths) for name, paths in collected.items()}


def _receiver_paths_from_instructions(
    *,
    instructions: tuple[dis.Instruction, ...],
    receiver_name: str,
    ignored_offsets: frozenset[int] = frozenset(),
) -> tuple[set[tuple[str, ...]], bool]:
    """Extract immediate attribute chains rooted at one loaded local name."""
    collected: set[tuple[str, ...]] = set()
    has_opaque_use = False
    for index, instruction in enumerate(instructions):
        if instruction.offset in ignored_offsets:
            continue
        loaded_names = _loaded_local_names(instruction)
        if receiver_name not in loaded_names:
            continue
        # A fused load pushes its names in order. Only the final value can be
        # the receiver consumed by an immediately following attribute chain.
        if receiver_name in loaded_names[:-1]:
            has_opaque_use = True
        if loaded_names[-1] != receiver_name:
            continue
        attributes: list[str] = []
        for following in instructions[index + 1 :]:
            if following.offset in ignored_offsets:
                break
            if following.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
                break
            attributes.append(str(following.argval))
        if attributes:
            collected.add(tuple(attributes))
        else:
            has_opaque_use = True
    return collected, has_opaque_use


def _loaded_local_names(instruction: dis.Instruction) -> tuple[str, ...]:
    """Return local/cell names pushed by one regular or fused load opcode."""
    is_local_load = instruction.opname.startswith("LOAD_FAST")
    is_cell_load = instruction.opname in {
        "LOAD_DEREF",
        "LOAD_CLASSDEREF",
        "LOAD_FROM_DICT_OR_DEREF",
    }
    if not is_local_load and not is_cell_load:
        return ()
    value = instruction.argval
    if isinstance(value, tuple):
        return tuple(str(name) for name in value)
    return (str(value),)


def _referenced_global_names(code: types.CodeType) -> frozenset[str]:
    """Collect global-name candidates from a function and nested code."""
    return frozenset(_referenced_global_attribute_paths(code=code))


def _referenced_global_attribute_paths(  # noqa: C901
    *,
    code: types.CodeType,
    ignore_beartype_guards: bool = False,
) -> dict[str, frozenset[tuple[str, ...]]]:
    """Collect each global plus consecutive module attribute chains it reads."""
    collected: dict[str, set[tuple[str, ...]]] = {}
    instructions = tuple(dis.get_instructions(code))
    ignored_offsets = (
        _beartype_local_annotation_guard_offsets(instructions)
        if ignore_beartype_guards
        else frozenset()
    )
    for index, instruction in enumerate(instructions):
        if instruction.offset in ignored_offsets:
            continue
        if instruction.opname not in {"LOAD_GLOBAL", "LOAD_NAME"}:
            continue
        name = str(instruction.argval)
        attributes: list[str] = []
        for following in instructions[index + 1 :]:
            if following.offset in ignored_offsets:
                break
            if following.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
                break
            attributes.append(str(following.argval))
        collected.setdefault(name, set()).add(tuple(attributes))

    for constant in code.co_consts:
        if not isinstance(constant, types.CodeType):
            continue
        if _is_annotationlib_thunk(constant):
            continue
        nested = _referenced_global_attribute_paths(
            code=constant,
            ignore_beartype_guards=ignore_beartype_guards,
        )
        for name, paths in nested.items():
            collected.setdefault(name, set()).update(paths)

    return {name: frozenset(paths) for name, paths in collected.items()}


def _semantic_sort_key(value: object) -> tuple[str, str]:
    """Stable ordering key for unordered containers."""
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    try:
        digest = _semantic_fingerprint(value)
    except TypeError, ValueError:
        # Keys in pylcm's semantic mappings are strings, ints, tuples and small
        # frozen dataclasses. This fallback keeps diagnostics possible for an
        # unsupported user key without feeding its repr into the fingerprint.
        digest = type_name
    return type_name, digest


__all__ = ["fingerprint_model", "project_solution_params"]
