"""Collision-focused tests for durable solution fingerprints."""

import dataclasses
import inspect
from abc import ABCMeta
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from functools import partial
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import cast

import dags.exceptions as dags_exceptions
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pytest

import _lcm.solution.grid_search as grid_search_declarations
from _lcm.certainty_equivalent import CertaintyEquivalent
from _lcm.constraints.ir import And
from _lcm.engine import Regime as EngineRegime
from _lcm.identity_transition import _IdentityTransition
from _lcm.solution import fingerprint as fingerprints
from _lcm.typing import FlatParams, RegimeNamesToIds
from _lcm.utils.functools import allow_args
from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinearExpectation,
    LinSpacedGrid,
    Phased,
    case_boundary,
    categorical,
    ref,
)
from lcm.regime import Regime as UserRegime
from lcm.solver_api import (
    ArtifactChannel,
    ArtifactDescriptor,
    ArtifactKey,
    AxisDescriptor,
    AxisRole,
    PersistencePolicy,
    SolverIdentity,
)
from lcm.solvers import (
    FiniteOuterGrid,
    FUESEnvelope,
    SolutionKernels,
    Solver,
    SolverBuildContext,
)
from lcm.typing import ContinuousState, FloatND, IntND, ScalarInt
from tests.test_models.taste_shocks_toy import (
    get_model as get_toy_model,
)
from tests.test_models.taste_shocks_toy import (
    get_params as get_toy_params,
)

_fingerprint_model_for_test = inspect.unwrap(fingerprints.fingerprint_model)


@categorical(ordered=False)
class _LowHigh:
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _HighLow:
    high: ScalarInt
    low: ScalarInt


@categorical(ordered=True)
class _OrderedLowHigh:
    low: ScalarInt
    high: ScalarInt


_GLOBAL_OFFSET = 1
_CE_OFFSET = 1
_SPOOFED_CE_OFFSET = 1
_LIST_APPEND_DEPENDENCY = [1]
_HELPER_MODULE = ModuleType("tests.fingerprint_helper")
_HELPER_MODULE.OFFSET = 1  # ty: ignore[unresolved-attribute]


def _module_utility_a(value: int) -> int:
    return value + 1


def _module_utility_b(value: int) -> int:
    return value + 2


# keyword-only-exempt: library-callback=builtins.classmethod
def _module_class_utility_a(cls: type, value: int) -> int:  # noqa: ARG001
    return value + 1


_HELPER_MODULE.utility = _module_utility_a  # ty: ignore[unresolved-attribute]


class _DynamicModuleDependency(ModuleType):
    live_offset = 1

    def __getattribute__(self, name: str) -> object:
        if name == "OFFSET":
            return type(self).live_offset
        return super().__getattribute__(name)


_DYNAMIC_HELPER_MODULE = _DynamicModuleDependency("tests.dynamic_fingerprint_helper")
_DYNAMIC_HELPER_MODULE.OFFSET = 0  # ty: ignore[unresolved-attribute]


class _ClassDependency:
    utility = staticmethod(_module_utility_a)
    profiling_epoch = 1


@dataclass
class _ObjectDependency:
    offset: int
    profiling_epoch: int


_OBJECT_DEPENDENCY = _ObjectDependency(offset=1, profiling_epoch=1)


@dataclass
class _ObjectMethodDependency:
    offset: int
    profiling_epoch: int

    def utility(self, value: int) -> int:
        return value + self.offset


_OBJECT_METHOD_DEPENDENCY = _ObjectMethodDependency(offset=1, profiling_epoch=1)
_HELPER_MODULE.object_dependency = _OBJECT_METHOD_DEPENDENCY  # ty: ignore[unresolved-attribute]


class _StatelessObjectMethodDependency:
    offset = 1

    def utility(self, value: int) -> int:
        return value + self.offset


_STATELESS_OBJECT_METHOD_DEPENDENCY = _StatelessObjectMethodDependency()


class _ClassClosureMethodDependency:
    offset = 1

    def utility(self, value: int) -> int:
        return value + __class__.offset


_CLASS_CLOSURE_METHOD_DEPENDENCY = _ClassClosureMethodDependency()


class _BaseMethodDependency:
    def utility(self, value: int) -> int:
        return value + 1


class _SuperMethodDependency(_BaseMethodDependency):
    def utility(self, value: int) -> int:
        return super().utility(value)


_SUPER_METHOD_DEPENDENCY = _SuperMethodDependency()


class _ClassMethodDependency:
    offset = 1
    profiling_epoch = 1

    @classmethod
    def utility(cls, value: int) -> int:
        return value + cls.offset


class _MetaClassMethodDependency(type):
    offset = 1

    @classmethod
    def utility(cls, value: int) -> int:
        return value + cls.offset


class _ClassUsingMeta(metaclass=_MetaClassMethodDependency):
    offset = 100


class _RecursiveClassDependency:
    offset = 1

    @staticmethod
    def utility(value: int) -> int:
        if value:
            return _RecursiveClassDependency.utility(value - 1)
        return value + _RecursiveClassDependency.offset


class _DynamicDescriptor:
    # keyword-only-exempt: library-callback=builtins.object.__getattribute__
    def __get__(self, instance: object, owner: type) -> int:
        return 1


class _DescriptorDependency:
    dynamic = _DynamicDescriptor()


class _DynamicStaticMethod(staticmethod):
    # keyword-only-exempt: library-callback=builtins.staticmethod.__get__
    def __get__(self, instance: object, owner: type | None = None):
        return _module_utility_b


class _DynamicClassMethod(classmethod):
    # keyword-only-exempt: library-callback=builtins.classmethod.__get__
    def __get__(self, instance: object, owner: type | None = None):
        return _module_utility_b


class _DynamicMethodDescriptorDependency:
    static_utility = _DynamicStaticMethod(_module_utility_a)
    class_utility = _DynamicClassMethod(_module_class_utility_a)


class _CustomLookupDependency:
    value = 1

    def __getattribute__(self, name: str) -> object:
        if name == "value":
            return 2
        return object.__getattribute__(self, name)


_CUSTOM_LOOKUP_DEPENDENCY = _CustomLookupDependency()


class _DirectClassDependency:
    def __init__(self, value: int) -> None:
        self.value = value


_HELPER_MODULE.DirectClass = _DirectClassDependency  # ty: ignore[unresolved-attribute]


class _DirectCallableClassDependency:
    offset = 1

    def __new__(cls, value: int) -> int:
        return value + cls.offset


class _AnnotationOnlyDependency:
    metadata = 1


class _ModuleCallableDependency:
    def __call__(self, value: int) -> int:
        return value + 1


_ModuleCallableDependency.__module__ = "jax.numpy"
_HELPER_MODULE.callable_dependency = (  # ty: ignore[unresolved-attribute]
    _ModuleCallableDependency()
)
_OPAQUE_CALLABLE_DEPENDENCY = staticmethod(_module_utility_a)


class _SpoofedBuiltinClassDependency:
    def __init__(self, value: int) -> None:
        self.value = value


_SpoofedBuiltinClassDependency.__module__ = "builtins"


class _BuiltinEqualitySpoofMeta(type):
    def __hash__(cls) -> int:
        return hash(int)

    def __eq__(cls, other: object) -> bool:
        return cls is other or other is int


class _EqualitySpoofedBuiltinClassDependency(metaclass=_BuiltinEqualitySpoofMeta):
    pass


class _DirectObjectDependency:
    def __add__(self, value: int) -> int:
        return value + 1


_DIRECT_OBJECT_DEPENDENCY = _DirectObjectDependency()
_CYCLIC_CONTAINER_DEPENDENCY: list[object] = []
_CYCLIC_CONTAINER_DEPENDENCY.append(_CYCLIC_CONTAINER_DEPENDENCY)
_FRACTION_DEPENDENCY = Fraction(1, 3)


class _SemanticFraction(Fraction):
    offset = 1

    def __radd__(self, value: complex) -> complex:  # ty: ignore[invalid-method-override]
        return value + type(self).offset


_SEMANTIC_FRACTION = _SemanticFraction(1, 3)


class _SemanticArray(np.ndarray):
    offset = 1

    def __jax_array__(self):
        return jnp.asarray(type(self).offset)


_SEMANTIC_ARRAY = np.asarray(1.0).view(_SemanticArray)


class _SemanticMapping(dict[str, int]):
    offset = 1

    def shift(self) -> int:
        return type(self).offset


class _DictEqualitySpoofMeta(type):
    def __hash__(cls) -> int:
        return hash(dict)

    def __eq__(cls, other: object) -> bool:
        return cls is other or other is dict


class _EqualitySpoofedMapping(
    dict[str, int],
    metaclass=_DictEqualitySpoofMeta,
):
    pass


@dataclass(frozen=True)
class _ConfiguredFingerprintSolver(Solver):
    config: object

    @property
    def identity(self) -> SolverIdentity:
        return SolverIdentity(
            plugin_id="tests.configured-fingerprint",
            plugin_version="1",
        )

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        raise NotImplementedError


class _StatelessFingerprintSolver(Solver):
    @property
    def identity(self) -> SolverIdentity:
        return SolverIdentity(
            plugin_id="tests.stateless-fingerprint",
            plugin_version="1",
        )

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        raise NotImplementedError


class _DynamicLookupFingerprintSolver(_StatelessFingerprintSolver):
    def __getattribute__(self, name: str) -> object:
        return super().__getattribute__(name)


class _DiscreteGridSubclass(DiscreteGrid):
    pass


class _PhasedSubclass(Phased[object, object]):
    pass


class _EnumDependency(Enum):
    LEFT = 1
    RIGHT = 2


_ENUM_DEPENDENCY = _EnumDependency.LEFT


def _reads_global(value: int) -> int:
    return value + _GLOBAL_OFFSET


def _reads_module_constant(value: int) -> int:
    return value + _HELPER_MODULE.OFFSET


def _reads_module_function(value: int) -> int:
    return _HELPER_MODULE.utility(value)


def _reads_dynamic_module_constant(value: int) -> int:
    return value + cast("int", _DYNAMIC_HELPER_MODULE.OFFSET)


def _dispatch_module_constant(*, module: ModuleType, value: int) -> int:
    return value + module.OFFSET


def _passes_module_to_helper(value: int) -> int:
    return _dispatch_module_constant(module=_HELPER_MODULE, value=value)


def _reads_module_default(*, value: int, module: ModuleType = _HELPER_MODULE) -> int:
    return value + module.OFFSET


def _reads_instance_method_default(
    *, value: int, helper: _ObjectMethodDependency = _OBJECT_METHOD_DEPENDENCY
) -> int:
    return helper.utility(value)


def _reads_class_default(
    *, value: int, helper: type[_ClassDependency] = _ClassDependency
) -> int:
    return helper.utility(value)


def _reads_class_function(value: int) -> int:
    return _ClassDependency.utility(value)


def _reads_object_attribute(value: int) -> int:
    return value + _OBJECT_DEPENDENCY.offset


def _reads_object_method(value: int) -> int:
    return _OBJECT_METHOD_DEPENDENCY.utility(value)


def _reads_module_object_method(value: int) -> int:
    return _HELPER_MODULE.object_dependency.utility(value)


def _reads_class_closure_method(value: int) -> int:
    return _CLASS_CLOSURE_METHOD_DEPENDENCY.utility(value)


def _reads_super_method(value: int) -> int:
    return _SUPER_METHOD_DEPENDENCY.utility(value)


def _reads_class_method(value: int) -> int:
    return _ClassMethodDependency.utility(value)


def _reads_metaclass_method(value: int) -> int:
    return _ClassUsingMeta.utility(value)


def _reads_recursive_class_function(value: int) -> int:
    return _RecursiveClassDependency.utility(value)


def _reads_dynamic_descriptor() -> int:
    return _DescriptorDependency.dynamic


def _reads_custom_lookup() -> object:
    return _CUSTOM_LOOKUP_DEPENDENCY.value


def _constructs_direct_class(value: int) -> _DirectClassDependency:
    return _DirectClassDependency(value)


def _constructs_module_class(value: int) -> _DirectClassDependency:
    return _HELPER_MODULE.DirectClass(value)


def _calls_module_assigned_callable(value: int) -> int:
    return _HELPER_MODULE.callable_dependency(value)


def _constructs_spoofed_builtin_class(value: int) -> _SpoofedBuiltinClassDependency:
    return _SpoofedBuiltinClassDependency(value)


def _uses_direct_object(value: int) -> int:
    return _DIRECT_OBJECT_DEPENDENCY + value


def _reads_dynamic_static_method(value: int) -> int:
    return _DynamicMethodDescriptorDependency.static_utility(value)


def _reads_dynamic_class_method(value: int) -> int:
    return _DynamicMethodDescriptorDependency.class_utility(value)


def _annotated_identity(
    value: _AnnotationOnlyDependency,
) -> _AnnotationOnlyDependency:
    return value


def _uses_direct_cyclic_container() -> object:
    return _CYCLIC_CONTAINER_DEPENDENCY


def _uses_fraction_dependency() -> Fraction:
    return _FRACTION_DEPENDENCY


def _uses_fraction_subclass_dependency(value: int) -> complex:
    return value + _SEMANTIC_FRACTION


def _uses_array_subclass_dependency(value: int):
    return value + _SEMANTIC_ARRAY


def _uses_enum_dependency() -> _EnumDependency:
    return _ENUM_DEPENDENCY


def _uses_jnp_exp(value: float):
    return jnp.exp(value)


def _uses_jnp_maximum(value: float):
    return jnp.maximum(value, 0)


def _uses_jnp_logaddexp(value: float):
    return jnp.logaddexp(value, 0)


def _uses_np_exp(value: float):
    return np.exp(value)


def _uses_jnp_int32(value: float):
    return jnp.int32(value)


def _uses_jnp_where(value: float):
    return jnp.where(value > 0, value, 0)


def _uses_jnp_linalg_norm(value: float):
    return jnp.linalg.norm(value)


def _uses_jax_nn_sigmoid(value: float):
    return jax.nn.sigmoid(value)


def _uses_jax_scipy_expit(value: float):
    return jsp.special.expit(value)


def _uses_jax_segment_sum(value: float):
    return jax.ops.segment_sum(jnp.asarray([value]), jnp.asarray([0]))


def _closure(offset: int):
    def add(value: int) -> int:
        return value + offset

    return add


def _add(*, left: int, right: int) -> int:
    return left + right


def _dependency_closure(dependency: object) -> Callable[[], object]:
    def use_dependency() -> object:
        return dependency

    return use_dependency


def _forged_beartype_wrappee(value: int) -> int:
    return value


def _forge_beartype_wrapper(delta: int) -> Callable[[int], int]:
    # keyword-only-exempt: primary-argument=value
    def fake(
        value: int,
        *,
        __beartype_func=_forged_beartype_wrappee,  # ty: ignore[invalid-legacy-positional-parameter]
    ) -> int:
        return __beartype_func(value) + delta

    fake.__dict__["__beartype_wrapper"] = True
    fake.__wrapped__ = _forged_beartype_wrappee  # ty: ignore[unresolved-attribute]
    fake.__module__ = "tests.solution.test_solution_fingerprint"
    fake.__qualname__ = "forged"
    fake.__name__ = "forged"
    fake.__code__ = fake.__code__.replace(co_filename="<@beartype(forged)>")
    return fake


def _takes_zero_arg_callback(callback: Callable[[], float]) -> float:
    return callback()


def _uses_builtin_list_descriptor(value: int) -> None:
    _LIST_APPEND_DEPENDENCY.append(value)


def _custom_jvp_primal(value: FloatND) -> FloatND:
    return value


# keyword-only-exempt: library-callback=jax.custom_jvp.defjvp
def _custom_jvp_rule_a(primals, tangents):
    (value,), (tangent,) = primals, tangents
    return value, tangent


# keyword-only-exempt: library-callback=jax.custom_jvp.defjvp
def _custom_jvp_rule_b(primals, tangents):
    (value,), (tangent,) = primals, tangents
    return value, 2 * tangent


def _with_custom_jvp(rule: Callable[..., tuple[FloatND, FloatND]]) -> object:
    transform = jax.custom_jvp(_custom_jvp_primal)
    transform.defjvp(rule)
    return transform


class _FingerprintCertaintyEquivalent(CertaintyEquivalent):
    @property
    def param_names(self) -> frozenset[str]:
        return frozenset()

    def aggregate(
        self,
        *,
        values: FloatND,
        weights: FloatND,
        params: Mapping[str, FloatND],
    ) -> FloatND:
        del weights, params
        return jnp.asarray(values) + _CE_OFFSET

    def aggregate_scaled(
        self,
        *,
        values: FloatND,
        coefficients: FloatND,
        shifts: IntND,
        params: Mapping[str, FloatND],
    ) -> FloatND:
        return LinearExpectation().aggregate_scaled(
            values=values, coefficients=coefficients, shifts=shifts, params=params
        )


class _CertaintyEquivalentEqualitySpoofMeta(ABCMeta):
    def __hash__(cls) -> int:
        return hash(LinearExpectation)

    def __eq__(cls, other: object) -> bool:
        return cls is other or other is LinearExpectation


class _EqualitySpoofedCertaintyEquivalent(
    CertaintyEquivalent,
    metaclass=_CertaintyEquivalentEqualitySpoofMeta,
):
    @property
    def param_names(self) -> frozenset[str]:
        return frozenset()

    def aggregate(
        self,
        *,
        values: FloatND,
        weights: FloatND,
        params: Mapping[str, FloatND],
    ) -> FloatND:
        del weights, params
        return jnp.asarray(values) + _SPOOFED_CE_OFFSET

    def aggregate_scaled(
        self,
        *,
        values: FloatND,
        coefficients: FloatND,
        shifts: IntND,
        params: Mapping[str, FloatND],
    ) -> FloatND:
        return LinearExpectation().aggregate_scaled(
            values=values, coefficients=coefficients, shifts=shifts, params=params
        )


_EqualitySpoofedCertaintyEquivalent.__module__ = LinearExpectation.__module__
_EqualitySpoofedCertaintyEquivalent.__name__ = LinearExpectation.__name__
_EqualitySpoofedCertaintyEquivalent.__qualname__ = LinearExpectation.__qualname__


# keyword-only-exempt: primary-argument=helper
def _dispatch_object_method(
    helper: _StatelessObjectMethodDependency, *, value: int
) -> int:
    return helper.utility(value)


def _solve_law(value: int) -> int:
    return value


def _other_solve_law(value: int) -> int:
    return value + 1


def _truth_law_a(value: int) -> int:
    return value * 2


def _truth_law_b(value: int) -> int:
    return value * 3


def _terminal_utility() -> float:
    return 0.0


def _generic_solve_callable(**params: object) -> object:
    return params


@dataclass(frozen=True, slots=True)
class _CallableDataclass:
    scale: int
    batch_size: int

    def __call__(self, value: int) -> int:
        return self.scale * value + self.batch_size


@dataclass(frozen=True)
class _CallableDataclassClassDependency:
    marker: int = 0
    offset = 1

    def __call__(self, value: int) -> int:
        return value + self.offset


@dataclass(frozen=True)
class _SolverModuleNeighbor:
    batch_size: int


_SolverModuleNeighbor.__module__ = "_lcm.solution.nbegm"


@dataclass(frozen=True)
class _GridSearchNominalTwin:
    action_block_width: int


_GridSearchNominalTwin.__module__ = "_lcm.solution.grid_search"
_GridSearchNominalTwin.__qualname__ = "GridSearch"


class _SlotCallable:
    __slots__ = ("shift",)

    def __init__(self, shift: int) -> None:
        self.shift = shift

    def __call__(self, value: int) -> int:
        return value + self.shift

    def apply(self, value: int) -> int:
        return self(value)


class _StatelessCallableDependency:
    offset = 1

    def __call__(self, value: int) -> int:
        return value + self.offset


_STATELESS_CALLABLE_DEPENDENCY = _StatelessCallableDependency()


def _callable_with_params(*names: str):
    def func() -> None:
        return None

    func.__signature__ = inspect.Signature(  # ty: ignore[unresolved-attribute]
        [inspect.Parameter(name, kind=inspect.Parameter.KEYWORD_ONLY) for name in names]
    )
    return func


def _code_variant(*, increment: bool):
    if increment:

        def transform(value: int) -> int:
            return value + 1

    else:

        def transform(value: int) -> int:
            return value * 1

    return transform


def test_discrete_grid_fingerprint_binds_category_order_and_ordered_flag() -> None:
    baseline = fingerprints._semantic_fingerprint(DiscreteGrid(category_class=_LowHigh))

    assert baseline != fingerprints._semantic_fingerprint(
        DiscreteGrid(category_class=_HighLow)
    )
    assert baseline != fingerprints._semantic_fingerprint(
        DiscreteGrid(category_class=_OrderedLowHigh)
    )


def test_discrete_grid_protocol_subclass_fails_closed() -> None:
    with pytest.raises(TypeError, match="DiscreteGrid subclass"):
        fingerprints._semantic_fingerprint(
            _DiscreteGridSubclass(category_class=_LowHigh)
        )


def test_grid_execution_policy_does_not_change_semantic_fingerprint() -> None:
    unbatched = DiscreteGrid(category_class=_LowHigh, batch_size=0)
    batched = DiscreteGrid(category_class=_LowHigh, batch_size=1)

    assert fingerprints._semantic_fingerprint(
        unbatched
    ) == fingerprints._semantic_fingerprint(batched)


def test_builtin_execution_fields_are_excluded_only_on_their_owner_types() -> None:
    assert fingerprints._semantic_fingerprint(
        FUESEnvelope(scan_unroll=1)
    ) == fingerprints._semantic_fingerprint(FUESEnvelope(scan_unroll=4))

    mesh = LinSpacedGrid(start=0, stop=1, n_points=3)
    assert fingerprints._semantic_fingerprint(
        FiniteOuterGrid(grid=mesh, batch_size=1)
    ) == fingerprints._semantic_fingerprint(FiniteOuterGrid(grid=mesh, batch_size=2))

    assert fingerprints._semantic_fingerprint(
        _SolverModuleNeighbor(batch_size=1)
    ) != fingerprints._semantic_fingerprint(_SolverModuleNeighbor(batch_size=2))


def test_execution_field_exclusion_requires_exact_owner_type_identity() -> None:
    left = fingerprints._semantic_fingerprint(_GridSearchNominalTwin(1))
    right = fingerprints._semantic_fingerprint(_GridSearchNominalTwin(2))

    assert left != right


def test_execution_field_exclusion_ignores_module_rebinding(*, monkeypatch) -> None:
    original_grid_search_type = grid_search_declarations.GridSearch

    monkeypatch.setattr(
        grid_search_declarations,
        "GridSearch",
        _GridSearchNominalTwin,
    )

    assert fingerprints._exclude_field(
        owner=original_grid_search_type(),
        field_name="action_block_width",
    )

    left = fingerprints._semantic_fingerprint(_GridSearchNominalTwin(1))
    right = fingerprints._semantic_fingerprint(_GridSearchNominalTwin(2))
    assert left != right


def test_phased_fingerprint_binds_solve_but_not_simulate_truth() -> None:
    baseline = Phased(solve=_solve_law, simulate=_truth_law_a)
    new_truth = Phased(solve=_solve_law, simulate=_truth_law_b)
    new_belief = Phased(solve=_other_solve_law, simulate=_truth_law_a)

    assert fingerprints._semantic_fingerprint(
        baseline
    ) == fingerprints._semantic_fingerprint(new_truth)
    assert fingerprints._semantic_fingerprint(
        baseline
    ) != fingerprints._semantic_fingerprint(new_belief)


def test_phased_protocol_subclass_fails_closed() -> None:
    with pytest.raises(TypeError, match="Phased subclass"):
        fingerprints._semantic_fingerprint(
            _PhasedSubclass(solve=_solve_law, simulate=_truth_law_a)
        )


def test_regime_description_is_not_mathematical_identity() -> None:
    regime = UserRegime(
        transition=None,
        functions={"utility": _terminal_utility},
        description="first wording",
    )
    reworded = dataclasses.replace(regime, description="second wording")

    assert fingerprints._semantic_fingerprint(
        fingerprints._project_user_regime_declaration(regime)
    ) == fingerprints._semantic_fingerprint(
        fingerprints._project_user_regime_declaration(reworded)
    )


def test_callable_signature_type_alias_metadata_is_supported() -> None:
    transition = _IdentityTransition(
        state_name="wealth",
        annotation=ContinuousState,
    )

    assert fingerprints._semantic_fingerprint(transition)


@pytest.mark.parametrize(
    ("baseline", "changed"),
    [
        (slice(0, 3, 1), slice(0, 4, 1)),
        (
            LinSpacedGrid(start=0, stop=1, n_points=3),
            LinSpacedGrid(start=0, stop=2, n_points=3),
        ),
        (
            case_boundary(condition=ref("wealth") < 0, kind="jump"),
            case_boundary(condition=ref("wealth") < 1, kind="jump"),
        ),
    ],
    ids=["slice", "grid", "case-boundary"],
)
def test_trusted_frozen_dependencies_bind_their_semantic_state(
    *,
    baseline: object,
    changed: object,
) -> None:
    assert fingerprints._semantic_fingerprint(_dependency_closure(baseline)) != (
        fingerprints._semantic_fingerprint(_dependency_closure(changed))
    )


@pytest.mark.parametrize(
    "dependency",
    [
        LinSpacedGrid,
        np.dtype,
        partial,
        inspect.Signature.empty,
        And,
        dags_exceptions.InvalidFunctionArgumentsError,
        ContinuousState,
    ],
    ids=[
        "grid-class",
        "numpy-dtype-class",
        "partial-class",
        "inspect-empty",
        "constraint-class",
        "dags-exception-class",
        "type-alias",
    ],
)
def test_identity_gated_direct_dependencies_have_closed_fingerprints(
    dependency: object,
) -> None:
    assert fingerprints._semantic_fingerprint(_dependency_closure(dependency))


def test_callable_with_empty_argument_list_annotation_is_supported() -> None:
    assert fingerprints._semantic_fingerprint(_takes_zero_arg_callback)


def test_exact_builtin_method_descriptor_binds_its_receiver(*, monkeypatch) -> None:
    baseline = fingerprints._semantic_fingerprint(_uses_builtin_list_descriptor)
    monkeypatch.setitem(globals(), "_LIST_APPEND_DEPENDENCY", [2])

    assert baseline != fingerprints._semantic_fingerprint(_uses_builtin_list_descriptor)


def test_custom_jvp_fingerprint_binds_the_registered_derivative_rule() -> None:
    baseline = _with_custom_jvp(_custom_jvp_rule_a)
    changed = _with_custom_jvp(_custom_jvp_rule_b)

    assert fingerprints._semantic_fingerprint(baseline) != (
        fingerprints._semantic_fingerprint(changed)
    )


def test_shipped_beartype_local_annotation_guards_are_transparent() -> None:
    assert fingerprints._semantic_fingerprint(allow_args)


def test_forged_beartype_wrappers_fail_closed() -> None:
    baseline = _forge_beartype_wrapper(0)
    changed = _forge_beartype_wrapper(1)
    assert baseline(2) != changed(2)

    for function in (baseline, changed):
        with pytest.raises(TypeError, match="uncaptured transparent beartype wrapper"):
            fingerprints._semantic_fingerprint(function)


def test_captured_beartype_wrapper_code_mutation_fails_closed() -> None:
    original_code = allow_args.__code__
    try:
        allow_args.__code__ = _forged_beartype_wrappee.__code__
        with pytest.raises(TypeError, match="inexact transparent beartype wrapper"):
            fingerprints._semantic_fingerprint(allow_args)
    finally:
        allow_args.__code__ = original_code


def test_captured_inspect_signature_code_mutation_fails_closed() -> None:
    original_code = inspect.signature.__code__
    try:
        inspect.signature.__code__ = _forged_beartype_wrappee.__code__
        with pytest.raises(TypeError, match="captured code identity changed"):
            fingerprints._semantic_fingerprint(inspect.signature)
    finally:
        inspect.signature.__code__ = original_code


def test_stdlib_dataclasses_replace_has_a_closed_fingerprint() -> None:
    assert fingerprints._semantic_fingerprint(dataclasses.replace) == (
        fingerprints._semantic_fingerprint(dataclasses.replace)
    )


def test_nominal_dataclasses_field_marker_cannot_spoof_captured_identity(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = vars(dataclasses)["_FIELD_CLASSVAR"]
    nominal_marker = type(marker)(repr(marker))
    monkeypatch.setitem(vars(dataclasses), "_FIELD_CLASSVAR", nominal_marker)

    with pytest.raises(
        TypeError,
        match=r"direct object dependency dataclasses[.]_FIELD_BASE",
    ):
        fingerprints._semantic_fingerprint(dataclasses.replace)


def test_noncaptured_inspect_callable_remains_fail_closed() -> None:
    with pytest.raises(TypeError, match="ForwardRef"):
        fingerprints._semantic_fingerprint(inspect.get_annotations)


def test_default_model_declaration_has_a_closed_fingerprint() -> None:
    model = get_toy_model()
    flat_params = model._process_params(get_toy_params(scale=1.0))

    digest = fingerprints.fingerprint_model(
        ages=model.ages,
        regimes=model._regimes,
        user_regimes=model.user_regimes,
        regime_names_to_ids=model.regime_names_to_ids,
        flat_params=flat_params,
    )

    assert len(digest) == 64


def test_user_certainty_equivalent_fingerprint_binds_protocol_code(
    *, monkeypatch
) -> None:
    dependency = _FingerprintCertaintyEquivalent()
    baseline = fingerprints._semantic_fingerprint(_dependency_closure(dependency))
    monkeypatch.setitem(globals(), "_CE_OFFSET", 2)

    assert baseline != fingerprints._semantic_fingerprint(
        _dependency_closure(dependency)
    )


def test_ce_type_name_and_metaclass_equality_cannot_spoof_shipped_identity(
    *, monkeypatch
) -> None:
    dependency = _EqualitySpoofedCertaintyEquivalent()
    baseline = fingerprints._semantic_fingerprint(_dependency_closure(dependency))
    monkeypatch.setitem(globals(), "_SPOOFED_CE_OFFSET", 2)

    assert baseline != fingerprints._semantic_fingerprint(
        _dependency_closure(dependency)
    )


def test_function_fingerprint_binds_globals_and_closure_cells(*, monkeypatch) -> None:
    baseline = fingerprints._semantic_fingerprint(_reads_global)
    monkeypatch.setitem(_reads_global.__globals__, "_GLOBAL_OFFSET", 2)

    assert baseline != fingerprints._semantic_fingerprint(_reads_global)
    assert fingerprints._semantic_fingerprint(_closure(1)) != (
        fingerprints._semantic_fingerprint(_closure(2))
    )


def test_function_fingerprint_binds_module_qualified_dependencies(
    *, monkeypatch
) -> None:
    constant_baseline = fingerprints._semantic_fingerprint(_reads_module_constant)
    function_baseline = fingerprints._semantic_fingerprint(_reads_module_function)

    monkeypatch.setattr(_HELPER_MODULE, "OFFSET", 2)
    assert constant_baseline != fingerprints._semantic_fingerprint(
        _reads_module_constant
    )

    monkeypatch.setattr(_HELPER_MODULE, "utility", _module_utility_b)
    assert function_baseline != fingerprints._semantic_fingerprint(
        _reads_module_function
    )


def test_module_passed_as_value_to_helper_fails_closed() -> None:
    """A helper-local attribute lookup cannot hide mutable module semantics."""
    with pytest.raises(TypeError, match="direct module dependency"):
        fingerprints._semantic_fingerprint(_passes_module_to_helper)


def test_module_used_through_a_function_default_fails_closed() -> None:
    """A default-local attribute lookup cannot hide mutable module semantics."""
    with pytest.raises(TypeError, match="default dependency for parameter 'module'"):
        fingerprints._semantic_fingerprint(_reads_module_default)


def test_module_subclass_with_dynamic_lookup_fails_closed() -> None:
    with pytest.raises(TypeError, match="dynamic module attribute lookup"):
        fingerprints._semantic_fingerprint(_reads_dynamic_module_constant)


@pytest.mark.parametrize(
    "function",
    [_reads_instance_method_default, _reads_class_default],
)
def test_unsealed_method_receivers_in_function_defaults_fail_closed(
    function: object,
) -> None:
    """A local default receiver cannot hide a replaced method implementation."""
    with pytest.raises(TypeError, match="default dependency for parameter 'helper'"):
        fingerprints._semantic_fingerprint(function)


def test_function_fingerprint_binds_only_referenced_class_attributes(
    *, monkeypatch
) -> None:
    baseline = fingerprints._semantic_fingerprint(_reads_class_function)

    monkeypatch.setattr(_ClassDependency, "profiling_epoch", 2)
    assert baseline == fingerprints._semantic_fingerprint(_reads_class_function)

    monkeypatch.setattr(_ClassDependency, "utility", staticmethod(_module_utility_b))
    assert baseline != fingerprints._semantic_fingerprint(_reads_class_function)


def test_function_fingerprint_binds_only_referenced_object_attributes(
    *, monkeypatch
) -> None:
    baseline = fingerprints._semantic_fingerprint(_reads_object_attribute)

    monkeypatch.setattr(_OBJECT_DEPENDENCY, "profiling_epoch", 2)
    assert baseline == fingerprints._semantic_fingerprint(_reads_object_attribute)

    monkeypatch.setattr(_OBJECT_DEPENDENCY, "offset", 2)
    assert baseline != fingerprints._semantic_fingerprint(_reads_object_attribute)


def test_referenced_bound_method_binds_only_used_receiver_attributes(
    *, monkeypatch
) -> None:
    baseline = fingerprints._semantic_fingerprint(_reads_object_method)

    monkeypatch.setattr(_OBJECT_METHOD_DEPENDENCY, "profiling_epoch", 2)
    assert baseline == fingerprints._semantic_fingerprint(_reads_object_method)

    monkeypatch.setattr(_OBJECT_METHOD_DEPENDENCY, "offset", 2)
    assert baseline != fingerprints._semantic_fingerprint(_reads_object_method)


def test_direct_bound_method_binds_class_fallback_attributes(*, monkeypatch) -> None:
    baseline = fingerprints._semantic_fingerprint(
        _STATELESS_OBJECT_METHOD_DEPENDENCY.utility
    )

    monkeypatch.setattr(_StatelessObjectMethodDependency, "offset", 2)
    assert baseline != fingerprints._semantic_fingerprint(
        _STATELESS_OBJECT_METHOD_DEPENDENCY.utility
    )


def test_direct_classmethod_binds_class_attributes(*, monkeypatch) -> None:
    baseline = fingerprints._semantic_fingerprint(_ClassMethodDependency.utility)

    monkeypatch.setattr(_ClassMethodDependency, "offset", 2)
    assert baseline != fingerprints._semantic_fingerprint(
        _ClassMethodDependency.utility
    )


def test_module_qualified_bound_method_binds_used_receiver_attributes(
    *, monkeypatch
) -> None:
    baseline = fingerprints._semantic_fingerprint(_reads_module_object_method)

    monkeypatch.setattr(_OBJECT_METHOD_DEPENDENCY, "profiling_epoch", 2)
    assert baseline == fingerprints._semantic_fingerprint(_reads_module_object_method)

    monkeypatch.setattr(_OBJECT_METHOD_DEPENDENCY, "offset", 2)
    assert baseline != fingerprints._semantic_fingerprint(_reads_module_object_method)


def test_bound_method_binds_class_closure_attributes(*, monkeypatch) -> None:
    baseline = fingerprints._semantic_fingerprint(_reads_class_closure_method)

    monkeypatch.setattr(_ClassClosureMethodDependency, "offset", 2)
    assert baseline != fingerprints._semantic_fingerprint(_reads_class_closure_method)


def test_zero_argument_super_dependency_fails_closed() -> None:
    with pytest.raises(TypeError, match="direct class dependency"):
        fingerprints._semantic_fingerprint(_reads_super_method)


def test_referenced_classmethod_binds_only_used_class_attributes(
    *, monkeypatch
) -> None:
    baseline = fingerprints._semantic_fingerprint(_reads_class_method)

    monkeypatch.setattr(_ClassMethodDependency, "profiling_epoch", 2)
    assert baseline == fingerprints._semantic_fingerprint(_reads_class_method)

    monkeypatch.setattr(_ClassMethodDependency, "offset", 2)
    assert baseline != fingerprints._semantic_fingerprint(_reads_class_method)


def test_metaclass_classmethod_binds_the_metaclass_receiver(*, monkeypatch) -> None:
    baseline = fingerprints._semantic_fingerprint(_reads_metaclass_method)

    monkeypatch.setattr(_ClassUsingMeta, "offset", 101)
    assert baseline == fingerprints._semantic_fingerprint(_reads_metaclass_method)

    monkeypatch.setattr(_MetaClassMethodDependency, "offset", 2)
    assert baseline != fingerprints._semantic_fingerprint(_reads_metaclass_method)


def test_recursive_class_dependency_has_a_deterministic_closed_fingerprint(
    *, monkeypatch
) -> None:
    baseline = fingerprints._semantic_fingerprint(_reads_recursive_class_function)

    assert baseline == fingerprints._semantic_fingerprint(
        _reads_recursive_class_function
    )
    monkeypatch.setattr(_RecursiveClassDependency, "offset", 2)
    assert baseline != fingerprints._semantic_fingerprint(
        _reads_recursive_class_function
    )


def test_dynamic_descriptor_dependency_fails_closed() -> None:
    with pytest.raises(TypeError, match="dynamic descriptor"):
        fingerprints._semantic_fingerprint(_reads_dynamic_descriptor)


@pytest.mark.parametrize(
    "function",
    [_reads_dynamic_static_method, _reads_dynamic_class_method],
)
def test_dynamic_method_descriptor_subclasses_fail_closed(function: object) -> None:
    with pytest.raises(TypeError, match="dynamic descriptor"):
        fingerprints._semantic_fingerprint(function)


def test_custom_attribute_lookup_dependency_fails_closed() -> None:
    with pytest.raises(TypeError, match="dynamic attribute lookup"):
        fingerprints._semantic_fingerprint(_reads_custom_lookup)


def test_direct_class_dependency_fails_closed() -> None:
    with pytest.raises(TypeError, match="direct class dependency"):
        fingerprints._semantic_fingerprint(_constructs_direct_class)


def test_direct_callable_class_fails_closed() -> None:
    with pytest.raises(TypeError, match="direct class dependency"):
        fingerprints._semantic_fingerprint(_DirectCallableClassDependency)


def test_direct_categorical_class_fails_closed() -> None:
    with pytest.raises(TypeError, match="direct class dependency"):
        fingerprints._semantic_fingerprint(_LowHigh)


def test_custom_class_annotations_remain_supported(*, monkeypatch) -> None:
    baseline = fingerprints._semantic_fingerprint(_annotated_identity)

    monkeypatch.setattr(_AnnotationOnlyDependency, "metadata", 2)
    assert baseline == fingerprints._semantic_fingerprint(_annotated_identity)


def test_module_assigned_direct_class_dependency_fails_closed() -> None:
    with pytest.raises(TypeError, match="direct class dependency"):
        fingerprints._semantic_fingerprint(_constructs_module_class)


def test_module_assigned_callable_cannot_spoof_jax_provenance() -> None:
    with pytest.raises(TypeError, match="direct object dependency"):
        fingerprints._semantic_fingerprint(_calls_module_assigned_callable)


def test_non_python_callable_object_fails_closed() -> None:
    with pytest.raises(TypeError, match="non-Python __call__"):
        fingerprints._semantic_fingerprint(_OPAQUE_CALLABLE_DEPENDENCY)


def test_direct_class_dependency_cannot_spoof_a_builtin_identity() -> None:
    with pytest.raises(TypeError, match="direct class dependency"):
        fingerprints._semantic_fingerprint(_constructs_spoofed_builtin_class)


def test_direct_class_cannot_spoof_builtin_equality_membership() -> None:
    with pytest.raises(TypeError, match="direct class dependency"):
        fingerprints._semantic_fingerprint(_EqualitySpoofedBuiltinClassDependency)


def test_direct_object_protocol_dependency_fails_closed() -> None:
    with pytest.raises(TypeError, match="direct object dependency"):
        fingerprints._semantic_fingerprint(_uses_direct_object)


def test_direct_cyclic_container_dependency_fails_closed() -> None:
    with pytest.raises(TypeError, match="direct object dependency"):
        fingerprints._semantic_fingerprint(_uses_direct_cyclic_container)


def test_direct_fraction_dependency_has_a_closed_fingerprint(*, monkeypatch) -> None:
    baseline = fingerprints._semantic_fingerprint(_uses_fraction_dependency)

    monkeypatch.setitem(globals(), "_FRACTION_DEPENDENCY", Fraction(2, 3))
    assert baseline != fingerprints._semantic_fingerprint(_uses_fraction_dependency)


def test_fraction_protocol_subclass_dependency_fails_closed() -> None:
    with pytest.raises(TypeError, match="Fraction subclass"):
        fingerprints._semantic_fingerprint(_uses_fraction_subclass_dependency)


def test_numpy_protocol_subclass_dependency_fails_closed() -> None:
    with pytest.raises(TypeError, match="NumPy protocol subclass"):
        fingerprints._semantic_fingerprint(_uses_array_subclass_dependency)


def test_direct_enum_dependency_fails_closed() -> None:
    with pytest.raises(TypeError, match="custom Enum semantic value"):
        fingerprints._semantic_fingerprint(_uses_enum_dependency)


def test_engine_descriptor_enums_and_solver_identity_remain_supported() -> None:
    descriptor = ArtifactDescriptor(
        key=ArtifactKey(type_id="tests.compatibility", schema_version=1),
        channel=ArtifactChannel.REPLAY,
        persistence=PersistencePolicy.MODEL_VERIFIABLE,
        payload_type_id="jax.Array",
        named_axes=(
            AxisDescriptor(
                name="wealth",
                length=1,
                role=AxisRole.STATE,
            ),
        ),
        state_roles=("wealth",),
    )
    identity = SolverIdentity(
        plugin_id="tests.compatibility",
        plugin_version="1",
    )

    assert fingerprints._semantic_fingerprint(descriptor)
    assert fingerprints._semantic_fingerprint(identity)


def test_custom_mapping_in_solver_configuration_fails_closed() -> None:
    solver = _ConfiguredFingerprintSolver(config=_SemanticMapping(mode=1))

    with pytest.raises(TypeError, match="custom Mapping semantic value"):
        fingerprints._semantic_fingerprint(solver)


def test_mapping_type_equality_cannot_spoof_exact_classifier() -> None:
    with pytest.raises(TypeError, match="custom Mapping semantic value"):
        fingerprints._semantic_fingerprint(_EqualitySpoofedMapping(mode=1))


def test_opaque_solver_configuration_fails_closed_but_stateless_solver_is_sealed() -> (
    None
):
    with pytest.raises(TypeError, match="opaque semantic value"):
        fingerprints._semantic_fingerprint(
            _ConfiguredFingerprintSolver(config=object())
        )

    assert fingerprints._semantic_fingerprint(_StatelessFingerprintSolver())


def test_solver_with_dynamic_instance_lookup_fails_closed() -> None:
    with pytest.raises(TypeError, match="dynamic attribute lookup"):
        fingerprints._semantic_fingerprint(_DynamicLookupFingerprintSolver())


@pytest.mark.parametrize(
    "function",
    [
        _uses_jnp_exp,
        _uses_jnp_maximum,
        _uses_jnp_logaddexp,
        _uses_np_exp,
        _uses_jnp_int32,
        _uses_jnp_where,
        _uses_jnp_linalg_norm,
        _uses_jax_nn_sigmoid,
        _uses_jax_scipy_expit,
        _uses_jax_segment_sum,
    ],
)
def test_native_numeric_module_dependencies_have_closed_fingerprints(
    function: object,
) -> None:
    baseline = fingerprints._semantic_fingerprint(function)

    assert baseline == fingerprints._semantic_fingerprint(function)


def test_jax_ops_numeric_function_allowlist_is_identity_sealed(*, monkeypatch) -> None:
    genuine_segment_sum = jax.ops.segment_sum

    def nominal_segment_sum(*args: object, **kwargs: object) -> object:  # noqa: ARG001
        return object()

    nominal_segment_sum.__module__ = genuine_segment_sum.__module__
    nominal_segment_sum.__name__ = genuine_segment_sum.__name__
    nominal_segment_sum.__qualname__ = genuine_segment_sum.__qualname__

    assert (
        fingerprints._native_numeric_callable_kind(genuine_segment_sum)
        == "jax-function"
    )
    monkeypatch.setattr(jax.ops, "segment_sum", nominal_segment_sum)
    assert fingerprints._native_numeric_callable_kind(nominal_segment_sum) is None


def test_dataclasses_missing_terminal_support_is_identity_sealed(
    *, monkeypatch
) -> None:
    genuine_missing = dataclasses.MISSING
    nominal_missing = type(genuine_missing)()
    baseline = fingerprints._semantic_fingerprint(genuine_missing)

    with pytest.raises(TypeError, match="opaque semantic value"):
        fingerprints._semantic_fingerprint(nominal_missing)

    monkeypatch.setattr(dataclasses, "MISSING", nominal_missing)
    assert fingerprints._semantic_fingerprint(genuine_missing) == baseline
    with pytest.raises(TypeError, match="opaque semantic value"):
        fingerprints._semantic_fingerprint(nominal_missing)


def test_function_fingerprint_binds_effective_signature() -> None:
    assert fingerprints._semantic_fingerprint(_callable_with_params("left")) != (
        fingerprints._semantic_fingerprint(_callable_with_params("right"))
    )


def test_function_fingerprint_binds_executable_code() -> None:
    assert fingerprints._semantic_fingerprint(_code_variant(increment=True)) != (
        fingerprints._semantic_fingerprint(_code_variant(increment=False))
    )


def test_partial_method_and_callable_state_are_fingerprinted() -> None:
    assert fingerprints._semantic_fingerprint(partial(_add, right=1)) != (
        fingerprints._semantic_fingerprint(partial(_add, right=2))
    )
    assert fingerprints._semantic_fingerprint(_SlotCallable(1).apply) != (
        fingerprints._semantic_fingerprint(_SlotCallable(2).apply)
    )
    assert fingerprints._semantic_fingerprint(_SlotCallable(1)) != (
        fingerprints._semantic_fingerprint(_SlotCallable(2))
    )


def test_partial_with_unsealed_bound_object_argument_fails_closed() -> None:
    transform = partial(
        _dispatch_object_method,
        _STATELESS_OBJECT_METHOD_DEPENDENCY,
    )

    with pytest.raises(TypeError, match="partial bound positional argument 0"):
        fingerprints._semantic_fingerprint(transform)


def test_callable_object_binds_class_fallback_attributes(*, monkeypatch) -> None:
    baseline = fingerprints._semantic_fingerprint(_STATELESS_CALLABLE_DEPENDENCY)

    monkeypatch.setattr(_StatelessCallableDependency, "offset", 2)
    assert baseline != fingerprints._semantic_fingerprint(
        _STATELESS_CALLABLE_DEPENDENCY
    )


def test_callable_dataclass_binds_class_fallback_attributes(*, monkeypatch) -> None:
    dependency = _CallableDataclassClassDependency()
    baseline = fingerprints._semantic_fingerprint(dependency)

    monkeypatch.setattr(_CallableDataclassClassDependency, "offset", 2)
    assert baseline != fingerprints._semantic_fingerprint(dependency)


def test_execution_like_field_name_on_user_callable_remains_semantic() -> None:
    baseline = _CallableDataclass(scale=2, batch_size=1)
    changed = _CallableDataclass(scale=2, batch_size=3)

    assert fingerprints._semantic_fingerprint(baseline) != (
        fingerprints._semantic_fingerprint(changed)
    )


def _fingerprint_space(*, regime_params: object) -> SimpleNamespace:  # noqa: ARG001
    """Return one fixed representative space for the focused fingerprint test."""
    return SimpleNamespace(
        states={"wealth": np.asarray([1.0, 2.0])},
        discrete_actions={},
        continuous_actions={},
    )


def _fingerprint_regime(
    *,
    second_period_nodes: tuple[float, ...],
    artifact_authorities: dict[ArtifactKey, object] | None = None,
) -> EngineRegime:
    """Build a minimal canonical-regime shape with age-specific state support."""
    solution = SimpleNamespace(
        state_names=("wealth",),
        action_names=(),
        functions={},
        continuation_functions={},
        constraints={},
        transitions={},
        compute_regime_transition_probs=None,
        pareto_weights=None,
        artifact_authorities=artifact_authorities or {},
        state_action_space=_fingerprint_space,
        period_state_axes={
            0: {"wealth": np.asarray([1.0, 2.0])},
            1: {"wealth": np.asarray(second_period_nodes)},
        },
    )
    regime = object.__new__(EngineRegime)
    for name, value in {
        "active_periods": (0, 1),
        "solution": solution,
        "simulation": SimpleNamespace(
            transitions={},
            compute_regime_transition_probs=None,
            external_replay_route=None,
        ),
        "fold_state_names": (),
        "stakeholders": None,
        "resolved_fixed_params": MappingProxyType({}),
    }.items():
        object.__setattr__(regime, name, value)
    return regime


def test_model_fingerprint_binds_each_periods_age_specialized_support() -> None:
    """Equal representative grids cannot hide a later period's moved nodes."""
    solver = SimpleNamespace(identity=("test-solver", 1))
    user_regimes = {"alive": SimpleNamespace(solver=solver)}
    flat_params = cast("FlatParams", MappingProxyType({"alive": MappingProxyType({})}))
    ages = AgeGrid(start=0, stop=1, step="Y")

    baseline = _fingerprint_model_for_test(
        ages=ages,
        regimes=cast(
            "dict",
            {"alive": _fingerprint_regime(second_period_nodes=(2.0, 3.0))},
        ),
        user_regimes=cast("dict", user_regimes),
        regime_names_to_ids=cast(
            "RegimeNamesToIds", MappingProxyType({"alive": jnp.int32(0)})
        ),
        flat_params=flat_params,
    )
    moved = _fingerprint_model_for_test(
        ages=ages,
        regimes=cast(
            "dict",
            {"alive": _fingerprint_regime(second_period_nodes=(2.0, 4.0))},
        ),
        user_regimes=cast("dict", user_regimes),
        regime_names_to_ids=cast(
            "RegimeNamesToIds", MappingProxyType({"alive": jnp.int32(0)})
        ),
        flat_params=flat_params,
    )

    assert baseline != moved


def test_model_fingerprint_treats_artifact_authorities_as_keyed_mapping() -> None:
    """Authority declaration order cannot change an otherwise identical model."""
    left = ArtifactKey(type_id="example.left", schema_version=1)
    right = ArtifactKey(type_id="example.right", schema_version=1)
    left_authority = SimpleNamespace(descriptor=("left", 1))
    right_authority = SimpleNamespace(descriptor=("right", 1))
    solver = SimpleNamespace(identity=("test-solver", 1))
    user_regimes = {"alive": SimpleNamespace(solver=solver)}
    flat_params = cast("FlatParams", MappingProxyType({"alive": MappingProxyType({})}))
    regime_names_to_ids = cast(
        "RegimeNamesToIds", MappingProxyType({"alive": jnp.int32(0)})
    )
    ages = AgeGrid(start=0, stop=1, step="Y")

    forward = _fingerprint_model_for_test(
        ages=ages,
        regimes=cast(
            "dict",
            {
                "alive": _fingerprint_regime(
                    second_period_nodes=(2.0, 3.0),
                    artifact_authorities={
                        left: left_authority,
                        right: right_authority,
                    },
                )
            },
        ),
        user_regimes=cast("dict", user_regimes),
        regime_names_to_ids=regime_names_to_ids,
        flat_params=flat_params,
    )
    reversed_order = _fingerprint_model_for_test(
        ages=ages,
        regimes=cast(
            "dict",
            {
                "alive": _fingerprint_regime(
                    second_period_nodes=(2.0, 3.0),
                    artifact_authorities={
                        right: right_authority,
                        left: left_authority,
                    },
                )
            },
        ),
        user_regimes=cast("dict", user_regimes),
        regime_names_to_ids=regime_names_to_ids,
        flat_params=flat_params,
    )

    assert forward == reversed_order


def test_model_fingerprint_binds_exact_regime_name_to_id_mapping() -> None:
    """Equal names and regimes cannot hide a different categorical code assignment."""
    solver = SimpleNamespace(identity=("test-solver", 1))
    regimes = cast(
        "dict",
        {
            name: _fingerprint_regime(second_period_nodes=(2.0, 3.0))
            for name in ("alive", "dead")
        },
    )
    user_regimes = cast(
        "dict",
        {name: SimpleNamespace(solver=solver) for name in ("alive", "dead")},
    )
    flat_params = cast(
        "FlatParams",
        MappingProxyType({name: MappingProxyType({}) for name in ("alive", "dead")}),
    )
    ages = AgeGrid(start=0, stop=1, step="Y")

    baseline = _fingerprint_model_for_test(
        ages=ages,
        regimes=regimes,
        user_regimes=user_regimes,
        regime_names_to_ids=cast(
            "RegimeNamesToIds",
            MappingProxyType({"alive": jnp.int32(0), "dead": jnp.int32(1)}),
        ),
        flat_params=flat_params,
    )
    swapped = _fingerprint_model_for_test(
        ages=ages,
        regimes=regimes,
        user_regimes=user_regimes,
        regime_names_to_ids=cast(
            "RegimeNamesToIds",
            MappingProxyType({"alive": jnp.int32(1), "dead": jnp.int32(0)}),
        ),
        flat_params=flat_params,
    )

    assert baseline != swapped


def test_project_solution_params_removes_only_proven_transition_truth() -> None:
    belief = _callable_with_params("next_stock__belief", "helper__shared")
    truth = _callable_with_params(
        "next_stock__truth", "helper__shared", "helper__current_value"
    )
    utility = _callable_with_params("helper__current_value")
    solution = SimpleNamespace(
        functions={"nested": {"utility": utility}},
        continuation_functions={},
        constraints={},
        transitions={"alive": {"next_stock": belief}},
        compute_regime_transition_probs=None,
        pareto_weights=None,
    )
    simulation = SimpleNamespace(
        transitions={"alive": {"next_stock": truth}},
        compute_regime_transition_probs=None,
    )
    regimes = {"alive": SimpleNamespace(solution=solution, simulation=simulation)}
    flat_params = cast(
        "FlatParams",
        MappingProxyType(
            {
                "alive": MappingProxyType(
                    {
                        "next_stock__belief": 1,
                        "next_stock__truth": 2,
                        "helper__shared": 3,
                        "helper__current_value": 4,
                        "uninspectable__kept": 5,
                    }
                )
            }
        ),
    )

    projected = fingerprints.project_solution_params(
        flat_params=flat_params, regimes=cast("dict", regimes)
    )

    assert projected == {
        "alive": {
            "next_stock__belief": 1,
            "helper__shared": 3,
            "helper__current_value": 4,
            "uninspectable__kept": 5,
        }
    }

    solution.functions = {
        "nested": {"utility": utility},
        "generic": _generic_solve_callable,
    }
    conservative = fingerprints.project_solution_params(
        flat_params=flat_params, regimes=cast("dict", regimes)
    )
    assert conservative["alive"] == flat_params["alive"]
