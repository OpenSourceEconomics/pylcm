"""Named, scalar borrowing floors and their model-owned dependency contract."""

import inspect
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, cast, no_type_check, overload

import jax
import jax.numpy as jnp
from dags import concatenate_functions, get_annotations
from numpy.typing import DTypeLike

from _lcm.constraints.ir import Const, Ref
from _lcm.constraints.processed import (
    ConstraintLike,
    ProcessedConstraint,
    normalize_constraints,
)
from _lcm.dtypes import canonical_float_dtype
from _lcm.post_decision_bound import (
    BoundDependencyEntry,
    PostDecisionBoundSpec,
    _PostDecisionLowerBound,
)
from _lcm.regime_building.age_specialization import resolve_node
from _lcm.regime_law import RegimeLaw
from _lcm.time import TimeAxis, coordinate_kind, specialization_coordinate_at
from _lcm.variables import from_regime
from lcm.exceptions import ModelInitializationError
from lcm.transition import AgeSpecializedFunction
from lcm.typing import (
    FloatND,
    FunctionName,
    ParameterName,
    ReferenceName,
    RegimeName,
    UserFunction,
)

if TYPE_CHECKING:
    from lcm.regime import Regime as UserRegime
else:
    # Regime imports constraint normalization, so the claw needs a runtime fallback.
    type UserRegime = object


@dataclass(frozen=True, kw_only=True, eq=False)
class ScalarFloor:
    """Preserve a floor's signature while checking its scalar floating result."""

    function: UserFunction
    """Original producer, composed through the ordinary DAG."""
    constraint_name: str
    """Declaration named in result diagnostics."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "__signature__", inspect.signature(self.function))
        object.__setattr__(self, "__annotations__", get_annotations(self.function))
        object.__setattr__(self, "__name__", self.constraint_name + "_lower")

    @no_type_check
    def __call__(self, *args: FloatND, **kwargs: FloatND) -> FloatND:
        value = jnp.asarray(self.function(*args, **kwargs))
        if value.ndim != 0 or not jnp.issubdtype(value.dtype, jnp.floating):
            raise ModelInitializationError(
                f"Borrowing bound '{self.constraint_name}' must return a real "
                f"floating scalar; "
                f"got shape {value.shape}, dtype {value.dtype}."
            )
        return value.astype(canonical_float_dtype())


@dataclass(frozen=True, kw_only=True)
class _ScalarFloorFactory:
    """Check each concrete producer selected by existing age normalization."""

    function: AgeSpecializedFunction
    constraint_name: str

    def __call__(self, age: float) -> UserFunction:
        return ScalarFloor(
            function=self.function.build(age), constraint_name=self.constraint_name
        )


def get_floor_node(*, function: UserFunction, constraint_name: str) -> UserFunction:
    """Keep temporal markers visible and check every concrete scalar producer."""
    if isinstance(function, ScalarFloor):
        return function
    if isinstance(function, AgeSpecializedFunction):
        return replace(
            function,
            build=_ScalarFloorFactory(
                function=function, constraint_name=constraint_name
            ),
        )
    return ScalarFloor(function=function, constraint_name=constraint_name)


@overload
def get_floor_source(*, function: UserFunction) -> UserFunction: ...


@overload
def get_floor_source(*, function: None) -> None: ...


def get_floor_source(*, function: UserFunction | None) -> UserFunction | None:
    """Return the declared producer behind a scalar-result check."""
    if isinstance(function, ScalarFloor):
        return function.function
    if isinstance(function, AgeSpecializedFunction) and isinstance(
        function.build, _ScalarFloorFactory
    ):
        return function.build.function
    return function


def _bound_surface(
    *,
    user_regime: UserRegime,
    regime_name: RegimeName,
    name: str,
    constraint: ProcessedConstraint,
) -> tuple[tuple[str, str, str, str], Const | Ref] | None:
    """Identify a single contextual floor and retain its complete liquid margin."""
    declaration = constraint.declaration
    margin = getattr(user_regime, "liquid", None)
    if isinstance(declaration, _PostDecisionLowerBound):
        roles = (
            declaration.liquid_state,
            declaration.action,
            declaration.resources_target,
            declaration.post_decision,
        )
        if margin is not None and roles != (
            margin.state,
            margin.action,
            margin.resources_name,
            margin.post_decision_state,
        ):
            raise ModelInitializationError(
                f"Borrowing bound '{name}' in regime '{regime_name}' uses a "
                f"different liquid margin."
            )
    elif margin is not None:
        roles = (
            margin.state,
            margin.action,
            margin.resources_name,
            margin.post_decision_state,
        )
    else:
        return None
    surfaces = constraint.boundary_surfaces
    if surfaces is None or len(surfaces) != 1:
        return None
    surface = surfaces[0]
    post = roles[3]
    if (
        surface.op != ">="
        or not isinstance(surface.left, Ref)
        or surface.left.name != post
    ):
        return None
    if not isinstance(surface.right, (Const, Ref)):
        return None
    return roles, surface.right


def validate_declared_bounds(
    *,
    user_regime: UserRegime,
    regime_name: RegimeName,
    law: RegimeLaw,
    ages: TimeAxis,
    periods: tuple[int, ...],
) -> tuple[PostDecisionBoundSpec, ...]:
    """Validate both floor spellings at every active temporal/phase resolution."""
    normalized = normalize_constraints(
        constraints=cast(
            "Mapping[str, ConstraintLike]", user_regime.decomposed_constraints
        )
    )
    # A stochastic process has discrete topology but is read at its float node
    # values, like a continuous state; only genuinely discrete variables are codes.
    known_dtypes: dict[str, DTypeLike] = {
        name: canonical_float_dtype()
        if info.topology == "continuous" or info.is_process
        else jnp.int32
        for name, info in from_regime(user_regime=user_regime).info.items()
    }
    known_dtypes["period"] = jnp.int32
    if coordinate_kind(ages) == "age":
        known_dtypes["age"] = ages.values.dtype
    specs = []
    for name, constraint in normalized.items():
        recognized = _bound_surface(
            user_regime=user_regime,
            regime_name=regime_name,
            name=name,
            constraint=constraint,
        )
        if recognized is None:
            continue
        roles, right = recognized
        post = roles[3]
        if isinstance(right, Const):
            if not bool(
                jnp.isfinite(jnp.asarray(right.value, dtype=canonical_float_dtype()))
            ):
                raise ModelInitializationError(
                    f"Borrowing bound '{name}' in regime '{regime_name}' must "
                    f"be finite in the canonical float dtype."
                )
            specs.append(
                PostDecisionBoundSpec(
                    post_decision=post,
                    liquid_state=roles[0],
                    action=roles[1],
                    resources_target=roles[2],
                    kind="literal",
                    literal_value=float(right.value),
                    lower_target=None,
                    lower_inputs=(),
                    dependency_identity=(((), right.value),),
                    declaration_names=(name,),
                )
            )
            continue
        target = right.name
        forbidden = {roles[0], *user_regime.actions}
        outer = getattr(user_regime, "outer_continuous", None)
        if outer is not None:
            forbidden.update((outer.action, outer.post_decision_state))
        current_identity, lower_inputs = _resolved_rhs_contract(
            user_regime=user_regime,
            regime_name=regime_name,
            law=law,
            ages=ages,
            periods=periods,
            name=name,
            target=target,
            forbidden=frozenset(forbidden),
            known_dtypes=known_dtypes,
        )
        specs.append(
            PostDecisionBoundSpec(
                post_decision=post,
                liquid_state=roles[0],
                action=roles[1],
                resources_target=roles[2],
                kind="dag",
                literal_value=None,
                lower_target=target,
                lower_inputs=tuple(sorted(lower_inputs)),
                dependency_identity=tuple(current_identity),
                declaration_names=(name,),
            )
        )
    return tuple(specs)


def _resolved_rhs_contract(
    *,
    user_regime: UserRegime,
    regime_name: RegimeName,
    law: RegimeLaw,
    ages: TimeAxis,
    periods: tuple[int, ...],
    name: str,
    target: FunctionName,
    forbidden: frozenset[str],
    known_dtypes: Mapping[str, DTypeLike],
) -> tuple[list[tuple[tuple[str, ...], tuple[BoundDependencyEntry, ...]]], set[str]]:
    """Resolve every active RHS and compare canonical phase identities."""
    current_identity = []
    lower_inputs = set()
    for period in periods or (0,):
        age = specialization_coordinate_at(ages=ages, period=period)
        phase_paths = []
        for phase in ("solve", "simulate"):
            declared_pool = user_regime.get_all_functions(phase=phase, law=law)
            pool = {
                key: resolve_node(node=node, age=age)
                for key, node in declared_pool.items()
            }
            canonical_producers = {
                key: (source, source.signature(age))
                for key, node in declared_pool.items()
                if isinstance(
                    source := get_floor_source(function=node),
                    AgeSpecializedFunction,
                )
            }
            paths, identity = _walk_rhs(
                pool=pool,
                target=target,
                canonical_producers=canonical_producers,
                fixed_inputs=frozenset(
                    {
                        *user_regime.states,
                        *user_regime.actions,
                        *user_regime.derived_categoricals,
                        "age",
                        "period",
                    }
                ),
                forbidden=frozenset(forbidden),
                constraint_name=name,
                regime_name=regime_name,
            )
            phase_paths.append(identity)
            for path in paths:
                leaf = path[-1]
                if leaf in forbidden:
                    raise ModelInitializationError(
                        f"Borrowing bound '{name}' in regime '{regime_name}' "
                        f"reads forbidden leaf "
                        f"'{leaf}' through {' -> '.join(path)}."
                    )
                lower_inputs.add(leaf)
            _validate_scalar_trace(
                pool=pool, target=target, name=name, known_dtypes=known_dtypes
            )
        if phase_paths[0] != phase_paths[1]:
            raise ModelInitializationError(
                f"Borrowing bound '{name}' in regime '{regime_name}' has "
                f"phase-varying RHS "
                f"dependencies at '{target}': "
                f"{sorted({p[0][-1] for p in phase_paths[0] if p[0]})}."
            )
        current_identity.append(((str(period),), tuple(phase_paths[0])))
    return current_identity, lower_inputs


def _walk_rhs(
    *,
    pool: Mapping[str, UserFunction],
    target: FunctionName,
    canonical_producers: Mapping[FunctionName, tuple[UserFunction, Hashable]],
    fixed_inputs: frozenset[str],
    forbidden: frozenset[str],
    constraint_name: str,
    regime_name: RegimeName,
) -> tuple[list[tuple[ReferenceName, ...]], list[BoundDependencyEntry]]:
    """Retain leaf paths and producer identities without walking the comparison LHS."""
    paths = []
    identity: list[BoundDependencyEntry] = []

    def visit(*, name: ReferenceName, path: tuple[ReferenceName, ...]) -> None:
        if name in path:
            raise ModelInitializationError(
                f"Borrowing floor has a dependency cycle: {' -> '.join((*path, name))}."
            )
        function = pool.get(name)
        new_path = (*path, name)
        if name in forbidden:
            raise ModelInitializationError(
                f"Borrowing bound '{constraint_name}' in regime "
                f"'{regime_name}' reads forbidden dependency "
                f"'{name}' through {' -> '.join(new_path)}."
            )
        if function is None:
            paths.append(new_path)
            owner: tuple[
                Literal["parameter"], RegimeName, FunctionName, ParameterName
            ] = ("parameter", regime_name, path[-1] if path else target, name)
            identity.append((new_path[1:], name if name in fixed_inputs else owner))
            return
        original = canonical_producers.get(name, get_floor_source(function=function))
        identity.append((new_path[1:], original))
        for argument in sorted(inspect.signature(function).parameters):
            visit(name=argument, path=new_path)

    visit(name=target, path=())
    return paths, identity


def _validate_scalar_trace(
    *,
    pool: Mapping[str, UserFunction],
    target: FunctionName,
    name: str,
    known_dtypes: Mapping[str, DTypeLike],
) -> None:
    """Check output shape/dtype using abstract leaves rather than numerical probes."""
    function = concatenate_functions(
        functions=pool, targets=target, enforce_signature=False, set_annotations=True
    )
    parameters = inspect.signature(function).parameters
    if any(key not in known_dtypes for key in parameters):
        # Parameter shapes are known only at the actual runtime trace.
        return
    inputs = {key: jax.ShapeDtypeStruct((), known_dtypes[key]) for key in parameters}
    output = jax.eval_shape(function, **inputs)
    if (
        not hasattr(output, "ndim")
        or output.ndim != 0
        or not jnp.issubdtype(output.dtype, jnp.floating)
    ):
        raise ModelInitializationError(
            f"Borrowing bound '{name}' must return a real floating scalar."
        )
