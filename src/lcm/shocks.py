"""IID laws and weighted rules on named base innovations.

This module declares and evaluates shocks without attaching them to a model. Numerical
kernels return an explicit validity channel: an engine consuming a `Realisation` must
check it at its runtime boundary before using any value or weight. `raise_if_invalid`
is the eager boundary. Signed provenance must separately be admitted for each reader.
"""

import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import NamedTuple, cast

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import Num

from _lcm.beartype_conf import SHOCK_CONF
from _lcm.grids.categorical import DataclassInstance
from _lcm.probability import is_represented_zero
from _lcm.processes.base import StateConditioned, sealed
from _lcm.processes.joint_normal import (
    _exact_zero_pivots,
    admitted_correlation_factor,
)
from lcm.exceptions import GridInitializationError, ShockDeclarationError
from lcm.grids import DiscreteGrid
from lcm.typing import (
    ActionName,
    BoolND,
    Float1D,
    Float2D,
    FloatND,
    FunctionName,
    IntND,
    ParameterName,
    ReferenceName,
    StateName,
)

type ShockName = str
type ShockGroupName = str
type HostArray = Num[np.ndarray, "..."]
type LawValue = float | int | FloatND | IntND | HostArray | tuple[float, ...]


@beartype(conf=SHOCK_CONF)
@dataclass(frozen=True, kw_only=True)
class BaseInnovations:
    """Named independent innovations, before correlation and physical transforms.

    Normal-base values are finite real numbers. Uniform-law support is [0,1];
    categorical bases use [0,1), as does the random uniform generator.
    Validation against the producing law occurs when the engine binds the input.
    Arrays remain explicit pytree leaves for placement and residency accounting.
    """

    values: FloatND | IntND | HostArray
    """Scalar cell, entry vector, or common/subject-specific dated input array."""

    def __post_init__(self) -> None:
        _validate_override_array(values=self.values)
        if self.values.dtype.kind != "f":
            raise ShockDeclarationError(
                "Base innovations require floating-point coordinates."
            )


@beartype(conf=SHOCK_CONF)
@dataclass(frozen=True, kw_only=True)
class PhysicalValues:
    """Physical observations to invert under their producing law and context.

    A supplied value does not declare that a policy can observe it. Correlated
    physical inputs require a complete supported frame, without mixed base inputs.
    """

    values: FloatND | IntND | HostArray
    """Physical values; categorical observations use integer category codes."""

    def __post_init__(self) -> None:
        _validate_override_array(values=self.values)


def _validate_override_array(*, values: FloatND | IntND | HostArray) -> None:
    """Reject narrowing before pytree tracing can erase the caller's dtype."""
    dtype = values.dtype
    if values.ndim > 2 or dtype.kind not in "fiu":  # noqa: PLR2004 - common paths or subject-by-period paths
        raise ShockDeclarationError(
            "Shock inputs require real scalar, vector, or matrix arrays."
        )
    target_bytes = (
        np.dtype(jnp.result_type(0.0)).itemsize
        if dtype.kind == "f"
        else np.dtype(np.int32).itemsize
    )
    if dtype.itemsize > target_bytes or dtype.kind == "u":
        raise ShockDeclarationError(
            "Shock input dtype would narrow; provide active-precision floats "
            "or int32 codes."
        )


jax.tree_util.register_dataclass(
    BaseInnovations, data_fields=("values",), meta_fields=()
)
jax.tree_util.register_dataclass(
    PhysicalValues, data_fields=("values",), meta_fields=()
)


type ShockOverride = BaseInnovations | PhysicalValues


@dataclass(frozen=True)
class _ConditionedLookup:
    """Category-code lookup with a fixed declared conditioning-state dependency."""

    on: StateName
    """Conditioning state name."""
    values: tuple[float, ...]
    """Values in categorical code order."""

    def __call__(self, **inputs: LawValue) -> FloatND:
        code = jnp.asarray(inputs[self.on])
        if code.dtype.kind not in "iu":
            return jnp.full(code.shape, jnp.nan)
        index = jnp.where(
            (code >= 0) & (code < len(self.values)), code, len(self.values)
        )
        return jnp.take(
            jnp.asarray(self.values), index, mode="fill", fill_value=jnp.nan
        )


type LawCallable = Callable[..., FloatND]
type LawParameter = (
    LawValue | LawCallable | StateConditioned | _ConditionedLookup | None
)


class _Independence(Enum):
    DEFAULT = "independent"


class ShockReader(Enum):
    """Semantic readers, including nonlinear consumers of a stored channel."""

    LINEAR_EXPECTATION = "linear expectation"
    DECISION = "decision"
    DISCRETE_LAW = "discrete law"
    CONSTRAINT = "constraint"
    REGIME_TRANSITION = "regime transition"
    EXIT_RULE = "exit rule"
    AVAILABILITY = "availability"
    GATE = "gate"
    SIMULATION_DRAW = "simulation draw"
    NONLINEAR_CE = "nonlinear certainty equivalent"
    INVERSE_TRANSFORM = "inverse transform"
    EULER_INVERSION = "Euler inversion"
    ROOT = "solver root"


@dataclass(frozen=True)
class _Law:
    """A law's parameters are fixed, runtime-supplied, or current-state callables."""

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if (
                item.name == "categories"
                or value is None
                or callable(value)
                or isinstance(value, StateConditioned)
            ):
                continue
            array = np.asarray(value)
            if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
                raise ShockDeclarationError(f"{item.name} must be finite and numeric.")
            if item.name != "probs" and array.ndim != 0:
                raise ShockDeclarationError(f"Fixed {item.name} must be scalar.")
            if item.name == "probs" and array.ndim != 1:
                raise ShockDeclarationError("Fixed probabilities must be a vector.")
            if array.ndim:
                object.__setattr__(self, item.name, tuple(float(x) for x in array))
            else:
                object.__setattr__(self, item.name, float(array))
        _validate_fixed_law(self)


@beartype(conf=SHOCK_CONF)
@dataclass(frozen=True, kw_only=True)
class Normal(_Law):
    """Normal law, including a deterministic zero-scale limit."""

    mu: LawParameter = None
    """Location, in physical units."""
    sigma: LawParameter = None
    """Nonnegative scale, in physical units."""


@beartype(conf=SHOCK_CONF)
@dataclass(frozen=True, kw_only=True)
class LogNormal(_Law):
    """Exponential of a normal variable; parameters belong to its logarithm."""

    mu: LawParameter = None
    """Underlying normal location."""
    sigma: LawParameter = None
    """Nonnegative underlying normal scale."""


@beartype(conf=SHOCK_CONF)
@dataclass(frozen=True, kw_only=True)
class Uniform(_Law):
    """Uniform law, including equal endpoints as a deterministic limit."""

    start: LawParameter = None
    """Lower endpoint."""
    stop: LawParameter = None
    """Upper endpoint."""


@beartype(conf=SHOCK_CONF)
@dataclass(frozen=True, kw_only=True)
class NormalMixture(_Law):
    """Two normal components mapped from one common normal-base innovation.

    Active components require strictly positive scales. An inactive component may have
    zero scale. Point masses in active components are outside this smooth quantile law.
    """

    p1: LawParameter = None
    """First component's probability."""
    mu1: LawParameter = None
    """First component's location."""
    sigma1: LawParameter = None
    """First component's scale."""
    mu2: LawParameter = None
    """Second component's location."""
    sigma2: LawParameter = None
    """Second component's scale."""


@beartype(conf=SHOCK_CONF)
@dataclass(frozen=True, kw_only=True)
class Discrete(_Law):
    """Categorical law with exact solve enumeration and common-uniform draws."""

    categories: type[DataclassInstance]
    """A pylcm categorical class; its codes order the probability axis."""
    probs: LawParameter = None
    """Probability vector, with categories on the final axis."""

    def __post_init__(self) -> None:
        try:
            DiscreteGrid(self.categories)
        except GridInitializationError as error:
            raise ShockDeclarationError(str(error)) from error
        super().__post_init__()


type Law = Normal | LogNormal | Uniform | NormalMixture | Discrete


@beartype(conf=SHOCK_CONF)
@dataclass(frozen=True)
class GaussHermite:
    """Standard-normal Gaussian rule, exact through degree `2*n-1` analytically."""

    n: int
    """Positive node count."""

    def __post_init__(self) -> None:
        _positive_count(self.n)


@beartype(conf=SHOCK_CONF)
@dataclass(frozen=True)
class EqualSpaced:
    """Normal CDF bins on inclusive nodes spanning `[-n_std, n_std]`."""

    n: int
    """Node count, at least two."""
    n_std: float
    """Positive finite normal-base support halfwidth."""

    def __post_init__(self) -> None:
        _positive_count(self.n)
        if self.n < 2 or not math.isfinite(self.n_std) or self.n_std <= 0:  # noqa: PLR2004 - CDF bins need adjacent endpoints
            raise ShockDeclarationError(
                "Normal CDF bins need two nodes and positive finite n_std."
            )


@beartype(conf=SHOCK_CONF)
@dataclass(frozen=True)
class EqualSpacedUniform:
    """Equal weights on inclusive unit endpoints; a singleton uses the midpoint."""

    n: int
    """Positive node count."""

    def __post_init__(self) -> None:
        _positive_count(self.n)


type ScalarRule = GaussHermite | EqualSpaced | EqualSpacedUniform


@dataclass(frozen=True, init=False)
class Product:
    """Tensor product in shock declaration order, last coordinate varying fastest."""

    rules: Mapping[ShockName, ScalarRule]
    """One compatible rule per continuous shock."""

    @beartype(conf=SHOCK_CONF)
    def __init__(self, **per_shock_rule: ScalarRule) -> None:
        object.__setattr__(self, "rules", MappingProxyType(dict(per_shock_rule)))


@beartype(conf=SHOCK_CONF)
@dataclass(frozen=True)
class MonteCarlo:
    """Equal-weight PCG64 uniforms, regenerated for each exact age and group."""

    n: int
    """Positive sample count."""
    seed: int
    """Solve seed, separate from simulation randomness."""

    def __post_init__(self) -> None:
        _positive_count(self.n)
        _validate_seed(self.seed)


@beartype(conf=SHOCK_CONF)
@dataclass(frozen=True)
class QuasiMonteCarlo:
    """Scrambled Sobol LMS+shift points, retaining original point order."""

    n: int
    """Power-of-two sample count."""
    seed: int
    """Solve scramble seed."""
    sequence: str = "sobol"
    """Supported sequence name."""

    def __post_init__(self) -> None:
        _positive_count(self.n)
        _validate_seed(self.seed)
        if self.sequence != "sobol" or self.n & (self.n - 1):
            raise ShockDeclarationError(
                "Scrambled Sobol requires a power-of-two n and sequence='sobol'."
            )


@beartype(conf=SHOCK_CONF)
@dataclass(frozen=True)
class Smolyak:
    """Zero-based sparse rule using Genz-Keister-19 and Clenshaw-Curtis.

    Normal levels zero through three have 1, 3, 9 and 19 nodes. Uniform level zero
    is the midpoint; subsequent levels have `2**level+1` inclusive unit nodes.
    Difference products with total component level at most `level` define the rule.
    Concrete negative weights require `signed=True`, independently of family names.
    """

    level: int
    """Nonnegative total difference level."""
    signed: bool = False
    """Declare signed linear quadrature and its reader restrictions."""

    def __post_init__(self) -> None:
        if isinstance(self.level, bool) or self.level < 0:
            raise ShockDeclarationError("Smolyak level must be nonnegative.")


@beartype(conf=SHOCK_CONF)
@dataclass(frozen=True)
class UserNodes:
    """Owned joint base nodes and weights, preserved in caller order."""

    nodes: Float2D | HostArray
    """Finite matrix with joint nodes on rows and named shocks on columns."""
    weights: Float1D | HostArray
    """Finite vector summing to unity; probabilities only when nonnegative."""
    signed: bool = False
    """Permit signed linear quadrature and propagate its reader restrictions."""

    def __post_init__(self) -> None:
        nodes, weights = (
            np.array(self.nodes, copy=True),
            np.array(self.weights, copy=True),
        )
        if (
            nodes.ndim != 2  # noqa: PLR2004 - joint nodes are a matrix
            or not nodes.shape[0]
            or nodes.shape[1] < 1
            or not np.isfinite(nodes).all()
        ):
            raise ShockDeclarationError("UserNodes needs finite nonempty (K,d) nodes.")
        _validate_weights(weights=weights, signed=self.signed)
        if weights.shape != (nodes.shape[0],):
            raise ShockDeclarationError("UserNodes weights must have shape (K,).")
        nodes.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "weights", weights)


type NodeRule = Product | MonteCarlo | QuasiMonteCarlo | Smolyak | UserNodes


class Realisation(NamedTuple):
    """Numerical values, weights and an explicit mandatory validity channel.

    `valid` covers law domains, finite outputs, probabilities and quantile
    convergence.
    A transformed engine must route it into its runtime checks. It is never a mask
    authorizing a consumer to replace an invalid economic operand with zero.
    """

    values: Mapping[ShockName, FloatND | IntND]
    weights: FloatND
    valid: BoolND

    def raise_if_invalid(self) -> None:
        """Enforce the law status outside JAX transformations."""
        if not bool(np.asarray(self.valid)):
            raise ShockDeclarationError(
                "Invalid shock law, weights, or unresolved mixture quantile."
            )


@dataclass(frozen=True)
class NodeSet:
    """Resolved base rule, exact category enumeration and durable provenance."""

    base_nodes: Float2D
    """Joint base coordinates, `(K,d)` in shock order."""
    weights: Float1D
    """Parameter-independent continuous-rule weights, `(K,)`."""
    discrete_codes: Mapping[ShockName, IntND]
    """Exact category codes for discrete coordinates, rather than innovations."""
    group_name: ShockGroupName
    """Producer used in reader diagnostics and solve random addressing."""
    signed: bool
    """Signed provenance, retained even when concrete weights happen to be positive."""
    sum_abs_weights: float
    """Sum of absolute concrete weights, a conditioning diagnostic."""
    fingerprint: str
    """Versioned rule, coordinate, ordering, algorithm and concrete-support digest."""

    @beartype(conf=SHOCK_CONF)
    def admit(self, *, reader: ShockReader) -> None:
        """Refuse nonlinear or probabilistic consumers of signed provenance."""
        if not isinstance(reader, ShockReader) or (
            self.signed and reader is not ShockReader.LINEAR_EXPECTATION
        ):
            raise ShockDeclarationError(
                f"Signed shock group {self.group_name!r} cannot feed reader {reader!r}."
            )


jax.tree_util.register_dataclass(
    NodeSet,
    data_fields=("base_nodes", "weights", "discrete_codes"),
    meta_fields=("group_name", "signed", "sum_abs_weights", "fingerprint"),
)


@beartype(conf=SHOCK_CONF)
@dataclass(frozen=True, kw_only=True)
class ShockGroup:
    """Named independent base innovations, physical laws and one joint node rule.

    Omitted correlation means independence. Explicit `correlation=None` requests a
    runtime matrix under `corr` in this group's parameter namespace. Fixed matrices
    must be represented positive semidefinite correlation matrices. Correlation is
    confined to normal-base shocks; uniforms and categoricals do not declare copulas.
    """

    shocks: Mapping[ShockName, Law]
    """Nonempty names in the order of base columns and correlation rows."""
    nodes: NodeRule | None = None
    """Required for continuous shocks; categoricals are enumerated exactly."""
    correlation: (
        tuple[tuple[float, ...], ...] | Float2D | HostArray | _Independence | None
    ) = _Independence.DEFAULT
    """Independent by omission, fixed matrix, or explicit runtime `None`."""
    _correlation_support: tuple[bool, ...] | None = field(
        init=False, default=None, repr=False, compare=False
    )
    """Prepared exact pivot pattern; runtime correlation values remain parameters."""

    def __post_init__(self) -> None:  # noqa: C901, PLR0912 - validate each independent declaration surface
        if not self.shocks or any(
            not name.isidentifier() or name.startswith("next_") or "__" in name
            for name in self.shocks
        ):
            raise ShockDeclarationError("Shocks need nonempty ordinary unique names.")
        if any(
            not isinstance(law, Normal | LogNormal | Uniform | NormalMixture | Discrete)
            for law in self.shocks.values()
        ):
            raise ShockDeclarationError("Every shock must declare a supported law.")
        object.__setattr__(self, "shocks", MappingProxyType(dict(self.shocks)))
        continuous = {
            name for name, law in self.shocks.items() if not isinstance(law, Discrete)
        }
        if continuous and self.nodes is None:
            raise ShockDeclarationError(
                "Continuous shocks require an explicit node rule."
            )
        if isinstance(self.nodes, Product):
            if self.nodes.rules.keys() != continuous:
                raise ShockDeclarationError(
                    "Product rules must name exactly the continuous shocks."
                )
            for name in continuous:
                uniform = isinstance(self.shocks[name], Uniform)
                if uniform != isinstance(self.nodes.rules[name], EqualSpacedUniform):
                    raise ShockDeclarationError(
                        f"Rule for {name!r} is incompatible with its base measure."
                    )
        if isinstance(self.nodes, UserNodes):
            if self.nodes.nodes.shape[1] != len(self.shocks):
                raise ShockDeclarationError(
                    "UserNodes dimension must match the shock group."
                )
            for column, law in enumerate(self.shocks.values()):
                if isinstance(law, Uniform | Discrete) and np.any(
                    (self.nodes.nodes[:, column] < 0)
                    | (self.nodes.nodes[:, column] > 1)
                ):
                    raise ShockDeclarationError(
                        "Uniform-base nodes must lie in the unit interval."
                    )
        if self.signed and any(
            isinstance(law, Discrete) for law in self.shocks.values()
        ):
            raise ShockDeclarationError("A signed rule cannot feed a discrete law.")
        if self.correlation is not _Independence.DEFAULT:
            if any(isinstance(law, Uniform | Discrete) for law in self.shocks.values()):
                raise ShockDeclarationError(
                    "Gaussian correlation requires normal-base continuous laws."
                )
            if self.correlation is not None:
                _validate_correlation(
                    correlation=self.correlation, size=len(self.shocks)
                )
                object.__setattr__(
                    self,
                    "correlation",
                    tuple(tuple(float(x) for x in row) for row in self.correlation),
                )

    @property
    def signed(self) -> bool:
        """Whether this declaration carries signed integration provenance."""
        return isinstance(self.nodes, Smolyak | UserNodes) and self.nodes.signed

    @property
    def fingerprint(self) -> str:
        """Fingerprint both laws and the separately represented rule declaration."""
        return _fingerprint(
            (
                "shock-group",
                1,
                tuple(self.shocks.items()),
                self.nodes,
                _correlation_record(self.correlation),
                self._correlation_support,
            )
        )

    def has_same_laws_as(self, other: ShockGroup) -> bool:
        """Compare ordered named law semantics independently of the node rule.

        Column order is conservatively part of joint-law identity, so a reordered
        declaration cannot silently reinterpret the rows of its correlation matrix.
        """
        return self is other or _fingerprint(
            (tuple(self.shocks.items()), _correlation_record(self.correlation))
        ) == _fingerprint(
            (tuple(other.shocks.items()), _correlation_record(other.correlation))
        )

    def resolve_nodes(
        self,
        *,
        group_name: ShockGroupName,
        age: Fraction | int,
        coordinate_kind: str = "age",
    ) -> NodeSet:
        """Resolve a rule for an exact age or an explicitly labelled period."""
        from _lcm.shock_nodes import (  # noqa: PLC0415 - rules import declarations
            resolve_nodes,
        )

        if isinstance(age, bool) or coordinate_kind not in {"age", "period"}:
            raise ShockDeclarationError(
                "Node coordinates must be exact ages or explicit periods."
            )
        return resolve_nodes(
            group=self,
            group_name=group_name,
            age=Fraction(age),
            coordinate_kind=coordinate_kind,
        )

    def transform(
        self,
        *,
        base_nodes: FloatND,
        params: Mapping[ParameterName, LawValue] = MappingProxyType({}),
        inputs: Mapping[ReferenceName, LawValue] = MappingProxyType({}),
    ) -> Realisation:
        """Map realised base innovations; this operation itself never samples a rule.

        Parameters use group-local names, e.g. `eps__sigma` and `corr`. The model
        registry adds the enclosing `params[group]` namespace. Inputs provide current
        states, time and arguments of callable parameters. A base column belongs to a
        named innovation, irrespective of its physical value or category code.
        """
        return self._realise(
            base_nodes=base_nodes, params=params, inputs=inputs, nodes=None
        )[0]

    def realise(
        self,
        *,
        nodes: NodeSet,
        params: Mapping[ParameterName, LawValue] = MappingProxyType({}),
        inputs: Mapping[ReferenceName, LawValue] = MappingProxyType({}),
    ) -> Realisation:
        """Evaluate all quadrature nodes, enumerating discrete outcomes exactly."""
        return self._realise(
            base_nodes=nodes.base_nodes, params=params, inputs=inputs, nodes=nodes
        )[0]

    def bind_correlation(
        self, *, correlation: tuple[tuple[float, ...], ...] | Float2D | HostArray
    ) -> ShockGroup:
        """Prepare exact support while retaining correlation as a traced parameter.

        Each invocation's host boundary must call `validate_runtime_correlation`.
        Compiled evaluation retains the supplied values and their derivatives; changing
        rank or pivot support requires a new binding. Exact host admission is mandatory.
        """
        if self.correlation is not None:
            raise ShockDeclarationError(
                "Only a runtime-correlation group needs binding."
            )
        support = self.validate_runtime_correlation(correlation=correlation)
        prepared = replace(self)
        object.__setattr__(prepared, "_correlation_support", support)
        return prepared

    def validate_runtime_correlation(
        self, *, correlation: tuple[tuple[float, ...], ...] | Float2D | HostArray
    ) -> tuple[bool, ...]:
        """Enforce exact PSD and prepared support before every compiled invocation."""
        _validate_correlation(correlation=correlation, size=len(self.shocks))
        rounded = np.asarray(correlation, dtype=np.dtype(jnp.result_type(0.0)))
        _validate_correlation(correlation=rounded, size=len(self.shocks))
        support = _correlation_pivots(rounded)
        if (
            self._correlation_support is not None
            and support != self._correlation_support
        ):
            raise ShockDeclarationError("Runtime corr support changed; bind it again.")
        return support

    def _has_identity_correlation(
        self, *, params: Mapping[ParameterName, LawValue]
    ) -> bool | BoolND:
        """Prove exact identity at the current binding, without admitting a factor.

        Fixed declarations use host metadata rounded to the active dtype. Runtime
        matrices remain dynamic, so the proof is recomputed on every invocation.
        Consumers must separately retain canonical correlation/factor validity.
        """
        size = len(self.shocks)
        if self.correlation is _Independence.DEFAULT:
            return True
        if self.correlation is not None:
            matrix = np.asarray(self.correlation, dtype=np.dtype(jnp.result_type(0.0)))
            return bool(np.array_equal(matrix, np.eye(size, dtype=matrix.dtype)))
        matrix = params.get("corr")
        if matrix is None:
            raise ShockDeclarationError(
                "Runtime corr must be supplied in the group namespace."
            )
        corr = jnp.asarray(matrix)
        if corr.shape != (size, size):
            raise ShockDeclarationError("corr must have a row and column per shock.")
        return jnp.all(corr == jnp.eye(size, dtype=corr.dtype))

    def _correlation_factor(
        self,
        *,
        params: Mapping[ParameterName, LawValue],
        admitted_correlation_support: BoolND | None = None,
    ) -> tuple[FloatND, BoolND]:
        """Share the admitted active-precision factor between forward and inverse."""
        valid = jnp.asarray(a=True)
        matrix = params.get("corr") if self.correlation is None else self.correlation
        if matrix is None:
            raise ShockDeclarationError(
                "Runtime corr must be supplied in the group namespace."
            )
        corr = jnp.asarray(matrix)
        if corr.shape != (len(self.shocks), len(self.shocks)):
            raise ShockDeclarationError("corr must have a row and column per shock.")
        if self.correlation is None and admitted_correlation_support is not None:
            zero_pivots = admitted_correlation_support
        elif self.correlation is None:
            if isinstance(matrix, jax.core.Tracer):
                if self._correlation_support is None:
                    raise ShockDeclarationError(
                        "Bind runtime corr support before tracing the transform."
                    )
                zero_pivots = self._correlation_support
                valid &= _runtime_support_validity(corr=corr, support=zero_pivots)
            else:
                zero_pivots = self.validate_runtime_correlation(correlation=corr)
        else:
            concrete = np.asarray(matrix, dtype=np.dtype(jnp.result_type(0.0)))
            _validate_correlation(correlation=concrete, size=len(self.shocks))
            zero_pivots = _correlation_pivots(concrete)
        factor, factor_valid = admitted_correlation_factor(
            correlation=corr,
            zero_pivots=jnp.asarray(zero_pivots, dtype=jnp.bool_),
        )
        return factor, valid & factor_valid

    def _realise(  # noqa: C901, PLR0912, PLR0915 - each law contributes values and validity once
        self,
        *,
        base_nodes: FloatND,
        params: Mapping[ParameterName, LawValue],
        inputs: Mapping[ReferenceName, LawValue],
        nodes: NodeSet | None,
        admitted_correlation_support: BoolND | None = None,
        scalar_cell_group: ShockGroupName | None = None,
    ) -> tuple[Realisation, tuple[FloatND, ...]]:
        if base_nodes.ndim < 2 or base_nodes.shape[-1] != len(self.shocks):  # noqa: PLR2004 - final axes are nodes and shock columns
            raise ShockDeclarationError(
                "Base nodes must have a final column per named shock."
            )
        valid = jnp.all(jnp.isfinite(base_nodes))
        base = base_nodes
        if self.correlation is not _Independence.DEFAULT:
            factor, factor_valid = self._correlation_factor(
                params=params,
                admitted_correlation_support=admitted_correlation_support,
            )
            valid &= factor_valid
            base = jnp.matmul(base, factor.T)
        weights = jnp.ones(base.shape[:-1]) if nodes is None else nodes.weights
        weight_factors = [weights]
        values: dict[ShockName, FloatND | IntND] = {}
        cell_shape: tuple[int, ...] = ()
        for column, (name, law) in enumerate(self.shocks.items()):
            parameters = {
                item.name: _resolve_parameter(
                    value=getattr(law, item.name),
                    name=f"{name}__{item.name}",
                    params=params,
                    inputs=inputs,
                )
                for item in fields(law)
                if item.name != "categories"
            }
            if scalar_cell_group is not None:
                _check_scalar_law_parameters(
                    parameters=parameters, group_name=scalar_cell_group, shock_name=name
                )
            shapes = [
                value.shape[:-1] if key == "probs" else value.shape
                for key, value in parameters.items()
            ]
            try:
                cell_shape = np.broadcast_shapes(cell_shape, *shapes)
                if nodes is None:
                    np.broadcast_shapes(base.shape[:-1], *shapes)
            except ValueError as error:
                raise ShockDeclarationError(
                    f"Shock {name!r} parameter shapes do not broadcast to its cells."
                ) from error
            valid &= _law_validity(law=law, parameters=parameters)
            if nodes is not None and not isinstance(law, Discrete):
                parameters = {
                    key: value[..., None] for key, value in parameters.items()
                }
            coordinate = base[..., column]
            if isinstance(law, Discrete):
                probs = parameters["probs"]
                if nodes is None:
                    code, code_valid = _sample_discrete(
                        coordinate=coordinate, probs=probs
                    )
                    valid &= code_valid
                    values[name] = code
                else:
                    codes = nodes.discrete_codes[name]
                    values[name] = codes
                    categorical_weights = jnp.take(probs, codes, axis=-1)
                    weight_factors.append(categorical_weights)
                    weights = weights * categorical_weights
            else:
                value, law_valid = _continuous_value(
                    law=law, coordinate=coordinate, parameters=parameters
                )
                # Ordinary consumers must see these rounded physical bits.
                values[name] = sealed(value)
                valid &= law_valid & jnp.all(jnp.isfinite(value))
                if isinstance(law, Uniform):
                    valid &= jnp.all((coordinate >= 0) & (coordinate <= 1))
        try:
            shape = np.broadcast_shapes(
                weights.shape, *(value.shape for value in values.values())
            )
        except ValueError as error:
            raise ShockDeclarationError(
                "Shock values have incompatible cell shapes."
            ) from error
        realised = Realisation(
            MappingProxyType(
                {name: jnp.broadcast_to(value, shape) for name, value in values.items()}
            ),
            jnp.broadcast_to(weights, shape),
            valid & jnp.all(jnp.isfinite(weights)),
        )
        if scalar_cell_group is not None and shape != base_nodes.shape[:-1]:
            raise ShockDeclarationError(
                f"Group {scalar_cell_group!r} must resolve to "
                "one scalar cell's node rows."
            )
        return realised, tuple(
            jnp.broadcast_to(value, shape) for value in weight_factors
        )

    def validate_dependencies(  # noqa: C901 - traverse named callable and lookup dependencies
        self,
        *,
        states: frozenset[StateName],
        actions: frozenset[ActionName],
        functions: Mapping[FunctionName, LawCallable],
        shock_names: frozenset[ShockName] = frozenset(),
        downstream_nodes: frozenset[FunctionName] = frozenset(),
    ) -> frozenset[ReferenceName]:
        """Return current-state inputs, refusing choice and realised-draw ancestry."""
        forbidden = (
            actions
            | {f"{name}_code" for name in actions}
            | self.shocks.keys()
            | shock_names
            | {f"{name}_code" for name in (*self.shocks, *shock_names)}
            | downstream_nodes
        )
        leaves: set[ReferenceName] = set()

        def visit(*, name: ReferenceName, stack: tuple[ReferenceName, ...]) -> None:
            if (
                name in forbidden
                or name.startswith(("next_", "V_", "D_", "Q_"))
                or name in {"V", "D", "Q", "CE", "policy"}
            ):
                raise ShockDeclarationError(
                    f"Shock law cannot read choice or realised-shock input {name!r}."
                )
            if name in stack:
                raise ShockDeclarationError(f"Cycle in shock law dependency {name!r}.")
            if name in functions:
                for argument in _arguments(functions[name]):
                    visit(name=argument, stack=(*stack, name))
            elif name.endswith("_code") and name[:-5] in states:
                visit(name=name[:-5], stack=stack)
            elif name in states:
                leaves.add(name)

        for law in self.shocks.values():
            for item in fields(law):
                value = getattr(law, item.name)
                if isinstance(value, StateConditioned | _ConditionedLookup):
                    visit(name=value.on, stack=())
                elif callable(value) and item.name != "categories":
                    for argument in _arguments(value):
                        visit(name=argument, stack=())
        return frozenset(leaves)

    def bind_conditioning(
        self, *, state_grids: Mapping[StateName, DiscreteGrid]
    ) -> ShockGroup:
        """Resolve `StateConditioned` lookups in the conditioning grid's code order."""
        bound: dict[ShockName, Law] = {}
        for name, law in self.shocks.items():
            changes: dict[ParameterName, _ConditionedLookup] = {}
            for item in fields(law):
                value = getattr(law, item.name)
                if not isinstance(value, StateConditioned):
                    continue
                if value.on not in state_grids:
                    raise ShockDeclarationError(
                        f"Conditioning state {value.on!r} needs a discrete grid."
                    )
                grid = state_grids[value.on]
                if set(value.by) != set(grid.categories):
                    raise ShockDeclarationError(
                        "StateConditioned.by must match the grid's categories exactly."
                    )
                ordered = sorted(zip(grid.codes, grid.categories, strict=True))
                changes[item.name] = _ConditionedLookup(
                    on=value.on,
                    values=tuple(value.by[category] for _, category in ordered),
                )
            bound[name] = replace(law, **changes)
        prepared = replace(self, shocks=bound)
        object.__setattr__(prepared, "_correlation_support", self._correlation_support)
        return prepared


def _positive_count(n: int) -> None:
    if isinstance(n, bool) or n < 1:
        raise ShockDeclarationError("Node count must be positive.")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or seed < 0:
        raise ShockDeclarationError("Solve seed must be a nonnegative integer.")


def _validate_weights(*, weights: HostArray, signed: bool) -> None:
    if weights.ndim != 1 or not len(weights) or not np.isfinite(weights).all():
        raise ShockDeclarationError("Weights must be a finite nonempty vector.")
    tolerance = max(
        1e-12,
        64 * np.finfo(weights.dtype if weights.dtype.kind == "f" else np.float64).eps,
    )
    if not math.isclose(
        math.fsum(float(x) for x in weights), 1.0, rel_tol=0.0, abs_tol=tolerance
    ):
        raise ShockDeclarationError("Weights must sum to unity within dtype tolerance.")
    if not signed and np.any(weights < 0):
        raise ShockDeclarationError("Negative concrete weights require signed=True.")


def _arguments(function: LawCallable) -> tuple[ReferenceName, ...]:
    try:
        parameters = inspect.signature(function).parameters
    except (ValueError, TypeError) as error:
        raise ShockDeclarationError(
            "Shock law needs an inspectable dependency signature."
        ) from error
    if any(
        value.kind
        in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        }
        for value in parameters.values()
    ):
        raise ShockDeclarationError(
            "Shock law dependency signatures must name keyword arguments."
        )
    return tuple(parameters)


def _resolve_parameter(
    *,
    value: LawParameter,
    name: ParameterName,
    params: Mapping[ParameterName, LawValue],
    inputs: Mapping[ReferenceName, LawValue],
) -> FloatND:
    if value is None:
        if name not in params:
            raise ShockDeclarationError(f"Missing runtime shock parameter {name!r}.")
        value = params[name]
    elif isinstance(value, StateConditioned):
        raise ShockDeclarationError(
            "Bind StateConditioned to its discrete grid before evaluation."
        )
    elif isinstance(value, _ConditionedLookup):
        value = value(**inputs)
    elif callable(value):
        arguments = {}
        function = cast("LawCallable", value)
        for argument in _arguments(function):
            if argument not in inputs and argument not in params:
                raise ShockDeclarationError(
                    f"Missing shock callable input {argument!r}."
                )
            arguments[argument] = (
                inputs[argument] if argument in inputs else params[argument]
            )
        value = function(**arguments)
    try:
        array = jnp.asarray(value)
    except (ValueError, TypeError) as error:
        raise ShockDeclarationError(
            f"Shock parameter {name!r} must be numeric."
        ) from error
    if not (
        jnp.issubdtype(array.dtype, jnp.floating)
        or jnp.issubdtype(array.dtype, jnp.integer)
    ):
        raise ShockDeclarationError(f"Shock parameter {name!r} must be real numeric.")
    return array.astype(jnp.result_type(0.0))


def _validate_fixed_law(law: _Law) -> None:
    numeric = {
        item.name: jnp.asarray(getattr(law, item.name), dtype=jnp.result_type(0.0))
        for item in fields(law)
        if item.name != "categories"
        and getattr(law, item.name) is not None
        and not callable(getattr(law, item.name))
        and not isinstance(getattr(law, item.name), StateConditioned)
    }
    if len(numeric) == len(fields(law)) - isinstance(law, Discrete) and not bool(
        np.asarray(_law_validity(law=law, parameters=numeric))
    ):
        raise ShockDeclarationError(
            "Invalid fixed shock law parameters or probabilities."
        )
    for name in ("sigma", "sigma1", "sigma2"):
        if name in numeric and bool(np.asarray(numeric[name] < 0)):
            raise ShockDeclarationError(f"{name} must be nonnegative.")


def _check_scalar_law_parameters(
    *,
    parameters: Mapping[ParameterName, FloatND],
    group_name: str,
    shock_name: ShockName,
) -> None:
    """Keep one common scalar-cell parameter boundary for forward and support."""
    for parameter, value in parameters.items():
        expected_rank = 1 if parameter == "probs" else 0
        if value.ndim != expected_rank:
            raise ShockDeclarationError(
                f"Group {group_name!r}, shock {shock_name!r}: "
                f"{parameter} must resolve to a scalar cell parameter "
                "(or a single categorical probability vector)."
            )


def _law_validity(*, law: _Law, parameters: Mapping[ParameterName, FloatND]) -> BoolND:
    valid = jnp.ones((), dtype=bool)
    for value in parameters.values():
        valid &= jnp.all(jnp.isfinite(value))
    if isinstance(law, Normal | LogNormal):
        return valid & jnp.all(parameters["sigma"] >= 0)
    if isinstance(law, Uniform):
        return valid & jnp.all(parameters["stop"] >= parameters["start"])
    if isinstance(law, NormalMixture):
        p, first, second = parameters["p1"], parameters["sigma1"], parameters["sigma2"]
        return valid & jnp.all(
            (p >= 0)
            & (p <= 1)
            & (first >= 0)
            & (second >= 0)
            & ((p <= 0) | (first > 0))
            & ((p >= 1) | (second > 0))
        )
    if isinstance(law, Discrete):
        probs = parameters["probs"]
        if not probs.ndim or probs.shape[-1] != len(DiscreteGrid(law.categories).codes):
            raise ShockDeclarationError(
                "Discrete probabilities need the category count on their last axis."
            )
        tolerance = jnp.maximum(1e-12, 64 * jnp.finfo(probs.dtype).eps)
        return (
            valid
            & jnp.all(probs >= 0)
            & jnp.all(jnp.abs(jnp.sum(probs, axis=-1) - 1) <= tolerance)
        )
    raise ShockDeclarationError("Unsupported shock law.")


def _discrete_cdf(*, probs: FloatND) -> FloatND:
    """Share the represented category intervals between sampling and support."""
    last_positive = jnp.max(
        jnp.where(probs > 0, jnp.arange(probs.shape[-1]), -1), axis=-1
    )
    return jnp.where(
        jnp.arange(probs.shape[-1]) >= last_positive[..., None],
        1.0,
        jnp.cumsum(probs, axis=-1),
    )


def _sample_discrete(*, coordinate: FloatND, probs: FloatND) -> tuple[IntND, BoolND]:
    """Apply the existing inverse CDF to an explicitly supplied coordinate."""
    cdf = _discrete_cdf(probs=probs)
    code = jnp.sum(coordinate[..., None] >= cdf, axis=-1).astype(jnp.int32)
    valid = jnp.all((coordinate >= 0) & (coordinate < 1)) & jnp.all(
        code < probs.shape[-1]
    )
    return code, valid


def _continuous_value(
    *, law: _Law, coordinate: FloatND, parameters: Mapping[ParameterName, FloatND]
) -> tuple[FloatND, BoolND]:
    if isinstance(law, Uniform):
        value = (
            parameters["start"]
            + (parameters["stop"] - parameters["start"]) * coordinate
        )
    elif isinstance(law, Normal | LogNormal):
        value = parameters["mu"] + parameters["sigma"] * coordinate
        if isinstance(law, LogNormal):
            value = jnp.exp(value)
            return value, jnp.all(~is_represented_zero(value))
    elif isinstance(law, NormalMixture):
        from _lcm.shock_quantile import (  # noqa: PLC0415 - isolate derivative kernel
            mixture_quantile,
        )

        value, resolved = mixture_quantile(
            z=coordinate,
            p=parameters["p1"],
            mu1=parameters["mu1"],
            sigma1=parameters["sigma1"],
            mu2=parameters["mu2"],
            sigma2=parameters["sigma2"],
        )
        return value, jnp.all(resolved)
    else:
        raise ShockDeclarationError("Unsupported continuous law.")
    return value, jnp.ones((), dtype=bool)


def _correlation_pivots(corr: Float2D | HostArray) -> tuple[bool, ...]:
    try:
        return _exact_zero_pivots(cov=np.asarray(corr), size=len(corr))
    except GridInitializationError as error:
        raise ShockDeclarationError(str(error)) from error


def _validate_correlation(
    *, correlation: tuple[tuple[float, ...], ...] | Float2D | HostArray, size: int
) -> None:
    matrix = np.asarray(correlation, dtype=float)
    if matrix.shape != (size, size) or not np.array_equal(
        np.diag(matrix), np.ones(size)
    ):
        raise ShockDeclarationError(
            "Correlation must be square with unit diagonal in shock order."
        )
    _correlation_pivots(matrix)


def _fingerprint(value: tuple[object, ...]) -> str:  # noqa: PAN001 - semantic fingerprints classify arbitrary nested declarations
    from _lcm.solution.fingerprint import (  # noqa: PLC0415 - hasher imports model
        _semantic_fingerprint,
    )

    return _semantic_fingerprint(value)


def _runtime_support_validity(*, corr: Float2D, support: tuple[bool, ...]) -> BoolND:
    """A traced support check supplements mandatory exact host admission."""
    residual = corr
    valid = (
        jnp.all(jnp.isfinite(corr))
        & jnp.all(corr == corr.T)
        & jnp.all(jnp.diag(corr) == 1)
    )
    for column, zero in enumerate(support):
        if zero:
            # Exact host admission owns zero pivots, rather than rounded residuals.
            continue
        pivot = residual[column, column]
        valid &= pivot > 0
        denominator = jnp.where(pivot > 0, pivot, 1.0)
        vector = residual[:, column]
        residual = residual - vector[:, None] * vector[None, :] / denominator
    return valid


def _correlation_record(
    correlation: tuple[tuple[float, ...], ...]
    | Float2D
    | HostArray
    | _Independence
    | None,
) -> tuple[str, tuple[tuple[float, ...], ...] | None]:
    if correlation is _Independence.DEFAULT:
        return "independent", None
    if correlation is None:
        return "runtime", None
    return "fixed", tuple(tuple(float(x) for x in row) for row in correlation)
