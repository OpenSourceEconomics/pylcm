"""User-facing type aliases.

The model-authoring aliases — jaxtyping array shapes, `Period`, `Age` — the
string-label aliases (`RegimeName`, `StateName`, ...) that document which kind
of name a string slot carries, and the boundary `User*` forms accepted by
user-constructor methods (`Model.__init__`, `Model.solve`, `Model.simulate`,
`AgeGrid.__init__`). The compound mapping aliases and the structural protocols
live in `_lcm.typing`.
"""

from collections.abc import Callable, Mapping
from fractions import Fraction
from typing import TYPE_CHECKING, Literal, TypeAliasType

import numpy as np
import pandas as pd
from jax import Array
from jaxtyping import Bool, Float, Int, Int32, Scalar, Shaped

# Any model dtype; use only for genuinely dtype-polymorphic slots. Parameter
# wrappers import this alias, so define it before importing them below.
type ValueND = Shaped[Array, "..."]

# The coordinate labels of a `TimeVarying`, defined before the parameter
# wrappers for the same reason:
# - a computational period label; a boolean passes here and is refused when
#   the labels are validated;
# - a numeric age label; finiteness is checked when the labels are validated;
# - either of the two.
type PeriodLabel = int | np.integer
type AgeLabel = int | float | Fraction | np.integer | np.floating
type TimeLabel = PeriodLabel | AgeLabel

from lcm.params import TimeVarying, UserMappingLeaf, UserSequenceLeaf  # noqa: E402

if TYPE_CHECKING:
    from _lcm.typing import (
        EconFunction,
        EconFunctionsMapping,  # noqa: F401 - Public lazy re-export.
        FlatParams,  # noqa: F401 - Public lazy re-export.
        FlatRegimeParams,  # noqa: F401 - Public lazy re-export.
    )
    from lcm.initial_nodes import InitialNodes, UserInitialNodes  # noqa: F401

    # Defined beside `AgeRange` in `lcm.transition`, which imports this module;
    # `lcm.__init__` binds it here through `_bind_forward_refs`.
    from lcm.transition import AgeSelector  # noqa: F401  (re-exported)

    # A protocol class or a type alias that `__getattr__` resolves lazily.
    type _EngineAlias = type[EconFunction] | TypeAliasType
    type _InitialNodesClass = type[InitialNodes]
else:
    # The engine's typing module and `lcm.initial_nodes` import this module, so
    # their names exist only for type checkers.
    type _EngineAlias = object
    type _InitialNodesClass = type[object]

type ContinuousState = Float[Array, "..."]
type ContinuousAction = Float[Array, "..."]
type DiscreteState = Int32[Array, "..."]
type DiscreteAction = Int32[Array, "..."]

type FloatND = Float[Array, "..."]
type IntND = Int32[Array, "..."]
type BoolND = Bool[Array, "..."]

type Float1D = Float[Array, "_"]  # noqa: F821
type Int1D = Int32[Array, "_"]  # noqa: F821
type Bool1D = Bool[Array, "_"]  # noqa: F821

type Float2D = Float[Array, "_ _"]
type Int2D = Int32[Array, "_ _"]

# Zero-dimensional JAX scalars — pylcm's canonical scalar form post boundary cast.
type ScalarInt = Int32[Scalar, ""]
type ScalarFloat = Float[Scalar, ""]
type ScalarBool = Bool[Scalar, ""]

# Keep equivalent period annotations identical when DAGs reconcile their names.
Period = ScalarInt
type Age = ScalarInt | ScalarFloat

# `jax.lax.fori_loop` body index. BOTH forms are admitted deliberately: with
# static Python-int bounds the loop really runs in Python under
# `jax.disable_jit()` and hands the body a plain `int`; only under trace does the
# index become a tracer. Since `beartype_package` is registered unconditionally
# (see `lcm/__init__.py`), an array-only hint makes every EAGER call raise a type
# violation rather than compute a wrong answer. The traced counter is also
# materialized at JAX's default int dtype (weak int64 under x64), not pylcm's
# canonical int32, so the array arm stays dtype-agnostic rather than `ScalarInt`.
type LoopIndex = int | Int[Scalar, ""]


# String-label aliases. Runtime-equivalent to `str`; they exist purely to make
# signatures self-documenting about which kind of name a string slot carries.
type RegimeName = str
type Phase = Literal["solve", "simulate"]
type StateName = str
type ActionName = str
type StateOrActionName = str
type ProcessName = str
type FunctionName = str
type ParameterName = str
type MarginRoleName = str
type ReferenceName = (
    StateName | ActionName | FunctionName | ParameterName | MarginRoleName
)
type TransitionFunctionName = str


# Boundary form accepted by `AgeGrid.__init__` for `start`, `inclusive_stop`, and
# `exact_values` entries — converted to canonical JAX scalars internally.
type UserAge = int | Fraction

# Boundary form accepted by `AgeGrid.__init__` for `step`: a string matching
# the grammar `(\d+)?[YQM]` — an optional positive-integer multiplier followed
# by a unit (`Y` year, `Q` quarter, `M` month). Examples: `"Y"`, `"2Q"`, `"6M"`.
type AgeStep = str


# Boundary form of initial conditions — accepted by `Model.simulate` and
# canonicalized by `canonicalize_initial_conditions`. Keys are state names plus
# the literal `"regime_id"`.
type UserInitialConditions = Mapping[
    StateName | Literal["regime_id"], Array | np.ndarray
]


# Boundary leaf type — accepted by `Model.__init__` / `Model.solve` /
# `Model.simulate` and canonicalized by `cast_params_to_canonical_dtypes`.
type UserParamsLeaf = (
    bool
    | int
    | float
    # `AgeGrid.exact_values` yields `int | Fraction`, and an age read off the grid
    # is the natural way to write an age-valued parameter — `lcm_examples.tiny`,
    # `.mortality` and `.iskhakov_et_al_2017` all do exactly that.
    | Fraction
    | FloatND
    # Integer arrays of any width: with x64 enabled, `jnp.array([0])` is int64.
    | Int[Array, "..."]
    | BoolND
    | np.ndarray
    | pd.Series
    | TimeVarying
    | UserMappingLeaf
    | UserSequenceLeaf
)
# Parameter namespaces are recursively nested. Ordinary functions stop after
# `{regime: {function: {parameter: value}}}`; target-owned laws add a target
# level, and a `JointTransition` kernel adds its `support`/`probabilities` role
# before reaching parameter leaves.
type UserParamsNode = UserParamsLeaf | Mapping[str, UserParamsNode]
type UserParams = Mapping[str, UserParamsNode]


# User-facing templates keep the first regime and function/target levels
# structurally visible to type checkers; below them a branch nests as deep as its
# declaration path does — a joint kernel's `support`/`probabilities` role, or an
# `edges` slot's target, gate reference or route — before rendered annotation
# leaves. The `edges` branch maps each source regime to its slots.
type _UserFacingTemplateNode = str | dict[str, _UserFacingTemplateNode]
type UserFacingParamsTemplate = dict[
    RegimeName,
    dict[FunctionName | RegimeName, dict[str, _UserFacingTemplateNode]],
]


# What a user function returns: an array or a scalar, or mappings and tuples of
# them (a regime transition returns probabilities keyed by regime name).
type UserFunctionResult = (
    ValueND
    | float
    | int
    | bool
    | Mapping[str, UserFunctionResult]
    | tuple[UserFunctionResult, ...]
)

# A function provided by the user. Its parameters are resolved by name, so any
# callable qualifies whatever parameters it declares.
type UserFunction = Callable[..., UserFunctionResult]


outer_unchanged: FunctionName = "__outer_unchanged__"
# Sentinel declaring that an outer state is unchanged without adjustment.
# Use it as ``OuterContinuousMargin.no_adjustment`` when the no-adjustment map is
# literally the identity. Any other value is a function name and must resolve in
# the assembled regime DAG. A sentinel, rather than a generated callable, keeps
# the public declaration serialisable and avoids callable-wrapper behaviour under
# the project's beartype claw.


# The engine's typing module imports this one, so the four solver-contract
# aliases it defines are resolved lazily rather than imported at module load.

_ENGINE_ALIASES = frozenset(
    {"EconFunction", "EconFunctionsMapping", "FlatParams", "FlatRegimeParams"}
)


def __getattr__(name: str) -> _EngineAlias:
    """Resolve the solver-contract aliases the engine defines."""
    if name in _ENGINE_ALIASES:
        import _lcm.typing as engine_typing  # noqa: PLC0415

        return getattr(engine_typing, name)
    msg = f"module 'lcm.typing' has no attribute {name!r}"
    raise AttributeError(msg)


def _bind_forward_refs(
    *,
    age_selector: TypeAliasType,
    initial_nodes_cls: _InitialNodesClass,
    user_initial_nodes: TypeAliasType,
) -> None:
    """Bind public declaration types after their modules finish importing.

    The declaration modules import this module themselves. `lcm.__init__`
    calls this helper once they are loaded to make the re-exports available.
    """
    globals()["AgeSelector"] = age_selector
    globals()["InitialNodes"] = initial_nodes_cls
    globals()["UserInitialNodes"] = user_initial_nodes
