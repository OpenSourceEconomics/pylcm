"""User-facing type aliases.

The model-authoring aliases — jaxtyping array shapes, `Period`, `Age` — and the
boundary `User*` forms accepted by user-constructor methods (`Model.__init__`,
`Model.solve`, `Model.simulate`, `AgeGrid.__init__`). Engine-internal aliases
and the structural protocols live in `_lcm.typing`.
"""

from collections.abc import Mapping
from fractions import Fraction
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from jax import Array
from jaxtyping import Bool, Float, Int32, Scalar

from lcm.params import UserMappingLeaf, UserSequenceLeaf

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

# Zero-dimensional JAX scalars — pylcm's canonical scalar form post boundary cast.
type ScalarInt = Int32[Scalar, ""]
type ScalarFloat = Float[Scalar, ""]
type ScalarBool = Bool[Scalar, ""]

type Period = ScalarInt
type Age = ScalarInt | ScalarFloat


# Boundary form accepted by `AgeGrid.__init__` for `start`, `stop`, and
# `exact_values` entries — converted to canonical JAX scalars internally.
type UserAge = int | Fraction


# Boundary form accepted by `AgeGrid.__init__` for `step`: a string matching
# the grammar `(\d+)?[YQM]` — an optional positive-integer multiplier followed
# by a unit (`Y` year, `Q` quarter, `M` month). Examples: `"Y"`, `"2Q"`, `"6M"`.
type AgeStep = str


# Boundary form of initial conditions — accepted by `Model.simulate` and
# canonicalized by `canonicalize_initial_conditions`. Keys are state names plus
# the literal `"regime_id"`.
type UserInitialConditions = Mapping[str, Array | np.ndarray]


# Boundary leaf type — accepted by `Model.__init__` / `Model.solve` /
# `Model.simulate` and canonicalized by `cast_params_to_canonical_dtypes`.
type _UserParamsLeaf = (
    bool
    | int
    | float
    | FloatND
    | IntND
    | BoolND
    | np.ndarray
    | pd.Series
    | UserMappingLeaf
    | UserSequenceLeaf
)
type UserParams = Mapping[
    str,
    _UserParamsLeaf | Mapping[str, _UserParamsLeaf | Mapping[str, _UserParamsLeaf]],
]


# User-facing template; types rendered as strings. Keys are regime names, then
# function names.
type UserFacingParamsTemplate = dict[str, dict[str, dict[str, str]]]


@runtime_checkable
class UserFunction(Protocol):
    """A function provided by the user.

    Used for both type checking and beartype runtime checks on perimeter
    constructors. Any callable satisfies this protocol structurally.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401
