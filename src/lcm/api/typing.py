"""User-facing type aliases.

Boundary forms accepted by user-constructor methods (`Model.__init__`,
`Model.solve`, `Model.simulate`, `AgeGrid.__init__`). Each alias prefixed
`User*` is the input form before canonicalization; the canonical
post-processing form lives in `lcm.typing`.
"""

from collections.abc import Mapping
from fractions import Fraction
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from jax import Array

from lcm.params import UserMappingLeaf
from lcm.params.sequence_leaf import UserSequenceLeaf
from lcm.typing import (
    BoolND,
    FloatND,
    FunctionName,
    IntND,
    RegimeName,
    StateName,
)

# Boundary form accepted by `AgeGrid.__init__` for `start`, `stop`, and
# `exact_values` entries — converted to canonical JAX scalars internally.
type UserAge = int | Fraction


# Boundary form of initial conditions — accepted by `Model.simulate` and
# canonicalized by `canonicalize_initial_conditions`.
type UserInitialConditions = Mapping[
    StateName | Literal["regime_id"], Array | np.ndarray
]


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


# User-facing template; types rendered as strings.
type UserFacingParamsTemplate = dict[RegimeName, dict[FunctionName, dict[str, str]]]


@runtime_checkable
class UserFunction(Protocol):
    """A function provided by the user.

    Used for both type checking and beartype runtime checks on perimeter
    constructors. Any callable satisfies this protocol structurally.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401
