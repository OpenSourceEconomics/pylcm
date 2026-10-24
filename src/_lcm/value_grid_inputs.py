"""Read the values a caller gives for a value grid as the codes the engine uses.

A `DiscreteGrid(values=...)` variable is a code `0..n-1` inside the engine. At
the user-facing boundaries (initial conditions, params indices, policy lookup) a
caller names its nodes by their values. The rules are strict:

- an integer grid matches integers exactly; a float equal to an integer counts;
- a float grid matches within the grid's `tolerance`, exactly when it is zero;
- a pandas Categorical must list exactly the grid's values, in order, as its
  categories, as `SimulationResult.to_dataframe` publishes them;
- anything else, strings and booleans included, raises naming the value.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

import numpy as np
import pandas as pd
from jax import Array
from jax.typing import ArrayLike

from _lcm.grids import DiscreteGrid
from _lcm.grids.discrete import MISSING_CAT_CODE
from _lcm.typing import RegimeName, RegimeNamesToIds
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    StateOrActionName,
    UserInitialConditions,
    UserInitialConditionsInput,
)

# What a caller may give for the nodes of one value-grid variable.
type UserGridValues = ArrayLike | pd.Series | pd.Categorical | pd.Index


def encode_grid_values(
    *,
    grid: DiscreteGrid,
    values: UserGridValues,
    name: str,
    error_type: type[Exception] = ValueError,
) -> np.ndarray:
    """Return the code of the node each value names.

    Args:
        grid: A grid declared with `values`.
        values: The values, as numbers or as a Categorical over the grid's values.
        name: What the values are, for the error message.
        error_type: The exception raised for values that name no node.

    Returns:
        The codes as an int32 NumPy array of the shape of `values`.

    Raises:
        error_type: If a value names no node, is not a number, or a Categorical's
            categories are not exactly the grid's values in order.

    """
    declared = grid.values  # noqa: PD011
    if declared is None:
        msg = f"{name}: only a grid declared with `values` reads values."
        raise TypeError(msg)
    if _is_categorical(values):
        return _codes_of_categorical(
            declared=declared,
            is_integer=grid.values_to_jax().dtype.kind == "i",
            categorical=cast("pd.Categorical | pd.Series", values),
            name=name,
            error_type=error_type,
        )
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise error_type(
            f"{name}: the values of a value grid must be numbers; got {array.dtype} "
            f"values {_sample(array)}."
        )
    if grid.values_to_jax().dtype.kind == "i" and array.dtype.kind == "f":
        integral = np.isfinite(array) & (array == np.trunc(array))
        codes = np.asarray(grid.encode(np.where(integral, array, 0).astype(np.int64)))
        codes = np.where(integral, codes, MISSING_CAT_CODE)
    else:
        codes = np.asarray(grid.encode(array))
    off = codes == MISSING_CAT_CODE
    if off.any():
        raise error_type(
            f"{name}: {_sample(array[off])} are not values of its grid "
            f"{list(declared)}."
        )
    return codes.astype(np.int32)


def encode_grid_codes(
    *,
    grid: DiscreteGrid,
    codes: ArrayLike | pd.Series | pd.Index,
    name: str,
    error_type: type[Exception] = ValueError,
) -> np.ndarray:
    """Return integer codes after checking each lies in `0..n-1`.

    Raises:
        error_type: If a code is not an integer of the grid's code range.

    """
    if _is_categorical(codes):
        raise error_type(f"{name}: give integer codes, not a Categorical.")
    array = np.asarray(codes)
    valid = (
        array.dtype.kind in "iu"
        or (array.dtype.kind == "f" and bool(np.all(array == np.trunc(array))))
    ) and bool(np.all((array >= 0) & (array < len(grid.codes))))
    if not valid:
        raise error_type(
            f"{name}: codes must be integers in [0, {len(grid.codes)}); got "
            f"{_sample(array)}."
        )
    return array.astype(np.int32)


def value_grids_by_regime(
    user_regimes: Mapping[RegimeName, UserRegime],
) -> MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, DiscreteGrid]]:
    """Return each regime's states and actions declared with `values`.

    A carried state (`Phased(solve=..., simulate=grid)`) counts by its simulate
    grid, the one its initial conditions and outputs use.
    """
    return MappingProxyType(
        {
            regime_name: MappingProxyType(
                {
                    name: grid
                    for name, entry in {**regime.states, **regime.actions}.items()
                    for grid in [entry.simulate if isinstance(entry, Phased) else entry]
                    if isinstance(grid, DiscreteGrid) and grid.values is not None  # noqa: PD011
                }
            )
            for regime_name, regime in user_regimes.items()
        }
    )


def encode_initial_values(
    *,
    initial_conditions: UserInitialConditionsInput,
    user_regimes: Mapping[RegimeName, UserRegime],
    regime_names_to_ids: RegimeNamesToIds,
    use_categorical_codes: bool,
    error_type: type[Exception] = ValueError,
) -> UserInitialConditions:
    """Return initial conditions as arrays, value-grid states as their codes.

    A pandas Series becomes its NumPy array. Without `use_categorical_codes` the
    values of a value-grid state become codes; a row is read only if its
    `"regime_id"` names a regime declaring the state, and every other row of that
    state gets `MISSING_CAT_CODE`, like a state a subject's regime does not have.
    Without `"regime_id"` every row is read. With `use_categorical_codes` every
    entry already holds codes, so a Categorical, which holds values, is refused.

    Raises:
        error_type: If a read value names no node of its grid, or a Categorical
            is given with `use_categorical_codes`.

    """
    grids = value_grids_by_regime(user_regimes)
    regime_ids = (
        np.asarray(initial_conditions["regime_id"])
        if "regime_id" in initial_conditions
        else None
    )
    encoded: dict[str, Array | np.ndarray] = {}
    for name, values in initial_conditions.items():
        label = f"initial_conditions[{name!r}]"
        holders = {
            regime_name: regime_grids[name]
            for regime_name, regime_grids in grids.items()
            if name in regime_grids
        }
        if use_categorical_codes and _is_categorical(values):
            raise error_type(
                f"{label}: with `use_categorical_codes=True` give integer codes, "
                "not a Categorical."
            )
        if use_categorical_codes or not holders:
            encoded[name] = (
                np.asarray(values)
                if isinstance(values, pd.Series | pd.Categorical)
                else values
            )
            continue
        # A name declared with values carries the same grid in every regime.
        grid = next(iter(holders.values()))
        if regime_ids is None:
            encoded[name] = encode_grid_values(
                grid=grid, values=values, name=label, error_type=error_type
            )
            continue
        read = np.isin(
            regime_ids, [int(regime_names_to_ids[holder]) for holder in holders]
        )
        codes = np.full(read.shape, MISSING_CAT_CODE, dtype=np.int32)
        codes[read] = encode_grid_values(
            grid=grid,
            values=_select_rows(values=values, rows=read),
            name=label,
            error_type=error_type,
        )
        encoded[name] = codes
    return MappingProxyType(encoded)


def _codes_of_categorical(
    *,
    declared: tuple[int | float, ...],
    is_integer: bool,
    categorical: pd.Categorical | pd.Series,
    name: str,
    error_type: type[Exception],
) -> np.ndarray:
    """Return a Categorical's codes, or raise unless it lists `declared`."""
    categorical = pd.Categorical(categorical)
    categories = categorical.categories
    kinds = "iu" if is_integer else "f"
    if categories.dtype.kind not in kinds or categories.tolist() != list(declared):
        raise error_type(
            f"{name}: a Categorical for a value grid must have the grid's values "
            f"{list(declared)}, in that order, as its categories; got "
            f"{categories.tolist()}."
        )
    codes = np.asarray(categorical.codes)
    if (codes < 0).any():
        raise error_type(f"{name}: missing entries are not values of its grid.")
    return codes.astype(np.int32)


def _select_rows(*, values: UserGridValues, rows: np.ndarray) -> UserGridValues:
    """Return the entries of `values` at the boolean `rows`, keeping a Categorical."""
    if isinstance(values, pd.Series):
        return values[rows]
    if isinstance(values, pd.Categorical):
        return values[rows]
    return np.asarray(values)[rows]


def _is_categorical(values: UserGridValues) -> bool:
    """Return whether `values` is a pandas Categorical or a categorical Series."""
    return isinstance(values, pd.Categorical) or (
        isinstance(values, pd.Series) and isinstance(values.dtype, pd.CategoricalDtype)
    )


def _sample(array: np.ndarray) -> list[float | int | str | bool]:
    """Return up to eight distinct offending values, in order of appearance."""
    return list(dict.fromkeys(np.asarray(array).ravel().tolist()))[:8]
