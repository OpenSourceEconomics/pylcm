"""Validate public simulation coordinates before lowering to engine inputs."""

import numpy as np
import pandas as pd

from _lcm.time import TimeAxis, coordinate_kind
from lcm.exceptions import InvalidInitialConditionsError
from lcm.typing import UserInitialConditions


def lower_initial_time(
    *, initial_conditions: UserInitialConditions | pd.DataFrame, ages: TimeAxis
) -> UserInitialConditions | pd.DataFrame:
    """Check the coordinate kind and preserve exact integer period starts.

    The simulation engine's historical `age` slot transports the clock
    coordinate. Renaming the public period column here does not allocate a
    device array or assign an age to the model.
    """
    kind = coordinate_kind(ages)
    wrong = "age" if kind == "period" else "period"
    if wrong in initial_conditions:
        raise InvalidInitialConditionsError(
            f"This {kind} model requires the {kind!r} coordinate; {wrong!r} is invalid."
        )
    if kind == "age":
        return initial_conditions
    if "period" not in initial_conditions:
        raise InvalidInitialConditionsError("Period models require a 'period' column.")
    values = np.asarray(initial_conditions["period"])
    if (
        values.ndim != 1
        or values.dtype.kind not in "iu"
        or np.any(values < 0)
        or np.any(values >= ages.n_periods)
    ):
        raise InvalidInitialConditionsError(
            f"Initial period must contain integers in [0, {ages.n_periods}); "
            "booleans and fractional values are invalid."
        )
    if isinstance(initial_conditions, pd.DataFrame):
        return initial_conditions.rename(columns={"period": "age"})
    return {
        ("age" if name == "period" else name): value
        for name, value in initial_conditions.items()
    }
