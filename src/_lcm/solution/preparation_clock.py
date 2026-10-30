"""Original period upload and age gather shared by dated preparation."""

from collections.abc import Mapping
from functools import partial

import numpy as np

from _lcm.materialization import DatedGenerationMaterializer
from _lcm.time import TimeAxis
from lcm.typing import FloatND, IntND, ReferenceName, ScalarFloat, ScalarInt


def _case_u_clock_age(
    *, values: FloatND | IntND, index: int
) -> ScalarFloat | ScalarInt:
    """Original clock-array indexing, before any economic shape callback."""
    return values[index]


def prepare_case_u_clock(
    *, period: int, ages: TimeAxis, materializer: DatedGenerationMaterializer
) -> Mapping[ReferenceName, ScalarFloat | ScalarInt]:
    """Admit the original scalar upload and separate age gather in that order."""
    actual_period = materializer(
        value=np.asarray(period, dtype=np.int32),
        dtype=np.dtype(np.int32),
        name="case_u_period",
    )
    actual_age = materializer.evaluate_validation(
        function=partial(_case_u_clock_age, index=period),
        arguments={"values": ages.values},
    )
    return {"period": actual_period, "age": actual_age}
