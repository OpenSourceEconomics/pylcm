"""Host cohort indices retain their int32 contract under runtime type checking."""

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from _lcm.simulation.initial_conditions import _gather_feasibility_inputs


def test_feasibility_gather_accepts_host_int32_indices() -> None:
    """An empty state mapping exercises the typed boundary without a model solve."""
    got = _gather_feasibility_inputs(
        states={},
        indices=np.asarray([0], dtype=np.int32),
        periods=np.asarray([0], dtype=np.int32),
        needs_period=False,
    )
    assert got == {}


@pytest.mark.parametrize("argument", ["indices", "periods"])
@pytest.mark.parametrize("dtype", [np.int64, np.float32])
def test_feasibility_gather_rejects_wrong_host_dtype(
    *, argument: str, dtype: type[np.int64 | np.float32]
) -> None:
    """Host dtype checks must not disappear when fixing annotation decoration."""
    inputs = {
        "indices": np.asarray([0], dtype=np.int32),
        "periods": np.asarray([0], dtype=np.int32),
    }
    inputs[argument] = np.asarray([0], dtype=dtype)
    with pytest.raises(BeartypeCallHintParamViolation):
        _gather_feasibility_inputs(states={}, needs_period=False, **inputs)
