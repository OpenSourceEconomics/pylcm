"""Host cohort indices retain their int32 contract under runtime type checking."""

import numpy as np

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
