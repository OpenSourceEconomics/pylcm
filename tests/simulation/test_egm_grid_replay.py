"""Explicit Cartesian replay is assessed separately from original EGM candidates."""

import numpy as np
import pytest

from lcm import (
    IrregSpacedGrid,
)
from lcm.exceptions import UnsupportedOperationError
from lcm.solvers import DCEGM
from tests.conftest import EXACT_KERNEL_SKIP_REASON

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def test_grid_replay_refuses_certified_at_declaration() -> None:
    """Certified replay refuses Cartesian omission before a Model is prepared."""
    with pytest.raises(UnsupportedOperationError, match=r"certified.*grid"):
        DCEGM(
            savings_grid=IrregSpacedGrid(points=np.asarray([0.125, 0.5, 1.0])),
            replay="grid",
            accuracy="certified",
        )
