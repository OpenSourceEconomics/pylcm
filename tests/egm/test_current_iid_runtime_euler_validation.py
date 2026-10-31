"""A runtime-supplied Euler-state grid is refused by an ordinary EGM model."""

from dataclasses import replace

import pytest

from lcm import IrregSpacedGrid, Model
from lcm.exceptions import ModelInitializationError
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.solution.test_current_iid_off_node_replay import _interior_declarations

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def test_ordinary_model_retains_runtime_euler_grid_refusal() -> None:
    """A model without a current shock refuses a child grid known only at runtime."""
    declarations = _interior_declarations()
    regimes = declarations["regimes"]
    child = replace(
        regimes["child"],
        states={"wealth": IrregSpacedGrid(n_points=2)},
        shocks=(),
    )
    declarations["regimes"] = {**regimes, "child": child}
    declarations["shocks"] = {}
    with pytest.raises(
        ModelInitializationError, match="supplies its points only at runtime"
    ):
        Model(**declarations)
