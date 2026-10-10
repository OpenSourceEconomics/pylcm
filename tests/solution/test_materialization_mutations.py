"""Admission of single block-major value reads across value shapes and accessors."""

import pytest

from lcm.exceptions import ExecutionPlanningError
from lcm.typing import FloatND
from tests.solution.test_materialization_admission import _leaf_bytes, _solve

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("n_wealth", [384, 768])
@pytest.mark.parametrize("access", ["value", "mapping"])
def test_materialization_admission_covers_shapes_and_accessors(
    *, n_wealth: int, access: str
) -> None:
    """A value one byte larger than the budget is refused through either accessor."""
    solution = _solve(budget=_leaf_bytes(n_wealth=n_wealth) - 1, n_wealth=n_wealth)

    def read() -> FloatND:
        if access == "value":
            return solution.value(period=0, regime="working")
        return solution.values[0]["working"]

    with pytest.raises(ExecutionPlanningError):
        read()
