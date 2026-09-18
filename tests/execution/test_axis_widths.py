"""`ExecutionConfig.axis_widths` fixes a planner axis width per axis name."""

from types import MappingProxyType

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from lcm import ExecutionConfig
from lcm_examples import precautionary_savings


def test_axis_widths_default_is_an_empty_read_only_mapping() -> None:
    """Without overrides the mapping is empty and immutable."""
    assert ExecutionConfig().axis_widths == MappingProxyType({})


def test_axis_widths_rejects_a_non_positive_width() -> None:
    """A width of zero is refused at construction, naming the axis."""
    with pytest.raises(
        ValueError, match=r"axis_widths\['action_product'\] must be positive"
    ):
        ExecutionConfig(axis_widths={"action_product": 0})


def test_axis_widths_rejects_a_bool_width() -> None:
    """Widths are exact ints, so a bool is refused."""
    with pytest.raises(TypeError, match="exact int"):
        ExecutionConfig(axis_widths={"action_product": True})


def test_axis_widths_is_frozen_after_construction() -> None:
    """The stored mapping cannot be mutated by the caller's original dict."""
    widths = {"action_product": 4}
    config = ExecutionConfig(axis_widths=widths)

    widths["action_product"] = 8

    assert config.axis_widths["action_product"] == 4


@pytest.mark.parametrize("cell_width", [3, 7, 20, 40])
def test_a_cell_tile_width_does_not_move_a_single_bit(*, cell_width: int) -> None:
    """Tiling the cell axis concatenates tiles, so every width returns one result.

    A tile covering a whole coordinate reads that grid directly instead of
    enumerating it by index, which changes the compiled program but must not
    change one bit of the result.
    """

    def solve(*, width: int) -> dict[str, np.ndarray]:
        model = precautionary_savings.create_model(
            n_periods=3,
            shock_type="rouwenhorst",
            wealth_n_points=8,
            consumption_n_points=6,
            execution_config=ExecutionConfig(
                axis_widths={"cell": width, "action_product": 4}
            ),
        )
        params = precautionary_savings.get_params(
            shock_type="rouwenhorst", sigma=0.2, rho=0.9
        )
        result = model.solve(params=params, log_level="off")
        return {
            f"{regime}/{key}": np.asarray(value)
            for regime, per_regime in result.values.items()
            for key, value in per_regime.items()
        }

    got = solve(width=cell_width)
    expected = solve(width=40)
    assert sorted(got) == sorted(expected)
    for key, value in got.items():
        assert_array_equal(value, expected[key], err_msg=key)
