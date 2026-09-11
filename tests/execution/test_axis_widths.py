"""`ExecutionConfig.axis_widths` fixes a planner axis width per axis name."""

from types import MappingProxyType

import pytest

from lcm import ExecutionConfig


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
