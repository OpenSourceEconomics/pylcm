"""Smallest budgeted simulation with a value-only pair in the solved domain."""

from tests.simulation.test_forward_domain_budget import (
    test_budgeted_forward_domain as _run_case,
)


def test_value_only_node_does_not_enter_budgeted_forward_planning():
    """A value-only pair is never profiled as a forward unit."""
    _run_case(promote=False, reverse=False, width=2, workers=1)
