"""Bookkeeping guards, not restrictions on the original cost family."""
from collections.abc import Mapping
from math import inf

DEFAULTS = {'segments': 2_000_000, 'groupings': 100_000, 'frontier_points': 5000,
            'forest_states': 2_000_000, 'forest_points': 5_000_000,
            'symbolic_nodes': 4_000_000, 'symbolic_operations': 200_000_000,
            'symbolic_leaf_candidates': 2_000_000, 'separator_points': 50_000}


def count_limits(overrides=None):
    if overrides is not None and not isinstance(overrides, Mapping):
        raise ValueError('Cost-search bookkeeping limits require an object')
    supplied = dict(overrides or {})
    if supplied.keys() - DEFAULTS.keys():
        raise ValueError('Unknown cost-search bookkeeping limit')
    if any(value is not None and not (type(value) is int and value > 0)
           and not (type(value) is float and value == inf) for value in supplied.values()):
        raise ValueError('Cost-search bookkeeping limits must be positive integers or null for no limit')
    return {key: inf if value is None else value for key, value in (DEFAULTS | supplied).items()}
