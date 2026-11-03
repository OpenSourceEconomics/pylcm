"""Ride-discrete NB-EGM kernels agree with the independent scalar oracle."""

import pytest

from tests.solution import test_nbegm_direct_oracle as oracle


@pytest.mark.parametrize(
    "route",
    tuple(route for route in oracle._ROUTES if route.name.startswith("ride_discrete")),
    ids=lambda route: route.name,
)
def test_direct_oracle_matches_the_tiled_core(route: oracle._Route) -> None:
    """Value, carry rows, and consumption agree on every ride-discrete route."""
    oracle._assert_route_agrees_with_oracle(route)
