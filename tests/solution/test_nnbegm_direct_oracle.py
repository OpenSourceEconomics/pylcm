"""Nested NB-EGM inner kernels agree with the independent scalar oracle."""

import pytest

from tests.solution import test_nbegm_direct_oracle as oracle


@pytest.mark.parametrize("route", oracle._NNBEGM_ROUTES, ids=lambda route: route.name)
def test_direct_oracle_covers_the_nnbegm_inner_contexts(route: oracle._Route) -> None:
    """The nested solver's keeper and adjuster kernels agree with the oracle."""
    oracle._assert_nnbegm_route_agrees_with_oracle(route)
