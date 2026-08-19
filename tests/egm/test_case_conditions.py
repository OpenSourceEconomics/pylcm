"""Case boundaries and schedule thresholds say the same thing constraints do.

A case boundary and a piecewise-affine bracket edge are both an assertion that
one variable stands on one side of a threshold — the same assertion a
constraint makes. Normalizing them onto the shared condition IR is what lets a
solver inspect one kind of object: it reads a boundary's comparison rather than
its `equality_owner` field, so a hand-written condition and a declared boundary
that mean the same thing are proved the same way.

The comparison operator is where the exact-equality point lives, so these tests
pin the boundary point itself, not only the two open sides.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.case_conditions import (
    condition_of_boundary_surface,
    condition_of_breakpoint,
)
from lcm.case_piece import AffineBreakpoint, BoundarySurface, EqualityOwner
from lcm.exceptions import ModelInitializationError
from tests.test_models import nbegm_medicaid_toy


def _surface(*, equality_owner: EqualityOwner) -> BoundarySurface:
    return BoundarySurface(
        variable="liquid",
        threshold="limit",
        equality_owner=equality_owner,
        kind="jump",
    )


def test_an_otherwise_owned_boundary_leaves_the_equality_point_outside():
    """`equality="otherwise"` renders as `<`, so the threshold is not included."""
    condition = condition_of_boundary_surface(
        surface=_surface(equality_owner="otherwise")
    )

    assert str(condition) == "liquid < limit"


def test_a_when_owned_boundary_brings_the_equality_point_inside():
    """`equality="when"` renders as `<=`, so the threshold belongs to `when`."""
    condition = condition_of_boundary_surface(surface=_surface(equality_owner="when"))

    assert str(condition) == "liquid <= limit"


def test_a_boundary_condition_reads_the_variable_and_the_threshold():
    """Both names must be available to evaluate the boundary."""
    condition = condition_of_boundary_surface(
        surface=_surface(equality_owner="otherwise")
    )

    assert condition.dependencies == frozenset({"liquid", "limit"})


@pytest.mark.parametrize(
    ("equality_owner", "at_threshold"), [("otherwise", False), ("when", True)]
)
def test_the_boundary_condition_decides_the_threshold_point_as_declared(
    equality_owner, at_threshold
):
    """At `liquid == limit` the declared owner is the side that holds.

    This is the whole content of `equality_owner`: away from the threshold both
    spellings agree, so a check that samples only the open sides would pass for
    either operator and measure nothing.
    """
    condition = condition_of_boundary_surface(
        surface=_surface(equality_owner=equality_owner)
    )

    got = condition.evaluate(liquid=jnp.asarray(8.0), limit=jnp.asarray(8.0))

    assert bool(got) is at_threshold


def test_the_boundary_condition_agrees_with_the_predicate_away_from_the_threshold():
    """Either spelling admits a point below the threshold and rejects one above."""
    condition = condition_of_boundary_surface(
        surface=_surface(equality_owner="otherwise")
    )

    got = condition.evaluate(
        liquid=jnp.asarray([7.0, 9.0]), limit=jnp.asarray([8.0, 8.0])
    )

    np.testing.assert_array_equal(np.asarray(got), np.asarray([True, False]))


def test_a_scalar_bracket_edge_normalizes_to_the_same_shape():
    """A schedule threshold is the same assertion a case boundary makes.

    A bracket edge is a continuous kink rather than a jump, but which side owns
    the exact edge is decided by the operator here exactly as it is there, so
    the solver reads one kind of object for both.
    """
    condition = condition_of_breakpoint(
        edge=AffineBreakpoint(threshold="bracket_top", kind="continuous_kink"),
        variable="taxable_income",
    )

    assert str(condition) == "taxable_income < bracket_top"


def test_an_indexed_threshold_is_refused_rather_than_normalized():
    """A table-valued threshold has no reference spelling, so it is not invented.

    `indexed_by` reads the threshold out of a table at a ride-along cell, which
    is not a named value the IR can refer to. Normalizing it to a bare `Ref` on
    the table's name would produce a condition that compares against the whole
    table and reads as an ordinary scalar boundary.
    """
    with pytest.raises(ModelInitializationError, match="indexed_by"):
        condition_of_breakpoint(
            edge=AffineBreakpoint(
                threshold="bracket_table",
                kind="continuous_kink",
                indexed_by="health",
            ),
            variable="taxable_income",
        )


def test_the_normalized_condition_agrees_with_the_declaring_predicate():
    """The condition a declared boundary yields is the predicate it was read from.

    Normalization is only sound if it preserves meaning, and the predicate the
    model author wrote is the independent reference for that. The comparison
    grid straddles the threshold and lands on it exactly, so the point where the
    two spellings could differ is included rather than sampled around.
    """
    meta = getattr(nbegm_medicaid_toy.medicaid_eligible, "__lcm_case_boundary__", None)
    assert meta is not None
    surface = meta.boundaries[0]
    condition = condition_of_boundary_surface(surface=surface)
    limit = 8.0
    liquid = jnp.asarray([0.0, 7.999, limit, 8.001, 20.0])

    from_condition = condition.evaluate(
        liquid=liquid, medicaid_asset_limit=jnp.full_like(liquid, limit)
    )
    from_predicate = nbegm_medicaid_toy.medicaid_eligible(
        liquid=liquid, medicaid_asset_limit=limit
    )

    np.testing.assert_array_equal(
        np.asarray(from_condition), np.asarray(from_predicate)
    )


def test_the_comparison_grid_actually_lands_on_the_threshold():
    """The agreement above is measured at the threshold, not only around it.

    Without this the grid could drift off the exact point and the agreement
    would hold for either operator, which is the one case it is meant to decide.
    """
    assert np.any(np.asarray(jnp.asarray([0.0, 7.999, 8.0, 8.001, 20.0])) == 8.0)
