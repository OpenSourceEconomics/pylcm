"""BQSEGM's all-jump route requires an additive (unit-slope) budget.

The pure-jump step recovers the budget from per-interval intercepts assuming
cash-on-hand has unit slope in the liquid (Euler) state — the shape of an additive
subsidy/tax cliff. A budget with a non-unit liquid slope that declares only jump
breakpoints would be silently mis-solved, so it is rejected at model build. The
unit-slope check is scoped to exactly this path — liquid-direct, non-ride, all-jump
— so it does not touch the legitimate cases where a non-unit slope is expected:
derived-variable and ride-along schedules (whose asset-space slope is recovered by
the preimage machinery) and floored/clipped budgets (which carry a `continuous_kink`
and route to the mixed step).
"""

import pytest

from lcm.exceptions import RegimeInitializationError
from tests.test_models import bqsegm_jump_schedule_toy as toy


def test_all_jump_schedule_with_a_non_unit_liquid_slope_is_rejected_at_build() -> None:
    """A jump-only schedule over a non-additive budget fails model build."""
    with pytest.raises(RegimeInitializationError, match=r"slope|additive"):
        toy.build_model(variant="bqsegm", non_additive=True, n_liquid=40, n_savings=40)


def test_schedule_over_a_nonlinear_budget_is_rejected_at_build() -> None:
    """A budget that is smooth but not affine in the liquid state fails model build.

    The per-interval affine segment is recovered from the budget's slope and value
    at one interior point, exact only for an affine budget; a curved budget is
    mis-tangented everywhere else, so the build refuses it."""
    with pytest.raises(RegimeInitializationError, match=r"affine|second derivative"):
        toy.build_model(variant="bqsegm", nonlinear=True, n_liquid=40, n_savings=40)


def test_all_jump_schedule_with_a_unit_liquid_slope_builds() -> None:
    """The additive jump-only schedule (unit liquid slope) builds without error."""
    toy.build_model(variant="bqsegm", non_additive=False, n_liquid=40, n_savings=40)
