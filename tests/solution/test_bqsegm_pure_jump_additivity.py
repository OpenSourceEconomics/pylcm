"""BQSEGM's all-jump route assumes an additive budget — a guard is still owed.

The pure-jump step recovers the budget from per-interval intercepts assuming
cash-on-hand has unit slope in the liquid (Euler) state — the shape of an additive
subsidy/tax cliff. A budget with a non-unit liquid slope that declares only jump
breakpoints is silently mis-solved. A build-time guard should reject it, but a
sound one is non-trivial: the naive "grad of coh in liquid must be 1" check
false-rejects the legitimate cases where the assumption does not apply verbatim —
derived-variable and ride-along schedules (whose asset-space slope is handled by
the preimage machinery, not a unit-slope assumption) and floored/clipped budgets
(whose liquid derivative is 0 below the floor). The rejection is captured here as a
strict `xfail` so the contract is on record and the day a sound guard lands the
test flips to a failure that prompts removing the marker.
"""

import pytest

from lcm.exceptions import RegimeInitializationError
from tests.test_models import bqsegm_jump_schedule_toy as toy


@pytest.mark.xfail(
    reason="Sound build-time additivity guard not yet implemented (see docstring).",
    strict=True,
)
def test_all_jump_schedule_with_a_non_unit_liquid_slope_is_rejected_at_build() -> None:
    """A jump-only schedule over a non-additive budget should fail model build."""
    with pytest.raises(RegimeInitializationError, match=r"slope|additive"):
        toy.build_model(variant="bqsegm", non_additive=True, n_liquid=40, n_savings=40)


def test_all_jump_schedule_with_a_unit_liquid_slope_builds() -> None:
    """The additive jump-only schedule (unit liquid slope) builds without error."""
    toy.build_model(variant="bqsegm", non_additive=False, n_liquid=40, n_savings=40)
