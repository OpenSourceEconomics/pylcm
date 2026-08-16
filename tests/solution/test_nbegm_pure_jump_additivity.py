"""NBEGM's all-jump route requires an additive (unit-slope) budget.

The pure-jump step recovers the budget from per-interval intercepts assuming
cash-on-hand has unit slope in the liquid (Euler) state — the shape of an additive
subsidy/tax cliff. A budget with a non-unit liquid slope that declares only jump
breakpoints would be silently mis-solved, so it is refused. The
unit-slope check is scoped to exactly this path — liquid-direct, non-ride, all-jump
— so it does not touch the legitimate cases where a non-unit slope is expected:
derived-variable and ride-along schedules (whose asset-space slope is recovered by
the preimage machinery) and floored/clipped budgets (which carry a `continuous_kink`
and route to the mixed step).
"""

import pytest

from _lcm.solution.preconditions import check_solver_params
from lcm.exceptions import RegimeInitializationError
from lcm.model import Model
from tests.test_models import nbegm_jump_schedule_toy as toy


def _check_probes(model: Model, params: dict) -> None:
    """Run the solver's parameter-dependent preconditions, and nothing else."""
    check_solver_params(
        regimes=model._regimes,
        flat_params=model._process_params(params),
    )


def test_all_jump_schedule_with_a_non_unit_liquid_slope_is_refused() -> None:
    """A jump-only schedule over a non-additive budget is refused."""
    model = toy.build_model(
        variant="nbegm", non_additive=True, n_liquid=40, n_savings=40
    )

    with pytest.raises(RegimeInitializationError, match=r"slope|additive"):
        _check_probes(model, toy.build_params(non_additive=True))


def test_schedule_over_a_nonlinear_budget_is_refused() -> None:
    """A budget that is smooth but not affine in the liquid state is refused.

    The per-interval affine segment is recovered from the budget's slope and value
    at one interior point, exact only for an affine budget; a curved budget is
    mis-tangented everywhere else, so the solve refuses it."""
    model = toy.build_model(variant="nbegm", nonlinear=True, n_liquid=40, n_savings=40)

    with pytest.raises(RegimeInitializationError, match=r"affine|second derivative"):
        _check_probes(model, toy.build_params(nonlinear=True))


def test_all_jump_schedule_with_a_unit_liquid_slope_builds() -> None:
    """The additive jump-only schedule (unit liquid slope) builds without error."""
    toy.build_model(variant="nbegm", non_additive=False, n_liquid=40, n_savings=40)
