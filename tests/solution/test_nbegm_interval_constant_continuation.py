"""NBEGM's continuation is interval-constant when a co-state law reads liquid.

When a carried state's law of motion reads the current liquid (Euler) state, NBEGM
binds the liquid state to each interval's node and reuses that continuation row
across the interval. That is exact only when the law's liquid dependence is
piecewise-constant — a level switched at a threshold, whose derivative between
breakpoints is zero. A smoothly varying dependence makes the midpoint-bound row
wrong for the interval's other liquid points, so it is refused at model build.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from _lcm.solution.nbegm import (
    _fail_if_liquid_reading_next_state_varies_within_interval,
)
from lcm.exceptions import RegimeInitializationError
from tests.test_models import nbegm_ride_discrete_toy as ride_toy


def test_costate_law_varying_smoothly_in_liquid_is_rejected_at_build() -> None:
    """A co-state whose law varies smoothly in the liquid state fails model build."""
    with pytest.raises(
        RegimeInitializationError, match=r"liquid|interval|continuation"
    ):
        ride_toy.build_model(
            variant="nbegm",
            n_liquid=12,
            liquid_max=30.0,
            n_savings=20,
            savings_max=28.0,
            n_consumption=8,
            costate_reads_liquid=True,
            costate_smooth=True,
        )


def test_costate_law_piecewise_constant_in_liquid_builds() -> None:
    """A co-state whose law switches at a liquid threshold builds without error."""
    ride_toy.build_model(
        variant="nbegm",
        n_liquid=12,
        liquid_max=30.0,
        n_savings=20,
        savings_max=28.0,
        n_consumption=8,
        costate_reads_liquid=True,
        costate_smooth=False,
    )


def test_transition_prob_varying_smoothly_in_liquid_is_rejected_at_build() -> None:
    """A regime-transition probability varying smoothly in liquid fails model build."""
    with pytest.raises(
        RegimeInitializationError, match=r"regime-transition probabilities"
    ):
        ride_toy.build_model(
            variant="nbegm",
            n_liquid=12,
            liquid_max=30.0,
            n_savings=20,
            savings_max=28.0,
            n_consumption=8,
            transition_reads_liquid=True,
            transition_smooth=True,
        )


def test_transition_prob_piecewise_constant_in_liquid_builds() -> None:
    """A survival probability switched at a liquid threshold builds without error."""
    ride_toy.build_model(
        variant="nbegm",
        n_liquid=12,
        liquid_max=30.0,
        n_savings=20,
        savings_max=28.0,
        n_consumption=8,
        transition_reads_liquid=True,
        transition_smooth=False,
    )


def test_costate_law_the_probe_cannot_evaluate_is_rejected_at_build() -> None:
    """A liquid-reading co-state law the constancy probe cannot differentiate
    fails model build — the interval path never assumes an unverifiable law is
    piecewise-constant."""
    with pytest.raises(RegimeInitializationError, match=r"probe|verify|constan"):
        ride_toy.build_model(
            variant="nbegm",
            n_liquid=12,
            liquid_max=30.0,
            n_savings=20,
            savings_max=28.0,
            n_consumption=8,
            costate_reads_liquid=True,
            costate_unprobeable=True,
        )


def test_unprobeable_budget_builds_with_warning_under_assume_declared() -> None:
    """`probe_failure="assume_declared"` turns an unverifiable-probe rejection
    into a loud warning: the model builds and the warning names the asserted
    precondition."""
    with pytest.warns(UserWarning, match=r"assume_declared"):
        ride_toy.build_model(
            variant="nbegm",
            n_liquid=12,
            liquid_max=30.0,
            n_savings=20,
            savings_max=28.0,
            n_consumption=8,
            costate_reads_liquid=True,
            costate_unprobeable=True,
            probe_failure="assume_declared",
        )


def test_constancy_probe_sweeps_each_discrete_arguments_actual_grid_codes():
    """A law that is liquid-dependent only at an unswept discrete code is rejected.

    The probe fills integer-coded arguments from a small set of synthetic
    constants and ramps. A law whose liquid derivative vanishes at every one of
    those values but is nonzero at another valid grid code is interval-varying on
    real cells, so the probe must sweep each discrete argument over its grid's
    actual codes to catch it.
    """

    def next_tracker(tracker, liquid, phase):
        # d/d liquid = 0.1 * (phase-1)(phase-3)(phase-5)(phase-7): zero at every
        # synthetic integer fill the probe's constants and ramps produce, nonzero
        # at the valid codes 0 and 2.
        gate = (phase - 1) * (phase - 3) * (phase - 5) * (phase - 7)
        return tracker + 0.1 * liquid * gate

    def compute_regime_transition_probs(age):
        return jnp.asarray(age) * 0.0

    plan = SimpleNamespace(
        carry_targets=("tracker",),
        child_reads={"tracker": SimpleNamespace(next_state_func=next_tracker)},
        compute_regime_transition_probs=compute_regime_transition_probs,
    )
    with pytest.raises(RegimeInitializationError, match="varies smoothly"):
        _fail_if_liquid_reading_next_state_varies_within_interval(
            continuation_plan=plan,
            liquid_name="liquid",
            regime_name="toy",
            int_arg_values={"phase": (0, 1, 2, 3)},
        )
