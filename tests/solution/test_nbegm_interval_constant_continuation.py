"""NBEGM's continuation is interval-constant when a co-state law reads liquid.

When a carried state's law of motion reads the current liquid (Euler) state, NBEGM
binds the liquid state to each interval's node and reuses that continuation row
across the interval. That is exact only when the law's liquid dependence is
piecewise-constant — a level switched at a threshold, whose derivative between
breakpoints is zero. A smoothly varying dependence makes the midpoint-bound row
wrong for the interval's other liquid points, so it is refused.

The probe needs parameter values to differentiate the law, so it runs on the first
solve rather than at model build. These tests drive it directly, without the
backward induction that would follow it.
"""

import inspect
from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace
from typing import cast

import jax.numpy as jnp
import pytest

from _lcm.solution.nbegm import (
    _fail_if_liquid_reading_next_state_varies_within_interval,
    _ProbeArguments,
)
from _lcm.solution.preconditions import check_solver_params
from lcm.exceptions import RegimeInitializationError
from lcm.model import Model
from lcm.transition import MarkovTransition
from tests.test_models import nbegm_ride_discrete_toy as ride_toy


def _check_probes(model: Model) -> None:
    """Run the solver's parameter-dependent preconditions, and nothing else."""
    check_solver_params(
        regimes=model._regimes,
        flat_params=model._process_params(ride_toy.build_params()),
    )


def test_costate_law_varying_smoothly_in_liquid_is_refused() -> None:
    """A co-state whose law varies smoothly in the liquid state is refused."""
    model = ride_toy.build_model(
        variant="nbegm",
        n_liquid=12,
        liquid_max=30.0,
        n_savings=20,
        savings_max=28.0,
        n_consumption=8,
        costate_reads_liquid=True,
        costate_smooth=True,
    )

    with pytest.raises(
        RegimeInitializationError, match=r"liquid|interval|continuation"
    ):
        _check_probes(model)


def test_costate_law_piecewise_constant_in_liquid_builds() -> None:
    """A co-state whose law switches at a liquid threshold builds, carrying the state.

    Asserting the co-state is actually declared pins down that the guard passed on
    the ride-along configuration under test, not on a model that quietly lost it.
    """
    model = ride_toy.build_model(
        variant="nbegm",
        n_liquid=12,
        liquid_max=30.0,
        n_savings=20,
        savings_max=28.0,
        n_consumption=8,
        costate_reads_liquid=True,
        costate_smooth=False,
    )
    assert "tracker" in model.user_regimes["alive"].states


def test_transition_prob_varying_smoothly_in_liquid_is_refused() -> None:
    """A regime-transition probability varying smoothly in liquid is refused."""
    model = ride_toy.build_model(
        variant="nbegm",
        n_liquid=12,
        liquid_max=30.0,
        n_savings=20,
        savings_max=28.0,
        n_consumption=8,
        transition_reads_liquid=True,
        transition_smooth=True,
    )

    with pytest.raises(
        RegimeInitializationError, match=r"regime-transition probabilities"
    ):
        _check_probes(model)


def test_transition_prob_piecewise_constant_in_liquid_builds() -> None:
    """A survival probability switched at a liquid threshold builds, reading liquid.

    Asserting the transition really reads the liquid state pins down that the guard
    passed on the configuration under test, not on a liquid-independent fallback.
    """
    model = ride_toy.build_model(
        variant="nbegm",
        n_liquid=12,
        liquid_max=30.0,
        n_savings=20,
        savings_max=28.0,
        n_consumption=8,
        transition_reads_liquid=True,
        transition_smooth=False,
    )
    transition = cast(
        "Mapping[str, MarkovTransition]", model.user_regimes["alive"].transition
    )
    assert "liquid" in inspect.signature(transition["alive"].func).parameters


def test_costate_law_the_probe_cannot_evaluate_is_refused() -> None:
    """A liquid-reading co-state law the constancy probe cannot differentiate
    is refused — the interval path never assumes an unverifiable law is
    piecewise-constant."""
    model = ride_toy.build_model(
        variant="nbegm",
        n_liquid=12,
        liquid_max=30.0,
        n_savings=20,
        savings_max=28.0,
        n_consumption=8,
        costate_reads_liquid=True,
        costate_unprobeable=True,
    )

    # Prefix match, so it covers both "constant" and "constants".
    message = r"probe|verify|constan"  # codespell:ignore
    with pytest.raises(RegimeInitializationError, match=message):
        _check_probes(model)


def test_unprobeable_law_warns_under_assume_declared() -> None:
    """`probe_failure="assume_declared"` turns an unverifiable-probe rejection
    into a loud warning: the solve proceeds and the warning names the asserted
    precondition."""
    model = ride_toy.build_model(
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

    with pytest.warns(UserWarning, match=r"assume_declared"):
        _check_probes(model)


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
        stateful_targets=("tracker",),
        child_reads={"tracker": SimpleNamespace(next_state_func=next_tracker)},
        compute_regime_transition_probs=compute_regime_transition_probs,
    )
    with pytest.raises(RegimeInitializationError, match="varies smoothly"):
        _fail_if_liquid_reading_next_state_varies_within_interval(
            continuation_plan=plan,
            liquid_name="liquid",
            regime_name="toy",
            probe_arguments=_ProbeArguments(
                int_arg_values=MappingProxyType({"phase": (0, 1, 2, 3)})
            ),
        )
