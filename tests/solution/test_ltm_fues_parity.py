"""Acceptance: the LTM backend reproduces FUES on the DC-EGM battery.

The brute local-upper-bound backend (`upper_envelope="ltm"`) must produce the
same value function as the Fast Upper-Envelope Scan (`upper_envelope="fues"`) on
the existing DC-EGM solve tests, within a documented tolerance.

Both backends compute the upper envelope of the same EGM candidate cloud; they
differ only in algorithm and cost (LTM is `O(K^2)`, FUES is a single scan), not
in the refined value they represent. FUES inserts the exact segment-crossing
abscissa; LTM evaluates the envelope at the candidate abscissae, so a kink lands
between two LTM output nodes and the downstream linear/Hermite read recovers it
to within the local grid spacing — a second-order error. On a concave segment
(no crossing) the two backends agree to machine precision; the tolerance only
absorbs the kink-placement delta where a discrete choice switches. The
brute-force-unstable low-wealth nodes are excluded exactly as the underlying
FUES tests do.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, MarkovTransition, Model
from lcm.typing import BoolND, DiscreteAction
from lcm_examples.iskhakov_et_al_2017 import dead
from tests.test_models.deterministic import base, retirement_only
from tests.test_models.deterministic.dcegm_variants import (
    dcegm_retirement,
    dcegm_retirement_full,
    dcegm_working_life,
    get_full_params,
    get_retirement_only_params,
)

# The no-exact-crossing delta is a kink-placement error of order the local grid
# spacing, propagated through the exact-slope Hermite carry. On the cubically
# clustered savings grid the retirement battery uses, that bounds the per-node V
# difference between the backends well inside this tolerance.
_PARITY_ATOL = 5e-3
_PARITY_RTOL = 1e-3


def _with_backend(regime, *, upper_envelope):
    """Rebuild a DC-EGM regime with the chosen upper-envelope backend."""
    solver = dataclasses.replace(regime.solver, upper_envelope=upper_envelope)
    return regime.replace(solver=solver)


def _retirement_only_model(*, upper_envelope, n_periods):
    ages = AgeGrid(start=40, stop=40 + (n_periods - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    return Model(
        regimes={
            "retirement": _with_backend(
                dcegm_retirement, upper_envelope=upper_envelope
            ).replace(active=lambda age, la=last_age: age < la),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=retirement_only.RetirementOnlyRegimeId,
    )


def _full_model(*, upper_envelope, n_periods):
    ages = AgeGrid(start=40, stop=40 + (n_periods - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    return Model(
        regimes={
            "working_life": _with_backend(
                dcegm_working_life, upper_envelope=upper_envelope
            ).replace(active=lambda age, la=last_age: age < la),
            "retirement": _with_backend(
                dcegm_retirement_full, upper_envelope=upper_envelope
            ).replace(active=lambda age, la=last_age: age < la),
            "dead": base.dead,
        },
        ages=ages,
        regime_id_class=base.RegimeId,
    )


@pytest.mark.parametrize("n_periods", [3, 5])
def test_ltm_matches_fues_on_concave_retirement(n_periods):
    """The pure-concave retirement solve agrees between LTM and FUES.

    No discrete choice means no segment crossing, so the two backends refine
    the same candidate set identically — agreement holds tightly on every
    wealth node.
    """
    params = get_retirement_only_params(n_periods)

    fues = _retirement_only_model(upper_envelope="fues", n_periods=n_periods).solve(
        params=params, log_level="debug"
    )
    ltm = _retirement_only_model(upper_envelope="ltm", n_periods=n_periods).solve(
        params=params, log_level="debug"
    )

    for period in sorted(fues)[:-1]:
        np.testing.assert_allclose(
            np.asarray(ltm[period]["retirement"]),
            np.asarray(fues[period]["retirement"]),
            atol=_PARITY_ATOL,
            rtol=_PARITY_RTOL,
            err_msg=f"period={period}",
        )


def _nothing_is_feasible(labor_supply: DiscreteAction) -> BoolND:
    return jnp.zeros_like(labor_supply, dtype=bool)


def test_ltm_publishes_neg_inf_for_all_infeasible_combo_like_fues():
    """An all-infeasible worker regime publishes `-inf` V under LTM too.

    A discrete-only constraint false everywhere makes the worker's value `-inf`
    at every state; the dead candidates must stay deleted from the envelope
    scan (`-inf`/NaN poisoning discipline), so LTM publishes `-inf` exactly as
    FUES does, never NaN.
    """
    n_periods = 4
    retirement_transition = {
        "retirement": MarkovTransition(
            lambda age, final_age_alive: jnp.where(age >= final_age_alive, 0.0, 1.0)
        ),
        "dead": MarkovTransition(
            lambda age, final_age_alive: jnp.where(age >= final_age_alive, 1.0, 0.0)
        ),
    }

    def build(upper_envelope):
        ages = AgeGrid(start=40, stop=40 + (n_periods - 1) * 10, step="10Y")
        return Model(
            regimes={
                "working_life": _with_backend(
                    dcegm_working_life, upper_envelope=upper_envelope
                ).replace(
                    constraints={"nothing_is_feasible": _nothing_is_feasible},
                    active=lambda age: age < 70,
                ),
                "retirement": _with_backend(
                    dcegm_retirement_full, upper_envelope=upper_envelope
                ).replace(
                    transition=retirement_transition,
                    state_transitions={
                        "wealth": dcegm_retirement_full.state_transitions["wealth"],
                    },
                    active=lambda age: age < 70,
                ),
                "dead": base.dead,
            },
            ages=ages,
            regime_id_class=base.RegimeId,
        )

    params = get_full_params(n_periods, discount_factor=0.98, wage=20.0)
    ltm = build("ltm").solve(params=params, log_level="debug")

    for period in sorted(ltm)[:-1]:
        working_V = np.asarray(ltm[period]["working_life"])
        assert bool(np.isneginf(working_V).all()), f"period={period}"
        assert bool(np.isfinite(ltm[period]["retirement"]).all())


@pytest.mark.parametrize("n_periods", [4])
def test_ltm_matches_fues_on_discrete_choice_working_life(n_periods):
    """The work/retire discrete-choice solve agrees between LTM and FUES.

    The labour-supply choice creates a non-concave kink in the worker's value
    correspondence. FUES inserts the exact crossing; LTM evaluates the envelope
    at the candidate abscissae and lets the downstream read recover the kink.
    The published value functions agree within the no-insertion tolerance on the
    wealth nodes where both are well-defined.
    """
    params = get_full_params(n_periods, discount_factor=0.98, wage=20.0)

    fues = _full_model(upper_envelope="fues", n_periods=n_periods).solve(
        params=params, log_level="debug"
    )
    ltm = _full_model(upper_envelope="ltm", n_periods=n_periods).solve(
        params=params, log_level="debug"
    )

    n_unstable_low_nodes = 10
    for period in sorted(fues)[:-1]:
        for regime in ["working_life", "retirement"]:
            np.testing.assert_allclose(
                np.asarray(ltm[period][regime])[..., n_unstable_low_nodes:],
                np.asarray(fues[period][regime])[..., n_unstable_low_nodes:],
                atol=_PARITY_ATOL,
                rtol=_PARITY_RTOL,
                err_msg=f"period={period}, regime={regime}",
            )
