"""Acceptance: the MSS backend reproduces FUES on the DC-EGM battery.

HARK's EGM upper envelope (`upper_envelope="mss"`) must produce the same value
function as the Fast Upper-Envelope Scan (`upper_envelope="fues"`) on the
existing DC-EGM solve tests, within a documented tolerance.

Both backends insert the exact segment-crossing abscissa where a discrete choice
switches, so MSS tracks the FUES envelope tightly — tighter than LTM, which
evaluates the envelope at the candidate abscissae only and lets a kink land
between output nodes. The tolerance therefore absorbs only the residual
difference in scan ordering and floating-point crossing arithmetic, not a
kink-placement error. The brute-force-unstable low-wealth nodes are excluded
exactly as the underlying FUES tests do.
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

# MSS inserts the same exact crossing FUES does, so the published value functions
# agree far tighter than the LTM no-insertion tolerance (5e-3): the residual is
# floating-point crossing arithmetic plus scan ordering, well inside this bound.
_PARITY_ATOL = 1e-5
_PARITY_RTOL = 1e-5


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
def test_mss_matches_fues_on_concave_retirement(n_periods):
    """The pure-concave retirement solve agrees between MSS and FUES.

    No discrete choice means no segment crossing, so the two backends refine
    the same candidate set identically — agreement holds tightly on every
    wealth node.
    """
    params = get_retirement_only_params(n_periods)

    fues = _retirement_only_model(upper_envelope="fues", n_periods=n_periods).solve(
        params=params, log_level="debug"
    )
    mss = _retirement_only_model(upper_envelope="mss", n_periods=n_periods).solve(
        params=params, log_level="debug"
    )

    for period in sorted(fues)[:-1]:
        np.testing.assert_allclose(
            np.asarray(mss[period]["retirement"]),
            np.asarray(fues[period]["retirement"]),
            atol=_PARITY_ATOL,
            rtol=_PARITY_RTOL,
            err_msg=f"period={period}",
        )


def _nothing_is_feasible(labor_supply: DiscreteAction) -> BoolND:
    return jnp.zeros_like(labor_supply, dtype=bool)


def test_mss_publishes_neg_inf_for_all_infeasible_combo_like_fues():
    """An all-infeasible worker regime publishes `-inf` V under MSS too.

    A discrete-only constraint false everywhere makes the worker's value `-inf`
    at every state; the dead candidates must stay deleted from the envelope
    sweep (`-inf`/NaN poisoning discipline), so MSS publishes `-inf` exactly as
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
    mss = build("mss").solve(params=params, log_level="debug")

    for period in sorted(mss)[:-1]:
        working_V = np.asarray(mss[period]["working_life"])
        assert bool(np.isneginf(working_V).all()), f"period={period}"
        assert bool(np.isfinite(mss[period]["retirement"]).all())


@pytest.mark.parametrize("n_periods", [4])
def test_mss_matches_fues_on_discrete_choice_working_life(n_periods):
    """The work/retire discrete-choice solve agrees between MSS and FUES.

    The labour-supply choice creates a non-concave kink in the worker's value
    correspondence. Both backends insert the exact crossing abscissa, so the
    published value functions agree tightly on the wealth nodes where both are
    well-defined.
    """
    params = get_full_params(n_periods, discount_factor=0.98, wage=20.0)

    fues = _full_model(upper_envelope="fues", n_periods=n_periods).solve(
        params=params, log_level="debug"
    )
    mss = _full_model(upper_envelope="mss", n_periods=n_periods).solve(
        params=params, log_level="debug"
    )

    n_unstable_low_nodes = 10
    for period in sorted(fues)[:-1]:
        for regime in ["working_life", "retirement"]:
            np.testing.assert_allclose(
                np.asarray(mss[period][regime])[..., n_unstable_low_nodes:],
                np.asarray(fues[period][regime])[..., n_unstable_low_nodes:],
                atol=_PARITY_ATOL,
                rtol=_PARITY_RTOL,
                err_msg=f"period={period}, regime={regime}",
            )
