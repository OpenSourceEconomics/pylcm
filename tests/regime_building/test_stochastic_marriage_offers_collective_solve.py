"""Integration tests for slice 5: the stochastic marriage/job-offer machinery.

EKL 2019's marriage market (eqs. 24-26): a single may receive a marriage OFFER
— a stochastic draw of a potential spouse's attributes, with an offer
distribution conditioned on the single's own education. Consent (eq. 27,
slice 4's `GatedEdge`) then compares the household's married value (given the
DRAWN spouse) against each partner's own outside option.

**What this slice adds vs. what was already there.** Slice 4 (E3') built the
gated-edge fold `Wbar = jnp.where(gate, V_target, V_fallback)` on the TARGET
regime's full state grid, and slice 2 (E1 continuation) already averages a
non-terminal regime's continuation over any `MarkovTransition`-declared
stochastic state, INCLUDING a "target-only" state declared solely in a
per-target `state_transitions` dict entry (a state that is BORN at the
transition — not carried from the source's own state space; see
`_lcm.regime_building.transitions.collect_state_transitions`, "Target-only
states"). Nothing in either mechanism is gated-edge-specific or
collective-specific: the gated edge only replaces the raw target V array with
Wbar in `next_regime_to_V_arr`, and the ordinary `get_Q_and_F` /
`get_Q_and_F_collective` continuation reads whichever array sits there through
the SAME stochastic-weights/productmap logic either way.

So the marriage-offer mechanism below is closing a GAP IN TESTING, not a gap
in the engine: a spouse-type offer, drawn via a `MarkovTransition` on a
target-only state conditioned on the single's own education, feeding a
gated (mutual-consent) edge. `test_stochastic_marriage_offer_*` proves this
composition numerically, hand-computed. `test_job_offer_gates_...` pins the
adjacent (already-generic) claim that ordinary discrete stochastic states can
gate action FEASIBILITY, not just shift utility (the mood test in
`test_nonterminal_collective_solve.py` already covers the utility-shift case).
`test_endogenous_offer_distribution_is_rejected` pins that an offer
distribution reading the household's OWN solved values (an equilibrium-
flavored, self-referential offer) is not representable through this
primitive.

Memory note (deferred to slice 5b): the spouse-type draw here is an ordinary
GRID AXIS of the stored `married_terminal` V array (`(*states, n_stakeholders)`
with a `spouse_type` state axis), exactly like every other stochastic state in
this engine. The design doc's "transient categorical shocks that fold before
storage" (§2, Stage-B F2) is an explicit MEMORY optimization for full-scale
EKL (avoiding the node blow-up across married's ~58M-cell grid) and is OUT OF
SCOPE here: this slice is about correctness/expressibility on small grids, not
performance at EKL's actual scale.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from _lcm.solution.backward_induction import solve
from _lcm.utils.logging import get_logger
from lcm import DiscreteGrid, Model, categorical, fixed_transition
from lcm.ages import AgeGrid
from lcm.regime import EdgeLeg, GatedEdge, Regime, SamePeriodRef
from lcm.transition import MarkovTransition
from lcm.typing import (
    BoolND,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
)

# --------------------------------------------------------------------------------------
# Stochastic marriage offer: single_f (singleton) -> married_terminal (collective),
# with a spouse-type draw feeding the mutual-consent gated edge.
# --------------------------------------------------------------------------------------


@categorical(ordered=True)
class Education:
    low: ScalarInt  # code 0
    high: ScalarInt  # code 1


@categorical(ordered=True)
class Work:
    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


_BETA = 0.95


def _wage(education: DiscreteState) -> FloatND:
    """Own-education wage AND (reused) spouse-type wage-equivalent: {1.0, 2.0}."""
    return jnp.where(education == 0, 1.0, 2.0)


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _u_single_f(education: DiscreteState, work: DiscreteAction) -> FloatND:
    return _wage(education) * work


def _u_single_f_terminal(education: DiscreteState) -> FloatND:
    return 1.5 * _wage(education)  # {1.5, 3.0}


def _u_single_m_terminal(education: DiscreteState) -> FloatND:
    return jnp.where(education == 0, 0.5, 3.0)  # {0.5, 3.0}


def _u_married(
    education: DiscreteState, spouse_type: DiscreteState, work: DiscreteAction
) -> FloatND:
    """Pooled household income; identical for both stakeholders (symmetric split)."""
    return _wage(education) + _wage(spouse_type) + 0.0 * work


def _identity_education(education: DiscreteState) -> DiscreteState:
    return education


def _spouse_type_as_education(spouse_type: DiscreteState) -> DiscreteState:
    """The offered spouse's own outside option is read at their drawn type."""
    return spouse_type


def _consent_gate(
    V_target_f: FloatND,
    V_target_m: FloatND,
    V_single_f_ref: FloatND,
    V_single_m_ref: FloatND,
) -> BoolND:
    # EKL eq. 27: strict, unanimous mutual consent.
    return (V_target_f > V_single_f_ref) & (V_target_m > V_single_m_ref)


def _offer_probs(education: DiscreteState) -> FloatND:
    """EKL eqs. 25-26 stand-in: spouse-type offer distribution over the joint

    spouse-attribute support, conditioned on the single's OWN education (eq.
    25's MNL education-conditioning collapsed to a hand-computable 2-point
    distribution; eq. 26's joint attribute draw collapsed to the 2-category
    `spouse_type` grid). Low own-education -> offer skews toward a
    low-type spouse; high own-education -> skews toward a high-type spouse.
    """
    low_educ_offer = jnp.array([0.7, 0.3])
    high_educ_offer = jnp.array([0.4, 0.6])
    return jnp.where(education == 0, low_educ_offer, high_educ_offer)


def _make_offer_regimes() -> dict[str, Regime]:
    single_f = Regime(
        transition={"married_terminal": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"education": DiscreteGrid(Education)},
        state_transitions={
            "education": fixed_transition("education"),
            # Target-only state: "spouse_type" does not exist in single_f's
            # own state space. It is BORN at the single_f -> married_terminal
            # transition, drawn from the offer distribution conditioned on
            # the single's own (carried) education.
            "spouse_type": {"married_terminal": MarkovTransition(_offer_probs)},
        },
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_single_f},
        gated_edges={
            "married_terminal": GatedEdge(
                gate=_consent_gate,
                legs={
                    "f": EdgeLeg(
                        target_stakeholder="f",
                        fallback=SamePeriodRef(
                            regime="single_f_terminal",
                            projection={"education": _identity_education},
                        ),
                    )
                },
                gate_refs={
                    "V_single_f_ref": SamePeriodRef(
                        regime="single_f_terminal",
                        projection={"education": _identity_education},
                    ),
                    "V_single_m_ref": SamePeriodRef(
                        regime="single_m_terminal",
                        projection={"education": _spouse_type_as_education},
                    ),
                },
            )
        },
    )
    single_f_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"education": DiscreteGrid(Education)},
        functions={"utility": _u_single_f_terminal},
    )
    single_m_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"education": DiscreteGrid(Education)},
        functions={"utility": _u_single_m_terminal},
    )
    married_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={
            "education": DiscreteGrid(Education),
            "spouse_type": DiscreteGrid(Education),
        },
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _u_married, "utility_m": _u_married},
    )
    return {
        "single_f": single_f,
        "single_f_terminal": single_f_terminal,
        "single_m_terminal": single_m_terminal,
        "married_terminal": married_terminal,
    }


# Hand computation (see module docstring for the mechanism).
#
# married_terminal value (pooled income, work irrelevant): V(edu, spouse) =
# wage(edu) + wage(spouse):
#   (0,0)=2  (0,1)=3  (1,0)=3  (1,1)=4
#
# Gate refs: V_single_f_ref(edu) = 1.5*wage(edu) -> {1.5, 3.0}
#            V_single_m_ref(spouse) = u_single_m_terminal(spouse) -> {0.5, 3.0}
#
# Consent gate (V_target_f > V_single_f_ref) & (V_target_m > V_single_m_ref),
# both components equal V_married here:
#   edu=0 (ref=1.5): spouse=0 -> 2>1.5 & 2>0.5   -> OPEN
#                     spouse=1 -> 3>1.5 & 3>3.0   -> CLOSED (strict, tie on m)
#   edu=1 (ref=3.0): spouse=0 -> 3>3.0 & ...      -> CLOSED (strict, tie on f)
#                     spouse=1 -> 4>3.0 & 4>3.0   -> OPEN
#
# Wbar_f(edu, spouse) = where(gate, V_married, V_single_f_ref(edu)):
#   (0,0)=2 (OPEN)      (0,1)=1.5 (CLOSED, fallback)
#   (1,0)=3.0 (CLOSED, fallback)   (1,1)=4 (OPEN)
#
# Offer distribution conditioned on OWN education:
#   edu=0: P(spouse=0)=0.7, P(spouse=1)=0.3
#   edu=1: P(spouse=0)=0.4, P(spouse=1)=0.6
#
# E[Wbar_f | edu=0] = 0.7*2 + 0.3*1.5   = 1.85
# E[Wbar_f | edu=1] = 0.4*3.0 + 0.6*4   = 3.6
#
# V_single_f(period 0, edu) = wage(edu)*work + beta*E[Wbar_f | edu], work=1 always
# optimal (wage(edu) > 0, continuation independent of the source's own action):
#   edu=0: 1.0 + 0.95*1.85 = 2.7575
#   edu=1: 2.0 + 0.95*3.6  = 5.42
_EXPECTED_V_SINGLE_F_PERIOD_0 = np.array([2.7575, 5.42])


def _solve_offer_regimes(*, enable_jit: bool = False):
    ages = AgeGrid(start=0, stop=2, step="Y")
    regime_names = list(_make_offer_regimes())
    regimes = process_regimes(
        user_regimes=finalize_regimes(
            user_regimes=_make_offer_regimes(), derived_categoricals={}
        ),
        ages=ages,
        regime_names_to_ids=MappingProxyType(
            {name: jnp.int32(i) for i, name in enumerate(regime_names)}
        ),
        enable_jit=enable_jit,
    )
    flat_params = MappingProxyType(
        {
            "single_f": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "single_f_terminal": MappingProxyType({}),
            "single_m_terminal": MappingProxyType({}),
            "married_terminal": MappingProxyType({}),
        }
    )
    return solve(
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=enable_jit,
    )


def test_stochastic_marriage_offer_forms_couple_via_kernel():
    """Kernel-level: process_regimes -> backward_induction.solve, real path.

    The single's continuation, at each education level, is the offer-
    probability-weighted mixture over the two drawn spouse types of the
    consent-gated marriage value vs. the single fallback.
    """
    solution, _sim, _divorce_flags = _solve_offer_regimes()
    V_single_f = np.asarray(solution[0]["single_f"])
    np.testing.assert_allclose(V_single_f, _EXPECTED_V_SINGLE_F_PERIOD_0, rtol=1e-6)


def test_stochastic_marriage_offer_matches_public_model_api():
    """Model-level: the same topology through the public `Model(...).solve()`."""

    @categorical(ordered=False)
    class OfferRegimeId:
        single_f: ScalarInt
        single_f_terminal: ScalarInt
        single_m_terminal: ScalarInt
        married_terminal: ScalarInt

    ages = AgeGrid(start=0, stop=2, step="Y")
    model = Model(
        regimes=_make_offer_regimes(),
        ages=ages,
        regime_id_class=OfferRegimeId,
    )
    solution = model.solve(params={"discount_factor": _BETA}, log_level="off")
    np.testing.assert_allclose(
        np.asarray(solution[0]["single_f"]),
        _EXPECTED_V_SINGLE_F_PERIOD_0,
        rtol=1e-6,
    )


def test_stochastic_marriage_offer_matches_under_jit():
    """The jitted kernel path folds and averages the same offer draw."""
    solution, _sim, _divorce = _solve_offer_regimes(enable_jit=True)
    np.testing.assert_allclose(
        np.asarray(solution[0]["single_f"]),
        _EXPECTED_V_SINGLE_F_PERIOD_0,
        rtol=1e-6,
    )


# --------------------------------------------------------------------------------------
# Job-offer stochastic state: feasibility (not just felicity) conditions on a
# MarkovTransition-drawn discrete state (EKL eq. 24). The collective +
# MarkovTransition-into-utility case is already covered end-to-end by
# `test_nonterminal_collective_stochastic_state_expectation_is_per_stakeholder`
# in `test_nonterminal_collective_solve.py` (nothing collective-specific about
# it); this test pins the FEASIBILITY-gating half.
# --------------------------------------------------------------------------------------


@categorical(ordered=True)
class Offer:
    none: ScalarInt  # code 0
    available: ScalarInt  # code 1


@categorical(ordered=False)
class JobRegimeId:
    job: ScalarInt
    job_terminal: ScalarInt


def _offer_arrival_probs() -> FloatND:
    """Unconditional offer arrival: P(offer'=0)=0.4, P(offer'=1)=0.6."""
    return jnp.array([0.4, 0.6])


def _work_requires_offer(offer: DiscreteState, work: DiscreteAction) -> BoolND:
    """EKL eq. 24: the "work" action is feasible only if an offer arrived."""
    return (offer == 1) | (work == 0)


def _u_job(offer: DiscreteState, work: DiscreteAction) -> FloatND:  # noqa: ARG001
    return 5.0 * work + 1.0 * (1.0 - work)


def _make_job_offer_regimes() -> dict[str, Regime]:
    job = Regime(
        transition=lambda age: JobRegimeId.job_terminal,  # noqa: ARG005
        active=lambda age: age < 1,
        states={"offer": DiscreteGrid(Offer)},
        state_transitions={"offer": MarkovTransition(_offer_arrival_probs)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_job},
        constraints={"work_requires_offer": _work_requires_offer},
    )
    job_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"offer": DiscreteGrid(Offer)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_job},
        constraints={"work_requires_offer": _work_requires_offer},
    )
    return {"job": job, "job_terminal": job_terminal}


# Hand computation.
# Terminal: offer=0 -> work infeasible -> V=1.0; offer=1 -> max(5,1)=5.0 -> V=5.0.
# Period 0: E[V'] = 0.4*1.0 + 0.6*5.0 = 3.4, identical at every (offer, work)
# cell since the offer draw here does not depend on the current cell.
#   offer=0: work infeasible -> V = 1.0 + 0.95*3.4 = 4.23
#   offer=1: work feasible -> max(5.0, 1.0) + 0.95*3.4 = 5.0 + 3.23 = 8.23
_EXPECTED_V_JOB_PERIOD_0 = np.array([4.23, 8.23])


def test_job_offer_gates_feasible_actions_and_solves():
    """A drawn discrete job-offer state gates action feasibility (EKL eq. 24)."""
    ages = AgeGrid(start=0, stop=2, step="Y")
    model = Model(
        regimes=_make_job_offer_regimes(),
        ages=ages,
        regime_id_class=JobRegimeId,
    )
    solution = model.solve(params={"discount_factor": _BETA}, log_level="off")
    np.testing.assert_allclose(
        np.asarray(solution[0]["job"]), _EXPECTED_V_JOB_PERIOD_0, rtol=1e-6
    )


# --------------------------------------------------------------------------------------
# Scope fence: an endogenous / self-referential offer distribution (reading the
# household's OWN solved value) is not representable through `MarkovTransition`
# — it can only read states/actions/params through the ordinary DAG, never a
# `Q_<s>` action value (that access is `value_constraints`-only, E2). This pins
# the natural failure mode rather than a bespoke check.
# --------------------------------------------------------------------------------------


def _self_referential_offer_probs(Q_f: FloatND) -> FloatND:  # noqa: ARG001
    """Ill-formed: an offer distribution reading the household's own Q (E2-only)."""
    return jnp.array([0.5, 0.5])


def test_endogenous_offer_distribution_is_rejected():
    """A `Q_<s>`-conditioned (self-referential) offer distribution is unrepresentable.

    `Q_<s>` is injected only into `value_constraints` predicates (E2); an
    ordinary state-transition / `MarkovTransition` function has no such
    injection — it resolves through the regime's plain DAG (states, actions,
    params, helper functions). `process_regimes` itself does not eagerly call
    the transition closures, so the ill-formed `Q_f` argument is NOT caught at
    build time: the DAG machinery falls back to treating the unresolved name
    as an ordinary flat regime PARAMETER (`married_terminal__next_spouse_type
    __Q_f`), so it surfaces only once `solve` actually invokes the closure and
    finds no such parameter was supplied — a `ValueError` naming exactly the
    missing (and unsuppliable-as-a-live-value) argument. There is no way to
    bind it to the household's own in-solve `Q_f` array: the params dict can
    only ever hold a caller-supplied CONSTANT, never the solved value, so the
    "endogenous offer distribution" is unrepresentable through this
    primitive — confirmed, not merely asserted, by this failure mode.
    """
    regimes = _make_offer_regimes()
    regimes["single_f"] = regimes["single_f"].replace(
        state_transitions={
            "education": fixed_transition("education"),
            "spouse_type": {
                "married_terminal": MarkovTransition(_self_referential_offer_probs)
            },
        },
    )
    ages = AgeGrid(start=0, stop=2, step="Y")
    # Build succeeds — the ill-formed argument is not caught until solve.
    processed = process_regimes(
        user_regimes=finalize_regimes(user_regimes=regimes, derived_categoricals={}),
        ages=ages,
        regime_names_to_ids=MappingProxyType(
            {n: jnp.int32(i) for i, n in enumerate(regimes)}
        ),
        enable_jit=False,
    )
    flat_params = MappingProxyType(
        {
            "single_f": MappingProxyType({"H__discount_factor": jnp.asarray(_BETA)}),
            "single_f_terminal": MappingProxyType({}),
            "single_m_terminal": MappingProxyType({}),
            "married_terminal": MappingProxyType({}),
        }
    )
    with pytest.raises(ValueError, match="Q_f"):
        solve(
            flat_params=flat_params,
            ages=ages,
            regimes=processed,
            logger=get_logger(log_level="off"),
            enable_jit=False,
        )
