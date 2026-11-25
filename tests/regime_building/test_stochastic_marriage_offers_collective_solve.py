"""Integration tests for the stochastic marriage/job-offer machinery.

EKL 2019's marriage market (eqs. 24-26): a single may receive a marriage OFFER
— a stochastic draw of a potential spouse's attributes, with an offer
distribution conditioned on the single's own education.

**What this adds on top of the surrounding machinery.** The continuation
already averages a non-terminal regime's continuation over any
`StochasticTransition`-declared stochastic state, INCLUDING a "target-only"
state declared solely in a per-target `state_transitions` dict entry (a state
that is BORN at the transition — not carried from the source's own state space;
see `_lcm.regime_building.transitions.collect_state_transitions`, "Target-only
states").

So the marriage-offer mechanism below is closing a GAP IN TESTING, not a gap
in the engine: a spouse-type offer, drawn via a `StochasticTransition` on a
target-only state conditioned on the single's own education, entering a
collective target through an always-open gate that names the single's role.
`test_stochastic_marriage_offer_*` proves this composition numerically,
hand-computed. Consent to the match is decided by an `Exit` of the single
before any transition draw (see `lcm_examples.collective_household`), so it is
not part of this composition. `test_job_offer_gates_...` pins the
adjacent (already-generic) claim that ordinary discrete stochastic states can
gate action FEASIBILITY, not just shift utility (the mood test in
`test_nonterminal_collective_solve.py` already covers the utility-shift case).
`test_endogenous_offer_distribution_is_rejected` pins that an offer
distribution reading the household's OWN solved values (an equilibrium-
flavored, self-referential offer) is not representable through this
primitive.

Memory note: the spouse-type draw here is an ordinary GRID AXIS of the stored
`married_terminal` V array (`(*states, n_members)` with a `spouse_type`
state axis), exactly like every other stochastic state in this engine. Folding
a transient categorical shock before storage instead is a MEMORY optimization
for full-scale EKL (it avoids a node blow-up across married's much larger
grid) and is OUT OF SCOPE here: these tests are about
correctness/expressibility on small grids, not performance at EKL's actual
scale.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.certainty_equivalent import LinearExpectation
from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from _lcm.regime_law import RegimeLaws
from _lcm.solution.backward_induction import solve
from _lcm.typing import PeriodToRegimeToSimulationPolicy, PeriodToRegimeToVArr
from _lcm.utils.logging import get_logger
from lcm import (
    AgeRange,
    ByAge,
    CollectiveUtility,
    DiscreteGrid,
    Gate,
    MemberRoute,
    Model,
    ProjectedRegimeValue,
    Regime,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.ages import AgeGrid
from lcm.koopmans_aggregation import LinearAggregator
from lcm.transition import StochasticTransition
from lcm.typing import (
    BoolND,
    DiscreteAction,
    DiscreteState,
    FloatND,
    RegimeName,
    ScalarInt,
)
from tests.collective_fixtures import recorded_dissolution_flags
from tests.conftest import bind_laws, build_prepared_structure

# Stochastic marriage offer: single_f (singleton) -> married_terminal (collective),
# with a spouse-type draw at the transition.


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


def _u_single_f(*, education: DiscreteState, work: DiscreteAction) -> FloatND:
    return _wage(education) * work


def _u_single_f_terminal(education: DiscreteState) -> FloatND:
    return 1.5 * _wage(education)  # {1.5, 3.0}


def _u_married(
    *, education: DiscreteState, spouse_type: DiscreteState, work: DiscreteAction
) -> FloatND:
    """Pooled household income; identical for both members (symmetric split)."""
    return _wage(education) + _wage(spouse_type) + 0.0 * work


def _identity_education(education: DiscreteState) -> DiscreteState:
    return education


def _always_open() -> BoolND:
    """Every offer is accepted; the gate only names the arriving member."""
    return jnp.ones((), dtype=bool)


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
        states={"education": DiscreteGrid(category_class=Education)},
        state_transitions={
            "education": fixed_transition("education"),
            # Target-only state: "spouse_type" does not exist in single_f's
            # own state space. It is BORN at the single_f -> married_terminal
            # transition, drawn from the offer distribution conditioned on
            # the single's own (carried) education.
            "spouse_type": {
                "married_terminal": StochasticTransition(func=_offer_probs)
            },
        },
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _u_single_f},
    )
    single_f_terminal = Regime(
        states={"education": DiscreteGrid(category_class=Education)},
        functions={"utility": _u_single_f_terminal},
    )
    married_terminal = Regime(
        states={
            "education": DiscreteGrid(category_class=Education),
            "spouse_type": DiscreteGrid(category_class=Education),
        },
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(utilities={"f": _u_married, "m": _u_married})
        },
    )
    return {
        "single_f": single_f,
        "single_f_terminal": single_f_terminal,
        "married_terminal": married_terminal,
    }


def _single_f_transition() -> Transition:
    """The single's age-0 transition: an always-accepted marriage offer.

    The always-open gate names the member `f` the single becomes in
    `married_terminal`; its fallback `single_f_terminal` is never taken.
    """
    return Transition(
        targets={"married_terminal": 0, "single_f_terminal": 0},
        law=ByAge(
            cases={
                AgeRange(exclusive_stop=1): {
                    "married_terminal": StochasticTransition(func=_prob_one)
                }
            }
        ),
        gates={
            "married_terminal": Gate(
                predicate=_always_open,
                routes={
                    "f": MemberRoute(
                        target_member="f",
                        fallback=ProjectedRegimeValue(
                            regime="single_f_terminal",
                            projection={"education": _identity_education},
                        ),
                    )
                },
            )
        },
    )


def _offer_laws(regimes: dict[str, Regime]) -> RegimeLaws:
    """The offer regimes' laws, bound as a model would bind them."""
    return bind_laws(
        {
            name: _single_f_transition() if name == "single_f" else None
            for name in regimes
        }
    )


# Hand computation (see module docstring for the mechanism).
#
# married_terminal value (pooled income, work irrelevant): V(edu, spouse) =
# wage(edu) + wage(spouse):
#   (0,0)=2  (0,1)=3  (1,0)=3  (1,1)=4
#
# Offer distribution conditioned on OWN education:
#   edu=0: P(spouse=0)=0.7, P(spouse=1)=0.3
#   edu=1: P(spouse=0)=0.4, P(spouse=1)=0.6
#
# E[V_married_f | edu=0] = 0.7*2 + 0.3*3   = 2.3
# E[V_married_f | edu=1] = 0.4*3 + 0.6*4   = 3.6
#
# V_single_f(period 0, edu) = wage(edu)*work + beta*E[V_married_f | edu], work=1
# always optimal (wage(edu) > 0, continuation independent of the source's own
# action):
#   edu=0: 1.0 + 0.95*2.3 = 3.185
#   edu=1: 2.0 + 0.95*3.6 = 5.42
_EXPECTED_V_SINGLE_F_PERIOD_0 = np.array([3.185, 5.42])


def _solve_offer_regimes(
    *, enable_jit: bool = False
) -> tuple[
    PeriodToRegimeToVArr,
    PeriodToRegimeToSimulationPolicy,
    dict[int, dict[RegimeName, BoolND]],
]:
    ages = AgeGrid(start=0, inclusive_stop=2, step="Y")
    user_regimes = _make_offer_regimes()
    laws = _offer_laws(user_regimes)
    finalized = finalize_regimes(
        user_regimes=user_regimes,
        laws=laws,
        derived_categoricals={},
        koopmans_aggregator=LinearAggregator(),
        certainty_equivalent=LinearExpectation(),
    )
    regimes = process_regimes(
        prepared_structure=build_prepared_structure(
            user_regimes=finalized, laws=laws, ages=ages
        ),
        user_regimes=finalized,
        ages=ages,
        regime_names_to_ids=MappingProxyType(
            {name: jnp.int32(i) for i, name in enumerate(user_regimes)}
        ),
        enable_jit=enable_jit,
    )
    flat_params = MappingProxyType(
        {
            "single_f": MappingProxyType(
                {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
            ),
            "single_f_terminal": MappingProxyType({}),
            "married_terminal": MappingProxyType({}),
        }
    )
    with recorded_dissolution_flags() as dissolution_flags:
        _bi_result = solve(
            program_fingerprint="test_stochastic_marriage_offers_collective_solve",
            flat_params=flat_params,
            ages=ages,
            regimes=regimes,
            logger=get_logger(log_level="off"),
            enable_jit=enable_jit,
        )
    return (
        _bi_result.value_functions,
        _bi_result.simulation_policies,
        dissolution_flags,
    )


def test_stochastic_marriage_offer_forms_couple_via_kernel() -> None:
    """Kernel-level: process_regimes -> backward_induction.solve, real path.

    The single's continuation, at each education level, is the offer-
    probability-weighted mixture of the marriage value over the two drawn
    spouse types.
    """
    solution, _sim, _dissolution_flags = _solve_offer_regimes()
    V_single_f = np.asarray(solution[0]["single_f"])[..., 0]
    np.testing.assert_allclose(V_single_f, _EXPECTED_V_SINGLE_F_PERIOD_0, rtol=1e-6)


def test_stochastic_marriage_offer_matches_public_model_api() -> None:
    """Model-level: the same topology through the public `Model(...).solve()`."""

    @categorical(ordered=False)
    class OfferRegimeId:
        single_f: ScalarInt
        single_f_terminal: ScalarInt
        married_terminal: ScalarInt

    ages = AgeGrid(start=0, inclusive_stop=2, step="Y")
    model = Model(
        regimes=_make_offer_regimes(),
        edges={"single_f": _single_f_transition()},
        ages=ages,
        regime_id_class=OfferRegimeId,
        initial_nodes={ages.exact_values[0]: "single_f"},
    )
    solution = model.solve(params={"discount_factor": _BETA}, log_level="off").values
    np.testing.assert_allclose(
        np.asarray(solution[0]["single_f"])[..., 0],
        _EXPECTED_V_SINGLE_F_PERIOD_0,
        rtol=1e-6,
    )


def test_stochastic_marriage_offer_matches_under_jit() -> None:
    """The jitted kernel path folds and averages the same offer draw."""
    solution, _sim, _dissolution = _solve_offer_regimes(enable_jit=True)
    np.testing.assert_allclose(
        np.asarray(solution[0]["single_f"])[..., 0],
        _EXPECTED_V_SINGLE_F_PERIOD_0,
        rtol=1e-6,
    )


# Job-offer stochastic state: feasibility (not just felicity) conditions on a
# StochasticTransition-drawn discrete state (EKL eq. 24). The collective +
# StochasticTransition-into-utility case is already covered end-to-end by
# `test_nonterminal_collective_stochastic_state_expectation_is_per_member`
# in `test_nonterminal_collective_solve.py` (nothing collective-specific about
# it); this test pins the FEASIBILITY-gating half.


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


def _work_requires_offer(*, offer: DiscreteState, work: DiscreteAction) -> BoolND:
    """EKL eq. 24: the "work" action is feasible only if an offer arrived."""
    return (offer == 1) | (work == 0)


def _u_job(*, offer: DiscreteState, work: DiscreteAction) -> FloatND:  # noqa: ARG001
    return 5.0 * work + 1.0 * (1.0 - work)


def _make_job_offer_regimes() -> dict[str, Regime]:
    job = Regime(
        states={"offer": DiscreteGrid(category_class=Offer)},
        state_transitions={"offer": StochasticTransition(func=_offer_arrival_probs)},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _u_job},
        constraints={"work_requires_offer": _work_requires_offer},
    )
    job_terminal = Regime(
        states={"offer": DiscreteGrid(category_class=Offer)},
        actions={"work": DiscreteGrid(category_class=Work)},
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


def test_job_offer_gates_feasible_actions_and_solves() -> None:
    """A drawn discrete job-offer state gates action feasibility (EKL eq. 24)."""
    ages = AgeGrid(start=0, inclusive_stop=2, step="Y")
    model = Model(
        regimes=_make_job_offer_regimes(),
        edges={"job": {"job_terminal": 0}},
        ages=ages,
        regime_id_class=JobRegimeId,
        initial_nodes={ages.exact_values[0]: "job"},
    )
    solution = model.solve(params={"discount_factor": _BETA}, log_level="off").values
    np.testing.assert_allclose(
        np.asarray(solution[0]["job"])[..., 0], _EXPECTED_V_JOB_PERIOD_0, rtol=1e-6
    )


# Scope fence: an endogenous / self-referential offer distribution (reading the
# household's OWN solved value) is not representable through `StochasticTransition`
# — it can only read states/actions/params through the ordinary DAG, never a
# `Q_<s>` action value, which the engine hands to no user function. This pins
# the natural failure mode rather than a bespoke check.


def _self_referential_offer_probs(Q_f: FloatND) -> FloatND:  # noqa: ARG001
    """Ill-formed: an offer distribution reading the household's own Q."""
    return jnp.array([0.5, 0.5])


def test_endogenous_offer_distribution_is_rejected() -> None:
    """A `Q_<s>`-conditioned (self-referential) offer distribution is unrepresentable.

    `Q_<s>` is injected into no user function; an ordinary state-transition /
    `StochasticTransition` function resolves its arguments through the regime's
    plain DAG (states, actions, params, helper functions). `process_regimes` itself
    does not eagerly call the transition closures, so the ill-formed `Q_f` argument
    is NOT caught at build time: the DAG machinery falls back to treating the
    unresolved name as an ordinary flat regime PARAMETER
    (`married_terminal__next_spouse_type __Q_f`), so it surfaces only once `solve`
    actually invokes the closure and finds no such parameter was supplied — a
    `KeyError` naming exactly the missing (and unsuppliable-as-a-live-value)
    argument. There is no way to bind it to the household's own in-solve `Q_f`
    array: the params dict can only ever hold a caller-supplied CONSTANT, never the
    solved value, so the "endogenous offer distribution" is unrepresentable through
    this primitive — confirmed, not merely asserted, by this failure mode.
    """
    regimes = _make_offer_regimes()
    regimes["single_f"] = regimes["single_f"].replace(
        state_transitions={
            "education": fixed_transition("education"),
            "spouse_type": {
                "married_terminal": StochasticTransition(
                    func=_self_referential_offer_probs
                )
            },
        },
    )
    ages = AgeGrid(start=0, inclusive_stop=2, step="Y")
    # Build succeeds — the ill-formed argument is not caught until solve.
    laws = _offer_laws(regimes)
    finalized = finalize_regimes(
        user_regimes=regimes,
        laws=laws,
        derived_categoricals={},
        koopmans_aggregator=LinearAggregator(),
        certainty_equivalent=LinearExpectation(),
    )
    processed = process_regimes(
        prepared_structure=build_prepared_structure(
            user_regimes=finalized, laws=laws, ages=ages
        ),
        user_regimes=finalized,
        ages=ages,
        regime_names_to_ids=MappingProxyType(
            {n: jnp.int32(i) for i, n in enumerate(regimes)}
        ),
        enable_jit=False,
    )
    flat_params = MappingProxyType(
        {
            "single_f": MappingProxyType(
                {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
            ),
            "single_f_terminal": MappingProxyType({}),
            "married_terminal": MappingProxyType({}),
        }
    )
    with pytest.raises(KeyError) as error:
        solve(
            program_fingerprint="test_stochastic_marriage_offers_collective_solve",
            flat_params=flat_params,
            ages=ages,
            regimes=processed,
            logger=get_logger(log_level="off"),
            enable_jit=False,
        )
    assert error.value.args == ("married_terminal__next_spouse_type__Q_f",)
