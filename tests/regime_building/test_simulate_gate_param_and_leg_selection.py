"""Regressions for two simulate-side gated-edge routing invariants.

Both live on the path `_lcm.simulation.gated_routing.route_gated_edges`
walks once per declared gated edge; the first is exercised here at the kernel
level (build + solve a miniature, then call
`substitute_gated_edge_continuations` and `route_gated_edges` directly on
hand-crafted realized states).

**A STATELESS gated target does not crash `vmap`.** With no states of its own
(a terminal scrap-value regime), the evaluator's batched kwargs are empty
after filtering, and `jax.vmap` cannot infer a batch size from zero batched
arguments ("vmap wrapped function must be passed at least one argument
containing an array or axis_size must be specified"), so `axis_size` is
threaded explicitly from the population size.

**A row's seeded role is checked against ITS OWN regime's members, not
against the model-wide vocabulary.** Roles are model-wide integer codes, so a
code being defined says nothing about whether the regime a row starts in has
that role — and the router picks a row's leg by matching the role, with the
first declared leg answering for a row that matches none. Checking the seed
per regime is what makes that match total. Arity does not decide it either:
the validator accepts a ONE-element `members` tuple, and `processing.py`'s
`leg_order = [(s, s) for s in source_member_axis]` gives that sole leg
`source_member="f"`, not `None`, so a single-leg collective source is
exactly as exposed as a two-leg one.
"""

import dataclasses
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.engine import Regime as EngineRegime
from _lcm.regime_building.collective import NO_ROLE
from _lcm.regime_building.Q_and_F import SAME_PERIOD_V_ARG
from _lcm.regime_law import RegimeLawDeclaration
from _lcm.simulation.gated_routing import (
    _call_vmapped_with_accepted_kwargs,
    route_gated_edges,
    substitute_gated_edge_continuations,
)
from _lcm.simulation.simulate import _initial_own_member
from _lcm.solution.backward_induction import solve
from _lcm.typing import FlatRegimeParams, PeriodToRegimeToVArr, RegimeNamesToIds
from _lcm.utils.logging import get_logger
from lcm import (
    AgeRange,
    ByAge,
    DiscreteGrid,
    Gate,
    LinSpacedGrid,
    MemberRoute,
    ProjectedRegimeValue,
    Regime,
    Transition,
    categorical,
)
from lcm.ages import AgeGrid
from lcm.exceptions import InvalidInitialConditionsError
from lcm.transition import StochasticTransition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    Int1D,
    RegimeName,
    ScalarInt,
)
from tests.regime_building.test_collective_regime_simulate import (
    _solve_and_process,
    _solve_dissolution,
)

_BETA = 0.95
_AGES = AgeGrid(start=0, inclusive_stop=2, step="Y")


@categorical(ordered=True)
class Work:
    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _identity_x(x: ContinuousState) -> ContinuousState:
    return x


def _u_src(*, x: ContinuousState, work: DiscreteAction) -> FloatND:
    return jnp.zeros_like(x) * work


# A STATELESS gated target leaves `vmap` with zero batched args.


def _u_stateless_target() -> FloatND:
    return jnp.asarray(1.0)


def _u_stateless_fallback() -> FloatND:
    return jnp.asarray(0.0)


def _stateless_gate() -> BoolND:
    # The stateless target offers the gate nothing to read -> OPEN.
    return jnp.asarray(1) == 1


def _make_f3_regimes() -> dict[str, Regime]:
    """A 3-regime model whose gated target is STATELESS (terminal scrap value)."""
    src = Regime(
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={"x": _identity_x},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _u_src},
    )
    stateless_target = Regime(
        functions={"utility": _u_stateless_target},
    )
    stateless_fallback = Regime(
        functions={"utility": _u_stateless_fallback},
    )
    return {
        "src": src,
        "stateless_target": stateless_target,
        "stateless_fallback": stateless_fallback,
    }


def _make_f3_laws() -> dict[RegimeName, RegimeLawDeclaration | Transition]:
    """`src` routes into the stateless target, falling back to a stateless regime."""
    return {
        "src": Transition(
            law=ByAge(
                cases={
                    AgeRange(exclusive_stop=1): {
                        "stateless_target": StochasticTransition(func=_prob_one)
                    }
                }
            ),
            gates={
                "stateless_target": Gate(
                    predicate=_stateless_gate,
                    routes={
                        "only": MemberRoute(
                            fallback=ProjectedRegimeValue(
                                regime="stateless_fallback", projection={}
                            )
                        )
                    },
                )
            },
        ),
        "stateless_target": None,
        "stateless_fallback": None,
    }


def _solve_f3_fixture() -> tuple[
    MappingProxyType[RegimeName, EngineRegime],
    RegimeNamesToIds,
    MappingProxyType[RegimeName, FlatRegimeParams],
    PeriodToRegimeToVArr,
]:
    regimes_dict = _make_f3_regimes()
    regimes, regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict,
        laws=_make_f3_laws(),
        ages=_AGES,
        regime_names=list(regimes_dict),
    )
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType(
                {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
            ),
            "stateless_target": MappingProxyType({}),
            "stateless_fallback": MappingProxyType({}),
        }
    )
    _bi_result = solve(
        program_fingerprint="test_simulate_gate_param_and_leg_selection",
        flat_params=flat_params,
        ages=_AGES,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    solution = _bi_result.value_functions
    _sim_policies = _bi_result.simulation_policies
    return regimes, regime_names_to_ids, flat_params, solution


def test_stateless_gated_target_routes_without_vmap_axis_size_error() -> None:
    """A stateless gated target routes without a vmap axis-size error.

    Such a target's evaluator accepts no batched arg at all, so a bare
    `jax.vmap(f)(batched)` with `batched == {}` raises `ValueError: vmap
    wrapped function must be passed at least one argument containing an array
    or axis_size must be specified`. The routing call below must simply work,
    and produce a per-subject gate of the right length.
    """
    regimes, regime_names_to_ids, flat_params, solution = _solve_f3_fixture()
    src = regimes["src"]
    target_id = regime_names_to_ids["stateless_target"]

    base_state_action_spaces = {
        name: regime.solution.state_action_space(regime_params=flat_params[name])
        for name, regime in regimes.items()
    }
    _substituted, same_period_mappings = substitute_gated_edge_continuations(
        regime=src,
        regime_name="src",
        regimes=regimes,
        period=0,
        next_regime_to_V_arr=solution[1],
        base_state_action_spaces=base_state_action_spaces,
        period_to_regime_to_V_arr=solution,
        flat_params=flat_params,
    )
    assert "stateless_target" in same_period_mappings

    n_subjects = 3
    next_states = MappingProxyType(
        {
            "stateless_target": MappingProxyType({}),
            "stateless_fallback": MappingProxyType({}),
        }
    )
    _states, routed_ids, _routed_roles, _closed_masks = route_gated_edges(
        # The source is simulated at period 0, so the gate is decided on
        # the value it would enter at period 1.
        fold_period=1,
        regime=src,
        same_period_mappings=same_period_mappings,
        next_states=next_states,
        regime_names_to_ids=regime_names_to_ids,
        new_subject_regime_ids=jnp.array([target_id] * n_subjects, dtype=jnp.int32),
        subjects_in_regime=jnp.array([True] * n_subjects),
        flat_params=flat_params,
        own_member=jnp.full_like(
            jnp.array([True] * n_subjects), NO_ROLE, dtype=jnp.int32
        ),
        new_own_member=jnp.full_like(
            jnp.array([True] * n_subjects), NO_ROLE, dtype=jnp.int32
        ),
    )
    # The gate is OPEN for every subject; the routing is per-subject
    # (length n_subjects), not a collapsed scalar.
    np.testing.assert_array_equal(np.asarray(routed_ids), [target_id] * n_subjects)


def test_vmap_without_axis_size_fails_for_a_stateless_target() -> None:
    """The axis-size argument is what makes the test above pass, not luck.

    Runs the bare call (`jax.vmap(f)(batched)`, no `axis_size`) on the SAME
    evaluator and the SAME empty batched kwargs, in this process — the
    production source is never touched.
    """
    regimes, _ids, flat_params, solution = _solve_f3_fixture()
    src = regimes["src"]
    base_state_action_spaces = {
        name: regime.solution.state_action_space(regime_params=flat_params[name])
        for name, regime in regimes.items()
    }
    _substituted, same_period_mappings = substitute_gated_edge_continuations(
        regime=src,
        regime_name="src",
        regimes=regimes,
        period=0,
        next_regime_to_V_arr=solution[1],
        base_state_action_spaces=base_state_action_spaces,
        period_to_regime_to_V_arr=solution,
        flat_params=flat_params,
    )
    evaluator = src.gated_edges["stateless_target"].simulate_gate_evaluator_at(period=1)
    static = {SAME_PERIOD_V_ARG: same_period_mappings["stateless_target"]}

    def _call_one_subject(one_subject_kwargs: dict[str, FloatND]) -> BoolND:
        return evaluator(**one_subject_kwargs, **static)

    # The same body, verbatim except for the missing `axis_size`.
    with pytest.raises(ValueError, match="at least one argument"):
        jax.vmap(_call_one_subject)({})

    # And the production body, on the identical inputs, does not raise.
    result = _call_vmapped_with_accepted_kwargs(
        func=evaluator,
        batched_kwargs={},
        static_kwargs=static,
        axis_size=4,
    )
    assert np.asarray(result).shape == (4,)


# A seeded role is checked against the starting regime's own members.


def _dissolution_setup() -> tuple[
    MappingProxyType[RegimeName, EngineRegime], RegimeNamesToIds
]:
    """The two-member dissolution miniature, as canonical regimes."""
    _ages, regimes, regime_names_to_ids, *_rest = _solve_dissolution()
    return regimes, regime_names_to_ids


def _seed(
    *,
    regimes: MappingProxyType[RegimeName, EngineRegime],
    regime_names_to_ids: RegimeNamesToIds,
    regime_name: RegimeName,
    role: int | None,
) -> Int1D:
    """Read the seeded roles of a one-subject cohort starting in `regime_name`."""
    return _initial_own_member(
        initial_conditions=MappingProxyType(
            {
                "regime_id": jnp.asarray(
                    [regime_names_to_ids[regime_name]], dtype=jnp.int32
                ),
                **({} if role is None else {"own_member": jnp.asarray([role])}),
            }
        ),
        regimes=regimes,
        regime_names_to_ids=regime_names_to_ids,
    )


def test_a_one_member_collective_source_rejects_a_role_it_does_not_have() -> None:
    """A single-leg collective source refuses a role that is not its own.

    Its sole leg carries `source_member="f"`, so the router would match
    no leg for a husband's row and fall through to that leg — sending him into
    his wife's regime at her state. The number of legs is not what protects
    against that; checking the seed against this regime's own members is.
    """
    regimes, regime_names_to_ids = _dissolution_setup()
    wife_only = dataclasses.replace(regimes["married"], members=("f",))
    regimes = MappingProxyType({**regimes, "married": wife_only})

    with pytest.raises(InvalidInitialConditionsError, match="married"):
        _seed(
            regimes=regimes,
            regime_names_to_ids=regime_names_to_ids,
            regime_name="married",
            role=regimes["married"].member_names_to_ids["m"],
        )


def test_a_collective_source_accepts_a_role_it_does_have() -> None:
    """The refusal does not over-fire onto a role the regime declares.

    Without this control the check above could be refusing every seed into a
    collective regime rather than the mismatched ones.
    """
    regimes, regime_names_to_ids = _dissolution_setup()
    role = regimes["married"].member_names_to_ids["m"]

    seeded = _seed(
        regimes=regimes,
        regime_names_to_ids=regime_names_to_ids,
        regime_name="married",
        role=role,
    )

    np.testing.assert_array_equal(np.asarray(seeded), [role])


def test_a_cohort_starting_singleton_needs_no_role() -> None:
    """A row starting in a singleton regime occupies none, and says so.

    This keeps the requirement scoped to where legs actually differ: a
    singleton regime has one leg carrying no role, so demanding one there
    would refuse a well-specified model.
    """
    regimes, regime_names_to_ids = _dissolution_setup()

    seeded = _seed(
        regimes=regimes,
        regime_names_to_ids=regime_names_to_ids,
        regime_name="single_f",
        role=None,
    )

    np.testing.assert_array_equal(np.asarray(seeded), [NO_ROLE])
