"""What forward simulation writes into a gated edge's fallback state slots.

A dissolution `GatedEdge` on a collective source carries one leg per partner,
and each leg says where in its own single regime that partner would land. The
router recomputes the gate at the household's realized candidate target state
and, for the households the gate turned away, writes each leg's projected
coordinates into that leg's fallback regime.

Two properties of that write are pinned here.

- **It belongs to the households that were turned away.** A household whose
  gate opened stays with its target regime and dissolves into nothing, so no
  leg's projected coordinates belong in its record — the fallback slots it
  never entered keep whatever they held.
- **It covers every state the fallback regime carries in simulation.** A state
  declared `Phased(solve=..., simulate=...)` is no axis of the fallback
  regime's solved value function but is a genuine per-subject state once
  simulation runs, so a household entering that regime needs a landing point on
  it exactly as it does on the regime's solved states.

The model is the same household throughout: `married` is a two-stakeholder
regime whose consent gate keeps the higher-wage households together and sends
the lowest-wage one to `single_f` (the wife's regime, whose leg the simulated
cohort follows) and `single_m` (the husband's).
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.certainty_equivalent import LinearExpectation
from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from _lcm.simulation.gated_routing import (
    route_gated_edges,
    substitute_gated_edge_continuations,
)
from _lcm.solution.backward_induction import solve
from _lcm.utils.logging import get_logger
from lcm import (
    AgeGrid,
    DiscreteGrid,
    EdgeLeg,
    GatedEdge,
    LinSpacedGrid,
    Model,
    Phased,
    Regime,
    SamePeriodRef,
    categorical,
    fixed_transition,
)
from lcm.koopmans_aggregation import LinearAggregator
from lcm.transition import MarkovTransition
from lcm.typing import BoolND, ContinuousState, DiscreteAction, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION, build_prepared_structure

_BETA = 0.95

# Three periods: the couple is married at age 0, either still married or single
# at age 1, and out of the model at age 2.
_AGES = AgeGrid(start=0, stop=3, step="Y")

# The wage every regime is defined on, and the three households' wages.
_WAGE = LinSpacedGrid(start=1.0, stop=3.0, n_points=3)
_HOUSEHOLD_WAGES = (1.0, 2.0, 3.0)

# The consent gate keeps a household together above this wage, so the wage-1
# household is the only one of the three that dissolves.
_STAY_TOGETHER_ABOVE = 1.5

# The wife's simulate-only career axis, and the career her leg projects her
# onto per unit of household wage.
_CAREER = LinSpacedGrid(start=0.0, stop=100.0, n_points=5)
_CAREER_PER_WAGE = 10.0

# Career the simulated cohort is seeded with. No household wage projects onto
# it, so a career slot left at this value was never written.
_SEEDED_CAREER = 99.0

# Wage handed to the fallback regimes' slots before the router runs, so a slot
# the router wrote is recognizable from one it left alone.
_UNWRITTEN_WAGE = -999.0


@categorical(ordered=True)
class Work:
    """The binary labor-supply action every regime here offers."""

    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


@categorical(ordered=False)
class RegimeId:
    """Regime ids of the household model in this module."""

    married: ScalarInt
    married_terminal: ScalarInt
    single_f: ScalarInt
    single_f_terminal: ScalarInt
    single_m: ScalarInt
    single_m_terminal: ScalarInt


def test_gate_open_household_keeps_the_wifes_fallback_slot_it_never_entered():
    """Only a household the consent gate turned away is given a wife's regime.

    The three households enter the edge at wages 1, 2 and 3, and the gate
    closes on the wage-1 household alone. It is therefore the only one that
    lands in `single_f`, at wage 1.0; the two that stay married keep the wage
    their `single_f` slot already held.
    """
    routed_states = _route_three_households()
    np.testing.assert_array_equal(
        np.asarray(routed_states["single_f"]["wage"]),
        [_HOUSEHOLD_WAGES[0], _UNWRITTEN_WAGE, _UNWRITTEN_WAGE],
    )


def test_gate_open_household_keeps_the_husbands_fallback_slot_it_never_entered():
    """Every leg of the edge writes the households the gate turned away.

    The husband's leg owns a fallback regime of its own and is subject to the
    same rule as the wife's: it gives a starting wage to the wage-1 household
    that dissolves, and to neither of the two that stay married.
    """
    routed_states = _route_three_households()
    np.testing.assert_array_equal(
        np.asarray(routed_states["single_m"]["wage"]),
        [_HOUSEHOLD_WAGES[0], _UNWRITTEN_WAGE, _UNWRITTEN_WAGE],
    )


def test_dissolving_household_starts_its_fallback_regime_at_the_projected_career():
    """A dissolving wife enters `single_f` at the career her leg projects.

    `single_f` declares `career` as a carried state — imputed while the regime
    is solved, evolved per subject once simulation runs — so the leg routing a
    household into it says where on that axis she lands. At the wage-1
    household the projection is 10.0, replacing the 99.0 the cohort started at.
    """
    result = _simulate_three_households()
    single_f = result.raw_results["single_f"][1]
    assert bool(np.asarray(single_f.in_regime)[0])
    aaae(
        np.asarray(single_f.states["career"])[0],
        _CAREER_PER_WAGE * _HOUSEHOLD_WAGES[0],
        decimal=DECIMAL_PRECISION,
    )


def _route_three_households() -> MappingProxyType:
    """Run the router on three households and return their state slots.

    Every household's ordinary draw is the edge's target and every fallback
    slot starts at `_UNWRITTEN_WAGE`, so the returned arrays say exactly which
    households the edge wrote a fallback state for.

    Returns:
        Immutable mapping of regime name to that regime's state slots, each
        carrying one entry per household.

    """
    regimes, regime_names_to_ids, flat_params, bi_result = _solve_kernel_level(
        carrying_fallback=False
    )
    married = regimes["married"]
    _substituted, same_period_mappings = substitute_gated_edge_continuations(
        regime=married,
        regime_name="married",
        regimes=regimes,
        period=0,
        next_regime_to_V_arr=bi_result.value_functions[1],
        base_state_action_spaces={
            name: regime.solution.state_action_space(regime_params=flat_params[name])
            for name, regime in regimes.items()
        },
        period_to_regime_to_V_arr=bi_result.value_functions,
        period_to_regime_to_dissolution_flags=bi_result.dissolution_flags,
        flat_params=flat_params,
    )
    n_households = len(_HOUSEHOLD_WAGES)
    next_states = MappingProxyType(
        {
            "married_terminal": MappingProxyType(
                {"wage": jnp.asarray(_HOUSEHOLD_WAGES)}
            ),
            "single_f": MappingProxyType(
                {"wage": jnp.full(n_households, _UNWRITTEN_WAGE)}
            ),
            "single_m": MappingProxyType(
                {"wage": jnp.full(n_households, _UNWRITTEN_WAGE)}
            ),
        }
    )
    routed_states, _routed_ids = route_gated_edges(
        # The source is simulated at period 0, so the gate is decided on
        # the value it would enter at period 1.
        fold_period=1,
        regime=married,
        same_period_mappings=same_period_mappings,
        next_states=next_states,
        regime_names_to_ids=regime_names_to_ids,
        new_subject_regime_ids=jnp.full(
            n_households, regime_names_to_ids["married_terminal"], dtype=jnp.int32
        ),
        subjects_in_regime=jnp.ones(n_households, dtype=bool),
        flat_params=flat_params,
        own_stakeholder="f",
    )
    return routed_states


def _simulate_three_households():
    """Solve and simulate the model whose `single_f` carries a career state."""
    model = Model(
        regimes=_make_regimes(carrying_fallback=True),
        ages=_AGES,
        regime_id_class=RegimeId,
    )
    params = {"discount_factor": _BETA}
    solution, dissolution_flags = model.solve(
        params=params, log_level="off", return_dissolution_flags=True
    )
    n_households = len(_HOUSEHOLD_WAGES)
    initial_conditions = MappingProxyType(
        {
            "wage": jnp.asarray(_HOUSEHOLD_WAGES),
            "career": jnp.full(n_households, _SEEDED_CAREER),
            "age": jnp.zeros(n_households),
            "regime_id": jnp.full(
                n_households, model.regime_names_to_ids["married"], dtype=jnp.int32
            ),
        }
    )
    return model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=dissolution_flags,
        own_stakeholder="f",
        log_level="off",
        seed=0,
    )


def _solve_kernel_level(*, carrying_fallback: bool):
    """Compile and solve the model, returning the pieces the router needs."""
    regimes_dict = _make_regimes(carrying_fallback=carrying_fallback)
    regime_names_to_ids = MappingProxyType(
        {name: jnp.int32(index) for index, name in enumerate(regimes_dict)}
    )
    user_regimes = finalize_regimes(
        user_regimes=regimes_dict,
        derived_categoricals={},
        koopmans_aggregator=LinearAggregator(),
        certainty_equivalent=LinearExpectation(),
    )
    regimes = process_regimes(
        prepared_structure=build_prepared_structure(
            user_regimes=user_regimes, ages=_AGES
        ),
        user_regimes=user_regimes,
        ages=_AGES,
        regime_names_to_ids=regime_names_to_ids,
        enable_jit=False,
    )
    flat_params = MappingProxyType(
        {
            name: MappingProxyType(
                {}
                if regime.terminal
                else {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
            )
            for name, regime in regimes_dict.items()
        }
    )
    bi_result = solve(
        flat_params=flat_params,
        ages=_AGES,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    return regimes, regime_names_to_ids, flat_params, bi_result


def _make_regimes(*, carrying_fallback: bool) -> dict[str, Regime]:
    """Build the household regimes, with or without the wife's carried career.

    Args:
        carrying_fallback: Whether `single_f` declares the carried `career`
            state and the wife's leg projects a household onto it.

    Returns:
        Dict of regime names to regimes, ready for `Model` or `process_regimes`.

    """
    if carrying_fallback:
        wife_projection = {"wage": _identity_wage, "career": _project_career}
        single_f = Regime(
            transition={"single_f_terminal": MarkovTransition(_prob_one)},
            active=lambda age: (age >= 1) & (age < 2),
            states={
                "wage": _WAGE,
                "career": Phased(solve=_impute_career, simulate=_CAREER),
            },
            state_transitions={
                "wage": fixed_transition("wage"),
                "career": _next_career,
            },
            actions={"work": DiscreteGrid(Work)},
            functions={"utility": _utility_single_f_with_career},
        )
    else:
        wife_projection = {"wage": _identity_wage}
        single_f = Regime(
            transition={"single_f_terminal": MarkovTransition(_prob_one)},
            active=lambda age: (age >= 1) & (age < 2),
            states={"wage": _WAGE},
            state_transitions={"wage": fixed_transition("wage")},
            actions={"work": DiscreteGrid(Work)},
            functions={"utility": _utility_single_f},
        )

    married = Regime(
        transition={"married_terminal": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _utility_married_f, "utility_m": _utility_married_m},
        gated_edges={
            "married_terminal": GatedEdge(
                gate=_consent_gate,
                legs={
                    "f": EdgeLeg(
                        target_stakeholder="f",
                        fallback=SamePeriodRef(
                            regime="single_f", projection=wife_projection
                        ),
                    ),
                    "m": EdgeLeg(
                        target_stakeholder="m",
                        fallback=SamePeriodRef(
                            regime="single_m", projection={"wage": _identity_wage}
                        ),
                    ),
                },
            )
        },
    )
    married_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _utility_married_f, "utility_m": _utility_married_m},
    )
    single_m = Regime(
        transition={"single_m_terminal": MarkovTransition(_prob_one)},
        active=lambda age: (age >= 1) & (age < 2),
        states={"wage": _WAGE},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility_single_m},
    )
    single_terminal = Regime(
        transition=None,
        active=lambda age: age >= 2,
        states={"wage": _WAGE},
        functions={"utility": _utility_no_payoff},
    )
    return {
        "married": married,
        "married_terminal": married_terminal,
        "single_f": single_f,
        "single_f_terminal": single_terminal,
        "single_m": single_m,
        "single_m_terminal": single_terminal.replace(),
    }


def _prob_one(age: FloatND) -> FloatND:
    """Regime transition taken with certainty."""
    return jnp.ones_like(age, dtype=float)


def _consent_gate(wage: ContinuousState) -> BoolND:
    """Consent holds above the wage at which the household stays together."""
    return wage > _STAY_TOGETHER_ABOVE


def _identity_wage(wage: ContinuousState) -> ContinuousState:
    """A partner keeps the household wage on entering a single regime."""
    return wage


def _project_career(wage: ContinuousState) -> FloatND:
    """The career the wife starts her own regime at, scaled by household wage."""
    return _CAREER_PER_WAGE * wage


def _impute_career() -> FloatND:
    """Career while `single_f` is solved: a constant, never a grid axis."""
    return jnp.asarray(0.0)


def _next_career(career: FloatND) -> FloatND:
    """Career law once simulation runs: one more year per period."""
    return career + 1.0


def _utility_married_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Wife's payoff while married: she values her leisure and the wage."""
    return 3.0 * (1.0 - work) + 2.0 * wage * work


def _utility_married_m(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Husband's payoff while married: he values household consumption."""
    return 0.5 * (1.0 - work) + wage * work


def _utility_single_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Wife's payoff in her own regime."""
    return 1.5 * wage * work


def _utility_single_f_with_career(
    wage: ContinuousState, work: DiscreteAction, career: FloatND
) -> FloatND:
    """Wife's payoff in her own regime, reading her career at a zero weight.

    The zero weight keeps every value function equal to the model without the
    carried state, so declaring the career changes what simulation writes, not
    what the solver computes.
    """
    return 1.5 * wage * work + 0.0 * career


def _utility_single_m(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Husband's payoff in his own regime."""
    return 1.0 * wage * work


def _utility_no_payoff(wage: ContinuousState) -> FloatND:
    """Terminal payoff: nothing, so the terminal period adds no value."""
    return 0.0 * wage
