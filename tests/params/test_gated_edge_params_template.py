"""A gated edge's own functions contribute parameters under `params["edges"]`.

A gated edge is declared with four kinds of user callable: the selection
probability, the gate predicate, one projection per gate reference, and one
projection per route fallback. Each is an ordinary DAG function, so every scalar
it reads beyond the values and states the engine wires in is a model parameter.
Its slot is its declaration path under `params["edges"][source][target]`, it is
listed by `get_params_template()`, and the value the user supplies for it is
what the solved model uses.

The topology is a mutual-consent edge. A single woman reaches a married couple
only if both partners prefer marriage to their own single life (the gate), each
partner's single life is valued by a gate reference read at a projected wage,
and a woman whose proposal is refused falls back to her own single value, also
read at a projected wage. Wages live on the two-point grid $\\{1, 2\\}$ and every
value below is exact on it.
"""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.params.edges import edge_params
from _lcm.regime_building.gated_edges import SOURCE_PARAMS, EdgeArgProvenance
from _lcm.simulation.gated_routing import bind_provenance_params
from _lcm.typing import FlatParams
from lcm import (
    CollectiveUtility,
    DiscreteGrid,
    Gate,
    LinSpacedGrid,
    Model,
    Phased,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.ages import AgeGrid
from lcm.exceptions import InvalidParamsError
from lcm.transition import StochasticTransition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    Period,
    ScalarInt,
    UserFunction,
)
from tests.collective_fixtures import DISCOUNT_FACTOR, Work
from tests.conftest import DECIMAL_PRECISION

# The declaration path of the one gated edge, below `params["edges"]`.
_EDGE = ("single_f", "married_terminal")


def test_gate_scalar_parameter_is_a_model_parameter():
    """A scalar the gate predicate reads is listed and used at the supplied value.

    A marriage premium of 1.5 lifts both partners' married values above their
    single ones at every wage, so the gate is open throughout and the wife's
    continuation is her married value everywhere:
    `V = wage + 0.95 * V_married_f = [1 + 0.95*2, 2 + 0.95*4] = [2.9, 5.8]`.
    Without the premium the husband refuses at the high wage and the value there
    is 4.85 instead.
    """
    model = _build_model(
        gate=_consent_gate_with_premium,
        husband_reference_projection=_wage_itself,
        wife_fallback=_wife_fallback(projection=_wage_itself),
    )
    assert (*_EDGE, "predicate", "marriage_premium") in _leaf_paths(
        model.get_params_template()["edges"]
    )
    solution = model.solve(
        params={"discount_factor": DISCOUNT_FACTOR, "marriage_premium": 1.5},
        log_level="debug",
    ).values
    aaae(
        np.asarray(solution[0]["single_f"]),
        np.array([2.9, 5.8]),
        decimal=DECIMAL_PRECISION,
    )


def test_gate_ref_projection_scalar_parameter_is_a_model_parameter():
    """A scalar a gate reference's projection reads is listed and used.

    A weight of 0.5 has the husband value his single life halfway between the
    couple's own wage and the top of the wage grid, which raises his outside
    option enough to refuse at both wages. The wife then takes her own fallback
    everywhere: `V = wage + 0.95 * 1.5 * wage = [2.425, 4.85]`. At a weight of
    zero he would accept at the low wage and the value there would be 2.9.
    """
    model = _build_model(
        gate=_consent_gate,
        husband_reference_projection=_husband_reference_wage,
        wife_fallback=_wife_fallback(projection=_wage_itself),
    )
    assert (
        *_EDGE,
        "references",
        "V_single_m_ref",
        "wage",
        "husband_reference_weight",
    ) in _leaf_paths(model.get_params_template()["edges"])
    solution = model.solve(
        params={
            "discount_factor": DISCOUNT_FACTOR,
            "husband_reference_weight": 0.5,
        },
        log_level="debug",
    ).values
    aaae(
        np.asarray(solution[0]["single_f"]),
        np.array([2.425, 4.85]),
        decimal=DECIMAL_PRECISION,
    )


def test_leg_fallback_projection_scalar_parameter_is_a_model_parameter():
    """A scalar a route fallback's projection reads is listed and used.

    The husband refuses at the high wage, so the wife falls back on her single
    value there. A weight of 0.4 has her read it at
    `2 - 0.4 * (2 - 1) = 1.6`, worth `1.5 * 1.6 = 2.4`, giving
    `V = [2.9, 2 + 0.95 * 2.4] = [2.9, 4.28]`. At a weight of zero she would
    read it at her own wage, worth 3.0, and the value there would be 4.85.
    """
    model = _build_model(
        gate=_consent_gate,
        husband_reference_projection=_wage_itself,
        wife_fallback=_wife_fallback(projection=_wife_fallback_wage),
    )
    assert (
        *_EDGE,
        "routes",
        "f",
        "fallback",
        "wage",
        "wife_fallback_weight",
    ) in _leaf_paths(model.get_params_template()["edges"])
    solution = model.solve(
        params={"discount_factor": DISCOUNT_FACTOR, "wife_fallback_weight": 0.4},
        log_level="debug",
    ).values
    aaae(
        np.asarray(solution[0]["single_f"]),
        np.array([2.9, 4.28]),
        decimal=DECIMAL_PRECISION,
    )


def test_gated_edge_template_holds_each_declaration_path():
    """Every edge callable's parameter sits at its declaration path, nowhere else."""
    model = _build_model(
        probability=_marry_at_meeting_rate,
        gate=_consent_gate_with_premium,
        husband_reference_projection=_husband_reference_wage,
        wife_fallback=_wife_fallback(projection=_wife_fallback_wage),
    )
    assert _leaf_paths(model.get_params_template()["edges"]) == {
        (*_EDGE, "meeting_rate"),
        (*_EDGE, "predicate", "marriage_premium"),
        (
            *_EDGE,
            "references",
            "V_single_m_ref",
            "wage",
            "husband_reference_weight",
        ),
        (*_EDGE, "routes", "f", "fallback", "wage", "wife_fallback_weight"),
    }


def test_source_regime_template_holds_no_edge_entry():
    """The source regime's branch names no target, law or gate entry."""
    model = _build_model(
        probability=_marry_at_meeting_rate,
        gate=_consent_gate_with_premium,
        husband_reference_projection=_husband_reference_wage,
        wife_fallback=_wife_fallback(projection=_wife_fallback_wage),
    )
    assert not {"married_terminal", "single_f_terminal", "next_regime", "gate"} & set(
        model.get_params_template()["single_f"]
    )


def test_phased_fallback_template_nests_each_phase_under_its_name():
    """A `Phased` fallback's two projections each own the parameters they read."""
    model = _build_model(
        gate=_consent_gate,
        husband_reference_projection=_wage_itself,
        wife_fallback=Phased(
            solve=_wife_fallback(projection=_wife_fallback_wage),
            simulate=_wife_fallback(projection=_wife_realized_wage),
        ),
    )
    route = (*_EDGE, "routes", "f", "fallback")
    assert _leaf_paths(model.get_params_template()["edges"]) == {
        (*route, "solve", "wage", "wife_fallback_weight"),
        (*route, "simulate", "wage", "wife_realized_weight"),
    }


@pytest.mark.parametrize(
    "edge_branch",
    [
        pytest.param(
            {
                "meeting_rate": 1.0,
                "predicate": {"marriage_premium": 0.25},
                "references": {
                    "V_single_m_ref": {"wage": {"husband_reference_weight": 0.1}}
                },
                "routes": {"f": {"fallback": {"wage": {"wife_fallback_weight": 0.4}}}},
            },
            id="declaration-path",
        ),
        pytest.param(None, id="edges-source-level"),
    ],
)
def test_gated_edge_values_do_not_depend_on_the_level_supplying_them(edge_branch):
    """The edge callables read the same values from every level, so solve agrees.

    The reference supplies each value at the model level. The declaration path
    and the `params["edges"][source]` level hand the same numbers to the same
    callables, so the solved values are identical, not merely close.
    """
    model = _build_model(
        probability=_marry_at_meeting_rate,
        gate=_consent_gate_with_premium,
        husband_reference_projection=_husband_reference_wage,
        wife_fallback=_wife_fallback(projection=_wife_fallback_wage),
    )
    values = {
        "meeting_rate": 1.0,
        "marriage_premium": 0.25,
        "husband_reference_weight": 0.1,
        "wife_fallback_weight": 0.4,
    }
    reference = model.solve(
        params={"discount_factor": DISCOUNT_FACTOR, **values}, log_level="debug"
    ).values
    source_branch = values if edge_branch is None else {"married_terminal": edge_branch}
    got = model.solve(
        params={
            "discount_factor": DISCOUNT_FACTOR,
            "edges": {"single_f": source_branch},
        },
        log_level="debug",
    ).values
    np.testing.assert_array_equal(
        np.asarray(got[0]["single_f"]), np.asarray(reference[0]["single_f"])
    )


def test_engine_edge_namespace_holds_exactly_the_template_slots():
    """`flat_params["edges"][source]` keys are the template's edge paths, joined."""
    model = _build_model(
        probability=_marry_at_meeting_rate,
        gate=_consent_gate_with_premium,
        husband_reference_projection=_husband_reference_wage,
        wife_fallback=_wife_fallback(projection=_wife_fallback_wage),
    )
    flat_params = model._process_params(
        {
            "discount_factor": DISCOUNT_FACTOR,
            "meeting_rate": 1.0,
            "marriage_premium": 0.25,
            "husband_reference_weight": 0.1,
            "wife_fallback_weight": 0.4,
        }
    )
    assert set(edge_params(flat_params, source="single_f")) == {
        "married_terminal__meeting_rate",
        "married_terminal__predicate__marriage_premium",
        (
            "married_terminal__references__V_single_m_ref__wage__"
            "husband_reference_weight"
        ),
        "married_terminal__routes__f__fallback__wage__wife_fallback_weight",
    }


def test_source_regime_flat_params_hold_no_edge_key():
    """The source regime's flat params carry its own functions' parameters only."""
    model = _build_model(
        probability=_marry_at_meeting_rate,
        gate=_consent_gate_with_premium,
        husband_reference_projection=_husband_reference_wage,
        wife_fallback=_wife_fallback(projection=_wife_fallback_wage),
    )
    flat_params = model._process_params(
        {
            "discount_factor": DISCOUNT_FACTOR,
            "meeting_rate": 1.0,
            "marriage_premium": 0.25,
            "husband_reference_weight": 0.1,
            "wife_fallback_weight": 0.4,
        }
    )
    assert not [key for key in flat_params["single_f"] if "married_terminal" in key]


def test_simulate_binder_names_the_user_path_of_a_missing_edge_slot():
    """A gate parameter missing from the edge namespace is reported by its path."""
    qname = "married_terminal__predicate__marriage_premium"
    provenance = EdgeArgProvenance(
        states=frozenset(),
        params=MappingProxyType({f"__source_param__{qname}": (SOURCE_PARAMS, qname)}),
    )
    flat_params = cast(
        "FlatParams",
        MappingProxyType(
            {
                "single_f": MappingProxyType({}),
                "married_terminal": MappingProxyType({}),
                "edges": MappingProxyType({"single_f": MappingProxyType({})}),
            }
        ),
    )
    with pytest.raises(
        KeyError,
        match=r"params\['edges'\]\['single_f'\]\['married_terminal'\]\['predicate'\]"
        r"\['marriage_premium'\]",
    ):
        bind_provenance_params(
            provenance=provenance,
            flat_params=flat_params,
            source_name="single_f",
            target_name="married_terminal",
        )


@pytest.mark.parametrize(
    "regime_params",
    [
        pytest.param(
            {"married_terminal": {"predicate": {"marriage_premium": 1.5}}},
            id="predicate",
        ),
        pytest.param({"marriage_premium": 1.5}, id="source-regime-level"),
    ],
)
def test_edge_parameter_under_the_source_regime_names_the_edges_path(regime_params):
    """A gate parameter written under the source regime is refused with its path."""
    model = _build_model(
        gate=_consent_gate_with_premium,
        husband_reference_projection=_wage_itself,
        wife_fallback=_wife_fallback(projection=_wage_itself),
    )
    with pytest.raises(
        InvalidParamsError,
        match=r"params\['edges'\]\['single_f'\]\['married_terminal'\]\['predicate'\]"
        r"\['marriage_premium'\]",
    ):
        model.solve(
            params={"discount_factor": DISCOUNT_FACTOR, "single_f": regime_params},
            log_level="off",
        )


def test_series_valued_gate_parameter_converts_and_solves():
    """An age-indexed Series for a gate parameter is read at the fold's age.

    The premium is 1.5 at every age, so the solution equals the scalar case:
    `V = [2.9, 5.8]`.
    """
    model = _build_model(
        gate=_consent_gate_with_premium_by_period,
        husband_reference_projection=_wage_itself,
        wife_fallback=_wife_fallback(projection=_wage_itself),
    )
    premium = pd.Series(
        [1.5, 1.5, 1.5],
        index=pd.MultiIndex.from_arrays([[0.0, 1.0, 2.0]], names=["age"]),
    )
    solution = model.solve(
        params={
            "discount_factor": DISCOUNT_FACTOR,
            "edges": {
                "single_f": {
                    "married_terminal": {"predicate": {"marriage_premium": premium}}
                }
            },
        },
        log_level="debug",
    ).values
    aaae(
        np.asarray(solution[0]["single_f"]),
        np.array([2.9, 5.8]),
        decimal=DECIMAL_PRECISION,
    )


@categorical(ordered=False)
class _RegimeId:
    """Regime ids of the mutual-consent model."""

    single_f: ScalarInt
    married_terminal: ScalarInt
    single_f_terminal: ScalarInt
    single_m_terminal: ScalarInt


# The one continuous state every regime carries.
_WAGE = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)

# `_WAGE`'s two nodes, so a projection can name the grid's ends directly.
_WAGE_LOW = 1.0
_WAGE_HIGH = 2.0

# Three ages: the single woman decides at age 0, everyone else pays out from age 1.
_AGES = AgeGrid(start=0, inclusive_stop=2, step="Y")


def _leaf_paths(branch: Mapping[str, object]) -> set[tuple[str, ...]]:
    """Return the path of every leaf of a params-template branch.

    Args:
        branch: A mapping of `Model.get_params_template()`.

    Returns:
        Set of the key paths, from `branch` down, that end in a parameter leaf.

    """
    paths: set[tuple[str, ...]] = set()
    for name, value in branch.items():
        if isinstance(value, Mapping):
            paths |= {(name, *path) for path in _leaf_paths(value)}
        else:
            paths.add((name,))
    return paths


def _wife_fallback(*, projection: UserFunction) -> ProjectedRegimeValue:
    """The wife's own single value, read at the wage `projection` returns."""
    return ProjectedRegimeValue(
        regime="single_f_terminal", projection={"wage": projection}
    )


def _build_model(
    *,
    gate: UserFunction,
    husband_reference_projection: UserFunction,
    wife_fallback: ProjectedRegimeValue | Phased,
    probability: Callable[..., FloatND] | None = None,
) -> Model:
    """Build the mutual-consent model around the edge callables given.

    Args:
        gate: The edge's boolean consent predicate.
        husband_reference_projection: Wage at which the husband's single value is
            read for the gate.
        wife_fallback: Where the wife lands when consent fails, for both phases
            or `Phased` per phase.
        probability: Probability with which the single woman meets the couple;
            certainty when `None`.

    Returns:
        The model, ready to solve.

    """
    single_f_law = {
        "married_terminal": StochasticTransition(
            func=_marry_for_sure if probability is None else probability
        )
    }
    single_f_gates = {
        "married_terminal": Gate(
            predicate=gate,
            routes={
                "f": StakeholderRoute(target_stakeholder="f", fallback=wife_fallback)
            },
            references={
                "V_single_f_ref": ProjectedRegimeValue(
                    regime="single_f_terminal",
                    projection={"wage": _wage_itself},
                ),
                "V_single_m_ref": ProjectedRegimeValue(
                    regime="single_m_terminal",
                    projection={"wage": husband_reference_projection},
                ),
            },
        )
    }
    single_f = Regime(
        states={"wage": _WAGE},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _utility_single_f},
    )
    married_terminal = Regime(
        states={"wage": _WAGE},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _utility_married_f, "m": _utility_married_m}
            )
        },
    )
    single_f_terminal = Regime(
        states={"wage": _WAGE},
        functions={"utility": _utility_single_f_terminal},
    )
    single_m_terminal = Regime(
        states={"wage": _WAGE},
        functions={"utility": _utility_single_m_terminal},
    )
    return Model(
        regimes={
            "single_f": single_f,
            "married_terminal": married_terminal,
            "single_f_terminal": single_f_terminal,
            "single_m_terminal": single_m_terminal,
        },
        ages=_AGES,
        regime_id_class=_RegimeId,
        initial_nodes={0: "single_f"},
        edges={"single_f": Transition(law=single_f_law, gates=single_f_gates)},
    )


def _consent_gate(
    *,
    V_target_f: FloatND,
    V_target_m: FloatND,
    V_single_f_ref: FloatND,
    V_single_m_ref: FloatND,
) -> BoolND:
    """Marriage happens only if both partners strictly prefer it."""
    return (V_target_f > V_single_f_ref) & (V_target_m > V_single_m_ref)


def _consent_gate_with_premium(
    *,
    V_target_f: FloatND,
    V_target_m: FloatND,
    V_single_f_ref: FloatND,
    V_single_m_ref: FloatND,
    marriage_premium: float,
) -> BoolND:
    """Both partners value marriage at its value plus a common premium."""
    return ((V_target_f + marriage_premium) > V_single_f_ref) & (
        (V_target_m + marriage_premium) > V_single_m_ref
    )


def _consent_gate_with_premium_by_period(
    *,
    V_target_f: FloatND,
    V_target_m: FloatND,
    V_single_f_ref: FloatND,
    V_single_m_ref: FloatND,
    marriage_premium: FloatND,
    period: Period,
) -> BoolND:
    """Both partners value marriage at its value plus the premium of the period."""
    premium = marriage_premium[period]
    return ((V_target_f + premium) > V_single_f_ref) & (
        (V_target_m + premium) > V_single_m_ref
    )


def _wage_itself(wage: ContinuousState) -> ContinuousState:
    """Read the referenced regime at the couple's own wage."""
    return wage


def _husband_reference_wage(
    *, wage: ContinuousState, husband_reference_weight: float
) -> ContinuousState:
    """Wage at which the husband values single life.

    A weight of zero is the couple's own wage; a weight of one is the top of the
    wage grid, where single life is worth most to him.
    """
    return wage + husband_reference_weight * (_WAGE_HIGH - wage)


def _wife_fallback_wage(
    *, wage: ContinuousState, wife_fallback_weight: float
) -> ContinuousState:
    """Wage at which a refused woman values single life.

    A weight of zero is the couple's own wage; a weight of one is the bottom of
    the wage grid, so a larger weight is a costlier refusal.
    """
    return wage + wife_fallback_weight * (_WAGE_LOW - wage)


def _wife_realized_wage(
    *, wage: ContinuousState, wife_realized_weight: float
) -> ContinuousState:
    """Wage at which a refused woman lands in simulation."""
    return wage + wife_realized_weight * (_WAGE_LOW - wage)


def _utility_single_f(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """A single woman earns her wage when she works and nothing otherwise."""
    return wage * work


def _utility_single_f_terminal(wage: ContinuousState) -> FloatND:
    """Terminal single-life payoff of the woman: 1.5 per unit of wage."""
    return 1.5 * wage


def _utility_single_m_terminal(wage: ContinuousState) -> FloatND:
    """Terminal single-life payoff of the man: 0.5 at the low wage, 3.0 at the high."""
    return jnp.where(wage < 1.5, 0.5, 3.0)


def _utility_married_f(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """The wife's married payoff: twice the wage, whatever the couple does."""
    return 2.0 * wage + 0.0 * work


def _utility_married_m(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """The husband's married payoff: the wage itself, whatever the couple does."""
    return wage + 0.0 * work


def _marry_for_sure(age: FloatND) -> FloatND:
    """The single woman faces the married couple with probability one."""
    return jnp.ones_like(age, dtype=float)


def _marry_at_meeting_rate(*, age: FloatND, meeting_rate: float) -> FloatND:
    """The single woman faces the married couple at the meeting rate."""
    return meeting_rate * jnp.ones_like(age, dtype=float)
