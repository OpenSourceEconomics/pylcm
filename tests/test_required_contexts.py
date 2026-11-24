"""Required contexts: missing prerequisites, zero-node regimes, folds and schemas.

A problem some start requires by value must have a law at the age it is read,
whatever kind of read requires it, and adding a start never repairs a missing
prerequisite. A registered regime no start reaches keeps its code, its place in
the regime vector and its absence from every period. A state-only gate folds
only where its source reaches the gated target. Parameters are collected over
the required cases, including value-only problems, and one parameter name has
one schema.
"""

from collections.abc import Callable, Mapping
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    AgeRange,
    ByAge,
    Gate,
    LinSpacedGrid,
    MemberRoute,
    Model,
    ProjectedRegimeValue,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ModelInitializationError
from lcm.initial_nodes import UserInitialNodes
from lcm.phased import Phased
from lcm.transition import AgeSelector, PhaseEdges, TransitionLaw
from lcm.typing import (
    BoolND,
    ContinuousState,
    FloatND,
    RegimeName,
    ScalarInt,
    UserFacingRegimeParamsTemplate,
    UserFunction,
)

_WEALTH = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
_PARAMS = {"discount_factor": 0.9}


def _utility(wealth: ContinuousState) -> FloatND:
    return wealth


def _nonterminal(*, utility: UserFunction = _utility) -> Regime:
    return Regime(
        states={"wealth": _WEALTH},
        state_transitions={"wealth": fixed_transition("wealth")},
        functions={"utility": utility},
    )


def _terminal() -> Regime:
    return Regime(
        states={"wealth": _WEALTH},
        functions={"utility": _utility},
    )


@categorical(ordered=False)
class _GatedId:
    source: ScalarInt
    target: ScalarInt
    reference: ScalarInt
    priced: ScalarInt
    fallback: ScalarInt


_GATED_AGES = AgeGrid(start=40, inclusive_stop=50, step="5Y")


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _wealth_is_positive(wealth: ContinuousState) -> BoolND:
    return wealth > 0.0


def _identity(wealth: ContinuousState) -> ContinuousState:
    return wealth


def _projected(regime: RegimeName) -> ProjectedRegimeValue:
    return ProjectedRegimeValue(regime=regime, projection={"wealth": _identity})


def _gated_law() -> dict[RegimeName, StochasticTransition]:
    return {"target": StochasticTransition(func=_prob_one)}


type Fallback = (
    ProjectedRegimeValue | Phased[ProjectedRegimeValue, ProjectedRegimeValue]
)
type GatedStructure = Mapping[RegimeName, Mapping[RegimeName, AgeSelector]]


def _target_gate(*, fallback: Fallback | None) -> dict[RegimeName, Gate]:
    """Gate the target on wealth on landing."""
    routes = {"only": MemberRoute(fallback=fallback or _projected("fallback"))}
    return {"target": Gate(predicate=_wealth_is_positive, routes=routes)}


def _with_source_law(
    *,
    phase: GatedStructure,
    law: TransitionLaw,
    gates: Mapping[RegimeName, Gate],
) -> PhaseEdges:
    return {
        **phase,
        "source": Transition(targets=phase["source"], law=law, gates=gates),
    }


def _gated_model(
    *,
    initial_nodes: UserInitialNodes | None = None,
    source: TransitionLaw | None = None,
    priced: Regime | None = None,
    fallback: Fallback | None = None,
    edges: GatedStructure | Phased[GatedStructure, GatedStructure] | None = None,
) -> Model:
    law = source or ByAge(cases={40: _gated_law()})
    gates = _target_gate(fallback=fallback)
    structure = edges or {"source": {"target": 40, "fallback": 40}}
    return Model(
        regimes={
            "source": _nonterminal(),
            "target": _terminal(),
            "reference": _terminal(),
            "priced": priced or _terminal(),
            "fallback": _terminal(),
        },
        ages=_GATED_AGES,
        regime_id_class=_GatedId,
        initial_nodes=initial_nodes or {40: "source"},
        edges=(
            Phased(
                solve=_with_source_law(phase=structure.solve, law=law, gates=gates),
                simulate=_with_source_law(
                    phase=structure.simulate, law=law, gates=gates
                ),
            )
            if isinstance(structure, Phased)
            else _with_source_law(phase=structure, law=law, gates=gates)
        ),
    )


def _solve_fallback_without_law(initial_nodes: UserInitialNodes) -> Model:
    return _gated_model(
        initial_nodes=initial_nodes,
        priced=_nonterminal(),
        fallback=Phased(solve=_projected("priced"), simulate=_projected("fallback")),
        edges=Phased(
            solve={
                "source": {"target": 40, "priced": 40},
                "priced": {"fallback": 40},
            },
            simulate={
                "source": {"target": 40, "fallback": 40},
                "priced": {"fallback": 40},
            },
        ),
    )


# The match names the kind of read, the requesting pair and the missing pair, in any
# order.
@pytest.mark.parametrize(
    ("build", "match"),
    [
        (
            _solve_fallback_without_law,
            r"(?is)(?=.*fallback)(?=.*\(40, 'source'\))(?=.*'priced' at age 45)",
        ),
    ],
    ids=["solve-fallback"],
)
def test_a_required_value_read_without_a_law_names_its_kind(
    *, build: Callable[[UserInitialNodes], Model], match: str
) -> None:
    """A valuation fallback lacking a law is named as such."""
    with pytest.raises(ModelInitializationError, match=match):
        build({40: "source"})


@pytest.mark.parametrize(
    ("build", "initial_nodes"),
    [(_solve_fallback_without_law, {40: "source", 45: "priced"})],
    ids=["solve-fallback"],
)
def test_an_added_root_does_not_repair_a_missing_prerequisite(
    *,
    build: Callable[[UserInitialNodes], Model],
    initial_nodes: Mapping[AgeSelector, RegimeName],
) -> None:
    """Declaring the missing pair as a start still fails on its missing law."""
    missing = next(name for age, name in initial_nodes.items() if age == 45)
    with pytest.raises(ModelInitializationError, match=rf"'{missing}' at age 45"):
        build(initial_nodes)


def test_an_unknown_target_in_an_unused_case_is_a_global_error() -> None:
    """An unknown regime name fails even in a case no start requires."""
    with pytest.raises(ModelInitializationError, match="'nowhere'"):
        _gated_model(
            source=ByAge(
                cases={
                    40: _gated_law(),
                    45: {"nowhere": StochasticTransition(func=_prob_one)},
                }
            ),
            edges={"source": {"target": 40, "fallback": 40, "nowhere": 45}},
        )


def test_a_gate_folds_at_every_age_its_target_is_reached() -> None:
    """A target reached by two schedule cases is gated after both of them."""
    model = _gated_model(
        source=ByAge(cases={40: _gated_law(), 45: _gated_law()}),
        edges={"source": {"target": (40, 45), "fallback": (40, 45)}},
        initial_nodes={40: "source", 45: "source"},
    )
    folds = model._regimes["source"].gated_edges["target"].folds_by_period
    assert set(folds) == {1, 2}


@categorical(ordered=False)
class _LifeId:
    working: ScalarInt
    retirement: ScalarInt
    dead: ScalarInt


_LIFE_AGES = AgeGrid(start=25, inclusive_stop=75, step="10Y")


def _stay() -> FloatND:
    return jnp.asarray(0.9)


def _die() -> FloatND:
    return jnp.asarray(0.1)


_LIFE_EDGES = {
    "working": Transition(
        targets={
            "working": (25, 35, 45),
            "dead": (25, 35, 45),
            "retirement": 55,
        },
        law=ByAge.until(
            stop_age_exclusive=65,
            law={
                "working": StochasticTransition(func=_stay),
                "dead": StochasticTransition(func=_die),
            },
            then="retirement",
        ),
    ),
    "retirement": {"dead": 65},
}


def _life_model(initial_nodes: UserInitialNodes) -> Model:
    return Model(
        regimes={
            "working": _nonterminal(),
            "retirement": _nonterminal(),
            "dead": _terminal(),
        },
        ages=_LIFE_AGES,
        regime_id_class=_LifeId,
        initial_nodes=initial_nodes,
        edges=_LIFE_EDGES,
    )


_ROOTS = pytest.mark.parametrize(
    "initial_nodes",
    [{25: "working"}, {65: "retirement"}],
    ids=["first-age-root", "late-root-with-zero-node-regime"],
)


@_ROOTS
def test_regime_codes_do_not_depend_on_the_starts(
    initial_nodes: UserInitialNodes,
) -> None:
    """Every registered regime keeps its code, reached or not."""
    assert _life_model(initial_nodes).regime_names_to_ids == {
        "working": 0,
        "retirement": 1,
        "dead": 2,
    }


@_ROOTS
def test_retirement_solves_to_the_dead_value_under_any_start(
    initial_nodes: UserInitialNodes,
) -> None:
    """Retirement moving along its only edge solves to the dead value."""
    values = _life_model(initial_nodes).solve(params=_PARAMS, log_level="off")
    np.testing.assert_allclose(
        np.asarray(values.values[4]["retirement"]), np.asarray([0.0, 1.9])
    )


def test_late_root_with_an_empty_first_period_simulates() -> None:
    """With no problem at the first age, subjects start at 65 and die at 75."""
    model = _life_model({65: "retirement"})
    panel = model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wealth": jnp.asarray([0.0, 1.0]),
            "age": jnp.asarray([65.0, 65.0]),
            "regime_id": jnp.full(2, model.regime_names_to_ids["retirement"]),
        },
        solution=model.solve(params=_PARAMS, log_level="off"),
        log_level="off",
        seed=0,
    ).to_dataframe()
    assert sorted(zip(panel["age"], panel["regime_name"], strict=True)) == [
        (65, "retirement"),
        (65, "retirement"),
        (75, "dead"),
        (75, "dead"),
    ]


def test_first_period_of_a_late_root_has_no_values() -> None:
    """Periods keep their global index; those before the start publish no key."""
    values = _life_model({65: "retirement"}).solve(params=_PARAMS, log_level="off")
    assert {period: set(by) for period, by in values.values.items()} == {
        4: {"retirement"},
        5: {"dead"},
    }


def _source_owned_fold_model() -> Model:
    return _gated_model(
        initial_nodes={40: "source", 45: "source"},
        source=ByAge(cases={40: "fallback", 45: _gated_law()}),
        edges={"source": {"fallback": (40, 45), "target": 45}},
    )


def test_a_gate_folds_only_where_its_source_reaches_the_target() -> None:
    """A source reaching its gated target only at 45 folds the gate only at 50."""
    folds = _source_owned_fold_model()._regimes["source"].gated_edges["target"]
    assert set(folds.folds_by_period) == {2}


def _rate_as_float(rate: float) -> FloatND:
    return jnp.asarray(rate)


def _rest_as_float(rate: float) -> FloatND:
    return 1 - jnp.asarray(rate)


def _rate_as_int(rate: ScalarInt) -> FloatND:
    return jnp.asarray(rate) * 0.5


def _rest_as_int(rate: ScalarInt) -> FloatND:
    return 1 - jnp.asarray(rate) * 0.5


def _two_schema_model(initial_nodes: UserInitialNodes) -> Model:
    return Model(
        regimes={
            "working": _nonterminal(),
            "retirement": _terminal(),
            "dead": _terminal(),
        },
        ages=_LIFE_AGES,
        regime_id_class=_LifeId,
        initial_nodes=initial_nodes,
        edges={
            "working": Transition(
                targets={
                    "working": AgeRange(exclusive_stop=65),
                    "dead": AgeRange(exclusive_stop=75),
                },
                law=ByAge(
                    cases={
                        AgeRange(start=25, exclusive_stop=45): {
                            "working": StochasticTransition(func=_rate_as_float),
                            "dead": StochasticTransition(func=_rest_as_float),
                        },
                        AgeRange(start=45, exclusive_stop=65): {
                            "working": StochasticTransition(func=_rate_as_int),
                            "dead": StochasticTransition(func=_rest_as_int),
                        },
                    },
                    default="dead",
                ),
            )
        },
    )


def test_conflicting_schemas_of_one_required_parameter_fail() -> None:
    """`rate` read as `float` in one required case and `ScalarInt` in another fails."""
    with pytest.raises(ModelInitializationError, match=r"'rate'.*float.*ScalarInt"):
        _two_schema_model({25: "working"})


def test_a_schema_conflict_in_an_unrequired_case_is_not_required() -> None:
    """With only the late case required, `rate` has the late case's schema."""
    template = cast(
        "dict[str, UserFacingRegimeParamsTemplate]",
        _two_schema_model({55: "working"}).get_params_template(),
    )
    assert template["edges"]["working"]["working"] == {"rate": "ScalarInt"}


@categorical(ordered=False)
class _PhasedId:
    source: ScalarInt
    perceived: ScalarInt
    realized: ScalarInt
    end: ScalarInt


def _bonus_utility(*, wealth: ContinuousState, bonus: float) -> FloatND:
    return wealth + bonus


def _value_only_model() -> Model:
    return Model(
        regimes={
            "source": _nonterminal(),
            "perceived": _nonterminal(utility=_bonus_utility),
            "realized": _nonterminal(),
            "end": _terminal(),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_PhasedId,
        initial_nodes={0: "source"},
        edges=Phased(
            solve={
                "source": {"perceived": 0},
                "perceived": {"end": 1},
                "realized": {"end": 1},
            },
            simulate={
                "source": {"realized": 0},
                "perceived": {"end": 1},
                "realized": {"end": 1},
            },
        ),
    )


def test_a_parameter_of_a_value_only_regime_is_kept() -> None:
    """The perceived regime is never visited, yet its `bonus` is required."""
    assert cast(
        "dict[str, UserFacingRegimeParamsTemplate]",
        _value_only_model().get_params_template(),
    )["perceived"]["utility"] == {"bonus": "float"}


def _grown(*, wealth: ContinuousState, growth: float) -> ContinuousState:
    return wealth * growth


def _producer_model(initial_nodes: UserInitialNodes) -> Model:
    return Model(
        regimes={
            "working": Regime(
                states={"wealth": _WEALTH},
                state_transitions={
                    "wealth": {
                        "working": _identity,
                        "retirement": _grown,
                        "dead": _identity,
                    }
                },
                functions={"utility": _utility},
            ),
            "retirement": _nonterminal(),
            "dead": _terminal(),
        },
        ages=_LIFE_AGES,
        regime_id_class=_LifeId,
        initial_nodes=initial_nodes,
        edges=_LIFE_EDGES,
    )


def test_a_late_root_keeps_the_producer_of_its_exit_target() -> None:
    """A start at 55 still requires the wealth producer into retirement."""
    template = cast(
        "dict[str, UserFacingRegimeParamsTemplate]",
        _producer_model({55: "working"}).get_params_template(),
    )
    assert template["working"]["retirement"]["next_wealth"] == {"growth": "float"}
