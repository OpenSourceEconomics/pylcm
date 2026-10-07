"""Parameters of a regime-transition law live at its declaration path.

`Model(edges=...)` declares where a source regime can go and, through a
`Transition` law, how it chooses. The law's parameters are found under
`params["edges"][source]`: directly there for a law over all targets, one level
deeper under the target for a per-target law. A value may also be given once for
every callable of a source (`params["edges"][source][arg]`) or at the model level;
a value written under the source regime itself never reaches the law.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from _lcm.params.edges import edge_params, regime_kernel_params
from _lcm.params.regime_template import iter_edge_callables
from lcm import (
    AgeGrid,
    AgeRange,
    ByAge,
    DeterministicTransition,
    LinSpacedGrid,
    Model,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import (
    InvalidNameError,
    InvalidParamsError,
    ModelInitializationError,
)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
    UserFunction,
    UserParams,
)

_AGES = AgeGrid(start=60, inclusive_stop=63, step="Y")
_WEALTH = LinSpacedGrid(start=1.0, stop=10.0, n_points=5)
_CONSUMPTION = LinSpacedGrid(start=0.5, stop=5.0, n_points=5)


@categorical(ordered=False)
class _MortalId:
    working: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class _RetirementId:
    working: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class _WorkRetireId:
    working: ScalarInt
    retired: ScalarInt


def test_coarse_law_slot_is_the_source_path():
    """A law choosing among all targets reads `params["edges"][source][arg]`."""
    model = _retirement_model()
    assert _leaf_paths(model.get_params_template()["edges"]) == {
        ("working", "retirement_age")
    }


def test_per_target_law_slots_nest_under_their_target():
    """A per-target law reads `params["edges"][source][target][arg]`."""
    model = _mortal_model()
    assert _leaf_paths(model.get_params_template()["edges"]) == {
        ("working", "working", "survival_probability"),
        ("working", "dead", "survival_probability"),
    }


def test_source_regime_template_holds_no_law_entry():
    """The source regime's branch names neither a target nor the law."""
    model = _mortal_model()
    assert not {"working", "dead", "next_regime"} & set(
        model.get_params_template()["working"]
    )


def test_source_without_a_law_has_no_edges_entry():
    """A source declared by its edges alone owns no parameter slot."""
    model = _retirement_model()
    assert "retired" not in model.get_params_template()["edges"]


@pytest.mark.parametrize("inclusive_stop", [62, 63, 64])
def test_edges_template_does_not_depend_on_the_horizon(inclusive_stop):
    """Every declared case of a schedule owns its slots, whatever the horizon.

    The early case covers the source ages before 62, the last source age exits
    into `dead`, and the default case covers every age in between, of which
    there is one only when the grid runs past 63.
    """
    last_source_age = inclusive_stop - 1
    model = _mortal_model(
        ages=AgeGrid(start=60, inclusive_stop=inclusive_stop, step="Y"),
        law=ByAge(
            cases={
                AgeRange(start=60, exclusive_stop=min(62, last_source_age)): _EARLY_LAW,
                last_source_age: "dead",
            },
            default=_LATE_LAW,
        ),
    )
    assert _leaf_paths(model.get_params_template()["edges"]) == {
        ("working", "working", "early_survival"),
        ("working", "dead", "early_survival"),
        ("working", "working", "late_survival"),
        ("working", "dead", "late_survival"),
    }


@pytest.mark.parametrize("n_periods", [2, 3, 4])
def test_edges_template_does_not_depend_on_how_many_edges_the_horizon_leaves(
    n_periods,
):
    """A law keeps its slot when the horizon leaves its source one destination.

    With two periods, retirement is the working regime's only destination; with
    more, staying at work is a second one before the last source age.
    """
    model = _horizon_retirement_model(n_periods=n_periods)
    assert _leaf_paths(model.get_params_template()["edges"]) == {
        ("working", "retirement_age")
    }


@pytest.mark.parametrize("death_probability", [0.0, 0.5])
def test_edges_template_does_not_depend_on_a_fixed_zero_cell(death_probability):
    """A fixed value that prunes a cell leaves the other slots unchanged."""
    model = _mortal_model(
        law={
            "working": StochasticTransition(func=_stay),
            "dead": StochasticTransition(func=_die_at),
        },
        fixed_params={
            "edges": {"working": {"dead": {"death_probability": death_probability}}}
        },
    )
    assert _leaf_paths(model.get_params_template()["edges"]) == {
        ("working", "working", "stay_probability"),
    }


@pytest.mark.parametrize("variable", ["x", "a"])
@pytest.mark.parametrize("n_periods", [2, 3, 4])
def test_variable_read_by_a_dormant_selector_is_no_edge_parameter(
    *, variable, n_periods
):
    """A state or action a declared selector reads is never an edge parameter.

    With two periods the selector's case selects no age and the model drops the
    variable from the regime; with more periods the selector runs and reads it.
    The selector's one free parameter is the only edge slot at every horizon.
    """
    model = _dormant_selector_model(variable=variable, n_periods=n_periods)
    assert _leaf_paths(model.get_params_template()["edges"]) == {("alive", "cutoff")}


@pytest.mark.parametrize("variable", ["x", "a"])
def test_short_horizon_drops_the_variable_only_a_dormant_selector_reads(variable):
    """The variable a dormant selector reads does not stay on the regime's grid."""
    model = _dormant_selector_model(variable=variable, n_periods=2)
    assert variable in model.pruned_variables["alive"]


@pytest.mark.parametrize("variable", ["x", "a"])
@pytest.mark.parametrize("n_periods", [2, 3])
def test_selector_parameter_alone_fills_the_edge_slots_at_every_horizon(
    *, variable, n_periods
):
    """Supplying the selector's free parameter is all its edges require."""
    model = _dormant_selector_model(variable=variable, n_periods=n_periods)
    flat_params = model._process_params(
        {"discount_factor": 0.95, "edges": {"alive": {"cutoff": 0.5}}}
    )
    assert float(
        cast("Array", edge_params(flat_params, source="alive")["cutoff"])
    ) == pytest.approx(0.5)


@pytest.mark.parametrize("function_level", ["regime", "model"])
@pytest.mark.parametrize("n_periods", [2, 3])
def test_state_read_through_a_dormant_selector_function_is_no_parameter(
    *, function_level, n_periods
):
    """A state a selector reads through a function stays a state at every horizon.

    The function's own parameter is its only slot in every regime that declares
    it, a model-level one reaching `dead` as well, and the selector's free
    parameter is the only edge slot, whether or not the selector runs.
    """
    model = _dormant_scored_selector_model(
        function_level=function_level, n_periods=n_periods
    )
    template = model.get_params_template()
    assert (
        {name: dict(branch.get("score", {})) for name, branch in template.items()},
        _leaf_paths(template["edges"]),
    ) == (
        {
            "alive": {"scale": "float"},
            "dead": {"scale": "float"} if function_level == "model" else {},
            "edges": {},
        },
        {("alive", "cutoff")},
    )


@pytest.mark.parametrize("function_level", ["regime", "model"])
@pytest.mark.parametrize(
    ("n_periods", "expected"),
    [
        # The only source age exits into `dead`: 1 + 0.95 * 1 at both nodes.
        pytest.param(2, [1.95, 1.95], id="dormant-selector"),
        # Below the cutoff the household stays alive for a last age that exits
        # into `dead` (1 + 0.95 * 1.95); above it, it dies at once.
        pytest.param(3, [2.8525, 1.95], id="running-selector"),
    ],
)
def test_selector_reading_a_function_solves_with_only_declared_parameters(
    *, function_level, n_periods, expected
):
    """The function's parameter and the selector's cutoff are all a solve needs."""
    model = _dormant_scored_selector_model(
        function_level=function_level, n_periods=n_periods
    )
    values = model.solve(
        params={
            "discount_factor": 0.95,
            **(
                {"alive": {"score": {"scale": 1.0}}}
                if function_level == "regime"
                else {"scale": 1.0}
            ),
            "edges": {"alive": {"cutoff": 0.5}},
        },
        log_level="debug",
    ).values
    np.testing.assert_allclose(
        np.broadcast_to(np.asarray(values[0]["alive"]), (2,)), expected, rtol=1e-6
    )


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(
            {
                "edges": {
                    "working": {
                        "working": {"survival_probability": 0.9},
                        "dead": {"survival_probability": 0.9},
                    }
                }
            },
            id="declaration-path",
        ),
        pytest.param(
            {"edges": {"working": {"survival_probability": 0.9}}},
            id="edges-source-level",
        ),
        pytest.param({"survival_probability": 0.9}, id="model-level"),
    ],
)
def test_each_resolution_level_fills_the_engine_slot(params):
    """The value reaches the engine under the joined declaration path."""
    model = _mortal_model()
    flat_params = model._process_params({"discount_factor": 0.95, **params})
    assert float(
        cast(
            "Array",
            edge_params(flat_params, source="working")["dead__survival_probability"],
        )
    ) == (pytest.approx(0.9))


def test_engine_edge_namespace_holds_exactly_the_template_slots():
    """`flat_params["edges"][source]` keys are the source's slots, joined."""
    model = _mortal_model()
    flat_params = model._process_params(
        {"discount_factor": 0.95, "survival_probability": 0.9}
    )
    assert set(edge_params(flat_params, source="working")) == {
        "working__survival_probability",
        "dead__survival_probability",
    }


def test_source_regime_flat_params_hold_no_law_key():
    """The source regime's flat params carry its own functions' parameters only."""
    model = _mortal_model()
    flat_params = model._process_params(
        {"discount_factor": 0.95, "survival_probability": 0.9}
    )
    assert not [key for key in flat_params["working"] if "survival_probability" in key]


def test_value_at_two_levels_is_ambiguous():
    """A slot filled at the source level and the model level is refused."""
    model = _mortal_model()
    with pytest.raises(InvalidNameError, match="Ambiguous"):
        model._process_params(
            {
                "discount_factor": 0.95,
                "survival_probability": 0.9,
                "edges": {"working": {"survival_probability": 0.9}},
            }
        )


def test_edges_level_without_a_source_is_unknown():
    """`params["edges"][arg]` is not a resolution level."""
    model = _mortal_model()
    with pytest.raises(InvalidParamsError, match="edges__survival_probability"):
        model._process_params(
            {"discount_factor": 0.95, "edges": {"survival_probability": 0.9}}
        )


@pytest.mark.parametrize(
    "regime_params",
    [
        pytest.param({"survival_probability": 0.9}, id="source-regime-level"),
        pytest.param(
            {"dead": {"next_regime": {"survival_probability": 0.9}}},
            id="per-target-next-regime",
        ),
        pytest.param(
            {"next_regime": {"survival_probability": 0.9}}, id="coarse-next-regime"
        ),
    ],
)
def test_law_parameter_under_the_source_regime_names_the_edges_path(regime_params):
    """A law parameter written under its source regime is refused with its path."""
    model = _mortal_model()
    with pytest.raises(
        InvalidParamsError,
        match=r"params\['edges'\]\['working'\]\['dead'\]\['survival_probability'\]",
    ):
        model._process_params({"discount_factor": 0.95, "working": regime_params})


def test_fixed_params_follow_the_declaration_path():
    """A law parameter fixed at its path leaves no slot to supply at solve time."""
    model = _mortal_model(
        fixed_params={"edges": {"working": {"survival_probability": 0.9}}}
    )
    assert "working" not in model.get_params_template().get("edges", {})


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(
            {
                "edges": {
                    "working": {
                        "working": {"survival_probability": 0.9},
                        "dead": {"survival_probability": 0.9},
                    }
                }
            },
            id="declaration-path",
        ),
        pytest.param(
            {"edges": {"working": {"survival_probability": 0.9}}},
            id="edges-source-level",
        ),
    ],
)
def test_solution_does_not_depend_on_the_level_supplying_a_law_value(params):
    """The same value reaches the same law from every level, so values agree exactly."""
    model = _mortal_model()
    reference = model.solve(
        params={"discount_factor": 0.95, "survival_probability": 0.9},
        log_level="debug",
    ).values
    got = model.solve(params={"discount_factor": 0.95, **params}, log_level="debug")
    np.testing.assert_array_equal(
        np.asarray(got.values[0]["working"]), np.asarray(reference[0]["working"])
    )


@pytest.mark.parametrize(
    ("regime_name", "functions"),
    [
        pytest.param("edges", {}, id="regime"),
        pytest.param("working", {"edges": lambda wealth: wealth}, id="function"),
    ],
)
def test_edges_is_reserved_as_a_regime_and_function_name(*, regime_name, functions):
    """`edges` names the edge namespace, so no regime or function may take it."""
    with pytest.raises(InvalidNameError, match="edges"):
        _named_model(regime_name=regime_name, functions=functions)


def test_edges_is_reserved_as_an_argument_name():
    """A function argument named `edges` would be a parameter of that name."""
    with pytest.raises(InvalidNameError, match="edges"):
        _named_model(
            regime_name="working",
            functions={"bonus": _bonus_from_edges},
        )


def test_law_argument_named_like_a_target_is_rejected():
    """A law argument may not share its name with a regime."""
    with pytest.raises(InvalidNameError, match="dead"):
        _mortal_model(
            law={
                "working": StochasticTransition(func=_stay_reading_dead),
                "dead": StochasticTransition(func=_die),
            }
        )


def test_kernel_params_refuse_a_key_both_the_regime_and_its_edges_hold():
    """A source's kernels cannot bind one key from two namespaces."""
    flat_params = MappingProxyType(
        {
            "working": MappingProxyType({"dead__rate": 1.0}),
            "edges": MappingProxyType(
                {"working": MappingProxyType({"dead__rate": 2.0})}
            ),
        }
    )
    with pytest.raises(InvalidNameError, match="'dead__rate'"):
        regime_kernel_params(flat_params, regime_name="working")


def test_edge_callables_refuse_a_value_that_is_no_law_form():
    """A declared law cell that is neither a law nor a callable is named by type."""
    with pytest.raises(TypeError, match="'float'"):
        list(iter_edge_callables(law={"dead": 0.5}, path=()))


def test_invalid_law_argument_name_is_reported_at_its_edges_path():
    """An argument name the path grammar refuses is reported where its slot lives."""
    with pytest.raises(
        InvalidNameError, match=r"params\['edges'\]\['working'\]\['dead'\]"
    ):
        _mortal_model(
            law={
                "working": StochasticTransition(func=_survive),
                "dead": StochasticTransition(func=_die_reading_a_nested_name),
            }
        )


def test_parametrized_coarse_case_beside_per_target_cases_is_rejected():
    """A law over all targets with parameters cannot share a source with cells.

    Its parameters would sit at `params["edges"][source][arg]`, the level that
    also broadcasts into every per-target slot, so the model refuses the mix and
    asks for one form.
    """
    with pytest.raises(ModelInitializationError, match="per-target mapping"):
        _mortal_model(
            law=ByAge(
                cases={
                    AgeRange(start=60, exclusive_stop=62): DeterministicTransition(
                        func=_stay_until
                    ),
                    62: "dead",
                },
                default=_LATE_LAW,
            )
        )


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


def _utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _bequest(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def _next_wealth(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption


def _feasible(*, wealth: ContinuousState, consumption: ContinuousAction) -> BoolND:
    return consumption < wealth


def _bonus_from_edges(*, wealth: ContinuousState, edges: float) -> FloatND:
    return wealth + edges


def _alive() -> Regime:
    return Regime(
        states={"wealth": _WEALTH},
        actions={"consumption": _CONSUMPTION},
        state_transitions={"wealth": _next_wealth},
        functions={"utility": _utility},
        constraints={"feasible": _feasible},
    )


def _dead() -> Regime:
    return Regime(states={"wealth": _WEALTH}, functions={"utility": _bequest})


def _survive(survival_probability: float) -> FloatND:
    return jnp.asarray(survival_probability)


def _die(survival_probability: float) -> FloatND:
    return 1.0 - jnp.asarray(survival_probability)


def _stay(stay_probability: float) -> FloatND:
    return jnp.asarray(stay_probability)


def _die_at(death_probability: float) -> FloatND:
    return jnp.asarray(death_probability)


def _stay_reading_dead(dead: float) -> FloatND:
    return 1.0 - jnp.asarray(dead)


def _die_reading_a_nested_name(survival__probability: float) -> FloatND:
    return 1.0 - jnp.asarray(survival__probability)


def _stay_until(*, age: float, last_working_age: float) -> ScalarInt:
    return jnp.where(age < last_working_age, _MortalId.working, _MortalId.dead)


def _survive_early(early_survival: float) -> FloatND:
    return jnp.asarray(early_survival)


def _die_early(early_survival: float) -> FloatND:
    return 1.0 - jnp.asarray(early_survival)


def _survive_late(late_survival: float) -> FloatND:
    return jnp.asarray(late_survival)


def _die_late(late_survival: float) -> FloatND:
    return 1.0 - jnp.asarray(late_survival)


_EARLY_LAW = {
    "working": StochasticTransition(func=_survive_early),
    "dead": StochasticTransition(func=_die_early),
}
_LATE_LAW = {
    "working": StochasticTransition(func=_survive_late),
    "dead": StochasticTransition(func=_die_late),
}


def _mortal_model(
    *,
    ages: AgeGrid = _AGES,
    law: object = None,
    fixed_params: UserParams | None = None,
) -> Model:
    """Working regime that survives a year at a time, then dies at the last age.

    Working is a destination at every source age but the last, where `dead` is
    the only one. A per-target `law` applies before the last source age, which
    exits into `dead`; a `ByAge` law states its own last-age case.
    """
    last_age = ages.exact_values[-1]
    last_source_age = ages.exact_values[-2]
    per_target_law = (
        {
            "working": StochasticTransition(func=_survive),
            "dead": StochasticTransition(func=_die),
        }
        if law is None
        else law
    )
    return Model(
        regimes={"working": _alive(), "dead": _dead()},
        ages=ages,
        regime_id_class=_MortalId,
        initial_nodes={60: "working"},
        fixed_params={} if fixed_params is None else fixed_params,
        edges={
            "working": Transition(
                targets={
                    "working": AgeRange(start=60, exclusive_stop=last_source_age),
                    "dead": AgeRange(start=60, exclusive_stop=last_age),
                },
                law=(
                    per_target_law
                    if isinstance(per_target_law, ByAge)
                    else ByAge.until(
                        stop_age_exclusive=last_age,
                        law=per_target_law,
                        then="dead",
                    )
                ),
            ),
        },
    )


def _retire_at(*, age: float, retirement_age: float) -> ScalarInt:
    return jnp.where(age < retirement_age, _RetirementId.working, _RetirementId.retired)


def _retirement_model() -> Model:
    """Working regime that retires at a parametrized age; retirement ends in death.

    Working retires at 61 at the latest, so retirement reaches `dead` at the last
    age.
    """
    return Model(
        regimes={"working": _alive(), "retired": _alive(), "dead": _dead()},
        ages=_AGES,
        regime_id_class=_RetirementId,
        initial_nodes={60: "working"},
        edges={
            "working": Transition(
                targets={"working": (60,), "retired": (60, 61)},
                law=ByAge.until(
                    stop_age_exclusive=62,
                    law=DeterministicTransition(func=_retire_at),
                    then="retired",
                ),
            ),
            "retired": {"retired": (60, 61), "dead": 62},
        },
    )


def _horizon_retirement_model(*, n_periods: int) -> Model:
    """Working regime that may stay at work until its last source age, then retires.

    The edges follow the age grid from 60 over `n_periods` years and retirement
    is terminal. The parametrized law chooses before the last source age, which
    retires; the declaration is the same at every horizon.
    """
    last_source_age = 58 + n_periods
    targets: dict[str, AgeRange] = {"retired": AgeRange(start=60)}
    if last_source_age > 60:
        targets["working"] = AgeRange(start=60, exclusive_stop=last_source_age)
    return Model(
        regimes={"working": _alive(), "retired": _dead()},
        ages=AgeGrid(start=60, inclusive_stop=last_source_age + 1, step="Y"),
        regime_id_class=_WorkRetireId,
        initial_nodes={60: "working"},
        edges={
            "working": Transition(
                targets=targets,
                law=ByAge.until(
                    stop_age_exclusive=last_source_age + 1,
                    law=DeterministicTransition(func=_retire_from_work_at),
                    then="retired",
                ),
            ),
        },
    )


def _retire_from_work_at(*, age: float, retirement_age: float) -> ScalarInt:
    return jnp.where(age < retirement_age, _WorkRetireId.working, _WorkRetireId.retired)


@categorical(ordered=False)
class _LifeId:
    alive: ScalarInt
    dead: ScalarInt


def _constant_utility() -> FloatND:
    return jnp.asarray(1.0)


def _choose_by_state(*, x: ContinuousState, cutoff: float) -> ScalarInt:
    return jnp.where(x < cutoff, _LifeId.alive, _LifeId.dead)


def _choose_by_action(*, a: ContinuousAction, cutoff: float) -> ScalarInt:
    return jnp.where(a < cutoff, _LifeId.alive, _LifeId.dead)


def _dormant_selector_model(*, variable: str, n_periods: int) -> Model:
    """An `alive` regime whose selector reads a model-level state or action.

    The selector runs at every source age but the last, which exits into `dead`.
    With two periods the only source age is the last, so the selector is dormant
    and the variable it reads has no other reader.
    """
    last_age = n_periods - 1
    last_source_age = last_age - 1
    targets: dict[str, AgeRange] = {"dead": AgeRange(start=0)}
    if last_source_age > 0:
        targets["alive"] = AgeRange(start=0, exclusive_stop=last_source_age)
    grid = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
    is_state = variable == "x"
    return Model(
        ages=AgeGrid(start=0, inclusive_stop=last_age, step="Y"),
        regimes={
            "alive": Regime(functions={"utility": _constant_utility}),
            "dead": Regime(functions={"utility": _constant_utility}),
        },
        regime_id_class=_LifeId,
        initial_nodes={0: "alive"},
        edges={
            "alive": Transition(
                targets=targets,
                law=ByAge.until(
                    stop_age_exclusive=last_age,
                    law=DeterministicTransition(
                        func=_choose_by_state if is_state else _choose_by_action
                    ),
                    then="dead",
                ),
            ),
        },
        states={"x": grid} if is_state else {},
        state_transitions={"x": fixed_transition("x")} if is_state else {},
        actions={} if is_state else {"a": grid},
    )


def _score(*, x: ContinuousState, scale: float) -> FloatND:
    return scale * x


def _choose_by_score(*, score: FloatND, cutoff: float) -> ScalarInt:
    return jnp.where(score < cutoff, _LifeId.alive, _LifeId.dead)


def _dormant_scored_selector_model(*, function_level: str, n_periods: int) -> Model:
    """An `alive` regime whose selector reads a model-level state via a function.

    `score` is declared on the regime or at the model level and is read only by
    the selector. The selector runs at every source age but the last, which exits
    into `dead`, so with two periods nothing reads `score` or the state `x`.
    """
    last_age = n_periods - 1
    last_source_age = last_age - 1
    targets: dict[str, AgeRange] = {"dead": AgeRange(start=0)}
    if last_source_age > 0:
        targets["alive"] = AgeRange(start=0, exclusive_stop=last_source_age)
    on_regime = function_level == "regime"
    return Model(
        ages=AgeGrid(start=0, inclusive_stop=last_age, step="Y"),
        regimes={
            "alive": Regime(
                functions={
                    "utility": _constant_utility,
                    **({"score": _score} if on_regime else {}),
                }
            ),
            "dead": Regime(functions={"utility": _constant_utility}),
        },
        regime_id_class=_LifeId,
        initial_nodes={0: "alive"},
        edges={
            "alive": Transition(
                targets=targets,
                law=ByAge.until(
                    stop_age_exclusive=last_age,
                    law=DeterministicTransition(func=_choose_by_score),
                    then="dead",
                ),
            ),
        },
        states={"x": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={"x": fixed_transition("x")},
        functions={} if on_regime else {"score": _score},
    )


@categorical(ordered=False)
class _EdgesNamedId:
    edges: ScalarInt
    dead: ScalarInt


def _named_model(*, regime_name: str, functions: Mapping[str, UserFunction]) -> Model:
    """A source regime with the given name and functions, then a terminal one."""
    source = Regime(
        states={"wealth": _WEALTH},
        actions={"consumption": _CONSUMPTION},
        state_transitions={"wealth": _next_wealth},
        functions={"utility": _utility, **functions},
        constraints={"feasible": _feasible},
    )
    return Model(
        regimes={regime_name: source, "dead": _dead()},
        ages=_AGES,
        regime_id_class=_EdgesNamedId if regime_name == "edges" else _MortalId,
        initial_nodes={60: regime_name},
        edges={
            regime_name: {
                regime_name: AgeRange(start=60, exclusive_stop=62),
                "dead": 62,
            }
        },
    )
