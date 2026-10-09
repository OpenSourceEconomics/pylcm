"""Explicit model topology separates admissible edges from transition kernels."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.utils.logging import LogLevel
from lcm import (
    AgeGrid,
    ByAge,
    DeterministicTransition,
    Gate,
    LinSpacedGrid,
    Model,
    Phased,
    ProjectedRegimeValue,
    StakeholderRoute,
    StochasticTransition,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import (
    InvalidRegimeTransitionProbabilitiesError,
    ModelInitializationError,
)
from lcm.regime import Regime
from lcm.typing import BoolND, ContinuousState, FloatND, ScalarInt
from tests.test_models import n_nbegm_toy


@categorical(ordered=False)
class _GraphRegimeId:
    work: ScalarInt
    perceived: ScalarInt
    realized: ScalarInt


def _graph_regimes() -> dict[str, Regime]:
    grid = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
    return {
        "work": Regime(
            states={"wealth": grid},
            actions={"investment": grid},
            state_transitions={"wealth": lambda investment: investment},
            functions={"utility": lambda wealth: 0.0 * wealth},
        ),
        "perceived": Regime(
            states={"wealth": grid},
            functions={"utility": lambda wealth: wealth},
        ),
        "realized": Regime(
            states={"wealth": grid},
            functions={"utility": lambda wealth: -wealth},
        ),
    }


def test_model_requires_explicit_edge_topology() -> None:
    """Require a model graph even when kernels can choose a destination."""
    with pytest.raises(TypeError, match="edges"):
        Model(  # ty: ignore[missing-argument] -- intentional missing topology
            regimes=_graph_regimes(),
            ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
            regime_id_class=_GraphRegimeId,
            initial_nodes=((0, "work"),),
        )


def test_graph_edges_price_perceived_choice_and_realize_other_destination() -> None:
    """Price an investment through solve edges and realize the simulation edge."""
    model = Model(
        regimes=_graph_regimes(),
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_GraphRegimeId,
        initial_nodes=((0, "work"),),
        edges=Phased(
            solve={"work": {"perceived": 0}},
            simulate={"work": {"realized": 0}},
        ),
        enable_jit=False,
    )
    assert model.initial_nodes == frozenset({(0, "work")})
    assert model.reachability.solution.targets(period=0, source="work") == (
        "perceived",
    )
    assert model.reachability.simulation.targets(period=0, source="work") == (
        "realized",
    )
    solution = model.solve(params={"discount_factor": 1.0}, log_level="debug")
    np.testing.assert_array_equal(solution.values[0]["work"], jnp.ones(2))
    frame = model.simulate(
        params={"discount_factor": 1.0},
        solution=solution,
        initial_conditions={
            "age": jnp.array([0]),
            "wealth": jnp.array([0.0]),
            "regime_id": jnp.array([_GraphRegimeId.work]),
        },
        log_level="debug",
        seed=7,
    ).to_dataframe()
    assert frame["regime_name"].tolist() == ["work", "realized"]
    assert frame.loc[frame["period"] == 0, "investment"].tolist() == [1.0]
    assert frame.loc[frame["period"] == 1, "wealth"].tolist() == [1.0]


@pytest.mark.parametrize(("source", "target"), [("work", "typo"), ("typo", "realized")])
def test_graph_unknown_names_are_rejected_even_at_final_source_age(
    *,
    source: str,
    target: str,
) -> None:
    """Validate topology names before discarding undemanded final-age edges."""
    with pytest.raises(ModelInitializationError, match="typo"):
        Model(
            regimes=_graph_regimes(),
            ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
            regime_id_class=_GraphRegimeId,
            initial_nodes=((0, "work"),),
            edges=Phased(
                solve={"work": {"perceived": 0}, source: {target: 1}},
                simulate={"work": {"realized": 0}},
            ),
            enable_jit=False,
        )


@pytest.mark.parametrize("wrapper", [DeterministicTransition, StochasticTransition])
def test_transition_kernel_rejects_embedded_topology(
    wrapper: type[DeterministicTransition | StochasticTransition],
) -> None:
    """A kernel cannot declare a second source of model edges."""
    with pytest.raises(TypeError, match="targets"):
        wrapper(
            func=lambda: _GraphRegimeId.realized,
            targets=("realized",),  # ty: ignore[unknown-argument] -- obsolete topology
        )


def test_graph_snapshot_owns_immutable_source_age_selectors() -> None:
    """Publish source ages without retaining mutable caller topology containers."""
    edges: dict[str, dict[str, tuple[int, ...]]] = {"work": {"perceived": (0, 1)}}
    model = Model(
        regimes=_graph_regimes(),
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_GraphRegimeId,
        initial_nodes=((1, "work"),),
        edges=edges,
        enable_jit=False,
    )
    edges["work"]["perceived"] = (0,)
    assert model.graph.edges.solve["work"]["perceived"] == frozenset({0, 1})
    assert model.graph.edges.simulate["work"]["perceived"] == frozenset({0, 1})
    assert model.graph.initial_nodes == frozenset({(1, "work")})
    assert model.graph.solution.targets(period=1, source="work") == ("perceived",)
    assert (2, "perceived") in model.graph.nodes
    with pytest.raises(TypeError):
        model.graph.edges.solve["work"]["perceived"] = frozenset({0})  # ty: ignore[invalid-assignment] -- intentional mutation
    with pytest.raises(AttributeError):
        model.graph.edges.solve["work"]["perceived"].add(2)  # ty: ignore[unresolved-attribute] -- intentional mutation


def test_fixed_zero_graph_edge_keeps_declared_topology_and_pruning_reason() -> None:
    """Distinguish declared support from a fixed-zero effective probability edge."""
    model = Model(
        regimes=_graph_regimes(),
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_GraphRegimeId,
        initial_nodes=((0, "work"),),
        edges={
            "work": Transition(
                targets={"perceived": 0, "realized": 0},
                law={
                    "perceived": StochasticTransition(func=lambda mass: mass),
                    "realized": StochasticTransition(func=lambda mass: 1.0 - mass),
                },
            )
        },
        fixed_params={"mass": 0.0},
        enable_jit=False,
    )
    assert model.graph.edges.solve["work"]["perceived"] == frozenset({0})
    assert model.graph.solution.targets(period=0, source="work") == ("realized",)
    assert model.graph.simulation.targets(period=0, source="work") == ("realized",)
    for phase in ("solve", "simulate"):
        assert model.graph.pruned_edges[phase][(0, "work", "perceived")] == (
            "fixed_zero_probability"
        )
        with pytest.raises(TypeError):
            model.graph.pruned_edges[phase][(0, "work", "perceived")] = "other"  # ty: ignore[invalid-assignment] -- intentional mutation


def test_scalar_probability_mapping_cannot_extend_declared_graph() -> None:
    """Reject a per-target law naming a destination its `targets` leave out."""
    with pytest.raises(
        ModelInitializationError,
        match=r"supplied: \{'perceived': \[0\], 'work': \[0\]\}; derived from the "
        r"law and its gates: \{'perceived': \[0\], 'realized': \[0\]\}",
    ):
        Model(
            regimes=_graph_regimes(),
            ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
            regime_id_class=_GraphRegimeId,
            initial_nodes=((0, "work"),),
            edges={
                "work": Transition(
                    targets={"perceived": 0, "work": 0},
                    law={
                        "perceived": StochasticTransition(
                            func=lambda: jnp.asarray(1.0)
                        ),
                        "realized": StochasticTransition(func=lambda: jnp.asarray(0.0)),
                    },
                )
            },
            enable_jit=False,
        )


def test_scalar_probability_mapping_naming_fewer_targets_than_supplied_is_refused():
    """Reject supplied `targets` that a per-target law does not name.

    A per-target law names its destinations, so a supplied destination without a
    cell is a second, conflicting declaration of the graph.
    """
    with pytest.raises(
        ModelInitializationError,
        match=r"supplied: \{'perceived': \[0\], 'realized': \[0\]\}; derived from "
        r"the law and its gates: \{'perceived': \[0\]\}",
    ):
        Model(
            regimes=_graph_regimes(),
            ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
            regime_id_class=_GraphRegimeId,
            initial_nodes=((0, "work"),),
            edges={
                "work": Transition(
                    targets={"perceived": 0, "realized": 0},
                    law={
                        "perceived": StochasticTransition(func=lambda: jnp.asarray(1.0))
                    },
                )
            },
            enable_jit=False,
        )


@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("log_level", ["off", "debug"])
def test_kernel_cannot_select_destination_outside_current_source_age_edges(
    *,
    stochastic: bool,
    log_level: LogLevel,
) -> None:
    """Reject a destination allowed at another source age, at every log level."""
    model = Model(
        regimes=_graph_regimes(),
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_GraphRegimeId,
        initial_nodes=((1, "work"),),
        edges={
            "work": Transition(
                targets={"perceived": 0, "realized": (0, 1)},
                law=(
                    StochasticTransition(func=lambda: jnp.array([0.0, 1.0, 0.0]))
                    if stochastic
                    else lambda: _GraphRegimeId.perceived
                ),
            )
        },
        enable_jit=False,
    )
    assert model.graph.solution.targets(period=1, source="work") == ("realized",)
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError):
        model.solve(params={"discount_factor": 1.0}, log_level=log_level)


def test_value_only_node_needs_only_its_perceived_graph_continuation() -> None:
    """Value a belief node without adding its continuation to physical visits."""
    regimes = _graph_regimes()
    regimes["perceived"] = regimes["perceived"].replace(
        state_transitions={"wealth": lambda wealth: wealth},
        functions={"utility": lambda wealth: 0.0 * wealth},
    )
    regimes["realized"] = regimes["realized"].replace(
        functions={"utility": lambda wealth: wealth},
    )
    model = Model(
        regimes=regimes,
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_GraphRegimeId,
        initial_nodes=((0, "work"),),
        edges=Phased(
            solve={"work": {"perceived": 0}, "perceived": {"realized": 1}},
            simulate={"work": {"realized": 0}},
        ),
        enable_jit=False,
    )
    assert model.graph.nodes == frozenset(
        {(0, "work"), (1, "perceived"), (1, "realized"), (2, "realized")}
    )
    assert model.graph.visited_nodes == frozenset({(0, "work"), (1, "realized")})
    assert model.graph.simulation.targets(period=1, source="perceived") == ()
    solution = model.solve(params={"discount_factor": 1.0}, log_level="debug")
    np.testing.assert_array_equal(solution.values[0]["work"], jnp.ones(2))


def test_explicit_initial_pair_requires_one_exact_age() -> None:
    """An explicit node pair cannot hide an age selector's Cartesian expansion."""
    with pytest.raises(ModelInitializationError, match="age"):
        Model(
            regimes=_graph_regimes(),
            ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
            regime_id_class=_GraphRegimeId,
            initial_nodes=(((0, 1), "perceived"),),
            edges={},
            enable_jit=False,
        )


def test_broadcast_graph_preserves_nnbegm_phase_invariance() -> None:
    """Broadcast topology preserves identical numerical declarations across phases."""
    model = n_nbegm_toy.build_model(variant="n_nbegm", n_periods=3)
    assert model.graph.edges.solve == model.graph.edges.simulate
    assert model.graph.solution.targets(period=0, source="alive") == ("alive", "dead")


def test_graph_destination_order_preserves_nnbegm_phase_invariance() -> None:
    """Destination insertion order cannot create numerical phase variation."""
    template = n_nbegm_toy.build_model(variant="n_nbegm", n_periods=3)
    solve_targets = {
        source: {target: tuple(sorted(ages)) for target, ages in destinations.items()}
        for source, destinations in template.graph.edges.solve.items()
    }
    solve_edges = {
        source: Transition(targets=targets, law=n_nbegm_toy.next_regime)
        for source, targets in solve_targets.items()
    }
    simulate_edges = {
        source: Transition(
            targets=dict(reversed(tuple(targets.items()))),
            law=n_nbegm_toy.next_regime,
        )
        for source, targets in solve_targets.items()
    }
    model = Model(
        regimes=template.user_regimes,
        ages=template.ages,
        regime_id_class=n_nbegm_toy.RegimeId,
        fixed_params=template.fixed_params,
        initial_nodes=tuple(template.initial_nodes),
        edges=Phased(solve=solve_edges, simulate=simulate_edges),
    )
    assert model.graph.edges.solve == model.graph.edges.simulate
    assert model.graph.solution.targets(period=0, source="alive") == ("alive", "dead")


@categorical(ordered=False)
class _DormantGateRegimeId:
    source: ScalarInt
    other: ScalarInt
    target: ScalarInt
    reference: ScalarInt
    fallback: ScalarInt
    terminal: ScalarInt


def test_value_only_source_age_does_not_activate_dormant_simulation_gate() -> None:
    """A gate used at one source age owes no references at a value-only age."""
    grid = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)

    def identity(x: ContinuousState) -> ContinuousState:
        return x

    def gate(*, V_reference: FloatND) -> BoolND:
        return V_reference >= 0.0

    one = StochasticTransition(func=lambda: jnp.asarray(1.0))
    zero = StochasticTransition(func=lambda: jnp.asarray(0.0))
    gates = {
        "target": Gate(
            predicate=gate,
            routes={
                "only": StakeholderRoute(
                    fallback=ProjectedRegimeValue(
                        regime="fallback", projection={"x": identity}
                    )
                )
            },
            references={
                "V_reference": ProjectedRegimeValue(
                    regime="reference", projection={"x": identity}
                )
            },
        )
    }
    solve_law = ByAge(cases={0: {"source": one, "target": zero}, 1: {"terminal": one}})
    simulate_law = ByAge(cases={0: {"target": one}})
    source = Regime(
        states={"x": grid},
        state_transitions={"x": fixed_transition("x")},
        functions={"utility": lambda x: 0.0 * x},
    )
    terminal = Regime(
        states={"x": grid},
        functions={"utility": identity},
    )
    model = Model(
        regimes={
            "source": source,
            "other": Regime(
                states={"x": grid},
                state_transitions={"x": fixed_transition("x")},
                functions={"utility": lambda x: 0.0 * x},
            ),
            "target": terminal,
            "reference": Regime(
                states={"x": grid},
                state_transitions={"x": fixed_transition("x")},
                functions={"utility": identity},
            ),
            "fallback": terminal,
            "terminal": terminal,
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_DormantGateRegimeId,
        initial_nodes=((0, "source"), (1, "other")),
        edges=Phased(
            solve={
                "source": Transition(
                    targets={"source": 0, "target": 0, "fallback": 0, "terminal": 1},
                    law=solve_law,
                    gates=gates,
                ),
                "other": {"target": 1},
                "reference": {"terminal": 1},
            },
            simulate={
                "source": Transition(
                    targets={"target": 0, "fallback": 0},
                    law=simulate_law,
                    gates=gates,
                ),
                "other": {"terminal": 1},
            },
        ),
        enable_jit=False,
    )
    assert (1, "source") in model.graph.nodes
    assert (1, "source") not in model.graph.visited_nodes
    assert (1, "reference") in model.graph.nodes
    assert (2, "reference") not in model.graph.nodes
    solution = model.solve(params={"discount_factor": 1.0}, log_level="debug")
    assert set(solution.values[2]) == {"target", "terminal"}


def test_required_age_without_an_edge_out_names_the_missing_edge() -> None:
    """A required regime at an age with no edge out names that missing edge."""
    with pytest.raises(
        ModelInitializationError,
        match=r"requires 'work' at age 1, where `edges` declares no edge out of "
        r"'work' at that age",
    ):
        Model(
            regimes=_graph_regimes(),
            ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
            regime_id_class=_GraphRegimeId,
            initial_nodes=((0, "work"),),
            edges={"work": {"work": 0}},
            enable_jit=False,
        )


def test_edge_whose_only_source_age_is_the_final_age_is_rejected() -> None:
    """No transition leaves the final age, so an edge selecting only it is invalid."""
    with pytest.raises(
        ModelInitializationError,
        match=r"'work' → 'realized' selects only the final age 1",
    ):
        Model(
            regimes=_graph_regimes(),
            ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
            regime_id_class=_GraphRegimeId,
            initial_nodes=((0, "work"),),
            edges={"work": {"perceived": 0, "realized": 1}},
            enable_jit=False,
        )


def _phased_graph_model(initial_nodes: object) -> Model:
    return Model(
        regimes=_graph_regimes(),
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_GraphRegimeId,
        initial_nodes=initial_nodes,  # ty: ignore[invalid-argument-type]
        edges=Phased(
            solve={"work": {"perceived": 0}},
            simulate={"work": {"realized": 0}},
        ),
        enable_jit=False,
    )


def test_published_initial_nodes_are_accepted_by_the_constructor() -> None:
    """`Model.initial_nodes` passed back to `Model` selects the same start pairs."""
    model = _phased_graph_model({0: "work"})
    rebuilt = _phased_graph_model(model.initial_nodes)
    assert rebuilt.initial_nodes == frozenset({(0, "work")})


def _declaration_model(*, edges: object, enable_jit: bool) -> Model:
    return Model(
        regimes={
            "work": Regime(functions={"utility": lambda: jnp.asarray(1.0)}),
            "perceived": Regime(functions={"utility": lambda: jnp.asarray(10.0)}),
            "realized": Regime(functions={"utility": lambda: jnp.asarray(20.0)}),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_GraphRegimeId,
        initial_nodes=((0, "work"),),
        edges=edges,
        enable_jit=enable_jit,
    )


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("container", ["dict", "proxy", "phased"])
@pytest.mark.parametrize("mutation", ["outer", "inner"])
def test_declared_edges_snapshot_reconstructs_original_value(
    *, enable_jit: bool, container: str, mutation: str
) -> None:
    """Caller topology edits cannot change the published model configuration."""
    targets = {"perceived": 0}
    edges = {"work": targets}
    declaration = (
        MappingProxyType(edges)
        if container == "proxy"
        else Phased(solve=edges, simulate=edges)
        if container == "phased"
        else edges
    )
    model = _declaration_model(edges=declaration, enable_jit=enable_jit)
    if mutation == "outer":
        edges["work"] = {"realized": 0}
    else:
        targets.clear()
        targets["realized"] = 0
    twin = _declaration_model(edges=model.edges, enable_jit=enable_jit)
    np.testing.assert_array_equal(
        [
            specimen.solve(params={"discount_factor": 0.5}, log_level="debug").values[
                0
            ]["work"]
            for specimen in (model, twin)
        ],
        [6.0, 6.0],
    )


@pytest.mark.parametrize(
    "wrapper", ["plain", "proxy", "case", "default", "until", "phased"]
)
def test_transition_snapshots_nested_laws_at_construction(wrapper: str) -> None:
    """Nested probability mappings are owned before a model binds their laws."""
    low = StochasticTransition(func=lambda: jnp.asarray(0.25))
    high = StochasticTransition(func=lambda: jnp.asarray(0.75))
    probabilities = {"perceived": low, "realized": high}
    laws = {
        "plain": probabilities,
        "proxy": MappingProxyType(probabilities),
        "case": ByAge(cases={0: probabilities}),
        "default": ByAge(cases={}, default=probabilities),
        "until": ByAge.until(
            stop_age_exclusive=1, law=probabilities, then=probabilities
        ),
        "phased": Phased(solve=probabilities, simulate=probabilities),
    }
    transition = Transition(law=laws[wrapper])
    probabilities["perceived"] = StochasticTransition(func=lambda: jnp.asarray(1.0))
    probabilities["realized"] = StochasticTransition(func=lambda: jnp.asarray(0.0))
    model = _declaration_model(edges={"work": transition}, enable_jit=False)
    np.testing.assert_array_equal(
        model.solve(params={"discount_factor": 0.5}, log_level="debug").values[0][
            "work"
        ],
        9.75,
    )


def test_declared_probability_law_preserves_callable_identity() -> None:
    """Snapshotting containers leaves the user's callable unchanged."""
    probability = StochasticTransition(func=lambda: jnp.asarray(1.0))
    transition = Transition(law={"perceived": probability})
    assert (
        cast("Mapping[str, StochasticTransition]", transition.law)["perceived"].func
        is probability.func
    )


def test_declared_probability_law_is_read_only() -> None:
    """A published probability mapping cannot be edited in place."""
    transition = Transition(
        law={"perceived": StochasticTransition(func=lambda: jnp.asarray(1.0))}
    )
    with pytest.raises(TypeError):
        cast("dict[str, object]", transition.law)["perceived"] = "realized"


def test_declared_edge_mapping_is_read_only() -> None:
    """A model publishes immutable declarations as well as an immutable graph."""
    model = _declaration_model(edges={"work": {"perceived": 0}}, enable_jit=False)
    with pytest.raises(TypeError):
        cast("dict[str, dict[str, int]]", model.edges)["work"]["realized"] = 0
