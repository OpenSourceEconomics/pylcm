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
    LinSpacedGrid,
    Model,
    Phased,
    StochasticTransition,
    Transition,
    categorical,
)
from lcm.exceptions import (
    InvalidRegimeTransitionProbabilitiesError,
    ModelInitializationError,
)
from lcm.initial_nodes import UserInitialNodes
from lcm.regime import Regime
from lcm.transition import ModelEdges
from lcm.typing import FloatND, RegimeName, ScalarInt
from tests.test_models import n_nbegm_toy


@categorical(ordered=False)
class _GraphRegimeId:
    work: ScalarInt
    perceived: ScalarInt
    realized: ScalarInt


def _invest(investment: float) -> float:
    return investment


def _keep_wealth(wealth: float) -> float:
    return wealth


def _zero_wealth_utility(wealth: float) -> float:
    return 0.0 * wealth


def _wealth_utility(wealth: float) -> float:
    return wealth


def _negative_wealth_utility(wealth: float) -> float:
    return -wealth


def _realized_regime() -> ScalarInt:
    return _GraphRegimeId.realized


def _perceived_regime() -> ScalarInt:
    return _GraphRegimeId.perceived


def _mass(mass: float) -> FloatND:
    return jnp.asarray(mass)


def _complement_mass(mass: float) -> FloatND:
    return jnp.asarray(1.0 - mass)


def _probability_zero() -> FloatND:
    return jnp.asarray(0.0)


def _probability_quarter() -> FloatND:
    return jnp.asarray(0.25)


def _probability_three_quarters() -> FloatND:
    return jnp.asarray(0.75)


def _probability_one() -> FloatND:
    return jnp.asarray(1.0)


def _realized_lottery() -> FloatND:
    return jnp.array([0.0, 1.0, 0.0])


def _utility_one() -> FloatND:
    return jnp.asarray(1.0)


def _utility_ten() -> FloatND:
    return jnp.asarray(10.0)


def _utility_twenty() -> FloatND:
    return jnp.asarray(20.0)


def _graph_regimes() -> dict[str, Regime]:
    grid = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
    return {
        "work": Regime(
            states={"wealth": grid},
            actions={"investment": grid},
            state_transitions={"wealth": _invest},
            functions={"utility": _zero_wealth_utility},
        ),
        "perceived": Regime(
            states={"wealth": grid},
            functions={"utility": _wealth_utility},
        ),
        "realized": Regime(
            states={"wealth": grid},
            functions={"utility": _negative_wealth_utility},
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


@pytest.mark.parametrize(("source", "target"), [("work", "typo"), ("typo", "realized")])
def test_graph_unknown_names_are_rejected_even_at_final_source_age(
    *,
    source: RegimeName,
    target: RegimeName,
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
            func=_realized_regime,
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
    assert model.graph.solution.targets(step=1, source="work") == ("perceived",)
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
                    "perceived": StochasticTransition(func=_mass),
                    "realized": StochasticTransition(func=_complement_mass),
                },
            )
        },
        fixed_params={"mass": 0.0},
        enable_jit=False,
    )
    assert model.graph.edges.solve["work"]["perceived"] == frozenset({0})
    assert model.graph.solution.targets(step=0, source="work") == ("realized",)
    assert model.graph.simulation.targets(step=0, source="work") == ("realized",)
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
                        "perceived": StochasticTransition(func=_probability_one),
                        "realized": StochasticTransition(func=_probability_zero),
                    },
                )
            },
            enable_jit=False,
        )


def test_scalar_probability_mapping_naming_fewer_targets_than_supplied_is_refused() -> (
    None
):
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
                    law={"perceived": StochasticTransition(func=_probability_one)},
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
                    StochasticTransition(func=_realized_lottery)
                    if stochastic
                    else _perceived_regime
                ),
            )
        },
        enable_jit=False,
    )
    assert model.graph.solution.targets(step=1, source="work") == ("realized",)
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError):
        model.solve(params={"discount_factor": 1.0}, log_level=log_level)


def test_value_only_node_needs_only_its_perceived_graph_continuation() -> None:
    """Value a belief node without adding its continuation to physical visits."""
    regimes = _graph_regimes()
    regimes["perceived"] = regimes["perceived"].replace(
        state_transitions={"wealth": _keep_wealth},
        functions={"utility": _zero_wealth_utility},
    )
    regimes["realized"] = regimes["realized"].replace(
        functions={"utility": _wealth_utility},
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
    assert model.graph.simulation.targets(step=1, source="perceived") == ()
    solution = model.solve(params={"discount_factor": 1.0}, log_level="debug")
    np.testing.assert_array_equal(solution.values[0]["work"][..., 0], jnp.ones(2))


def test_explicit_initial_pair_requires_one_exact_age() -> None:
    """An explicit node pair cannot hide an age selector's Cartesian expansion."""
    with pytest.raises(ModelInitializationError, match="age"):
        Model(
            regimes=_graph_regimes(),
            ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
            regime_id_class=_GraphRegimeId,
            initial_nodes=(((0, 1), "perceived"),),  # ty: ignore[invalid-argument-type]
            edges={},
            enable_jit=False,
        )


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
        initial_nodes=template.initial_nodes,
        edges=Phased(solve=solve_edges, simulate=simulate_edges),
    )
    assert model.graph.edges.solve == model.graph.edges.simulate
    assert model.graph.solution.targets(step=0, source="alive") == ("alive", "dead")


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


def _phased_graph_model(initial_nodes: UserInitialNodes) -> Model:
    return Model(
        regimes=_graph_regimes(),
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_GraphRegimeId,
        initial_nodes=initial_nodes,
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
    assert rebuilt.graph.initial_nodes == frozenset({(0, "work")})


def _declaration_model(*, edges: ModelEdges, enable_jit: bool) -> Model:
    return Model(
        regimes={
            "work": Regime(functions={"utility": _utility_one}),
            "perceived": Regime(functions={"utility": _utility_ten}),
            "realized": Regime(functions={"utility": _utility_twenty}),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_GraphRegimeId,
        initial_nodes=((0, "work"),),
        edges=edges,
        enable_jit=enable_jit,
    )


@pytest.mark.parametrize("enable_jit", [True])
@pytest.mark.parametrize("container", ["phased"])
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
            ]["work"][..., 0]
            for specimen in (model, twin)
        ],
        [6.0, 6.0],
    )


@pytest.mark.parametrize("wrapper", ["plain", "mapped-until", "phased"])
def test_transition_snapshots_nested_laws_at_construction(wrapper: str) -> None:
    """Nested probability mappings are owned before a model binds their laws."""
    low = StochasticTransition(func=_probability_quarter)
    high = StochasticTransition(func=_probability_three_quarters)
    probabilities = {"perceived": low, "realized": high}
    laws = {
        "plain": probabilities,
        "proxy": MappingProxyType(probabilities),
        "case": ByAge(cases={0: probabilities}),
        "default": ByAge(cases={}, default=probabilities),
        "until": ByAge.until(
            stop_age_exclusive=1, law=probabilities, then=probabilities
        ),
        "mapped-until": ByAge.until(
            stop_age_exclusive=1, law="perceived", then="perceived"
        ).with_mapped_laws(func=lambda _: probabilities),
        "phased": Phased(solve=probabilities, simulate=probabilities),
    }
    transition = Transition(law=laws[wrapper])
    probabilities["perceived"] = StochasticTransition(func=_probability_one)
    probabilities["realized"] = StochasticTransition(func=_probability_zero)
    model = _declaration_model(edges={"work": transition}, enable_jit=False)
    np.testing.assert_array_equal(
        model.solve(params={"discount_factor": 0.5}, log_level="debug").values[0][
            "work"
        ],
        9.75,
    )


def test_declared_probability_law_preserves_callable_identity() -> None:
    """Snapshotting containers leaves the user's callable unchanged."""
    probability = StochasticTransition(func=_probability_one)
    transition = Transition(law={"perceived": probability})
    assert (
        cast("Mapping[str, StochasticTransition]", transition.law)["perceived"].func
        is probability.func
    )


def test_declared_probability_law_is_read_only() -> None:
    """A published probability mapping cannot be edited in place."""
    transition = Transition(
        law={"perceived": StochasticTransition(func=_probability_one)}
    )
    with pytest.raises(TypeError):
        cast("dict[RegimeName, StochasticTransition | RegimeName]", transition.law)[
            "perceived"
        ] = "realized"


def test_declared_edge_mapping_is_read_only() -> None:
    """A model publishes immutable declarations as well as an immutable graph."""
    model = _declaration_model(edges={"work": {"perceived": 0}}, enable_jit=False)
    with pytest.raises(TypeError):
        cast("dict[str, dict[str, int]]", model.edges)["work"]["realized"] = 0
