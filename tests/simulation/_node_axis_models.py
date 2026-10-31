"""Small fixed-law current-shock models with analytical expectations."""

import jax.numpy as jnp
import numpy as np

from lcm import (
    AgeGrid,
    ConsumptionSavingsRegime,
    ExecutionConfig,
    InitialNodes,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    Transition,
    categorical,
)
from lcm.result import SimulationResult
from lcm.shocks import PhysicalValues, ShockGroup, Uniform, UserNodes
from lcm.solver_api import SolutionResult
from lcm.solvers import NBEGM
from lcm.typing import ContinuousAction, ContinuousState, FloatND, RegimeName, ScalarInt


@categorical(ordered=False)
class _ProbeIds:
    probe: ScalarInt


@categorical(ordered=False)
class _MixedIds:
    probe: ScalarInt
    child: ScalarInt
    done: ScalarInt


@categorical(ordered=False)
class _ChildIds:
    child: ScalarInt
    done: ScalarInt


def _utility(*, consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _resources(*, wealth: ContinuousState, eps: FloatND) -> FloatND:
    return wealth + eps


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _inverse(*, marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def _next_wealth(*, savings: FloatND) -> ContinuousState:
    return savings


def _terminal(*, wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def _income(*, nodes: tuple[float, ...], weights: tuple[float, ...]) -> ShockGroup:
    return ShockGroup(
        shocks={"eps": Uniform(start=0.0, stop=1.0)},
        nodes=UserNodes(
            nodes=np.asarray(nodes).reshape(-1, 1), weights=np.asarray(weights)
        ),
    )


def _child_regimes() -> dict[RegimeName, Regime]:
    savings = IrregSpacedGrid(
        points=np.asarray(
            [0.25, 0.375, 0.5, 0.6875, 0.75, 0.875, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
        )
    )
    return {
        "child": ConsumptionSavingsRegime(
            states={"wealth": IrregSpacedGrid(points=np.asarray([0.375, 1.0, 2.0]))},
            actions={"consumption": IrregSpacedGrid(points=np.asarray([0.1, 4.0]))},
            functions={
                "utility": _utility,
                "resources": _resources,
                "savings": _savings,
                "inverse_marginal_utility": _inverse,
            },
            state_transitions={"wealth": {"done": _next_wealth}},
            liquid=LiquidMargin(
                state="wealth",
                action="consumption",
                resources="resources",
                post_decision_state="savings",
            ),
            solver=NBEGM(savings_grid=savings, probe_schedule="never"),
            shocks=("income",),
        ),
        "done": Regime(states={"wealth": savings}, functions={"utility": _terminal}),
    }


def _probe_model(
    *,
    mixed: bool,
    nodes: tuple[float, ...],
    weights: tuple[float, ...],
    bad_node: float,
    payoff: float,
) -> Model:
    def utility(*, eps: FloatND) -> FloatND:
        return jnp.where(eps == bad_node, -jnp.inf, payoff)

    regimes = {"probe": Regime(functions={"utility": utility}, shocks=("income",))}
    if mixed:
        regimes.update(_child_regimes())
    return Model(
        regimes=regimes,
        ages=AgeGrid(exact_values=(0, 1, 2)),
        initial_nodes=InitialNodes(
            by_age={0: "probe", 1: "child"} if mixed else {0: "probe"}
        ),
        edges={"child": {"done": (1,)}} if mixed else {},
        regime_id_class=_MixedIds if mixed else _ProbeIds,
        shocks={"income": _income(nodes=nodes, weights=weights)},
    )


def _path_model(*, width: int = 2, budgeted: bool = False) -> Model:
    return Model(
        regimes=_child_regimes(),
        ages=AgeGrid(exact_values=(0, 1, 2)),
        initial_nodes=InitialNodes(by_age={1: "child"}),
        edges={"child": {"done": (1,)}},
        regime_id_class=_ChildIds,
        shocks={"income": _income(nodes=(0.0, 1.0), weights=(0.5, 0.5))},
        execution_config=ExecutionConfig(
            axis_widths={"subject": width},
            device_memory_bytes=2**27 if budgeted else None,
        ),
    )


def _replay(
    *, model: Model, solution: SolutionResult, path: np.ndarray, seed: int
) -> SimulationResult:
    n = path.shape[0] if path.ndim == 2 else 3
    return model.simulate(
        params={"discount_factor": 1.0},
        solution=solution,
        initial_conditions={
            "wealth": np.ones(n, dtype=jnp.asarray(0.0).dtype),
            "age": np.ones(n, dtype=np.int32),
            "regime_id": np.full(n, model.regime_names_to_ids["child"], dtype=np.int32),
        },
        shock_paths={"eps": PhysicalValues(values=path)},
        seed=seed,
        log_level="off",
    )


def _two_shock_resources(
    *, wealth: ContinuousState, eps: FloatND, eta: FloatND
) -> FloatND:
    return wealth + eps + eta


def _tensor_probe(*, offset: ContinuousState, eps: FloatND, eta: FloatND) -> FloatND:
    return jnp.where((eps == 1.0) | (eta == 1.0), -jnp.inf, 4.0 + offset)


def _two_axis_model(*, reverse: bool) -> Model:
    regimes = _child_regimes()
    groups = {
        "income": _income(nodes=(0.0, 1.0), weights=(1.0, 0.0)),
        "other": ShockGroup(
            shocks={"eta": Uniform(start=0.0, stop=1.0)},
            nodes=UserNodes(
                nodes=np.asarray([[0.0], [1.0]]), weights=np.asarray([1.0, 0.0])
            ),
        ),
    }
    if reverse:
        groups = dict(reversed(tuple(groups.items())))
    names = tuple(groups)
    child = regimes["child"]
    regimes["child"] = child.replace(
        functions={**child.functions, "resources": _two_shock_resources},
        shocks=names,
    )
    regimes["probe"] = Regime(
        states={"offset": IrregSpacedGrid(points=np.asarray([1.0, 2.0]))},
        functions={"utility": _tensor_probe},
        shocks=names,
    )
    return Model(
        regimes=regimes,
        ages=AgeGrid(exact_values=(0, 1, 2)),
        initial_nodes=InitialNodes(by_age={0: "probe", 1: "child"}),
        edges={"child": {"done": (1,)}},
        regime_id_class=_MixedIds,
        shocks=groups,
    )


def _heterogeneous_entry_model() -> Model:
    return Model(
        regimes=_child_regimes(),
        ages=AgeGrid(exact_values=(0, 1, 2)),
        initial_nodes=InitialNodes(by_age={1: "child", 2: "done"}),
        edges={"child": {"done": (1,)}},
        regime_id_class=_ChildIds,
        shocks={"income": _income(nodes=(0.0, 1.0), weights=(0.5, 0.5))},
    )


@categorical(ordered=False)
class _EntryIds:
    parent: ScalarInt
    child: ScalarInt
    done: ScalarInt


def _zero_utility(*, wealth: ContinuousState) -> FloatND:
    return wealth * 0


def _carry_wealth(*, wealth: ContinuousState) -> ContinuousState:
    return wealth


def _select_reader(*, wealth: ContinuousState) -> ScalarInt:
    return jnp.where(wealth < 1.5, _EntryIds.child, _EntryIds.done)


def _transition_model(*, budgeted: bool = False) -> Model:

    regimes = _child_regimes()
    regimes["parent"] = Regime(
        states={"wealth": IrregSpacedGrid(points=np.asarray([1.0, 2.0]))},
        functions={"utility": _zero_utility},
        state_transitions={"wealth": _carry_wealth},
    )
    return Model(
        regimes=regimes,
        ages=AgeGrid(exact_values=(0, 1, 2)),
        initial_nodes=InitialNodes(by_age={0: "parent"}),
        edges={
            "parent": Transition(
                targets={"child": (0,), "done": (0,)}, law=_select_reader
            ),
            "child": {"done": (1,)},
        },
        regime_id_class=_EntryIds,
        shocks={"income": _income(nodes=(0.0, 1.0), weights=(0.5, 0.5))},
        execution_config=ExecutionConfig(
            axis_widths={"subject": 1},
            device_memory_bytes=2**27 if budgeted else None,
        ),
    )
