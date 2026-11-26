"""Lottery contraction in composition action queries and collective regimes.

A consent marriage: each single's health and each spouse's health is a Markov draw whose
weights read only its own current value, so `GridSearch(lottery_contraction=True)` sums
it into the target's value array, also where a composition action query or a collective
continuation reads it. Income is a Rouwenhorst process whose weights read its own value,
so a query, which evaluates its regime at projected states, keeps it enumerated and
extends it beyond its nodes. The terminal couple's income follows another law, so the
solve's couple queries extend it too. A single's consent composition action is available
in good health only, so the query guards each lane.
"""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax.core import Tracer

from _lcm import composition_queries
from _lcm.outside_values import OutsideValueTable
from _lcm.policy_queries import (
    PolicyAnswer,
    ScalarPolicyQueries,
    _PolicyQueryKernel,
    _QueryChunks,
)
from _lcm.query_information import (
    Dependencies,
    InformationQueryCell,
    SourceStates,
)
from _lcm.regime_building import Q_and_F as q_and_f_module
from _lcm.regime_building.Q_and_F import (
    _ComputeCE,
    _contracts_despite_counting_or_members,
    _TargetContinuation,
)
from _lcm.regime_building.V import VInterpolationInfo
from _lcm.typing import (
    PytreeValue,
    QAndFFunction,
    QAndFKwargs,
    TransitionFunction,
)
from lcm import (
    CollectiveUtility,
    CompositionAction,
    Consent,
    Counterparty,
    Destination,
    DiscreteGrid,
    InitialNodes,
    LinSpacedGrid,
    Model,
    PCOnActions,
    Periods,
    Regime,
    RouwenhorstAR1Process,
    StochasticTransition,
)
from lcm.exceptions import UnsupportedOperationError
from lcm.shocks import GaussHermite, Normal, Product, ShockGroup
from lcm.solvers import GridSearch
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    OffGrid,
    TransitionFunctionName,
)
from tests.conftest import assert_agrees_to_ulp
from tests.solution.test_outside_value_truncation import _Marriage

_ROWS = np.array([[0.7, 0.3], [0.2, 0.8]])
_INCOME = RouwenhorstAR1Process(n_points=3, rho=0.6, sigma=0.8, mu=0.0)
# The same nodes under another law: both have unit stationary deviation.
_TERMINAL_COUPLE_INCOME = RouwenhorstAR1Process(n_points=3, rho=0.0, sigma=1.0, mu=0.0)
_PARAMS = {"discount_factor": 0.9}


def _probs(health: DiscreteState) -> FloatND:
    return jnp.asarray(_ROWS)[health]


def _probs_f(health_f: DiscreteState) -> FloatND:
    return jnp.asarray(_ROWS)[health_f]


def _offer_risk(*, health_f: DiscreteState, offer: FloatND) -> FloatND:
    return jnp.asarray(_ROWS)[health_f, 1] + 0.1 * offer


def _probs_f_with_offer(*, health_f: DiscreteState, offer: FloatND) -> FloatND:
    healthy = _offer_risk(health_f=health_f, offer=offer)
    return jnp.stack((1.0 - healthy, healthy))


def _probs_f_through_helper(offer_risk: FloatND) -> FloatND:
    return jnp.stack((1.0 - offer_risk, offer_risk))


def _probs_f_offer_only(offer: FloatND) -> FloatND:
    healthy = 0.55 + 0.1 * offer
    return jnp.stack((1.0 - healthy, healthy))


# How the wife's health draw reads the couple's current offer.
_OFFER_MOVES_HEALTH = {
    "directly": _probs_f_with_offer,
    "through-a-helper": _probs_f_through_helper,
    "alone": _probs_f_offer_only,
}


def _probs_m(health_m: DiscreteState) -> FloatND:
    return jnp.asarray(_ROWS)[health_m]


def _single(
    *, health: DiscreteState, work: DiscreteAction, income: ContinuousState
) -> FloatND:
    return 1.0 + health - 0.4 * work + 0.1 * income


def _wife(
    *, health_f: DiscreteState, work: DiscreteAction, income: ContinuousState
) -> FloatND:
    return 1.5 + health_f + 0.3 * work + 0.05 * income


def _wife_with_offer(
    *,
    health_f: DiscreteState,
    work: DiscreteAction,
    income: ContinuousState,
    offer: FloatND,
) -> FloatND:
    return 1.5 + health_f + 0.3 * work + 0.05 * income + 0.2 * offer


def _husband(
    *, health_m: DiscreteState, work: DiscreteAction, income: ContinuousState
) -> FloatND:
    return 1.2 + 0.5 * health_m - 0.2 * work + 0.15 * income


def _health(health: DiscreteState) -> DiscreteState:
    return health


def _health_f(health_f: DiscreteState) -> DiscreteState:
    return health_f


def _health_m(health_m: DiscreteState) -> DiscreteState:
    return health_m


def _income(income: ContinuousState) -> ContinuousState:
    return income


def _healthy(health: DiscreteState) -> BoolND:
    return health == 1


def _model(
    *,
    contract: bool,
    observe_offer: bool = False,
    offer_moves_health: str | None = None,
) -> Model:
    """Single_f marries by consent; the couple dissolves unilaterally; three periods.

    With `observe_offer`, the wife's flow reads a current offer that single_f's
    consent observes, so her query reads its information. `offer_moves_health` names
    how the wife's health draw reads it too (see `_OFFER_MOVES_HEALTH`).
    """
    health = DiscreteGrid(values=jnp.asarray([0, 1]))
    work = {"work": DiscreteGrid(values=jnp.asarray([0, 1]))}
    single = Regime(
        states={"health": health, "income": _INCOME},
        actions=work,
        functions={"utility": _single},
        state_transitions={"health": StochasticTransition(func=_probs)},
        solver=GridSearch(lottery_contraction=contract),
    )
    couple = Regime(
        states={"health_f": health, "health_m": health, "income": _INCOME},
        actions=work,
        functions={
            "utility": CollectiveUtility(
                utilities={
                    "f": _wife_with_offer if observe_offer else _wife,
                    "m": _husband,
                }
            ),
            **(
                {"offer_risk": _offer_risk}
                if offer_moves_health == "through-a-helper"
                else {}
            ),
        },
        shocks=("offers",) if observe_offer else (),
        state_transitions={
            "health_f": StochasticTransition(
                func=_OFFER_MOVES_HEALTH[offer_moves_health]
                if offer_moves_health
                else _probs_f
            ),
            "health_m": StochasticTransition(func=_probs_m),
        },
        solver=GridSearch(lottery_contraction=contract),
    )
    consent = Consent(
        counterparty=Counterparty(
            regime="single_m",
            role="m",
            projection={"health": _health, "income": _income},
        )
    )
    to_couple = Destination(
        regime="couple",
        role="f",
        projection={"health_f": _health, "health_m": _health, "income": _income},
    )
    regimes = {
        "single_f": single.replace(
            composition_action=CompositionAction(
                destinations={"self": to_couple},
                rule=consent,
                available=_healthy,
                observes=("offer",) if observe_offer else (),
            )
        ),
        "single_m": single,
        "couple": couple.replace(
            composition_action=CompositionAction(
                destinations={
                    "f": Destination(
                        regime="single_f",
                        projection={"health": _health_f, "income": _income},
                    ),
                    "m": Destination(
                        regime="single_m",
                        projection={"health": _health_m, "income": _income},
                    ),
                },
                rule=PCOnActions(),
            )
        ),
    }
    for name, regime in tuple(regimes.items()):
        regimes[f"{name}_terminal"] = regime.replace(
            composition_action=None, state_transitions={}
        )
    regimes["couple_terminal"] = regimes["couple_terminal"].replace(
        states={
            **regimes["couple_terminal"].states,
            "income": _TERMINAL_COUPLE_INCOME,
        },
        # Without a draw the terminal couple reads no offer risk.
        functions={"utility": regimes["couple_terminal"].functions["utility"]},
    )
    return Model(
        regimes=regimes,
        regime_id_class=_Marriage,
        n_periods=3,
        edges={
            name: {name: Periods(values=(0,)), f"{name}_terminal": Periods(values=(1,))}
            for name in ("single_f", "single_m", "couple")
        },
        initial_nodes=InitialNodes(by_period={0: ("single_f", "single_m", "couple")}),
        shocks=(
            {
                "offers": ShockGroup(
                    shocks={"offer": Normal(mu=0.0, sigma=1.0)},
                    nodes=Product(offer=GaussHermite(2)),
                )
            }
            if observe_offer
            else {}
        ),
    )


def _query_functions(model: Model) -> list[tuple[str, _ComputeCE]]:
    """Every query's target and the continuation aggregate of its Q_and_F."""
    found = []
    for regime in model._regimes.values():
        for kernel in regime.solution.period_kernels.values():
            if not isinstance(kernel, _PolicyQueryKernel):
                continue
            for node in kernel.definitions.values():
                function = node.context.Q_and_F_functions.get(node.read.period)
                compute_CE = getattr(function, "compute_CE", None)
                if isinstance(compute_CE, _ComputeCE):
                    found.append((node.read.target, compute_CE))
    return found


def _query_q_and_f(model: Model) -> list[QAndFFunction]:
    """Every query's Q_and_F."""
    return [
        node.context.Q_and_F_functions[node.read.period]
        for regime in model._regimes.values()
        for kernel in regime.solution.period_kernels.values()
        if isinstance(kernel, _PolicyQueryKernel)
        for node in kernel.definitions.values()
        if node.read.period in node.context.Q_and_F_functions
    ]


def _contracted(model: Model) -> dict[tuple[str, str], frozenset[str]]:
    """Per (query target, continuation target): the states its query Q_and_F sums."""
    return {
        (query_target, target.removesuffix("_terminal")): (
            frozenset(lot.state_name for lot in summed.lotteries)
            if (summed := continuation.lottery_contraction)
            else frozenset()
        )
        for query_target, compute_CE in _query_functions(model)
        for target, continuation in compute_CE.continuations.items()
    }


def test_query_and_collective_continuations_contract_exactly_when_requested() -> None:
    """With the option on, every health draw a query reads is summed; off, none is."""
    off = _contracted(_model(contract=False))
    on = _contracted(_model(contract=True))
    expected = {
        ("couple", "couple"): {"health_f", "health_m"},
        ("single_f", "single_f"): {"health"},
        ("single_m", "single_m"): {"health"},
    }
    assert (off, on) == (dict.fromkeys(expected, frozenset()), expected)


def test_couple_queries_extend_income_beyond_its_nodes() -> None:
    """The specimen exercises the edge extension inside a contracting query."""
    assert any(
        compute_CE.edge_extension is not None
        and any(
            continuation.lottery_contraction is not None
            for continuation in compute_CE.continuations.values()
        )
        for target, compute_CE in _query_functions(_model(contract=True))
        if target == "couple"
    )


def _keys(model: Model) -> frozenset[str]:
    return frozenset(
        continuation.lottery_contraction.key
        for _, compute_CE in _query_functions(model)
        for continuation in compute_CE.continuations.values()
        if continuation.lottery_contraction is not None
    )


def test_prepared_table_keys_are_fixed_by_what_each_reader_contracts() -> None:
    """Keys name the target, the summed states, their axes and the weights' inputs.

    The same literal in every process and build, so a traced program's structure
    and its compilation-cache entry do not depend on object identity.
    """
    assert _keys(_model(contract=True)) == {
        "__contracted__couple__health_f@0(health_f)__health_m@1(health_m)",
        "__contracted__couple_terminal__health_f@0(health_f)__health_m@1(health_m)",
        "__contracted__single_f__health@0(health)",
        "__contracted__single_f_terminal__health@0(health)",
        "__contracted__single_m__health@0(health)",
        "__contracted__single_m_terminal__health@0(health)",
    }


def _values(
    *, contract: bool, offer_moves_health: str | None = None
) -> dict[str, np.ndarray]:
    solution = _model(
        contract=contract,
        observe_offer=offer_moves_health is not None,
        offer_moves_health=offer_moves_health,
    ).solve(params=_PARAMS, log_level="off", runtime_checks=False)
    return {
        f"{period}/{regime}": np.asarray(value)
        for period, by_regime in solution.values.items()
        for regime, value in by_regime.items()
    }


@pytest.mark.parametrize(
    ("stored", "offer_moves_health"),
    [
        (True, None),
        (False, None),
        *((False, how) for how in _OFFER_MOVES_HEALTH),
    ],
    ids=[
        "stored",
        "computed",
        *(f"draw-reads-a-shock-{how}" for how in _OFFER_MOVES_HEALTH),
    ],
)
def test_contraction_agrees_with_enumeration_to_16_ulp(
    *, monkeypatch: pytest.MonkeyPatch, stored: bool, offer_moves_health: str | None
) -> None:
    """Values with summed draws agree with enumerated ones at the array's scale.

    A query builds its tables before it binds each information row's current
    shocks, so a draw whose weights read one, directly, through a helper or alone,
    stays enumerated there.
    """
    monkeypatch.setattr(composition_queries, "_READ_STORED_OUTSIDE_VALUES", stored)
    if offer_moves_health is not None:
        model = _model(
            contract=True, observe_offer=True, offer_moves_health=offer_moves_health
        )
        assert _contracted(model)[("couple", "couple")] == {"health_m"}
    off = _values(contract=False, offer_moves_health=offer_moves_health)
    on = _values(contract=True, offer_moves_health=offer_moves_health)
    assert on.keys() == off.keys()
    for key, expected in off.items():
        np.testing.assert_array_equal(np.isfinite(on[key]), np.isfinite(expected))
        assert_agrees_to_ulp(
            got=on[key],
            expected=expected,
            n_ulp=16,
            operand_magnitude=float(np.max(np.abs(expected[np.isfinite(expected)]))),
            err_msg=key,
        )


def _simulate(
    *,
    contract: bool,
    observe_offer: bool = False,
    offer_moves_health: str | None = None,
) -> pd.DataFrame:
    model = _model(
        contract=contract,
        observe_offer=observe_offer,
        offer_moves_health=offer_moves_health,
    )
    solution = model.solve(params=_PARAMS, log_level="off")
    return (
        model.simulate(
            params=_PARAMS,
            solution=solution,
            initial_conditions={
                "period": jnp.zeros(6, dtype=jnp.int32),
                "health": jnp.asarray([0, 1, 1, 0, 1, 1]),
                "health_f": jnp.asarray([0, 1, 0, 1, 1, 0]),
                "health_m": jnp.asarray([1, 0, 1, 0, 1, 1]),
                # Between the nodes and beyond the outer ones.
                "income": jnp.asarray([0.3, -2.0, 0.7, 1.8, -0.5, 1.2]),
                "regime_id": jnp.asarray(
                    [
                        _Marriage.single_f,
                        _Marriage.single_f,
                        _Marriage.single_m,
                        _Marriage.couple,
                        _Marriage.couple,
                        _Marriage.couple,
                    ],
                    dtype=jnp.int32,
                ),
                "own_member": jnp.asarray([0, 0, 0, 0, 1, 1], dtype=jnp.int32),
            },
            seed=0,
            log_level="off",
        )
        .to_dataframe(use_categorical_codes=True)
        .sort_values(["agent_id", "period"])
    )


def test_simulation_with_contraction_chooses_and_moves_as_without() -> None:
    """The simulation contracts what the solve contracts, couples included.

    Away from ties, both routes take the same actions and composition actions, and
    report the same values to rounding at the panel's scale.
    """
    off = _simulate(contract=False)
    on = _simulate(contract=True)
    discrete = ["agent_id", "period", "regime_name", "work"]
    pd.testing.assert_frame_equal(
        on[discrete].reset_index(drop=True), off[discrete].reset_index(drop=True)
    )
    expected = off["value"].to_numpy()
    assert_agrees_to_ulp(
        got=on["value"].to_numpy(),
        expected=expected,
        n_ulp=16,
        operand_magnitude=float(np.max(np.abs(expected[np.isfinite(expected)]))),
    )


def _query_tables_built_in_place(
    *,
    monkeypatch: pytest.MonkeyPatch,
    observe_offer: bool,
    offer_moves_health: str | None,
) -> tuple[int, int]:
    """Solve and simulate; count the tables query programs' readers build in place.

    A reader inside a query's loops and lane maps that builds its own table
    rebuilds it per chunk, and under a lane-mapped guard per lane, row and
    action, which at production size asks for terabytes. Also returns how often
    an information query was evaluated.
    """
    depth = 0
    count = 0
    information = 0

    def entering(*, method: Callable[..., PytreeValue]) -> Callable[..., PytreeValue]:
        def entered(*args: PytreeValue, **kwargs: PytreeValue) -> PytreeValue:
            nonlocal depth
            depth += 1
            try:
                return method(*args, **kwargs)
            finally:
                depth -= 1

        return entered

    for owner, name in (
        (_QueryChunks, "__call__"),
        (OutsideValueTable, "__call__"),
        (ScalarPolicyQueries, "evaluate"),
    ):
        monkeypatch.setattr(owner, name, entering(method=getattr(owner, name)))
    original = _TargetContinuation.with_contracted_draws
    original_information = InformationQueryCell.__call__

    # keyword-only-exempt: primary-argument=self
    def in_place(
        self: _TargetContinuation,
        *,
        values: FloatND,
        states_actions_params: QAndFKwargs,
    ) -> FloatND:
        nonlocal count
        count += depth > 0 and self.lottery_contraction is not None
        return original(
            self, values=values, states_actions_params=states_actions_params
        )

    # keyword-only-exempt: primary-argument=self
    def informed(
        self: InformationQueryCell,
        *,
        live: BoolND,
        source: SourceStates,
        dependencies: Dependencies,
    ) -> PolicyAnswer:
        nonlocal information
        information += 1
        return original_information(
            self, live=live, source=source, dependencies=dependencies
        )

    monkeypatch.setattr(_TargetContinuation, "with_contracted_draws", in_place)
    monkeypatch.setattr(InformationQueryCell, "__call__", informed)
    _simulate(
        contract=True,
        observe_offer=observe_offer,
        offer_moves_health=offer_moves_health,
    )
    return count, information


@pytest.mark.parametrize(
    ("observe_offer", "offer_moves_health", "reads_information"),
    [
        pytest.param(
            False, None, False, id="composition actions-couples-and-edge-corners"
        ),
        pytest.param(True, None, True, id="consent-observing-an-offer"),
        pytest.param(True, "directly", True, id="a-draw-reading-the-offer"),
    ],
)
def test_query_programs_read_only_prepared_tables(
    *,
    monkeypatch: pytest.MonkeyPatch,
    observe_offer: bool,
    offer_moves_health: str | None,
    reads_information: bool,
) -> None:
    """The queries of a composition action build no table in place.

    These are the composition action, consent, outside-value and simulated queries.
    The edge extension's corner reads included, and, where a consent observes an
    offer, the information queries. A draw whose weights read the offer stays
    enumerated there rather than being summed in place.
    """
    count, information = _query_tables_built_in_place(
        monkeypatch=monkeypatch,
        observe_offer=observe_offer,
        offer_moves_health=offer_moves_health,
    )
    assert (count, information > 0) == (0, reads_information)


def test_a_query_reader_without_its_table_refuses_to_build_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control: a query reader left without its table raises while tracing."""
    original = q_and_f_module.prepare_contracted_values

    def without_couple_tables(
        *,
        function: QAndFFunction | None,
        arguments: Mapping[str, PytreeValue],
        skip: frozenset[str] = frozenset(),
    ) -> Mapping[str, PytreeValue]:
        return original(function=function, arguments=arguments, skip=skip | {"couple"})

    monkeypatch.setattr(
        q_and_f_module, "prepare_contracted_values", without_couple_tables
    )
    with pytest.raises(
        UnsupportedOperationError,
        match=r"regime 'couple' finds no table under "
        r"'__contracted__couple__health_f@0\(health_f\)__health_m@1",
    ):
        _simulate(contract=True)


def test_every_corner_of_a_query_reads_a_table_its_continuation_prepares() -> None:
    """A corner's contraction is its continuation's: an edge draw is not contracted.

    A query keeps a process draw reading its own value enumerated, and only such
    a draw is extended beyond its nodes, so the corners sum exactly the draws, and
    read exactly the tables, that the prepared continuations do.
    """
    corners = [
        (
            {
                c.lottery_contraction.key
                for c in compute_CE.continuations.values()
                if c.lottery_contraction is not None
            },
            {
                c.lottery_contraction.key
                for c in compute_CE.edge_extension.corner.continuations.values()
                if c.lottery_contraction is not None
            },
        )
        for _, compute_CE in _query_functions(_model(contract=True))
        if compute_CE.edge_extension is not None
    ]
    assert corners
    assert all(corner <= prepared for prepared, corner in corners)


def test_with_the_option_off_queries_receive_their_arguments_untouched() -> None:
    """Preparation hands back the very mapping it was given, so no program changes."""
    arguments = {
        "next_regime_to_V_arr": MappingProxyType(
            {name: jnp.zeros(()) for name in _Marriage.__annotations__}
        )
    }
    assert all(
        q_and_f_module.prepare_contracted_values(function=function, arguments=arguments)
        is arguments
        for function in _query_q_and_f(_model(contract=False))
    )


def test_prepared_tables_are_built_inside_the_admitted_programs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each table is built from traced values, inside the program that reads it.

    Admission reserves every compiled program's arguments, outputs and
    temporaries, so a table built there is charged with the program; one built
    ahead of it, from concrete values, would be held outside that reservation.
    """
    original = q_and_f_module.prepare_contracted_values
    traced: list[bool] = []

    def recording(
        *,
        function: QAndFFunction | None,
        arguments: Mapping[str, PytreeValue],
        skip: frozenset[str] = frozenset(),
    ) -> Mapping[str, PytreeValue]:
        prepared = original(function=function, arguments=arguments, skip=skip)
        traced.extend(
            isinstance(table, Tracer)
            for key, table in cast(
                "Mapping[str, FloatND]", prepared.get("next_regime_to_V_arr", {})
            ).items()
            if key.startswith("__contracted__")
        )
        return prepared

    monkeypatch.setattr(q_and_f_module, "prepare_contracted_values", recording)
    _simulate(contract=True)
    assert set(traced) == {True}


_HEALTH = DiscreteGrid(values=jnp.asarray([0, 1]))


@pytest.mark.parametrize(
    ("off_grid", "count_raise_landings", "contracts"),
    [
        pytest.param("raise", True, False, id="counted-raise-grid-keeps-the-draw"),
        pytest.param("raise", False, True, id="uncounted-raise-grid-contracts"),
        pytest.param("extrapolate", True, True, id="no-raise-grid-contracts"),
    ],
)
def test_a_counting_reader_contracts_unless_its_target_counts_raise_landings(
    *, off_grid: OffGrid, count_raise_landings: bool, contracts: bool
) -> None:
    """A query counts landings on the nodes, so a raise-rule target keeps its axes."""
    info = VInterpolationInfo(
        state_names=("wealth", "health"),
        discrete_states=MappingProxyType({"health": _HEALTH}),
        continuous_states=MappingProxyType(
            {
                "wealth": LinSpacedGrid(
                    start=0.0, stop=1.0, n_points=3, off_grid=off_grid
                )
            }
        ),
    )
    assert (
        _contracts_despite_counting_or_members(
            v_interpolation_info=info,
            bundle=cast(
                "MappingProxyType[TransitionFunctionName, TransitionFunction]",
                MappingProxyType({"next_wealth": _income, "next_health": _health}),
            ),
            co_map_state_names=(),
            count_raise_landings=count_raise_landings,
        )
        is contracts
    )
