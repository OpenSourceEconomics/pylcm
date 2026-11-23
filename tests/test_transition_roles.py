"""A singleton source enters a collective target as a declared member.

`Transition(roles={"couple": "f"})` says that a row moving from a singleton source
into the collective regime `couple` continues there as its member `f`. It is
the always-open gate a model would otherwise write by hand, so the two declarations
solve and simulate identically. A row landing where the target has no feasible
action is refused as on a plain edge.
"""

from collections.abc import Mapping
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.model_graph import role_gate_is_open
from lcm import (
    AgeGrid,
    ByAge,
    CollectiveUtility,
    DiscreteGrid,
    Gate,
    JointTransition,
    LinSpacedGrid,
    MemberRoute,
    Model,
    Phased,
    ProjectedRegimeValue,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import (
    ModelInitializationError,
    NoFeasibleActionError,
)
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    RegimeName,
    ScalarInt,
)

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")

type LawForm = Literal["by_age", "phased_law", "phased_edges"]
type EdgeForm = Literal["roles", "hand_gate", "exact_gate", "plain"]


@categorical(ordered=True)
class Work:
    rest: ScalarInt
    work: ScalarInt


@categorical(ordered=False)
class RegimeId:
    single: ScalarInt
    couple: ScalarInt
    single_end: ScalarInt
    couple_end: ScalarInt


WEALTH = LinSpacedGrid(start=1.0, stop=3.0, n_points=3)
PARTNER_WEALTH = LinSpacedGrid(start=0.0, stop=2.0, n_points=3)


def utility_single(*, wealth: ContinuousState, work: DiscreteAction) -> FloatND:
    return wealth + 0.5 * work


def utility_f(
    *, wealth: ContinuousState, partner_wealth: ContinuousState, work: DiscreteAction
) -> FloatND:
    return 2.0 * wealth + 0.25 * partner_wealth + 0.1 * work


def utility_m(
    *, wealth: ContinuousState, partner_wealth: ContinuousState, work: DiscreteAction
) -> FloatND:
    return wealth + partner_wealth - 0.2 * work


def utility_single_end(wealth: ContinuousState) -> FloatND:
    return 0.5 * wealth


def utility_couple_end(
    *, wealth: ContinuousState, partner_wealth: ContinuousState
) -> FloatND:
    return 0.5 * wealth + 0.1 * partner_wealth


def rich_enough(wealth: ContinuousState) -> BoolND:
    return wealth > 2.5


def marries(*, age: FloatND, marriage_rate: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float) * marriage_rate


def stays_single(*, age: FloatND, marriage_rate: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float) * (1.0 - marriage_rate)


def marries_perceived(*, age: FloatND, marriage_rate: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float) * 0.5 * marriage_rate


def stays_single_perceived(*, age: FloatND, marriage_rate: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float) * (1.0 - 0.5 * marriage_rate)


def match_probabilities() -> FloatND:
    return jnp.asarray([0.25, 0.75])


def pooled_wealth(
    *, wealth: ContinuousState, match: Mapping[str, FloatND]
) -> ContinuousState:
    return jnp.minimum(wealth + 0.5 * match["partner_wealth"], 3.0)


def partner_wealth_at_match(match: Mapping[str, FloatND]) -> ContinuousState:
    return match["partner_wealth"]


def marriage_gate(wealth: ContinuousState) -> BoolND:
    return jnp.ones_like(wealth, dtype=bool)


def project_wealth(wealth: ContinuousState) -> ContinuousState:
    return wealth


def _single() -> Regime:
    return Regime(
        states={"wealth": WEALTH},
        state_transitions={"wealth": fixed_transition("wealth")},
        joint_transitions={
            "couple": {
                "match": JointTransition(
                    support_size=2,
                    support={"partner_wealth": jnp.asarray([0.0, 2.0])},
                    probabilities=match_probabilities,
                    outputs={
                        "wealth": pooled_wealth,
                        "partner_wealth": partner_wealth_at_match,
                    },
                )
            }
        },
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": utility_single},
    )


def _couple(*, collective: bool, infeasible_at_low_wealth: bool) -> Regime:
    return Regime(
        states={"wealth": WEALTH, "partner_wealth": PARTNER_WEALTH},
        state_transitions={
            "wealth": fixed_transition("wealth"),
            "partner_wealth": fixed_transition("partner_wealth"),
        },
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility": (
                CollectiveUtility(utilities={"f": utility_f, "m": utility_m})
                if collective
                else utility_f
            )
        },
        constraints=({"rich_enough": rich_enough} if infeasible_at_low_wealth else {}),
    )


def _couple_end(*, collective: bool) -> Regime:
    return Regime(
        states={"wealth": WEALTH, "partner_wealth": PARTNER_WEALTH},
        functions={
            "utility": (
                CollectiveUtility(
                    utilities={"f": utility_couple_end, "m": utility_couple_end}
                )
                if collective
                else utility_couple_end
            )
        },
    )


def _hand_gate() -> Gate:
    """The always-open gate a model writes by hand, falling back to the source."""
    return Gate(
        predicate=marriage_gate,
        routes={
            "own": MemberRoute(
                target_member="f",
                fallback=ProjectedRegimeValue(
                    regime="single", projection={"wealth": project_wealth}
                ),
            )
        },
    )


def _exact_gate() -> Gate:
    """The gate spelled exactly as `roles={"couple": "f"}` is lowered."""
    return Gate(
        predicate=role_gate_is_open,
        routes={
            "f": MemberRoute(
                target_member="f",
                fallback=ProjectedRegimeValue(
                    regime="couple",
                    projection={
                        "wealth": fixed_transition("wealth"),
                        "partner_wealth": fixed_transition("partner_wealth"),
                    },
                    member="f",
                ),
            )
        },
    )


def _gate(*, edge_form: EdgeForm) -> Gate | None:
    if edge_form == "hand_gate":
        return _hand_gate()
    if edge_form == "exact_gate":
        return _exact_gate()
    return None


def _single_transition(
    *, edge_form: EdgeForm, perceived: bool, phased_law: bool, gate: Gate | None
) -> Transition:
    stay = StochasticTransition(func=stays_single)
    marry = StochasticTransition(func=marries)
    if perceived:
        stay = StochasticTransition(func=stays_single_perceived)
        marry = StochasticTransition(func=marries_perceived)
    cells: dict[RegimeName, StochasticTransition] = {"single": stay, "couple": marry}
    law: dict[RegimeName, StochasticTransition] | Phased = cells
    if phased_law:
        law = Phased(
            solve={
                "single": StochasticTransition(func=stays_single_perceived),
                "couple": StochasticTransition(func=marries_perceived),
            },
            simulate=cells,
        )
    return Transition(
        law=ByAge.until(
            stop_age_exclusive=2,
            law=law,
            then="single_end",
            start_age_inclusive=0,
        ),
        gates={} if gate is None else {"couple": gate},
        roles={"couple": "f"} if edge_form == "roles" else {},
    )


def build_model(
    *,
    edge_form: EdgeForm,
    law_form: LawForm = "by_age",
    infeasible_at_low_wealth: bool = False,
) -> Model:
    """Single, who may marry at age 0 only, and a couple formed at age 1.

    The couple's states arrive through the source's joint match lottery; the
    single's marriage edge is selected by `ByAge.until` and is gone from age 1.
    """
    collective = edge_form != "plain"
    # One gate object serves both phases of `Phased` edges, as a gate must.
    gate = _gate(edge_form=edge_form)
    edges = {
        "single": _single_transition(
            edge_form=edge_form,
            perceived=False,
            phased_law=law_form == "phased_law",
            gate=gate,
        ),
        "couple": {"couple_end": 1},
    }
    model_edges = (
        Phased(
            solve={
                **edges,
                "single": _single_transition(
                    edge_form=edge_form,
                    perceived=True,
                    phased_law=False,
                    gate=gate,
                ),
            },
            simulate=edges,
        )
        if law_form == "phased_edges"
        else edges
    )
    return Model(
        regimes={
            "single": _single(),
            "couple": _couple(
                collective=collective,
                infeasible_at_low_wealth=infeasible_at_low_wealth,
            ),
            "single_end": Regime(
                states={"wealth": WEALTH}, functions={"utility": utility_single_end}
            ),
            "couple_end": _couple_end(collective=collective),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "single"},
        edges=model_edges,
    )


def _params(*, marriage_rate: float) -> dict[str, dict[str, dict[str, float]]]:
    return {
        "single": {"koopmans_aggregator": {"discount_factor": 0.9}},
        "couple": {"koopmans_aggregator": {"discount_factor": 0.9}},
        "single_end": {},
        "couple_end": {},
        "edges": {"single": {"marriage_rate": marriage_rate}},
    }


INITIAL_CONDITIONS = {
    "regime_id": jnp.zeros(8, dtype=jnp.int32),
    "age": jnp.zeros(8),
    "wealth": jnp.asarray([1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 2.0, 1.5]),
}


def _solve_and_simulate(
    *, model: Model, marriage_rate: float
) -> tuple[dict[tuple[int, RegimeName], FloatND], pd.DataFrame, str]:
    params = _params(marriage_rate=marriage_rate)
    solution = model.solve(params=params, log_level="off")
    result = model.simulate(
        params=params,
        initial_conditions=INITIAL_CONDITIONS,
        solution=solution,
        log_level="off",
        seed=11,
    )
    values = {
        (period, name): value
        for period, by_regime in solution.values.items()
        for name, value in by_regime.items()
    }
    return values, result.to_dataframe(), solution.metadata.model_fingerprint


@pytest.mark.parametrize("law_form", ["by_age", "phased_law", "phased_edges"])
@pytest.mark.parametrize("hand_form", ["hand_gate", "exact_gate"])
def test_roles_solve_bitwise_like_the_hand_written_always_open_gate(
    *, law_form: LawForm, hand_form: EdgeForm
) -> None:
    """Every solved value array is identical, bit for bit, to the gated model's."""
    roles, _, _ = _solve_and_simulate(
        model=build_model(edge_form="roles", law_form=law_form), marriage_rate=0.5
    )
    hand, _, _ = _solve_and_simulate(
        model=build_model(edge_form=hand_form, law_form=law_form), marriage_rate=0.5
    )
    assert roles.keys() == hand.keys()
    for key, value in roles.items():
        np.testing.assert_array_equal(value, hand[key], err_msg=str(key))


@pytest.mark.parametrize("law_form", ["by_age", "phased_law", "phased_edges"])
@pytest.mark.parametrize("hand_form", ["hand_gate", "exact_gate"])
def test_roles_simulate_the_same_panel_as_the_hand_written_always_open_gate(
    *, law_form: LawForm, hand_form: EdgeForm
) -> None:
    """The simulated panel under a fixed seed is identical to the gated model's."""
    _, roles, _ = _solve_and_simulate(
        model=build_model(edge_form="roles", law_form=law_form), marriage_rate=0.5
    )
    _, hand, _ = _solve_and_simulate(
        model=build_model(edge_form=hand_form, law_form=law_form), marriage_rate=0.5
    )
    pd.testing.assert_frame_equal(roles, hand, check_exact=True)


def test_roles_fingerprint_equals_the_gate_spelled_as_the_lowering() -> None:
    """A gate spelled exactly as `roles` is lowered gives the same model identity."""
    _, _, roles = _solve_and_simulate(
        model=build_model(edge_form="roles"), marriage_rate=0.5
    )
    _, _, exact = _solve_and_simulate(
        model=build_model(edge_form="exact_gate"), marriage_rate=0.5
    )
    assert roles == exact


def test_a_row_entering_through_roles_carries_the_declared_role() -> None:
    """Every row, all of whom marry, is the couple's member `f` at period 1."""
    _, panel, _ = _solve_and_simulate(
        model=build_model(edge_form="roles"), marriage_rate=1.0
    )
    married = panel.loc[panel["period"] == 1]
    assert (
        married[["regime_name", "own_member"]].astype(str).to_numpy().tolist()
        == [["couple", "f"]] * 8
    )


def test_roles_marriage_edge_fires_only_where_its_by_age_case_selects_it() -> None:
    """At period 2 every row is in a terminal regime: marriage stops at age 0."""
    _, panel, _ = _solve_and_simulate(
        model=build_model(edge_form="roles"), marriage_rate=0.5
    )
    at_two = panel.loc[panel["period"] == 2, "regime_name"].astype(str)
    assert set(at_two) == {"single_end", "couple_end"}


def _no_feasible_action_message(*, edge_form: EdgeForm) -> str:
    model = build_model(edge_form=edge_form, infeasible_at_low_wealth=True)
    with pytest.raises(
        NoFeasibleActionError,
        match=r"^Regime 'couple', period 1, state \{.*\} has no feasible action\.$",
    ) as raised:
        _solve_and_simulate(model=model, marriage_rate=1.0)
    return str(raised.value)


def test_landing_without_a_feasible_action_raises_as_on_a_plain_edge() -> None:
    """Landing where the couple has no feasible action raises the plain-edge error.

    The couple is feasible only at the top of its wealth grid, so some married
    rows land where no action is feasible. The roles model refuses them with the
    same message as the model whose couple is a singleton entered by a plain
    edge, instead of routing them anywhere.
    """
    roles = _no_feasible_action_message(edge_form="roles")
    plain = _no_feasible_action_message(edge_form="plain")
    assert roles == plain


def _edges_with_single(
    *,
    single: Transition | Mapping[RegimeName, int],
    couple: Transition | Mapping[RegimeName, int] | None = None,
) -> dict[RegimeName, Transition | Mapping[RegimeName, int]]:
    return {"single": single, "couple": couple or {"couple_end": 1}}


def _model_with_edges(
    edges: Mapping[RegimeName, Transition | Mapping[RegimeName, int]],
) -> Model:
    return Model(
        regimes={
            "single": _single(),
            "couple": _couple(collective=True, infeasible_at_low_wealth=False),
            "single_end": Regime(
                states={"wealth": WEALTH}, functions={"utility": utility_single_end}
            ),
            "couple_end": _couple_end(collective=True),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "single"},
        edges=edges,
    )


def _marriage_law() -> ByAge:
    return ByAge.until(
        stop_age_exclusive=2,
        law={
            "single": StochasticTransition(func=stays_single),
            "couple": StochasticTransition(func=marries),
        },
        then="single_end",
        start_age_inclusive=0,
    )


@pytest.mark.parametrize(
    ("roles", "match"),
    [
        (
            {"couple": "f", "single_end": "f"},
            r"'single'.*'single_end'.*'f'.*not collective",
        ),
        (
            {"couple": "x"},
            r"'single'.*'couple'.*'x'.*members of 'couple' are \('f', 'm'\)",
        ),
        (
            {"couple": "f", "couple_end": "f"},
            r"'single'.*'couple_end'.*'f'.*not a target",
        ),
    ],
)
def test_roles_are_refused_where_the_role_cannot_apply(
    *, roles: Mapping[RegimeName, str], match: str
) -> None:
    """Each refusal names the source, the target and the role."""
    with pytest.raises(ModelInitializationError, match=match):
        _model_with_edges(
            _edges_with_single(single=Transition(law=_marriage_law(), roles=roles))
        )


def test_roles_are_refused_on_a_collective_source() -> None:
    """A collective source already carries a role per row."""
    with pytest.raises(
        ModelInitializationError, match=r"'couple'.*'couple_end'.*'f'.*collective"
    ):
        _model_with_edges(
            _edges_with_single(
                single=Transition(law=_marriage_law(), roles={"couple": "f"}),
                couple=Transition(law="couple_end", roles={"couple_end": "f"}),
            )
        )


def test_a_target_with_both_a_role_and_a_gate_is_refused() -> None:
    """`roles` is the always-open gate, so a target takes one or the other."""
    with pytest.raises(
        ModelInitializationError, match=r"'single'.*'couple'.*'f'.*both"
    ):
        _model_with_edges(
            _edges_with_single(
                single=Transition(
                    law=_marriage_law(),
                    gates={"couple": _hand_gate()},
                    roles={"couple": "f"},
                )
            )
        )


def test_a_plain_edge_into_a_collective_target_points_at_roles() -> None:
    """Without `roles`, a gate or a participation check, the edge stays refused."""
    with pytest.raises(NotImplementedError, match=r"members.*roles="):
        _model_with_edges(_edges_with_single(single=Transition(law=_marriage_law())))
