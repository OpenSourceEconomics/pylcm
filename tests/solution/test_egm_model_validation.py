"""Model-stage applicability checks for the one-dimensional `EGM` solver."""

import jax.numpy as jnp
import pytest

from _lcm.egm.validation import _grid_sample
from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
    post_decision_lower_bound,
)
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime
from lcm.solvers import EGM, GridSearch
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)

_WEALTH_GRID = LinSpacedGrid(start=0.1, stop=4.0, n_points=8)
_ACTION_GRID = LinSpacedGrid(start=0.1, stop=4.0, n_points=8)
_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=4.0, n_points=12)


@categorical(ordered=False)
class RegimeId:
    saving: ScalarInt
    done: ScalarInt


@categorical(ordered=False)
class PreferenceType:
    patient: ScalarInt
    impatient: ScalarInt


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def terminal_utility(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def terminal_utility_with_action(
    *, wealth: ContinuousState, bequest: ContinuousAction
) -> FloatND:
    return jnp.log(bequest) + 0.0 * wealth


def terminal_utility_with_type(
    *, wealth: ContinuousState, preference_type: DiscreteState
) -> FloatND:
    return jnp.log(wealth) + 0.01 * preference_type


def savings(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption


def doubled_savings(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> FloatND:
    return 2.0 * (wealth - consumption)


def next_wealth(savings: FloatND) -> ContinuousState:
    return savings


def next_regime() -> ScalarInt:
    return RegimeId.done


def consumption_cap(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> BoolND:
    return consumption <= 0.2 * wealth


def _model(
    *,
    constraint=None,
    post_decision=savings,
    discrete_state: bool = False,
    terminal_action: bool = False,
    resources_func=None,
) -> Model:
    states = {"wealth": _WEALTH_GRID}
    state_transitions = {"wealth": {"done": next_wealth}}
    done_states = {"wealth": _WEALTH_GRID}
    done_functions = {"utility": terminal_utility}
    done_actions = {}
    if discrete_state:
        states["preference_type"] = DiscreteGrid(category_class=PreferenceType)
        state_transitions["preference_type"] = fixed_transition("preference_type")
        done_states["preference_type"] = DiscreteGrid(category_class=PreferenceType)
        done_functions = {"utility": terminal_utility_with_type}
    if terminal_action:
        done_actions = {"bequest": _ACTION_GRID}
        done_functions = {"utility": terminal_utility_with_action}

    saving_regime = ConsumptionSavingsRegime(
        actions={"consumption": _ACTION_GRID},
        states=states,
        state_transitions=state_transitions,
        constraints={} if constraint is None else {"cap": constraint},
        transition=next_regime,
        functions={
            "utility": utility,
            "savings": post_decision,
            **({"resources": resources_func} if resources_func else {}),
        },
        active=lambda age: age == 0,
        solver=EGM(
            savings_grid=_SAVINGS_GRID,
        ),
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources=("resources" if resources_func else "wealth"),
            post_decision_state="savings",
        ),
    )
    done_regime = Regime(
        actions=done_actions,
        transition=None,
        states=done_states,
        functions=done_functions,
        active=lambda age: age == 1,
        solver=GridSearch(),
    )
    return Model(
        regimes={"saving": saving_regime, "done": done_regime},
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=RegimeId,
    )


def test_egm_rejects_a_continuous_constraint_during_model_construction() -> None:
    """Plain EGM refuses a consumption cap that its numerical kernel cannot read.

    Matches the refused constraint's name rather than the wording, so the test
    survives any rewrite of the message and fails if a different constraint is
    named — which is the claim worth pinning.
    """
    with pytest.raises(ModelInitializationError, match="'cap'"):
        _model(constraint=consumption_cap)


def test_egm_accepts_a_lower_bound_enforced_by_its_savings_grid() -> None:
    """A declared borrowing limit builds when the EGM grid proves the same bound."""
    model = _model(
        constraint=post_decision_lower_bound(
            margin=LiquidMargin(
                state="wealth",
                action="consumption",
                resources="wealth",
                post_decision_state="savings",
            ),
            lower=0.0,
        )
    )

    assert "saving" in model.user_regimes


def test_egm_rejects_an_incompatible_post_decision_identity() -> None:
    """Post-decision assets must equal resources minus consumption."""
    with pytest.raises(ModelInitializationError, match="Consumption recovery fails"):
        _model(post_decision=doubled_savings)


def test_egm_rejects_a_discrete_state_during_model_construction() -> None:
    """A one-row EGM continuation cannot carry a discrete preference-type axis."""
    with pytest.raises(ModelInitializationError, match="discrete or process states"):
        _model(discrete_state=True)


def test_egm_rejects_a_terminal_target_with_an_action() -> None:
    """A terminal carry cannot stand in for a final-period optimization."""
    with pytest.raises(
        ModelInitializationError, match=r"terminal regime 'done'.*actions"
    ):
        _model(terminal_action=True)


def _sampled_wealth_nodes() -> list[float]:
    """The state nodes the post-decision spot check actually visits."""
    return [float(v) for v in _grid_sample(grid=_WEALTH_GRID)]


def test_resources_is_checked_at_every_represented_state_node() -> None:
    """A resources bump at an unvisited node is caught, not admitted.

    The spot check evaluates resources at every tabulated state, so a map that
    equals the state wherever a five-point sample would look and departs from
    it elsewhere is still rejected. It remains a diagnostic — nothing here
    establishes the identity between nodes — but a mistake at a represented
    state can no longer slip through on sampling alone.
    """
    nodes = jnp.asarray(_WEALTH_GRID.to_jax())
    unvisited = jnp.asarray(
        [n for n in nodes.tolist() if n not in _sampled_wealth_nodes()]
    )
    assert unvisited.size > 0, "grid too small to have an unvisited node"

    def bumped_resources(wealth: ContinuousState) -> FloatND:
        at_unvisited = jnp.any(jnp.abs(wealth[..., None] - unvisited) < 1e-9, axis=-1)
        return wealth + 0.25 * at_unvisited

    with pytest.raises(ModelInitializationError, match="must equal the liquid state"):
        _model(resources_func=bumped_resources)


def test_the_identity_resources_map_still_builds() -> None:
    """The control: an unperturbed model is unaffected by the wider check."""
    model = _model()

    assert "saving" in model.user_regimes
