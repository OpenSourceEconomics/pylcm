"""A declared lower bound on a post-decision state is checked, not guessed.

An endogenous-grid solve enforces its borrowing limit through the savings grid:
the lowest node of that grid *is* the solve-time limit. A regime that states its
own limit is therefore making a claim about the grid, and
`post_decision_lower_bound` is how it states one that can be checked. The same
declaration remains executable against phase-resolved values in simulation.
Where the declaration and grid disagree, the model is refused and both numbers
are named, so a limit can never be quietly replaced by whichever value the
grid happens to start at.
"""

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.grids import ContinuousGrid
from lcm import (
    AgeGrid,
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    categorical,
    post_decision_lower_bound,
    ref,
)
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime
from lcm.solvers import DCEGM, GridSearch
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
    UserFunction,
)
from tests.conftest import DECIMAL_PRECISION

_LIQUID = LiquidMargin(
    state="wealth",
    action="consumption",
    resources="wealth",
    post_decision_state="savings",
)

_WEALTH_GRID = LinSpacedGrid(start=0.1, stop=4.0, n_points=8)
_ACTION_GRID = LinSpacedGrid(start=0.1, stop=4.0, n_points=8)


@categorical(ordered=False)
class RegimeId:
    saving: ScalarInt
    done: ScalarInt


@categorical(ordered=False)
class Work:
    working: ScalarInt
    retired: ScalarInt


def utility(consumption: ContinuousAction, work: DiscreteAction) -> FloatND:
    return jnp.log(consumption) + 0.0 * work


def terminal_utility(wealth: ContinuousState) -> FloatND:
    return jnp.log(jnp.clip(wealth, 1e-8))


def savings(wealth: FloatND, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption


def next_wealth(savings: FloatND) -> FloatND:
    return savings


def next_regime(_age: float) -> ScalarInt:
    return RegimeId.done


def _model(
    *,
    savings_grid_start: float = -2.0,
    declared_lower_bound: float | None,
    opaque_constraint: UserFunction | None = None,
    savings_grid: ContinuousGrid | None = None,
) -> Model:
    """Build a one-period DC-EGM model, optionally declaring its own limit."""
    constraints: dict[str, UserFunction] = {}
    if declared_lower_bound is not None:
        constraints["borrowing_limit"] = post_decision_lower_bound(
            margin=_LIQUID, lower=declared_lower_bound
        )
    if opaque_constraint is not None:
        constraints["borrowing_constraint"] = opaque_constraint
    saving_regime = ConsumptionSavingsRegime(
        actions={"consumption": _ACTION_GRID, "work": DiscreteGrid(Work)},
        states={"wealth": _WEALTH_GRID},
        state_transitions={"wealth": {"done": next_wealth}},
        constraints=constraints,
        transition=next_regime,
        functions={
            "utility": utility,
            "savings": savings,
        },
        active=lambda age: age == 0,
        solver=DCEGM(
            savings_grid=savings_grid
            if savings_grid is not None
            else LinSpacedGrid(start=savings_grid_start, stop=4.0, n_points=12)
        ),
        liquid=_LIQUID,
    )
    done_regime = Regime(
        actions={},
        transition=None,
        states={"wealth": _WEALTH_GRID},
        functions={"utility": terminal_utility},
        active=lambda age: age == 1,
        solver=GridSearch(),
    )
    return Model(
        regimes={"saving": saving_regime, "done": done_regime},
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=RegimeId,
    )


def test_a_declared_lower_bound_matching_the_grid_is_accepted() -> None:
    """Stating the limit the grid already implies builds the model."""
    model = _model(savings_grid_start=-2.0, declared_lower_bound=-2.0)

    assert "saving" in model.user_regimes


def test_a_declared_lower_bound_the_grid_dtype_rounds_is_accepted() -> None:
    """A limit whose decimal the grid's dtype cannot hold exactly still matches.

    The regime declares the same number the grid was built from, so the two
    agree by construction whatever the grid's floating-point format does to
    that decimal on the way in.
    """
    model = _model(savings_grid_start=-0.1, declared_lower_bound=-0.1)

    assert "saving" in model.user_regimes


def test_a_runtime_supplied_savings_grid_is_refused_for_supplying_no_points() -> None:
    """A savings grid whose points arrive with the params is refused as a grid.

    The declared bound is checked against the grid's lowest node, and such a
    grid has no node to check against at model construction. What the regime
    gets back names the grid to supply points for, rather than reporting a
    failure to read them.
    """
    with pytest.raises(ModelInitializationError, match="only at runtime"):
        _model(
            savings_grid=IrregSpacedGrid(n_points=12),
            declared_lower_bound=-2.0,
        )


def test_a_declared_lower_bound_below_the_grid_is_refused() -> None:
    """A limit the grid cannot reach is an error, not a silent substitution."""
    with pytest.raises(ModelInitializationError, match="lower bound"):
        _model(savings_grid_start=-0.5, declared_lower_bound=-2.0)


def test_the_refusal_names_the_declared_bound() -> None:
    """The message carries the number the regime declared."""
    with pytest.raises(ModelInitializationError) as excinfo:
        _model(savings_grid_start=-0.5, declared_lower_bound=-2.0)

    assert "-2.0" in str(excinfo.value)


def test_the_refusal_names_the_grids_lowest_node() -> None:
    """The message carries the number the grid actually enforces."""
    with pytest.raises(ModelInitializationError) as excinfo:
        _model(savings_grid_start=-0.5, declared_lower_bound=-2.0)

    assert "-0.5" in str(excinfo.value)


def test_a_declared_lower_bound_above_the_grid_is_refused() -> None:
    """A limit stricter than the grid's is equally a disagreement."""
    with pytest.raises(ModelInitializationError, match="lower bound"):
        _model(savings_grid_start=-2.0, declared_lower_bound=-0.5)


def test_declaring_nothing_still_builds() -> None:
    """The declaration is optional; the grid keeps supplying the limit."""
    model = _model(savings_grid_start=-2.0, declared_lower_bound=None)

    assert "saving" in model.user_regimes


def _opaque_borrowing_constraint(savings: FloatND) -> FloatND:
    """A borrowing limit written as an ordinary predicate, carrying no proof."""
    return savings >= -2.0


def test_a_plain_callable_borrowing_constraint_is_refused() -> None:
    """An opaque predicate carries no proof, so it stays outside the contract.

    Matches the refused constraint's *name* rather than the wording of the
    message. Wording is free to improve; which constraint was refused is the
    claim being made. Matching the sentence instead would go red on any
    rewrite and stay green if the wrong constraint were named — backwards on
    both counts.

    The refusal is also attributable to the constraint rather than to the
    model: it differs by exactly one argument from the builds asserted above,
    so an unrelated build failure would not satisfy it.
    """
    with pytest.raises(ModelInitializationError, match="borrowing_constraint"):
        _model(
            savings_grid_start=-2.0,
            declared_lower_bound=None,
            opaque_constraint=_opaque_borrowing_constraint,
        )


def _filled_params(model: Model) -> dict:
    """Fill every free leaf of the params template with a usable number.

    A free leaf carries its annotation as a string rather than `None`, so the
    test substitutes for anything that is not already numeric — an annotation
    left in place reaches the dtype cast and is rejected there.
    """
    template = model.get_params_template()

    def fill(node: object, name: str = "") -> object:
        if isinstance(node, dict):
            return {key: fill(value, key) for key, value in node.items()}
        if isinstance(node, bool) or not isinstance(node, int | float):
            return 0.0 if name == "_age" else 0.95
        return node

    return fill(template)  # ty: ignore[invalid-return-type]


def test_declaring_the_bound_does_not_change_the_solution() -> None:
    """A proved declaration is inert — the grid already enforces it."""
    declared = _model(savings_grid_start=-2.0, declared_lower_bound=-2.0)
    bare = _model(savings_grid_start=-2.0, declared_lower_bound=None)

    with_declaration = declared.solve(params=_filled_params(declared), log_level="off")
    without = bare.solve(params=_filled_params(bare), log_level="off")

    for period, regime_to_V in without.items():
        for regime_name, V_arr in regime_to_V.items():
            aaae(
                with_declaration[period][regime_name],
                V_arr,
                decimal=DECIMAL_PRECISION,
            )


def _grid_search_model(
    *,
    declared_lower_bound: float | None,
    hand_written: bool = False,
) -> Model:
    """The same regime solved by grid search, which enforces nothing implicitly."""
    if hand_written and declared_lower_bound is not None:
        return _replace_constraints(
            constraints={"borrowing_limit": ref("savings") >= declared_lower_bound},
        )
    constraints = (
        {}
        if declared_lower_bound is None
        else {
            "borrowing_limit": post_decision_lower_bound(
                margin=_LIQUID, lower=declared_lower_bound
            )
        }
    )
    saving_regime = Regime(
        actions={"consumption": _ACTION_GRID, "work": DiscreteGrid(Work)},
        states={"wealth": _WEALTH_GRID},
        state_transitions={"wealth": {"done": next_wealth}},
        constraints=constraints,
        transition=next_regime,
        functions={
            "utility": utility,
            "savings": savings,
        },
        active=lambda age: age == 0,
        solver=GridSearch(),
    )
    done_regime = Regime(
        actions={},
        transition=None,
        states={"wealth": _WEALTH_GRID},
        functions={"utility": terminal_utility},
        active=lambda age: age == 1,
        solver=GridSearch(),
    )
    return Model(
        regimes={"saving": saving_regime, "done": done_regime},
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=RegimeId,
    )


def test_grid_search_keeps_the_declaration_as_a_real_constraint() -> None:
    """Grid search enforces nothing implicitly, so the bound must bind there.

    The same declaration is redundant under an endogenous-grid solver and
    load-bearing under grid search, so one spelling has to serve both arms of a
    model. Restricting the feasible set can only lower the attained value, so
    the declared solve must fall strictly below the unrestricted one somewhere;
    were the declaration dropped here too, the two would coincide.
    """
    declared = _grid_search_model(declared_lower_bound=1.0)
    bare = _grid_search_model(declared_lower_bound=None)

    with_declaration = declared.solve(params=_filled_params(declared), log_level="off")
    without = bare.solve(params=_filled_params(bare), log_level="off")

    # A binding constraint shrinks the feasible set, so the value it yields can
    # only fall. Asserting merely that the two differ would also accept a
    # declaration that perturbed the solution some other way.
    gaps = [
        float(jnp.max(V_arr - with_declaration[period][name]))
        for period, regime_to_V in without.items()
        for name, V_arr in regime_to_V.items()
    ]
    assert max(gaps) > 0.0


def _replace_constraints(*, constraints: dict) -> Model:
    """Build the grid-search model with an explicitly supplied constraint pool."""
    saving_regime = Regime(
        actions={"consumption": _ACTION_GRID, "work": DiscreteGrid(Work)},
        states={"wealth": _WEALTH_GRID},
        state_transitions={"wealth": {"done": next_wealth}},
        constraints=constraints,
        transition=next_regime,
        functions={
            "utility": utility,
            "savings": savings,
        },
        active=lambda age: age == 0,
        solver=GridSearch(),
    )
    done_regime = Regime(
        actions={},
        transition=None,
        states={"wealth": _WEALTH_GRID},
        functions={"utility": terminal_utility},
        active=lambda age: age == 1,
        solver=GridSearch(),
    )
    return Model(
        regimes={"saving": saving_regime, "done": done_regime},
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=RegimeId,
    )


def test_a_hand_written_condition_matches_the_declared_bound() -> None:
    """Writing the comparison out means what the convenience constructor means.

    `post_decision_lower_bound` is a way to spell a comparison, not a
    privileged form of one: an author who writes `ref("savings") >= 1.0`
    directly declares the same feasible set and must get the same solution.
    Were the two to diverge, the sugar would carry a meaning its own expansion
    does not.
    """
    sugared = _grid_search_model(declared_lower_bound=1.0)
    hand_written = _grid_search_model(declared_lower_bound=1.0, hand_written=True)

    from_sugar = sugared.solve(params=_filled_params(sugared), log_level="off")
    from_hand = hand_written.solve(params=_filled_params(hand_written), log_level="off")

    for period, regime_to_V in from_sugar.items():
        for regime_name, V_arr in regime_to_V.items():
            aaae(from_hand[period][regime_name], V_arr, decimal=DECIMAL_PRECISION)
