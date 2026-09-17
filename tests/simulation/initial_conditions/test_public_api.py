"""Public `Model` methods for validating initial conditions without simulating."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import (
    AgeGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    categorical,
)
from lcm.exceptions import (
    InvalidInitialConditionsError,
    ModelSealError,
    UnsupportedOperationError,
)
from lcm.regime import Regime as UserRegime
from lcm.transition import AgeSpecializedFunction
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND, ScalarInt
from tests.simulation.initial_conditions._models import (
    make_asymmetric_state_model,
    make_constraint_model,
    make_joint_constraint_model,
    make_period_constraint_model,
    make_state_only_constraint_model,
)
from tests.simulation.initial_conditions._oracle import exhaustive_scalar_feasibility

_CONSTRAINT_PARAMS = {
    "discount_factor": 0.95,
    "working_life": {"next_regime": {"final_age_alive": 1}},
}


def _constraint_model() -> Model:
    return make_constraint_model(
        wealth_grid=LinSpacedGrid(start=2.0, stop=10, n_points=15)
    )


def _mixed_population(model: Model) -> dict[str, jnp.ndarray]:
    working_life = model.regime_names_to_ids["working_life"]
    return {
        "age": jnp.array([0.0, 0.0, 0.0, 0.0]),
        "wealth": jnp.array([0.25, 5.0, 0.3, 7.0]),
        "regime_id": jnp.array([working_life] * 4),
    }


def test_feasibility_mask_reports_each_subject_in_caller_order() -> None:
    """`True` exactly for subjects admitting an action, in the order supplied."""
    model = _constraint_model()

    mask = model.initial_conditions_feasibility(
        initial_conditions=_mixed_population(model), params=_CONSTRAINT_PARAMS
    )

    assert mask.tolist() == [False, True, False, True]


def test_feasibility_mask_is_a_one_dimensional_boolean_array() -> None:
    """The mask is a boolean JAX array with one entry per subject."""
    model = _constraint_model()

    mask = model.initial_conditions_feasibility(
        initial_conditions=_mixed_population(model), params=_CONSTRAINT_PARAMS
    )

    assert (mask.shape, mask.dtype) == ((4,), jnp.dtype(bool))


def test_feasibility_mask_agrees_with_exhaustive_scalar_oracle() -> None:
    """The vectorized verdict equals brute-force scalar enumeration of the grid."""
    model = make_joint_constraint_model()
    df = pd.DataFrame(
        {
            "regime_name": ["alive"] * 6,
            "age": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "wealth": [1.0, 3.0, 4.0, 3.5, 12.0, 0.5],
        }
    )
    params = {"discount_factor": 0.95, "alive": {"at_least": {"lower": 4.0}}}

    mask = model.initial_conditions_feasibility(initial_conditions=df, params=params)

    np.testing.assert_array_equal(
        np.asarray(mask),
        exhaustive_scalar_feasibility(
            model=model, initial_conditions=df, params=params
        ),
    )


def test_jointly_impossible_constraints_mark_subject_infeasible() -> None:
    """Constraints each satisfiable alone but jointly impossible yield `False`."""
    model = make_joint_constraint_model()
    alive = model.regime_names_to_ids["alive"]

    mask = model.initial_conditions_feasibility(
        initial_conditions={
            "age": jnp.array([0.0, 0.0]),
            "wealth": jnp.array([3.0, 5.0]),
            "regime_id": jnp.array([alive, alive]),
        },
        params={"discount_factor": 0.95, "alive": {"at_least": {"lower": 4.0}}},
    )

    assert mask.tolist() == [False, True]


def test_period_constraint_binds_fixed_and_runtime_scalars() -> None:
    """A constraint reading `period`, a fixed and a runtime scalar evaluates per age.

    With `cap = 2` and fixed `floor = 0.5`, a subject with wealth 1 affords the
    minimum consumption of 1 only once `period * cap >= 0.5`, i.e. from age 1.
    """
    model = make_period_constraint_model()
    alive = model.regime_names_to_ids["alive"]

    mask = model.initial_conditions_feasibility(
        initial_conditions={
            "age": jnp.array([0.0, 1.0, 2.0]),
            "wealth": jnp.array([1.0, 1.0, 1.0]),
            "regime_id": jnp.array([alive, alive, alive]),
        },
        params={"discount_factor": 0.95, "alive": {"affordable": {"cap": 2.0}}},
    )

    assert mask.tolist() == [False, True, True]


def test_validate_raises_exactly_when_mask_has_a_false() -> None:
    """`validate_initial_conditions` rejects the population iff a subject is `False`."""
    model = _constraint_model()

    with pytest.raises(InvalidInitialConditionsError, match="infeasible for 2 subject"):
        model.validate_initial_conditions(
            initial_conditions=_mixed_population(model), params=_CONSTRAINT_PARAMS
        )


def test_validate_accepts_an_all_feasible_population() -> None:
    """`validate_initial_conditions` returns `None` when every subject is feasible."""
    model = _constraint_model()
    working_life = model.regime_names_to_ids["working_life"]

    result = model.validate_initial_conditions(
        initial_conditions={
            "age": jnp.array([0.0, 0.0]),
            "wealth": jnp.array([5.0, 7.0]),
            "regime_id": jnp.array([working_life, working_life]),
        },
        params=_CONSTRAINT_PARAMS,
    )

    assert result is None


def test_validate_message_matches_simulation_diagnostics() -> None:
    """The validator's message is the one `simulate(log_level="debug")` raises."""
    model = _constraint_model()
    initial = _mixed_population(model)

    with pytest.raises(InvalidInitialConditionsError) as from_validate:
        model.validate_initial_conditions(
            initial_conditions=initial, params=_CONSTRAINT_PARAMS
        )
    with pytest.raises(InvalidInitialConditionsError) as from_simulate:
        model.simulate(
            initial_conditions=initial, params=_CONSTRAINT_PARAMS, log_level="debug"
        )

    assert str(from_validate.value) == str(from_simulate.value)


def test_dataframe_and_mapping_inputs_give_the_same_mask() -> None:
    """A DataFrame with `regime_name` and ages yields the mapping form's mask."""
    model = make_state_only_constraint_model()
    alive = model.regime_names_to_ids["alive"]
    dead = model.regime_names_to_ids["dead"]
    params = {"discount_factor": 0.95}

    from_mapping = model.initial_conditions_feasibility(
        initial_conditions={
            "age": jnp.array([0.0, 2.0, 1.0, 2.0]),
            "wealth": jnp.array([0.5, -1.0, 60.0, 5.0]),
            "regime_id": jnp.array([alive, dead, alive, dead]),
        },
        params=params,
    )
    from_dataframe = model.initial_conditions_feasibility(
        initial_conditions=pd.DataFrame(
            {
                "regime_name": ["alive", "dead", "alive", "dead"],
                "age": [0.0, 2.0, 1.0, 2.0],
                "wealth": [0.5, -1.0, 60.0, 5.0],
            }
        ),
        params=params,
    )

    assert (
        from_dataframe.tolist() == from_mapping.tolist() == [False, False, True, True]
    )


def test_dataframe_with_labels_and_regime_specific_states_is_accepted() -> None:
    """Categorical labels and `NA` for states absent in a row's regime are handled."""
    model = make_asymmetric_state_model()

    mask = model.initial_conditions_feasibility(
        initial_conditions=pd.DataFrame(
            {
                "regime_name": ["alive", "dead", "alive"],
                "age": [0.0, 2.0, 1.0],
                "wealth": [10.0, 50.0, 20.0],
                "health": ["healthy", pd.NA, "sick"],
            }
        ),
        params={"discount_factor": 0.95},
    )

    assert mask.tolist() == [True, True, True]


@pytest.mark.parametrize(
    ("initial_conditions", "match"),
    [
        (
            {"wealth": jnp.array([5.0]), "regime_id": jnp.array([0])},
            "'age' must be provided",
        ),
        (
            {
                "age": jnp.array([0.0]),
                "wealth": jnp.array([5.0]),
                "regime_id": jnp.array([99]),
            },
            "Invalid regime IDs",
        ),
        (
            {
                "age": jnp.array([0.0, 0.0]),
                "wealth": jnp.array([5.0]),
                "regime_id": jnp.array([0, 0]),
            },
            "same length",
        ),
        (
            {
                "age": jnp.array([0.0]),
                "wealth": jnp.array([5.0]),
                "debt": jnp.array([1.0]),
                "regime_id": jnp.array([0]),
            },
            "Unknown initial states",
        ),
    ],
)
@pytest.mark.parametrize(
    "method", ["validate_initial_conditions", "initial_conditions_feasibility"]
)
def test_malformed_inputs_raise_from_both_methods(
    *, initial_conditions: dict[str, jnp.ndarray], match: str, method: str
) -> None:
    """Structural defects raise instead of being classified as infeasible."""
    model = _constraint_model()

    with pytest.raises(InvalidInitialConditionsError, match=match):
        getattr(model, method)(
            initial_conditions=initial_conditions, params=_CONSTRAINT_PARAMS
        )


@pytest.mark.parametrize(
    "method", ["validate_initial_conditions", "initial_conditions_feasibility"]
)
def test_unknown_categorical_label_raises(method: str) -> None:
    """An unknown categorical label is rejected, not marked infeasible."""
    model = make_asymmetric_state_model()

    with pytest.raises(ValueError, match="health"):
        getattr(model, method)(
            initial_conditions=pd.DataFrame(
                {
                    "regime_name": ["alive"],
                    "age": [0.0],
                    "wealth": [10.0],
                    "health": ["immortal"],
                }
            ),
            params={"discount_factor": 0.95},
        )


def test_runtime_supplied_grid_points_are_used() -> None:
    """Runtime-supplied irregular grid points bound the feasible set."""
    model = make_constraint_model(wealth_grid=IrregSpacedGrid(n_points=15))
    working_life = model.regime_names_to_ids["working_life"]

    mask = model.initial_conditions_feasibility(
        initial_conditions={
            "age": jnp.array([0.0, 0.0]),
            "wealth": jnp.array([0.3, 5.0]),
            "regime_id": jnp.array([working_life, working_life]),
        },
        params={
            "discount_factor": 0.95,
            "working_life": {
                "wealth": {"points": jnp.linspace(0.3, 10, 15)},
                "next_regime": {"final_age_alive": 1},
            },
        },
    )

    assert mask.tolist() == [False, True]


def test_inputs_are_not_mutated() -> None:
    """Neither the initial-condition arrays nor the params are changed by the call."""
    model = _constraint_model()
    initial = _mixed_population(model)
    before = {name: np.asarray(value).copy() for name, value in initial.items()}
    params = {
        "discount_factor": 0.95,
        "working_life": {"next_regime": {"final_age_alive": 1}},
    }
    params_before = repr(params)

    model.initial_conditions_feasibility(initial_conditions=initial, params=params)

    assert repr(params) == params_before
    for name, value in initial.items():
        np.testing.assert_array_equal(np.asarray(value), before[name])


@pytest.mark.parametrize(
    "method", ["validate_initial_conditions", "initial_conditions_feasibility"]
)
def test_methods_never_solve_or_simulate(
    *, method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public checks do not call `Model.solve` or `Model.simulate`."""
    model = _constraint_model()
    working_life = model.regime_names_to_ids["working_life"]

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("A feasibility check solved or simulated the model.")

    monkeypatch.setattr(Model, "solve", forbidden)
    monkeypatch.setattr(Model, "simulate", forbidden)

    getattr(model, method)(
        initial_conditions={
            "age": jnp.array([0.0]),
            "wealth": jnp.array([5.0]),
            "regime_id": jnp.array([working_life]),
        },
        params=_CONSTRAINT_PARAMS,
    )


def test_feasibility_mask_ignores_a_device_memory_budget() -> None:
    """The mask is computed eagerly even when the model carries a tiny memory budget."""
    model = make_state_only_constraint_model(device_memory_bytes=1024)
    dead = model.regime_names_to_ids["dead"]

    mask = model.initial_conditions_feasibility(
        initial_conditions={
            "age": jnp.array([2.0, 2.0]),
            "wealth": jnp.array([-1.0, 5.0]),
            "regime_id": jnp.array([dead, dead]),
        },
        params={"discount_factor": 0.95},
    )

    assert mask.tolist() == [False, True]


_UTILITY_SCALE = jnp.asarray(1.0)


def _sealed_model() -> Model:
    @categorical(ordered=False)
    class RegimeId:
        working: ScalarInt
        dead: ScalarInt

    def utility(consumption: ContinuousAction) -> FloatND:
        return _UTILITY_SCALE * jnp.log(consumption)

    def feasible(*, wealth: ContinuousState, consumption: ContinuousAction) -> BoolND:
        return consumption <= wealth

    def next_wealth(
        *, wealth: ContinuousState, consumption: ContinuousAction
    ) -> ContinuousState:
        return wealth - consumption + 1.0

    def next_regime(age: float) -> ScalarInt:
        return jnp.where(age >= 18, RegimeId.dead, RegimeId.working)

    working = UserRegime(
        transition=next_regime,
        states={"wealth": LinSpacedGrid(start=1, stop=3, n_points=3)},
        state_transitions={"wealth": next_wealth},
        actions={"consumption": LinSpacedGrid(start=0.5, stop=2.5, n_points=3)},
        functions={"utility": utility},
        constraints={"feasible": feasible},
        active=lambda age: age < 19,
    )
    dead = UserRegime(transition=None, functions={"utility": lambda: 0.0})
    return Model(
        regimes={"working": working, "dead": dead},
        ages=AgeGrid(start=18, stop=20, step="Y"),
        regime_id_class=RegimeId,
    )


@pytest.fixture
def restore_scale() -> object:
    yield None
    globals()["_UTILITY_SCALE"] = jnp.asarray(1.0)


@pytest.mark.parametrize(
    "method", ["validate_initial_conditions", "initial_conditions_feasibility"]
)
def test_rebound_global_after_build_is_refused(
    *, method: str, restore_scale: object
) -> None:
    """A model whose sealed bindings moved refuses the check like `solve` does."""
    del restore_scale
    model = _sealed_model()
    working = model.regime_names_to_ids["working"]
    globals()["_UTILITY_SCALE"] = jnp.asarray(7.0)

    with pytest.raises(ModelSealError, match="_UTILITY_SCALE"):
        getattr(model, method)(
            initial_conditions={
                "age": jnp.array([18.0]),
                "wealth": jnp.array([2.0]),
                "regime_id": jnp.array([working]),
            },
            params={"discount_factor": 0.95},
        )


def _age_specialized_model() -> Model:
    @categorical(ordered=False)
    class RegimeId:
        working_life: ScalarInt
        dead: ScalarInt

    def utility(consumption: ContinuousAction) -> FloatND:
        return jnp.log(consumption)

    def feasible(*, consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
        return consumption <= wealth

    def cap_of_age(age: float):
        def wealth_cap(
            *, consumption: ContinuousAction, wealth: ContinuousState
        ) -> BoolND:
            return consumption <= wealth + age

        return wealth_cap

    def next_wealth(
        *, wealth: ContinuousState, consumption: ContinuousAction
    ) -> ContinuousState:
        return wealth - consumption + 1.0

    def next_regime(age: float) -> ScalarInt:
        return jnp.where(age >= 65, RegimeId.dead, RegimeId.working_life)

    working_life = UserRegime(
        transition=next_regime,
        active=lambda age: age < 75,
        states={"wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=8)},
        actions={"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        state_transitions={"wealth": next_wealth},
        constraints={
            "feasible": feasible,
            "wealth_cap": AgeSpecializedFunction(
                build=cap_of_age, signature=lambda age: age
            ),
        },
        functions={"utility": utility},
    )
    dead = UserRegime(
        transition=None,
        active=lambda age: age >= 75,
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"working_life": working_life, "dead": dead},
        ages=AgeGrid(start=25, stop=75, step="10Y"),
        regime_id_class=RegimeId,
    )


@pytest.mark.parametrize(
    "method", ["validate_initial_conditions", "initial_conditions_feasibility"]
)
def test_unsupported_age_specialized_check_raises(method: str) -> None:
    """An age-specialized constraint with off-representative starts is unsupported."""
    model = _age_specialized_model()
    working_life = model.regime_names_to_ids["working_life"]

    with pytest.raises(UnsupportedOperationError, match="policy-specialized"):
        getattr(model, method)(
            initial_conditions={
                "age": jnp.array([25.0, 35.0]),
                "wealth": jnp.array([10.0, 50.0]),
                "regime_id": jnp.array([working_life, working_life]),
            },
            params={"discount_factor": 0.95},
        )


def test_budget_is_irrelevant_to_the_public_mask_but_not_to_simulate() -> None:
    """The same budgeted model simulates the feasible subjects the mask selects."""
    model = make_state_only_constraint_model(device_memory_bytes=2**24)
    dead = model.regime_names_to_ids["dead"]
    params = {"discount_factor": 0.95}
    initial = {
        "age": jnp.array([2.0, 2.0, 2.0]),
        "wealth": jnp.array([-1.0, 5.0, 8.0]),
        "regime_id": jnp.array([dead, dead, dead]),
    }

    mask = np.asarray(
        model.initial_conditions_feasibility(initial_conditions=initial, params=params)
    )
    result = model.simulate(
        params=params,
        initial_conditions={name: value[mask] for name, value in initial.items()},
        log_level="debug",
    )

    assert result.to_dataframe()["wealth"].tolist() == [5.0, 8.0]
