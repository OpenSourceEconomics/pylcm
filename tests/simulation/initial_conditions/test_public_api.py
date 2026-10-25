"""Public `Model` methods for validating initial conditions without simulating."""

from collections.abc import Callable, Generator, Mapping
from fractions import Fraction
from inspect import signature

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax import Array

from _lcm.engine import Regime as InternalRegime
from _lcm.simulation.current_iid_entry import SeededEntryFrames
from _lcm.simulation.initial_conditions import (
    _regime_feasibility_function,
    pad_initial_conditions_to_multiple,
    validate_simulation_inputs,
)
from _lcm.simulation.memory import SimulationMemory
from _lcm.time import TimeAxis
from _lcm.utils.logging import get_logger
from lcm import (
    AgeGrid,
    DeterministicTransition,
    InitialNodes,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    Transition,
    categorical,
)
from lcm.exceptions import (
    InvalidInitialConditionsError,
    InvalidSimulationInputError,
    ModelSealError,
    NoFeasibleActionError,
    ShockDeclarationError,
    UnsupportedOperationError,
)
from lcm.regime import Regime as UserRegime
from lcm.shocks import (
    BaseInnovations,
    Normal,
    PhysicalValues,
    ShockGroup,
    Uniform,
    UserNodes,
)
from lcm.transition import AgeSpecializedFunction
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    IntND,
    ReferenceName,
    RegimeName,
    ScalarInt,
)
from tests.regime_building.test_collective_regime_simulate import (
    _DISSOLUTION_PARAMS as DISSOLUTION_PARAMS,
)
from tests.regime_building.test_collective_regime_simulate import DissolutionRegimeId
from tests.regime_building.test_collective_regime_simulate import (
    _make_dissolution_regimes as make_dissolution_regimes,
)
from tests.regime_building.test_collective_regime_simulate import (
    _married_dissolution_transition as married_dissolution_transition,
)
from tests.simulation.initial_conditions._models import (
    make_asymmetric_state_model,
    make_constraint_model,
    make_joint_constraint_model,
    make_period_constraint_model,
    make_state_only_constraint_model,
)
from tests.simulation.initial_conditions._oracle import exhaustive_scalar_feasibility

_CONSTRAINT_PARAMS = {"discount_factor": 0.95, "final_age_alive": 1}


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
    """A state grid declared with runtime points can be checked through the public path.

    The runtime `wealth` points must reach `state_action_space` for the grid to be
    built at all; the verdict itself follows from the consumption grid's minimum
    against the supplied wealth values, not from the wealth grid.
    """
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
            "working_life": {"wealth": {"points": jnp.linspace(0.3, 10, 15)}},
            "final_age_alive": 1,
        },
    )

    assert mask.tolist() == [False, True]


def test_inputs_are_not_mutated() -> None:
    """Neither the initial-condition arrays nor the params are changed by the call."""
    model = _constraint_model()
    initial = _mixed_population(model)
    before = {name: np.asarray(value).copy() for name, value in initial.items()}
    params = {"discount_factor": 0.95, "final_age_alive": 1}
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

    def forbidden[Ignored](*args: Ignored, **kwargs: Ignored) -> None:
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

    working = UserRegime(
        states={"wealth": LinSpacedGrid(start=1, stop=3, n_points=3)},
        state_transitions={"wealth": next_wealth},
        actions={"consumption": LinSpacedGrid(start=0.5, stop=2.5, n_points=3)},
        functions={"utility": utility},
        constraints={"feasible": feasible},
    )
    dead = UserRegime(functions={"utility": lambda: 0.0})
    return Model(
        regimes={"working": working, "dead": dead},
        ages=AgeGrid(start=18, inclusive_stop=20, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={18: "working"},
        edges={"working": {"dead": 18}},
    )


@pytest.fixture
def restore_scale() -> Generator[None]:
    yield None
    globals()["_UTILITY_SCALE"] = jnp.asarray(1.0)


@pytest.mark.parametrize(
    "method", ["validate_initial_conditions", "initial_conditions_feasibility"]
)
def test_rebound_global_after_build_is_refused(
    *, method: str, restore_scale: None
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
            return consumption <= wealth - (age - 25.0)

        return wealth_cap

    def next_wealth(
        *, wealth: ContinuousState, consumption: ContinuousAction
    ) -> ContinuousState:
        return wealth - consumption + 1.0

    def next_regime(age: float) -> ScalarInt:
        return jnp.where(age >= 65, RegimeId.dead, RegimeId.working_life)

    working_life = UserRegime(
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
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"working_life": working_life, "dead": dead},
        ages=AgeGrid(start=25, inclusive_stop=75, step="10Y"),
        regime_id_class=RegimeId,
        initial_nodes={25: "working_life", 35: "working_life"},
        edges={
            "working_life": Transition(
                targets={
                    "working_life": (25, 35, 45, 55),
                    "dead": (25, 35, 45, 55, 65),
                },
                law=DeterministicTransition(func=next_regime),
            )
        },
    )


def test_age_specialized_constraint_admits_starts_at_every_active_age() -> None:
    """Subjects starting at different ages are each checked at their own start age.

    Both supplied rows admit an action under the cap at their own start age.
    """
    model = _age_specialized_model()
    working_life = model.regime_names_to_ids["working_life"]

    mask = model.initial_conditions_feasibility(
        initial_conditions={
            "age": jnp.array([25.0, 35.0]),
            "wealth": jnp.array([10.0, 50.0]),
            "regime_id": jnp.array([working_life, working_life]),
        },
        params={"discount_factor": 0.95},
    )

    assert np.asarray(mask).tolist() == [True, True]


def _start_age_scalar_feasibility(*, age: float, wealth: float) -> bool:
    """Enumerate this fixture's five actions using exact scalar arithmetic.

    This reference uses neither the model's constraint closures nor its period
    pools. All action nodes are exact binary fractions. At the tested boundary,
    subtracting ten from wealth is exact in both supported floating precisions.
    """
    age_exact = Fraction(age)
    wealth_exact = Fraction(wealth)
    for consumption in (
        Fraction(1),
        Fraction(13, 4),
        Fraction(11, 2),
        Fraction(31, 4),
        Fraction(10),
    ):
        if consumption <= wealth_exact and consumption <= (
            wealth_exact - (age_exact - 25)
        ):
            return True
    return False


@pytest.mark.parametrize(
    "order",
    [(0, 1, 2, 3, 4, 5), (5, 4, 3, 2, 1, 0), (2, 0, 5, 1, 4, 3)],
)
def test_age_specialized_public_checks_use_binding_caps_in_caller_order(
    *,
    order: tuple[int, ...],
) -> None:
    """Both public routes use the start-age cap, including adjacent boundaries."""
    model = _age_specialized_model()
    working_life = model.regime_names_to_ids["working_life"]
    dtype = np.asarray(jnp.asarray(11.0)).dtype
    boundary = dtype.type(11.0)
    below = np.nextafter(boundary, dtype.type(-np.inf))
    above = np.nextafter(boundary, dtype.type(np.inf))
    ages = np.asarray([35.0, 25.0, 35.0, 25.0, 35.0, 35.0], dtype=dtype)
    wealth = np.asarray([10.0, 1.0, boundary, 10.0, below, above], dtype=dtype)
    indices = np.asarray(order)
    ages, wealth = ages[indices], wealth[indices]
    expected = np.asarray(
        [
            _start_age_scalar_feasibility(age=float(age), wealth=float(value))
            for age, value in zip(ages, wealth, strict=True)
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(
        expected, np.asarray([False, True, True, True, False, True])[indices]
    )
    initial = {
        "age": jnp.asarray(ages),
        "wealth": jnp.asarray(wealth),
        "regime_id": jnp.full(len(ages), working_life),
    }
    params = {"discount_factor": 0.95}

    mask = model.initial_conditions_feasibility(
        initial_conditions=initial, params=params
    )
    assert np.asarray(mask).dtype == np.dtype(bool)
    assert mask.shape == (len(ages),)
    np.testing.assert_array_equal(np.asarray(mask), expected)

    with pytest.raises(
        InvalidInitialConditionsError, match="satisfies all constraints jointly"
    ):
        model.validate_initial_conditions(initial_conditions=initial, params=params)

    admitted = {
        name: jnp.asarray(np.asarray(values)[expected])
        for name, values in initial.items()
    }
    verdict = model.validate_initial_conditions(
        initial_conditions=admitted, params=params
    )
    assert verdict is None
    np.testing.assert_array_equal(
        np.asarray(
            model.initial_conditions_feasibility(
                initial_conditions=admitted, params=params
            )
        ),
        np.ones(int(expected.sum()), dtype=bool),
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


def _collective_population_without_roles(model: Model) -> dict[str, jnp.ndarray]:
    married = model.regime_names_to_ids["married"]
    return {
        "wage": jnp.asarray((1.0, 2.0, 3.0)),
        "age": jnp.zeros(3),
        "regime_id": jnp.full(3, married, dtype=jnp.int32),
    }


@pytest.mark.parametrize(
    "method", ["validate_initial_conditions", "initial_conditions_feasibility"]
)
def test_collective_start_without_roles_is_rejected_like_simulate(method: str) -> None:
    """A population `simulate` refuses for a missing role is refused by both methods."""
    model = Model(
        regimes=make_dissolution_regimes(),
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=DissolutionRegimeId,
        initial_nodes={0: "married"},
        edges={
            "married": married_dissolution_transition(),
            "married_ir": {"married_terminal": 1},
            "single_f": {"single_f_terminal": 1},
            "single_m": {"single_m_terminal": (0, 1, 2)},
        },
    )
    initial = _collective_population_without_roles(model)

    with pytest.raises(InvalidInitialConditionsError) as from_simulate:
        model.simulate(
            params=DISSOLUTION_PARAMS,
            initial_conditions=initial,
            log_level="debug",
            seed=0,
        )
    with pytest.raises(
        InvalidInitialConditionsError, match="own_stakeholder"
    ) as from_method:
        getattr(model, method)(initial_conditions=initial, params=DISSOLUTION_PARAMS)

    assert str(from_method.value) == str(from_simulate.value)


def _tagged_current_entry_model(
    *,
    shock_constraint: bool = True,
    positive_shock: bool = False,
    enable_jit: bool = False,
) -> Model:
    """A terminal current-shock problem with independent literal entry oracles."""

    @categorical(ordered=False)
    class EntryRegimeId:
        alive: ScalarInt

    def utility(*, consumption: ContinuousAction) -> FloatND:
        return consumption

    def affordable(
        *, consumption: ContinuousAction, wealth: ContinuousState, eps: FloatND
    ) -> BoolND:
        return consumption <= wealth + eps

    def utility_with_shock(*, consumption: ContinuousAction, eps: FloatND) -> FloatND:
        return consumption + eps

    def affordable_without_shock(
        *, consumption: ContinuousAction, wealth: ContinuousState
    ) -> BoolND:
        return consumption <= wealth

    return Model(
        regimes={
            "alive": UserRegime(
                states={"wealth": LinSpacedGrid(start=1, stop=2, n_points=2)},
                actions={"consumption": LinSpacedGrid(start=0, stop=1, n_points=2)},
                functions={
                    "utility": utility if shock_constraint else utility_with_shock
                },
                constraints={
                    "affordable": affordable
                    if shock_constraint
                    else affordable_without_shock
                },
                shocks=("entry",),
            )
        },
        shocks={
            "entry": ShockGroup(
                shocks={
                    "eps": Uniform(start=0.25, stop=0.75)
                    if positive_shock
                    else Normal(mu=0, sigma=1)
                },
                nodes=UserNodes(
                    nodes=np.array([[0.0], [0.5], [1.0]])
                    if positive_shock
                    else np.array([[-1.0], [0.0], [1.0]]),
                    weights=np.array([0.25, 0.5, 0.25]),
                ),
            )
        },
        n_periods=1,
        edges={},
        initial_nodes=InitialNodes(by_period={0: "alive"}),
        regime_id_class=EntryRegimeId,
        enable_jit=enable_jit,
    )


def test_tagged_current_physical_values_determine_public_entry_feasibility() -> None:
    """Identical states can have different feasibility under supplied entry shocks."""

    model = _tagged_current_entry_model()
    # These physical values are off the declared nodes. The zero action keeps
    # every declared quadrature node feasible at wealth one.
    initial: dict[str, Array | PhysicalValues] = {
        "period": jnp.zeros(4, dtype=jnp.int32),
        "wealth": jnp.ones(4),
        "regime_id": jnp.full(4, model.regime_names_to_ids["alive"], dtype=jnp.int32),
        "eps": PhysicalValues(values=jnp.array([0.25, -1.25, 0.75, -1.75])),
    }

    mask = model.initial_conditions_feasibility(initial_conditions=initial, params={})

    np.testing.assert_array_equal(mask, np.array([True, False, True, False]))
    with pytest.raises(InvalidInitialConditionsError):
        model.validate_initial_conditions(initial_conditions=initial, params={})


def test_seeded_simulation_preflight_uses_the_supplied_current_entry_frame(
    *, record_property: Callable[[str, str], None]
) -> None:
    """Enabled preflight judges the physical frame later used by original replay."""
    model = _tagged_current_entry_model()
    regime = model._regimes["alive"]
    processed_signature = signature(regime.simulation.constraints["affordable"])
    feasibility_signature = signature(
        _regime_feasibility_function(regime=regime, memory=None)
    )
    record_property("processed_affordable_signature", str(processed_signature))
    record_property("composed_feasibility_signature", str(feasibility_signature))
    assert "eps" in processed_signature.parameters
    assert "eps" in feasibility_signature.parameters

    solution = model.solve(params={}, runtime_checks=True, log_level="warning")
    physical = jnp.array([0.25, 0.75, 0.5, 0.125])
    initial = _tagged_entry_initial(model=model, value=PhysicalValues(values=physical))
    frame = model.simulate(
        params={},
        solution=solution,
        initial_conditions=initial,
        seed=17,
        runtime_checks=True,
        log_level="warning",
    ).to_dataframe(use_categorical_codes=True)
    rows = frame.sort_values("subject_id")
    np.testing.assert_array_equal(rows["subject_id"], [0, 1, 2, 3])
    np.testing.assert_array_equal(rows["period"], [0, 0, 0, 0])
    np.testing.assert_array_equal(rows["eps"], [0.25, 0.75, 0.5, 0.125])
    np.testing.assert_array_equal(rows["consumption"], [1.0, 1.0, 1.0, 1.0])
    assert np.all(rows["consumption"].to_numpy() <= 1.0 + physical)


def _tagged_entry_initial(
    *, model: Model, value: PhysicalValues | BaseInnovations | None
) -> dict[str, Array | PhysicalValues | BaseInnovations]:
    initial: dict[str, Array | PhysicalValues | BaseInnovations] = {
        "period": jnp.zeros(4, dtype=jnp.int32),
        "wealth": jnp.ones(4),
        "regime_id": jnp.full(4, model.regime_names_to_ids["alive"], dtype=jnp.int32),
    }
    if value is not None:
        initial["eps"] = value
    return initial


@pytest.mark.parametrize("seed", [29, None], ids=("explicit-seed", "fresh-seed"))
def test_seeded_entry_preflight_and_native_replay_use_the_same_frame(
    *, monkeypatch: pytest.MonkeyPatch, seed: int | None
) -> None:
    """A bounded positive law needs no guessed draw to keep every row feasible."""
    model = _tagged_current_entry_model(positive_shock=True)
    solution = model.solve(params={}, runtime_checks=True, log_level="warning")
    observed: dict[int, float] = {}
    providers: list[SeededEntryFrames] = []
    produce = SeededEntryFrames.physical_inputs

    # keyword-only-exempt: library-callback=SeededEntryFrames.physical_inputs
    def record(
        self: SeededEntryFrames,
        *,
        regime: InternalRegime,
        regime_name: RegimeName,
        initial_states: Mapping[ReferenceName, FloatND | IntND],
        indices: list[int],
        periods: np.ndarray,
        ages: TimeAxis,
        needed: frozenset[ReferenceName],
        memory: SimulationMemory | None,
    ) -> Mapping[ReferenceName, FloatND | IntND]:
        physical = produce(
            self,
            regime=regime,
            regime_name=regime_name,
            initial_states=initial_states,
            indices=indices,
            periods=periods,
            ages=ages,
            needed=needed,
            memory=memory,
        )
        providers.append(self)
        for row, value in zip(indices, np.asarray(physical["eps"]), strict=True):
            observed[row] = float(value)
        return physical

    monkeypatch.setattr(SeededEntryFrames, "physical_inputs", record)
    initial = _tagged_entry_initial(model=model, value=None)
    rows = (
        model.simulate(
            params={},
            solution=solution,
            initial_conditions=initial,
            seed=seed,
            runtime_checks=True,
            log_level="warning",
        )
        .to_dataframe(use_categorical_codes=True)
        .sort_values("subject_id")
    )
    assert set(observed) == {0, 1, 2, 3}
    np.testing.assert_array_equal(rows["subject_id"], [0, 1, 2, 3])
    np.testing.assert_array_equal(rows["period"], [0, 0, 0, 0])
    np.testing.assert_array_equal(rows["eps"], [observed[row] for row in range(4)])
    assert np.all((rows["eps"] >= 0.25) & (rows["eps"] <= 0.75))
    np.testing.assert_array_equal(rows["consumption"], [1.0, 1.0, 1.0, 1.0])
    canonical, _, _ = model._canonical_feasibility_inputs(
        initial_conditions=initial, params={}
    )
    selected = produce(
        providers[0],
        regime=model._regimes["alive"],
        regime_name="alive",
        initial_states=canonical,
        indices=[3, 1],
        periods=np.zeros(4, dtype=np.int32),
        ages=model._time,
        needed=frozenset({"eps"}),
        memory=None,
    )
    np.testing.assert_array_equal(selected["eps"], rows["eps"].to_numpy()[[3, 1]])


@pytest.mark.coverage(backends=("cpu", "gpu-small"), precisions="64")
@pytest.mark.parametrize("supplied", [False, True], ids=("seeded", "physical"))
def test_jitted_entry_preflight_matches_original_native_replay(
    *, monkeypatch: pytest.MonkeyPatch, supplied: bool
) -> None:
    """Default-device JIT retains supplied or genuinely seeded entry frames."""
    model = _tagged_current_entry_model(enable_jit=True, positive_shock=not supplied)
    assert model.enable_jit
    solution = model.solve(params={}, runtime_checks=True, log_level="warning")
    observed: dict[int, float] = {}
    produce = SeededEntryFrames.physical_inputs

    # keyword-only-exempt: library-callback=SeededEntryFrames.physical_inputs
    def record(
        self: SeededEntryFrames,
        *,
        regime: InternalRegime,
        regime_name: RegimeName,
        initial_states: Mapping[ReferenceName, FloatND | IntND],
        indices: list[int],
        periods: np.ndarray,
        ages: TimeAxis,
        needed: frozenset[ReferenceName],
        memory: SimulationMemory | None,
    ) -> Mapping[ReferenceName, FloatND | IntND]:
        physical = produce(
            self,
            regime=regime,
            regime_name=regime_name,
            initial_states=initial_states,
            indices=indices,
            periods=periods,
            ages=ages,
            needed=needed,
            memory=memory,
        )
        for row, value in zip(indices, np.asarray(physical["eps"]), strict=True):
            observed[row] = float(value)
        return physical

    monkeypatch.setattr(SeededEntryFrames, "physical_inputs", record)
    initial = _tagged_entry_initial(
        model=model,
        value=PhysicalValues(values=jnp.array([0.25, 0.75, 0.5, 0.125]))
        if supplied
        else None,
    )
    rows = (
        model.simulate(
            params={},
            solution=solution,
            initial_conditions=initial,
            seed=29,
            runtime_checks=True,
            log_level="warning",
        )
        .to_dataframe(use_categorical_codes=True)
        .sort_values("subject_id")
    )
    assert set(observed) == {0, 1, 2, 3}
    np.testing.assert_array_equal(rows["subject_id"], [0, 1, 2, 3])
    np.testing.assert_array_equal(rows["period"], [0, 0, 0, 0])
    np.testing.assert_array_equal(rows["eps"], [observed[row] for row in range(4)])
    np.testing.assert_array_equal(rows["consumption"], [1.0, 1.0, 1.0, 1.0])
    if supplied:
        np.testing.assert_array_equal(rows["eps"], [0.25, 0.75, 0.5, 0.125])
    else:
        assert np.all((rows["eps"] >= 0.25) & (rows["eps"] <= 0.75))


@pytest.mark.parametrize("last_physical", [0.75, -1.25], ids=("valid", "invalid"))
def test_seeded_preflight_ignores_padding_but_checks_the_last_real_row(
    *, last_physical: float
) -> None:
    """Alignment padding never becomes a new physical observation or subject."""
    model = _tagged_current_entry_model()
    initial: dict[str, Array | PhysicalValues] = {
        "period": jnp.zeros(3, dtype=jnp.int32),
        "wealth": jnp.ones(3),
        "regime_id": jnp.full(3, model.regime_names_to_ids["alive"], dtype=jnp.int32),
        "eps": PhysicalValues(values=jnp.array([0.25, 0.5, last_physical])),
    }
    canonical, flat_params, supplied = model._canonical_feasibility_inputs(
        initial_conditions=initial, params={}
    )
    padded, original = pad_initial_conditions_to_multiple(
        initial_conditions=canonical, multiple=4
    )
    assert original == 3
    assert len(padded["regime_id"]) == 4
    provider = SeededEntryFrames(
        entries=supplied.entries, flat_params=flat_params, seed=17, overrides=None
    )

    def validate() -> None:
        validate_simulation_inputs(
            initial_conditions=padded,
            regimes=model._regimes,
            regime_names_to_ids=model.regime_names_to_ids,
            flat_params=flat_params,
            ages=model._time,
            logger=get_logger(log_level="warning", runtime_checks=True),
            supplied_frames=provider,
            original_n_subjects=original,
        )

    if last_physical < -1.0:
        with pytest.raises(NoFeasibleActionError, match=r"for 1 subject\(s\)"):
            validate()
    else:
        validate()


@pytest.mark.parametrize("supplied", [False, True], ids=("seeded", "physical"))
def test_entry_law_context_matches_native_clock_and_period_dtype(
    *, supplied: bool
) -> None:
    """Near-grid entry times are admitted but laws read the computational clock."""

    @categorical(ordered=False)
    class DatedEntryRegimeId:
        alive: ScalarInt

    def lower(*, age: FloatND, period: IntND) -> FloatND:
        valid_clock = (age == 20.5 + period) & (period.dtype == jnp.int32)
        return jnp.where(valid_clock, age + period, jnp.nan)

    def upper(*, age: FloatND, period: IntND) -> FloatND:
        return lower(age=age, period=period) + 0.5

    template = _tagged_current_entry_model()
    model = Model(
        regimes=template.user_regimes,
        shocks={
            "entry": ShockGroup(
                shocks={"eps": Uniform(start=lower, stop=upper)},
                nodes=UserNodes(
                    nodes=np.array([[0.0], [0.5], [1.0]]),
                    weights=np.array([0.25, 0.5, 0.25]),
                ),
            )
        },
        ages=AgeGrid(exact_values=(Fraction(41, 2), Fraction(43, 2))),
        edges={},
        initial_nodes=InitialNodes(by_age={21.5: "alive"}),
        regime_id_class=DatedEntryRegimeId,
        enable_jit=False,
    )
    near_grid = jnp.full(2, 21.5 - 1e-4)
    assert np.all(np.asarray(near_grid) < 21.5)
    np.testing.assert_allclose(near_grid, [21.5, 21.5], rtol=1e-5, atol=1e-8)
    initial: dict[str, Array | PhysicalValues] = {
        "age": near_grid,
        "wealth": jnp.ones(2),
        "regime_id": jnp.full(2, model.regime_names_to_ids["alive"], dtype=jnp.int32),
    }
    if supplied:
        initial["eps"] = PhysicalValues(values=jnp.array([22.625, 22.875]))
        model.validate_initial_conditions(initial_conditions=initial, params={})
    solution = model.solve(params={}, runtime_checks=True, log_level="warning")
    rows = (
        model.simulate(
            params={},
            solution=solution,
            initial_conditions=initial,
            seed=29,
            runtime_checks=True,
            log_level="warning",
        )
        .to_dataframe(use_categorical_codes=True)
        .sort_values("subject_id")
    )
    canonical, flat_params, supplied_frames = model._canonical_feasibility_inputs(
        initial_conditions=initial, params={}
    )
    provider = SeededEntryFrames(
        entries=supplied_frames.entries,
        flat_params=flat_params,
        seed=29,
        overrides=None,
    )
    direct = provider.physical_inputs(
        regime=model._regimes["alive"],
        regime_name="alive",
        initial_states=canonical,
        indices=[1, 0],
        periods=np.ones(2, dtype=np.int64),
        ages=model._time,
        needed=frozenset({"eps"}),
        memory=None,
    )
    np.testing.assert_array_equal(rows["subject_id"], [0, 1])
    np.testing.assert_array_equal(rows["period"], [1, 1])
    np.testing.assert_array_equal(rows["age"], [21.5, 21.5])
    np.testing.assert_array_equal(rows["consumption"], [1.0, 1.0])
    np.testing.assert_array_equal(direct["eps"], rows["eps"].to_numpy()[[1, 0]])
    if supplied:
        np.testing.assert_array_equal(rows["eps"], [22.625, 22.875])
    else:
        assert np.all((rows["eps"] >= 22.5) & (rows["eps"] <= 23.0))


def test_supplied_entry_bases_forward_to_the_same_public_feasibility() -> None:
    """Normal(0,1) maps the supplied coordinates to the same physical values."""
    model = _tagged_current_entry_model()
    initial = _tagged_entry_initial(
        model=model, value=BaseInnovations(values=jnp.array([0.25, -1.25, 0.75, -1.75]))
    )
    mask = model.initial_conditions_feasibility(initial_conditions=initial, params={})
    np.testing.assert_array_equal(mask, np.array([True, False, True, False]))


def test_seedless_entry_helpers_require_their_actual_constraint_observation() -> None:
    """Helpers cannot invent a draw for a genuinely demanded physical leaf."""
    model = _tagged_current_entry_model()
    initial = _tagged_entry_initial(model=model, value=None)
    with pytest.raises(UnsupportedOperationError, match=r"alive.*eps.*period"):
        model.initial_conditions_feasibility(initial_conditions=initial, params={})
    with pytest.raises(UnsupportedOperationError, match=r"alive.*eps.*period"):
        model.validate_initial_conditions(initial_conditions=initial, params={})


def test_supplied_entry_tags_are_validated_even_without_a_shock_constraint() -> None:
    """A supplied tag is checked; an unneeded absent draw is not manufactured."""
    model = _tagged_current_entry_model(shock_constraint=False)
    ordinary = _tagged_entry_initial(model=model, value=None)
    np.testing.assert_array_equal(
        model.initial_conditions_feasibility(initial_conditions=ordinary, params={}),
        np.ones(4, dtype=bool),
    )
    invalid = _tagged_entry_initial(
        model=model, value=PhysicalValues(values=jnp.array([jnp.nan, 0.0, 0.0, 0.0]))
    )
    with pytest.raises(ShockDeclarationError, match="Invalid supplied entry frame"):
        model.initial_conditions_feasibility(initial_conditions=invalid, params={})
    malformed = _tagged_entry_initial(
        model=model, value=PhysicalValues(values=jnp.ones((2, 2)))
    )
    with pytest.raises(InvalidSimulationInputError, match="requires shape"):
        model.validate_initial_conditions(initial_conditions=malformed, params={})
