from _lcm.grids import DiscreteGrid
from _lcm.params.regime_template import (
    create_regime_params_template,
)
from _lcm.utils.containers import ensure_containers_are_immutable
from lcm import Phased, fixed_transition
from lcm.transition import AgeSpecialized
from tests.mock_regime import MockRegime


def test_create_params_without_processes(binary_category_class):
    regime = MockRegime(
        actions={
            "a": DiscreteGrid(binary_category_class),
        },
        states={
            "b": DiscreteGrid(binary_category_class),
        },
        state_transitions={"b": lambda b: b},
        transition=lambda: 0,
        functions={"utility": lambda a, b, c: None},  # noqa: ARG005
    )
    got = create_regime_params_template(regime)
    assert got == ensure_containers_are_immutable(
        {
            "H": {"discount_factor": "FloatND"},
            "utility": {"c": "no_annotation_found"},
            "next_b": {},
            "next_regime": {},
        }
    )


def test_create_params_resolves_age_specialized_function(binary_category_class):
    """An `AgeSpecialized` function contributes its concrete function's params.

    The wrapper's own call signature is `(*args, **kwargs)`; the template must be
    read off the concrete function `build(representative_age)` returns, so a real
    estimated parameter of an age-specialized function surfaces under its name.
    """

    def build(age):  # noqa: ARG001
        def net_income(a, b, tax_rate):  # noqa: ARG001
            return tax_rate

        return net_income

    regime = MockRegime(
        actions={"a": DiscreteGrid(binary_category_class)},
        states={"b": DiscreteGrid(binary_category_class)},
        state_transitions={"b": lambda b: b},
        transition=lambda: 0,
        functions={
            "utility": lambda a, b: None,  # noqa: ARG005
            "net_income": AgeSpecialized(build=build, signature=lambda age: age),
        },
    )
    got = create_regime_params_template(regime, representative_age=60.0)
    assert got["net_income"] == {"tax_rate": "no_annotation_found"}


def test_create_params_resolves_phased_age_specialized_function(binary_category_class):
    """A `Phased(AgeSpecialized, AgeSpecialized)` function contributes its params.

    Representative resolution must descend into both `Phased` sides so a real
    estimated parameter of a phase-split, age-specialized function surfaces in
    the template instead of the marker's `(*args, **kwargs)` signature.
    """

    def build(age):  # noqa: ARG001
        def net_income(a, b, tax_rate):  # noqa: ARG001
            return tax_rate

        return net_income

    phased_specialized = Phased(
        solve=AgeSpecialized(build=build, signature=lambda age: age),
        simulate=AgeSpecialized(build=build, signature=lambda age: age),
    )
    regime = MockRegime(
        actions={"a": DiscreteGrid(binary_category_class)},
        states={"b": DiscreteGrid(binary_category_class)},
        state_transitions={"b": lambda b: b},
        transition=lambda: 0,
        functions={
            "utility": lambda a, b: None,  # noqa: ARG005
            "net_income": phased_specialized,
        },
    )
    got = create_regime_params_template(regime, representative_age=60.0)
    assert got["net_income"] == {"tax_rate": "no_annotation_found"}


def test_create_params_with_custom_H_no_extra_params():
    """A custom H with no extra params beyond utility and E_next_V."""

    def custom_H(utility: float, E_next_V: float) -> float:
        return utility + E_next_V

    regime = MockRegime(
        actions={
            "a": None,
        },
        states={
            "b": None,
        },
        functions={"utility": lambda a, b, c: None, "H": custom_H},  # noqa: ARG005
    )
    got = create_regime_params_template(regime)
    assert got == ensure_containers_are_immutable(
        {"H": {}, "utility": {"c": "no_annotation_found"}}
    )


def test_default_H_with_state_named_discount_factor_is_allowed():
    """H params matching a state name are excluded from the template.

    pylcm wires state/action values through `states_actions_params` and
    filters into `H_kwargs` via the signature-derived `_H_accepted_params`.
    Names that match a state are therefore sourced from state values at
    runtime, not from the user-facing params dict, so they do not appear
    in the template.
    """
    regime = MockRegime(
        actions={"a": None},
        states={"discount_factor": None},
        state_transitions={"discount_factor": fixed_transition("discount_factor")},
        functions={"utility": lambda a, discount_factor: None},  # noqa: ARG005
        transition=lambda discount_factor: discount_factor,
    )
    got = create_regime_params_template(regime)
    assert got == ensure_containers_are_immutable(
        {
            "H": {},
            "utility": {},
            "next_discount_factor": {},
            "next_regime": {},
        }
    )


def test_custom_H_shadowing_state_is_allowed():
    """Custom H may declare a state in its signature to subscript it.

    This is how a model with a `pref_type` state can have a custom H that
    indexes a Series-valued param like `discount_factor_by_type[pref_type]`.
    The shadowed state name is excluded from the template and injected at
    call time from the state space.
    """

    def custom_H(utility: float, E_next_V: float, wealth: float) -> float:
        return utility + wealth * E_next_V

    regime = MockRegime(
        actions={"a": None},
        states={"wealth": None},
        functions={"utility": lambda a, wealth: None, "H": custom_H},  # noqa: ARG005
    )
    got = create_regime_params_template(regime)
    assert got == ensure_containers_are_immutable({"H": {}, "utility": {}})


def test_solve_simulate_pair_template_contains_union_of_params() -> None:
    """Template for a phase-variant (`Phased`) H contains params from both variants.

    The solve variant (`exponential_H`) takes `discount_factor`; the simulate
    variant (`beta_delta_H`) takes `beta` and `delta`. The template must contain
    all three so the user can provide a single flat params dict that satisfies
    both phases.
    """

    def exponential_h(utility: float, E_next_V: float, discount_factor: float) -> float:
        return utility + discount_factor * E_next_V

    def beta_delta_h(
        utility: float, E_next_V: float, beta: float, delta: float
    ) -> float:
        return utility + beta * delta * E_next_V

    regime = MockRegime(
        actions={"a": None},
        states={"b": None},
        functions={  # ty: ignore[invalid-argument-type]
            "utility": lambda a, b: None,  # noqa: ARG005
            "H": Phased(solve=exponential_h, simulate=beta_delta_h),
        },
    )
    got = create_regime_params_template(regime)
    assert set(got["H"]) == {"discount_factor", "beta", "delta"}


def test_regular_function_taking_state_as_argument_no_error(binary_category_class):
    """Regular functions that use states as arguments should not trigger the error."""
    regime = MockRegime(
        actions={
            "a": DiscreteGrid(binary_category_class),
        },
        states={
            "wealth": DiscreteGrid(binary_category_class),
        },
        state_transitions={"wealth": lambda wealth: wealth},
        transition=lambda: 0,
        functions={"utility": lambda a, wealth, risk_aversion: None},  # noqa: ARG005
    )
    got = create_regime_params_template(regime)
    assert got == ensure_containers_are_immutable(
        {
            "H": {"discount_factor": "FloatND"},
            "utility": {"risk_aversion": "no_annotation_found"},
            "next_wealth": {},
            "next_regime": {},
        }
    )


def test_state_transition_consuming_other_next_state_is_not_a_param(
    binary_category_class,
):
    """`next_<state>` names are exempt from param-template extraction.

    A state transition (here, `next_wealth`) that consumes another transition's
    output (here, `next_aime`) must not have `next_aime` classified as a
    regime-level fixed_param. dags resolves the chain at evaluation time
    (`get_next_state_function_for_solution` merges all transitions into a
    single dict before calling `concatenate_functions`).
    """

    def next_wealth(wealth: float, next_aime: float) -> float:
        return wealth + next_aime

    regime = MockRegime(
        actions={"a": DiscreteGrid(binary_category_class)},
        states={
            "wealth": DiscreteGrid(binary_category_class),
            "aime": DiscreteGrid(binary_category_class),
        },
        state_transitions={
            "wealth": next_wealth,
            "aime": lambda aime: aime,
        },
        transition=lambda: 0,
        functions={"utility": lambda a, wealth, aime: None},  # noqa: ARG005
    )
    got = create_regime_params_template(regime)
    assert got == ensure_containers_are_immutable(
        {
            "H": {"discount_factor": "FloatND"},
            "utility": {},
            "next_wealth": {},
            "next_aime": {},
            "next_regime": {},
        }
    )
