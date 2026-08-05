import inspect

from _lcm.grids import DiscreteGrid
from _lcm.params.regime_template import (
    create_regime_params_template,
)
from _lcm.utils.containers import ensure_containers_are_immutable
from lcm import Phased, fixed_transition
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
            "koopmans_aggregator": {"discount_factor": "FloatND"},
            "utility": {"c": "no_annotation_found"},
            "next_b": {},
            "next_regime": {},
        }
    )


def test_create_regime_params_template_has_no_representative_age_argument():
    """Age specialization is resolved upstream, so the template is age-unaware.

    `normalize_age_specialization` replaces every `AgeSpecializedFunction` by its
    first-active concrete function before this runs, so the template reads concrete
    signatures directly and needs no `representative_age` argument.
    """
    params = inspect.signature(create_regime_params_template).parameters
    assert "representative_age" not in params


def test_create_params_reads_concrete_function_params(binary_category_class):
    """A concrete (normalized) function contributes its params under its name.

    After normalization the regime carries the concrete first-active function in
    place of any marker, so a real estimated parameter surfaces under its name with
    no marker handling in the template.
    """

    def net_income(a, b, tax_rate):  # noqa: ARG001
        return tax_rate

    regime = MockRegime(
        actions={"a": DiscreteGrid(binary_category_class)},
        states={"b": DiscreteGrid(binary_category_class)},
        state_transitions={"b": lambda b: b},
        transition=lambda: 0,
        functions={
            "utility": lambda a, b: None,  # noqa: ARG005
            "net_income": net_income,
        },
    )
    got = create_regime_params_template(regime)
    assert got["net_income"] == {"tax_rate": "no_annotation_found"}


def test_create_params_unions_phased_variant_params(binary_category_class):
    """`Phased` entries contribute the union of both variants' parameters."""

    def solve_income(a, b, solve_rate):  # noqa: ARG001
        return solve_rate

    def simulate_income(a, b, simulate_rate):  # noqa: ARG001
        return simulate_rate

    regime = MockRegime(
        actions={"a": DiscreteGrid(binary_category_class)},
        states={"b": DiscreteGrid(binary_category_class)},
        state_transitions={"b": lambda b: b},
        transition=lambda: 0,
        functions={
            "utility": lambda a, b: None,  # noqa: ARG005
            "net_income": Phased(solve=solve_income, simulate=simulate_income),
        },
    )
    got = create_regime_params_template(regime)
    assert got["net_income"] == {
        "solve_rate": "no_annotation_found",
        "simulate_rate": "no_annotation_found",
    }


def test_create_params_with_custom_W_no_extra_params():
    """A custom H with no extra params beyond utility and CE."""

    def custom_W(utility: float, CE: float) -> float:
        return utility + CE

    regime = MockRegime(
        actions={
            "a": None,
        },
        states={
            "b": None,
        },
        functions={"utility": lambda a, b, c: None},  # noqa: ARG005
        koopmans_aggregator=custom_W,
    )
    got = create_regime_params_template(regime)
    assert got == ensure_containers_are_immutable(
        {"koopmans_aggregator": {}, "utility": {"c": "no_annotation_found"}}
    )


def test_default_H_with_state_named_discount_factor_is_allowed():
    """H params matching a state name are excluded from the template.

    pylcm wires state/action values through `states_actions_params` and
    filters into `W_kwargs` via the signature-derived `_H_accepted_params`.
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
            "koopmans_aggregator": {},
            "utility": {},
            "next_discount_factor": {},
            "next_regime": {},
        }
    )


def test_custom_W_shadowing_state_is_allowed():
    """Custom H may declare a state in its signature to subscript it.

    This is how a model with a `pref_type` state can have a custom H that
    indexes a Series-valued param like `discount_factor_by_type[pref_type]`.
    The shadowed state name is excluded from the template and injected at
    call time from the state space.
    """

    def custom_W(utility: float, CE: float, wealth: float) -> float:
        return utility + wealth * CE

    regime = MockRegime(
        actions={"a": None},
        states={"wealth": None},
        functions={"utility": lambda a, wealth: None},  # noqa: ARG005
        koopmans_aggregator=custom_W,
    )
    got = create_regime_params_template(regime)
    assert got == ensure_containers_are_immutable(
        {"koopmans_aggregator": {}, "utility": {}}
    )


def test_solve_simulate_pair_template_contains_union_of_params() -> None:
    """Template for a phase-variant (`Phased`) H contains params from both variants.

    The solve variant (`exponential_H`) takes `discount_factor`; the simulate
    variant (`beta_delta_H`) takes `beta` and `delta`. The template must contain
    all three so the user can provide a single flat params dict that satisfies
    both phases.
    """

    def exponential_h(utility: float, CE: float, discount_factor: float) -> float:
        return utility + discount_factor * CE

    def beta_delta_h(utility: float, CE: float, beta: float, delta: float) -> float:
        return utility + beta * delta * CE

    regime = MockRegime(
        actions={"a": None},
        states={"b": None},
        functions={"utility": lambda a, b: None},  # noqa: ARG005
        koopmans_aggregator=Phased(solve=exponential_h, simulate=beta_delta_h),
    )
    got = create_regime_params_template(regime)
    assert set(got["koopmans_aggregator"]) == {"discount_factor", "beta", "delta"}


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
            "koopmans_aggregator": {"discount_factor": "FloatND"},
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
            "koopmans_aggregator": {"discount_factor": "FloatND"},
            "utility": {},
            "next_wealth": {},
            "next_aime": {},
            "next_regime": {},
        }
    )
