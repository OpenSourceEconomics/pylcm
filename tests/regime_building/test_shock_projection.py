"""Shared discrete partitions require effective law-input identity proofs."""

import importlib
import importlib.util
from collections.abc import Callable
from typing import cast

import jax.numpy as jnp
import pytest

from _lcm.grids import Grid
from _lcm.shock_registry import ShockBinding, create_shock_registry
from lcm import DiscreteGrid, LinSpacedGrid, Regime, categorical
from lcm.exceptions import ShockDeclarationError
from lcm.shocks import Discrete, Product, ShockGroup
from lcm.typing import FloatND, ScalarInt


@categorical(ordered=True)
class _Outcome:
    low: ScalarInt
    high: ScalarInt


def _probabilities(state: FloatND) -> FloatND:
    high = 0.25 + state / 4
    return jnp.stack((1 - high, high))


def _payoff(eps: ScalarInt) -> FloatND:
    return 1.0 * eps


def _identity(state: FloatND) -> FloatND:
    return state


def _changed(state: FloatND) -> FloatND:
    return state / 2


def _binding(
    *, grid: Grid | None = None, probabilities: Callable[..., FloatND] = _probabilities
) -> ShockBinding:
    regime = Regime(
        states={"state": grid or LinSpacedGrid(start=0, stop=1, n_points=2)},
        functions={"utility": _payoff},
        shocks=("innovation",),
    )
    return create_shock_registry(
        shocks={
            "innovation": ShockGroup(
                shocks={"eps": Discrete(categories=_Outcome, probs=probabilities)},
                nodes=Product(),
            )
        },
        regimes={"alive": regime},
        legacy_regimes={"alive": regime},
        model_states={},
    ).bindings["alive", "solve", "innovation"]


def _validator() -> Callable[..., None]:
    assert importlib.util.find_spec("_lcm.shock_projection") is not None, (
        "Shared discrete projection admission must be implemented."
    )
    return cast(
        "Callable[..., None]",
        importlib.import_module(
            "_lcm.shock_projection"
        ).validate_shared_discrete_projection,
    )


def test_changed_shared_discrete_input_is_named_and_refused() -> None:
    """A projection changing an actually shared law input cannot reuse its category."""
    binding = _binding()
    with pytest.raises(ShockDeclarationError, match=r"eps.*Raw.*alive.*state"):
        _validator()(
            query_kind="Raw",
            target="alive",
            physical_inputs=frozenset({"eps"}),
            source_physical_inputs=frozenset({"eps"}),
            projection={"state": _changed},
            source_bindings={"innovation": binding},
            target_bindings={"innovation": binding},
            source_domains=binding.state_domains,
            target_domains=binding.state_domains,
        )


def test_identity_shared_discrete_input_is_admitted() -> None:
    """An effective direct return preserves the shared probability partition."""
    binding = _binding()
    _validator()(
        query_kind="Raw",
        target="alive",
        physical_inputs=frozenset({"eps"}),
        source_physical_inputs=frozenset({"eps"}),
        projection={"state": _identity},
        source_bindings={"innovation": binding},
        target_bindings={"innovation": binding},
        source_domains=binding.state_domains,
        target_domains=binding.state_domains,
    )


def test_formation_only_declaration_is_not_a_raw_read() -> None:
    """A raw stay problem ignores a partner law declared solely for formation."""
    binding = _binding()
    _validator()(
        query_kind="Raw",
        target="alive",
        physical_inputs=frozenset(),
        source_physical_inputs=frozenset({"eps"}),
        projection={"state": _changed},
        source_bindings={"innovation": binding},
        target_bindings={"innovation": binding},
        source_domains=binding.state_domains,
        target_domains=binding.state_domains,
    )


def _off_grid_change(state: FloatND) -> FloatND:
    return jnp.where((state == 0) | (state == 1), state, state + 1)


def _discarded_computation(state: FloatND) -> FloatND:
    unused = state * 2  # noqa: F841 - discarded equation must not certify continuous identity
    return state


def _identity_through_helper(mapped: FloatND) -> FloatND:
    return mapped


def _finite_identity(state: ScalarInt) -> ScalarInt:
    return jnp.where(state == 0, jnp.int32(0), jnp.int32(1))


def _finite_swap(state: ScalarInt) -> ScalarInt:
    return jnp.int32(1) - state


def _parameter_probabilities(*, state: FloatND, p: FloatND) -> FloatND:
    high = p + state / 4
    return jnp.stack((1 - high, high))


@pytest.mark.parametrize("project", [_off_grid_change, _discarded_computation])
def test_continuous_identity_is_not_a_grid_node_comparison(
    project: Callable[..., FloatND],
) -> None:
    """Off-grid changes and discarded equations cannot prove continuous identity."""
    binding = _binding()
    with pytest.raises(
        ShockDeclarationError, match=r"eps.*Participating.*alive.*state"
    ):
        _validator()(
            query_kind="Participating",
            target="alive",
            physical_inputs=frozenset({"eps"}),
            source_physical_inputs=frozenset({"eps"}),
            projection={"state": project},
            source_bindings={"innovation": binding},
            target_bindings={"innovation": binding},
            source_domains=binding.state_domains,
            target_domains=binding.state_domains,
        )


def test_effective_helper_composition_preserves_direct_return() -> None:
    """The complete effective projection, including helpers, is the proof subject."""
    binding = _binding()
    _validator()(
        query_kind="Raw",
        target="alive",
        physical_inputs=frozenset({"eps"}),
        source_physical_inputs=frozenset({"eps"}),
        projection={"state": _identity_through_helper},
        projection_functions={"mapped": _identity},
        source_bindings={"innovation": binding},
        target_bindings={"innovation": binding},
        source_domains=binding.state_domains,
        target_domains=binding.state_domains,
    )


@pytest.mark.parametrize("preserves", [True, False])
def test_discrete_identity_checks_the_complete_code_domain(*, preserves: bool) -> None:
    """Every valid code must map to itself under the same physical-value codec."""
    binding = _binding(grid=DiscreteGrid(values=jnp.array([10.0, 20.0])))
    kwargs = {
        "query_kind": "Raw",
        "target": "alive",
        "physical_inputs": frozenset({"eps"}),
        "source_physical_inputs": frozenset({"eps"}),
        "projection": {"state": _finite_identity if preserves else _finite_swap},
        "source_bindings": {"innovation": binding},
        "target_bindings": {"innovation": binding},
        "source_domains": binding.state_domains,
        "target_domains": binding.state_domains,
    }
    if preserves:
        _validator()(**kwargs)
    else:
        with pytest.raises(ShockDeclarationError, match=r"eps.*Raw.*alive.*state"):
            _validator()(**kwargs)


def test_matching_codes_with_different_value_domains_are_refused() -> None:
    """Equal integer positions cannot hide different physical state domains."""
    source = _binding(grid=DiscreteGrid(values=jnp.array([10.0, 20.0])))
    target = _binding(grid=DiscreteGrid(values=jnp.array([10.0, 30.0])))
    with pytest.raises(
        ShockDeclarationError, match=r"eps.*Raw.*alive.*(bindings|domain)"
    ):
        _validator()(
            query_kind="Raw",
            target="alive",
            physical_inputs=frozenset({"eps"}),
            source_physical_inputs=frozenset({"eps"}),
            projection={"state": _identity},
            source_bindings={"innovation": source},
            target_bindings={"innovation": target},
            source_domains=source.state_domains,
            target_domains=target.state_domains,
        )


def test_matching_parameter_values_cannot_replace_binding_identity() -> None:
    """A shared partition needs the same runtime operand path, not a sampled value."""
    binding = _binding(probabilities=_parameter_probabilities)
    with pytest.raises(ShockDeclarationError, match=r"eps.*Full.*alive.*bindings"):
        _validator()(
            query_kind="Full",
            target="alive",
            physical_inputs=frozenset({"eps"}),
            source_physical_inputs=frozenset({"eps"}),
            projection={"state": _identity},
            source_bindings={"innovation": binding},
            target_bindings={"innovation": binding},
            source_domains=binding.state_domains,
            target_domains=binding.state_domains,
            source_parameter_paths={"innovation__p": ("source", "p")},
            target_parameter_paths={"innovation__p": ("target", "p")},
        )


def test_target_only_hidden_category_has_no_source_partition() -> None:
    """A permission to reference a group does not create a source category read."""
    binding = _binding()
    _validator()(
        query_kind="Participating",
        target="alive",
        physical_inputs=frozenset({"eps"}),
        source_physical_inputs=frozenset(),
        projection={"state": _changed},
        source_bindings={"innovation": binding},
        target_bindings={"innovation": binding},
        source_domains=binding.state_domains,
        target_domains=binding.state_domains,
    )


def _helper_probabilities(share: FloatND) -> FloatND:
    return jnp.stack((1 - share, share))


def _share(state: FloatND) -> FloatND:
    return 0.25 + state / 4


def _different_share(state: FloatND) -> FloatND:
    return 0.25 + state / 8


def test_phase_specific_law_helper_bindings_must_match() -> None:
    """A shared group declaration does not equate different phase helper bodies."""
    from dataclasses import replace  # noqa: PLC0415

    base = _binding()
    group = ShockGroup(
        shocks={"eps": Discrete(categories=_Outcome, probs=_helper_probabilities)},
        nodes=Product(),
    )
    source = replace(base, group=group, functions={"share": _share})
    target = replace(base, group=group, functions={"share": _different_share})
    with pytest.raises(
        ShockDeclarationError, match=r"eps.*Participating.*alive.*bindings"
    ):
        _validator()(
            query_kind="Participating",
            target="alive",
            physical_inputs=frozenset({"eps"}),
            source_physical_inputs=frozenset({"eps"}),
            projection={"state": _identity},
            source_bindings={"innovation": source},
            target_bindings={"innovation": target},
            source_domains=source.state_domains,
            target_domains=target.state_domains,
        )


def _effectful_identity(state: FloatND) -> FloatND:
    import jax  # noqa: PLC0415

    jax.debug.print("state={state}", state=state)
    return state


def test_effectful_projection_is_not_an_identity_proof() -> None:
    """An effectful staged projection cannot certify shared-partition identity."""
    binding = _binding()
    with pytest.raises(ShockDeclarationError, match=r"eps.*Raw.*alive.*state"):
        _validator()(
            query_kind="Raw",
            target="alive",
            physical_inputs=frozenset({"eps"}),
            source_physical_inputs=frozenset({"eps"}),
            projection={"state": _effectful_identity},
            source_bindings={"innovation": binding},
            target_bindings={"innovation": binding},
            source_domains=binding.state_domains,
            target_domains=binding.state_domains,
        )


def _wide_probabilities(state: ScalarInt) -> FloatND:
    high = jnp.where(state == 0, 0.25, 0.5)
    return jnp.stack((1 - high, high))


def _bitwise_identity(state: ScalarInt) -> ScalarInt:
    return state | 0


@pytest.mark.parametrize("direct", [False, True])
def test_large_finite_domains_need_a_direct_identity_proof(*, direct: bool) -> None:
    """Bound exhaustive build-time work while admitting a direct return at any size."""
    import numpy as np  # noqa: PLC0415

    binding = _binding(
        grid=DiscreteGrid(values=np.arange(5000, dtype=np.int32)),
        probabilities=_wide_probabilities,
    )
    kwargs = {
        "query_kind": "Raw",
        "target": "alive",
        "physical_inputs": frozenset({"eps"}),
        "source_physical_inputs": frozenset({"eps"}),
        "projection": {"state": _identity if direct else _bitwise_identity},
        "source_bindings": {"innovation": binding},
        "target_bindings": {"innovation": binding},
        "source_domains": binding.state_domains,
        "target_domains": binding.state_domains,
    }
    if direct:
        _validator()(**kwargs)
    else:
        with pytest.raises(ShockDeclarationError, match=r"eps.*Raw.*alive.*state"):
            _validator()(**kwargs)
