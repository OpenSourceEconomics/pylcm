"""The terminal-carry decorator adds a carry and forwards everything else.

A terminal regime that a continuation-based parent transitions into gets its
carry published by an engine-owned decorator around the regime's own period
kernel. The decorator's job is additive: the wrapped kernel's generic outputs
have to reach the solve loop unchanged, or a solver publishing an off-grid
simulation policy would have it silently dropped when its regime happens to be
wrapped.
"""

from collections.abc import Callable, Mapping
from types import MappingProxyType

import jax.numpy as jnp

from _lcm.egm.carry import EGMCarry
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.engine import StateActionSpace
from _lcm.regime_building.processing import _TerminalCarryPeriodKernel
from _lcm.solution.contract import KernelResult
from lcm.ages import AgeGrid


def _carry() -> EGMCarry:
    grid = jnp.linspace(1.0, 4.0, 4)
    return EGMCarry(
        endog_grid=grid,
        value=grid,
        marginal_utility=jnp.ones_like(grid),
        taste_shock_scale=jnp.asarray(0.0),
    )


class _StubKernel:
    """Period kernel publishing a recognisable simulation policy."""

    def __init__(self, simulation_policy: object) -> None:
        self.simulation_policy = simulation_policy

    def cores(self) -> Mapping[str, Callable]:
        return MappingProxyType({})

    @property
    def core(self) -> Callable:
        return lambda: None

    def with_fixed_params(self, *, fixed_flat_params: object) -> _StubKernel:  # noqa: ARG002
        return self

    def build_lower_args(self, **kwargs: object) -> Mapping[str, object]:  # noqa: ARG002
        return MappingProxyType({})

    def __call__(self, **kwargs: object) -> KernelResult:  # noqa: ARG002
        return KernelResult(
            V_arr=jnp.zeros(4),
            simulation_policy=self.simulation_policy,  # ty: ignore[invalid-argument-type]
        )


def test_wrapped_simulation_policy_survives_the_terminal_carry_decorator():
    """A base kernel's published simulation policy is forwarded, not reset."""
    grid = jnp.linspace(1.0, 4.0, 4)
    published = EGMSimPolicy(
        endog_grid=grid,
        policy=grid,
        value=grid,
        marginal_utility=jnp.ones_like(grid),
    )
    kernel = _TerminalCarryPeriodKernel(
        base=_StubKernel(published),
        carry_producer=lambda **kwargs: _carry(),  # noqa: ARG005
        regime_name="dead",
    )
    result = kernel(
        compiled_cores=MappingProxyType({}),
        state_action_space=StateActionSpace(
            states=MappingProxyType({}),
            discrete_actions=MappingProxyType({}),
            continuous_actions=MappingProxyType({}),
            state_and_discrete_action_names=(),
        ),
        next_regime_to_V_arr=MappingProxyType({}),
        next_regime_to_continuation=MappingProxyType({}),
        flat_params=MappingProxyType({"dead": MappingProxyType({})}),
        period=0,
        ages=AgeGrid(start=0, stop=2, step="Y"),
    )
    assert result.simulation_policy is published
    assert result.continuation is not None
