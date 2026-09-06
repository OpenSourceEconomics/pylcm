"""The terminal-carry decorator adds a carry and forwards everything else.

A terminal regime that a continuation-based parent transitions into gets its
carry published by an engine-owned decorator around the regime's own period
kernel. The decorator's job is additive: every artifact the wrapped kernel
publishes reaches the solve loop on the channel it was published on, and the
carry joins the continuation channel under the EGM continuation key. A base
kernel that already publishes that key is refused rather than overwritten.
"""

from collections.abc import Mapping
from types import MappingProxyType

import jax.numpy as jnp
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.engine import StateActionSpace
from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
)
from _lcm.execution.output_layout import VALUE
from _lcm.regime_building.processing import _TerminalCarryPeriodKernel
from _lcm.utils.logging import get_logger
from lcm.ages import AgeGrid
from lcm.solver_api import (
    DISSOLUTION_FLAG,
    EGM_CONTINUATION,
    SIMULATION_POLICY,
    ArtifactKey,
    KernelOutput,
)


def _carry() -> EGMCarry:
    grid = jnp.linspace(1.0, 4.0, 4)
    return EGMCarry(
        endog_grid=grid,
        value=grid,
        marginal_utility=jnp.ones_like(grid),
        taste_shock_scale=jnp.asarray(0.0),
    )


class _StubKernel:
    """Period kernel returning a fixed output."""

    def __init__(self, output: object) -> None:
        self.output = output

    def with_fixed_params(self, *, fixed_flat_params: object) -> _StubKernel:  # noqa: ARG002
        return self

    def __call__(self, **kwargs: object) -> object:  # noqa: ARG002
        return self.output


class _CoreProgramProvider:
    """Base kernel declaring one recognisable core program."""

    def __init__(self, *, program: CoreProgram) -> None:
        self.program = program

    def core_programs(self) -> Mapping[str, CoreProgram]:
        return MappingProxyType({"main": self.program})


def _call(kernel: _TerminalCarryPeriodKernel) -> object:
    return kernel(
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
        logger=get_logger(log_level="off"),
    )


def _wrap(output: object) -> _TerminalCarryPeriodKernel:
    return _TerminalCarryPeriodKernel(
        base=_StubKernel(output),  # ty: ignore[invalid-argument-type]
        carry_producer=lambda **kwargs: _carry(),  # noqa: ARG005
        regime_name="dead",
    )


def test_terminal_carry_decorator_delegates_exact_core_program_graph():
    """A wrapped native kernel keeps its exact program declaration."""
    program = CoreProgram(
        name="main",
        function=lambda *, value: value,
        argument_builder=lambda _context: MappingProxyType(
            {"value": jnp.asarray([1.0, 2.0])}
        ),
        requirements=CoreExecutionRequirements(),
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.DENSE,
        disposition_reason="test_dense_terminal_carry",
    )
    wrapper = object.__new__(_TerminalCarryPeriodKernel)
    object.__setattr__(wrapper, "base", _CoreProgramProvider(program=program))

    actual = wrapper.core_programs()

    assert actual["main"] is program


def test_terminal_carry_decorator_rejects_program_unaware_base():
    """The decorator cannot recreate a second legacy declaration authority."""
    wrapper = object.__new__(_TerminalCarryPeriodKernel)
    object.__setattr__(wrapper, "base", object())

    with pytest.raises(TypeError, match="native core-program graph"):
        wrapper.core_programs()


def test_the_decorator_adds_the_carry_and_forwards_every_channel():
    """Every artifact stays on its channel; the carry joins the continuations."""
    grid = jnp.linspace(1.0, 4.0, 4)
    published = EGMSimPolicy(
        endog_grid=grid,
        policy=grid,
        value=grid,
        marginal_utility=jnp.ones_like(grid),
    )
    flag = jnp.zeros(4, dtype=jnp.bool_)
    auxiliary_key = ArtifactKey(type_id="example.auxiliary")
    auxiliary = object()
    base_output = KernelOutput(
        value=jnp.zeros(4),
        solve_time_artifacts={DISSOLUTION_FLAG: flag},
        replay={SIMULATION_POLICY: published},
        auxiliary={auxiliary_key: auxiliary},
    )

    output = _call(_wrap(base_output))

    assert isinstance(output, KernelOutput)
    assert output.value is base_output.value
    assert tuple(output.continuations) == (EGM_CONTINUATION,)
    assert isinstance(output.continuations[EGM_CONTINUATION], EGMCarry)
    assert output.replay[SIMULATION_POLICY] is published
    assert output.solve_time_artifacts[DISSOLUTION_FLAG] is flag
    assert output.auxiliary[auxiliary_key] is auxiliary


def test_the_decorator_keeps_a_base_continuation_under_another_key():
    other = ArtifactKey(type_id="example.continuation")
    payload = object()
    base_output = KernelOutput(value=jnp.zeros(4), continuations={other: payload})

    output = _call(_wrap(base_output))

    assert isinstance(output, KernelOutput)
    assert set(output.continuations) == {other, EGM_CONTINUATION}
    assert output.continuations[other] is payload


def test_the_decorator_refuses_a_base_that_already_publishes_the_carry():
    base_output = KernelOutput(
        value=jnp.zeros(4), continuations={EGM_CONTINUATION: _carry()}
    )

    with pytest.raises(RuntimeError, match=r"'dead'.*pylcm\.egm\.continuation"):
        _call(_wrap(base_output))


def test_the_decorator_refuses_a_base_that_returns_no_kernel_output():
    with pytest.raises(TypeError, match=r"'dead'.*KernelOutput"):
        _call(_wrap(object()))
