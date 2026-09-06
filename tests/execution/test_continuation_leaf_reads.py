"""One pytree leaf of a keyed continuation is an addressable stored value.

A parent that reads its target's carry names the artifact key and the leaf path,
so the leaf has an identity of its own for liveness and for transfer planning.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import jax.numpy as jnp
import pytest

from _lcm.egm.carry import build_template_egm_carry
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    ResolvedCoreProgram,
    ValueRead,
    materialize_core_program,
    resolve_core_program,
)
from _lcm.execution.output_layout import VALUE
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
)
from lcm.solver_api import EGM_CONTINUATION


@dataclass(frozen=True, kw_only=True)
class _Builder:
    """An argument builder returning a fixed argument tree."""

    arguments: Mapping[str, object]
    """The exact kwargs the program is materialized and called with."""

    def __call__(self, context: CoreBuildContext) -> Mapping[str, object]:
        """Return the fixed arguments, ignoring the build context."""
        del context
        return MappingProxyType(dict(self.arguments))


def _core(**_kwargs: object) -> object:
    """A core whose value is one row, independent of its arguments."""
    return jnp.zeros(1)


def _context() -> CoreBuildContext:
    """A build context with every channel empty."""
    return CoreBuildContext(
        state_action_space=object(),
        next_regime_to_V_arr=MappingProxyType({}),
        next_regime_to_continuation=MappingProxyType({}),
        flat_params=MappingProxyType({}),
        period=3,
        ages=object(),
    )


def _program(
    *, reads: tuple[ValueRead, ...], arguments: Mapping[str, object]
) -> CoreProgram:
    """A dense one-program graph declaring `reads` over `arguments`."""
    return CoreProgram(
        name="main",
        function=_core,
        argument_builder=_Builder(arguments=arguments),
        requirements=CoreExecutionRequirements(value_reads=reads),
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.DENSE,
        disposition_reason="test_dense_route",
    )


def _resolve(
    *, reads: tuple[ValueRead, ...], arguments: Mapping[str, object]
) -> ResolvedCoreProgram:
    """Materialize and resolve one dense program declaring `reads`."""
    program = _program(reads=reads, arguments=arguments)
    return resolve_core_program(
        program=materialize_core_program(program=program, context=_context())
    )


def _leaf_target(*, leaf: str) -> ValueArtifactAddress:
    """The `retired` regime's period-4 carry leaf named `leaf`."""
    return ValueArtifactAddress(
        kind=ValueArtifactKind.CONTINUATION_LEAF,
        period=4,
        regime="retired",
        artifact_key=EGM_CONTINUATION,
        leaf_path=(leaf,),
    )


def _leaf_read(*, leaf: str, argument: str | None) -> ValueRead:
    """One continuation-leaf read of `retired` by a `working` source at period 3."""
    return ValueRead(
        target=_leaf_target(leaf=leaf),
        source=ValueConsumerAddress(
            source_period=3,
            source_regime="working",
            core_key="main",
            channel=ValueInputChannel.CONTINUATION_LEAF,
            argument=argument,
            path=() if argument else ("retired", leaf),
        ),
    )


def test_a_continuation_leaf_address_names_its_artifact_key() -> None:
    """A continuation leaf is identified by the key of the artifact holding it."""
    assert _leaf_target(leaf="marginal_utility").artifact_key is EGM_CONTINUATION


def test_a_continuation_leaf_address_names_its_leaf_path() -> None:
    """A continuation leaf is identified by its pytree path inside that artifact."""
    assert _leaf_target(leaf="marginal_utility").leaf_path == ("marginal_utility",)


def test_a_regime_value_address_may_not_name_a_continuation_leaf() -> None:
    """Only a continuation-leaf artifact carries a key and a leaf path."""
    with pytest.raises(ValueError, match="leaf_path"):
        ValueArtifactAddress(
            kind=ValueArtifactKind.REGIME_VALUE,
            period=4,
            regime="retired",
            leaf_path=("value",),
        )


def test_a_continuation_leaf_address_requires_a_leaf_path() -> None:
    """An addressed leaf without a path would name the whole artifact."""
    with pytest.raises(ValueError, match="leaf_path"):
        ValueArtifactAddress(
            kind=ValueArtifactKind.CONTINUATION_LEAF,
            period=4,
            regime="retired",
            artifact_key=EGM_CONTINUATION,
        )


def test_a_mapping_channel_read_resolves_a_carry_attribute_leaf() -> None:
    """`next_regime_to_continuation[target].value` is one addressable leaf."""
    read = _leaf_read(leaf="value", argument=None)

    resolved = _resolve(
        reads=(read,),
        arguments={
            ValueInputChannel.CONTINUATION_LEAF.value: MappingProxyType(
                {"retired": build_template_egm_carry(n_rows=5)}
            )
        },
    )

    assert resolved.requirements.value_reads == (read,)


def test_three_direct_argument_reads_share_no_locator() -> None:
    """A builder that flattens carry rows into named arguments addresses each by
    its argument name, and three such reads are three distinct locators."""
    reads = (
        _leaf_read(leaf="endog_grid", argument="next_liquid_grid"),
        _leaf_read(leaf="value", argument="next_value"),
        _leaf_read(leaf="marginal_utility", argument="next_marginal"),
    )

    resolved = _resolve(
        reads=reads,
        arguments={
            "next_liquid_grid": jnp.zeros(5),
            "next_value": jnp.zeros(5),
            "next_marginal": jnp.zeros(5),
        },
    )

    assert resolved.requirements.value_reads == reads


def test_two_reads_with_one_locator_are_refused() -> None:
    """One argument leaf is read once, whether it is named or indexed."""
    read = _leaf_read(leaf="value", argument="next_value")

    with pytest.raises(ValueError, match="duplicate"):
        _resolve(reads=(read, read), arguments={"next_value": jnp.zeros(5)})


def test_a_direct_argument_read_names_an_argument_the_builder_supplies() -> None:
    """An argument the program never builds is refused while it resolves."""
    with pytest.raises(ValueError, match="absent"):
        _resolve(
            reads=(_leaf_read(leaf="marginal_utility", argument="absent"),),
            arguments={"next_marginal": jnp.zeros(5)},
        )
