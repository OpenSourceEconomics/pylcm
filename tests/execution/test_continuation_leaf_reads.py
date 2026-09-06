"""One pytree leaf of a keyed continuation is an addressable stored value.

A parent that reads its target's carry names the artifact key and the leaf path,
so the leaf has an identity of its own for liveness and for transfer planning.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.egm.carry import build_template_egm_carry
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    MaterializedCoreProgram,
    ProgramScope,
    ResolvedCoreProgram,
    ValueRead,
    _value_read_argument_leaf,
    core_program_graph,
    materialize_core_program,
    resolve_core_program,
)
from _lcm.execution.output_layout import VALUE
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
    ValueTransferKind,
)
from _lcm.solution.backward_induction import (
    _classify_dispatch_value_artifacts,
    _ProgramExecutionMetadata,
)
from lcm.solver_api import EGM_CONTINUATION
from lcm.solvers import EGM
from tests.conftest import DECIMAL_PRECISION
from tests.solution.test_egm_solver import _SAVINGS_GRID, _model
from tests.test_models.dcegm_paper_twin import build_dcegm_model


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


def _replicated_sharding() -> jax.NamedSharding:
    """A replicated sharding over every available device."""
    return jax.NamedSharding(
        mesh=jax.sharding.Mesh(np.asarray(jax.devices()), ("device",)), spec=jax.P()
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


def _materialize(
    *, reads: tuple[ValueRead, ...], arguments: Mapping[str, object]
) -> MaterializedCoreProgram:
    """Materialize one dense program declaring `reads` over `arguments`."""
    return materialize_core_program(
        program=_program(reads=reads, arguments=arguments), context=_context()
    )


def _resolve(
    *, reads: tuple[ValueRead, ...], arguments: Mapping[str, object]
) -> ResolvedCoreProgram:
    """Materialize and resolve one dense program declaring `reads`."""
    return resolve_core_program(program=_materialize(reads=reads, arguments=arguments))


def _carry_arguments() -> Mapping[str, object]:
    """The continuation channel holding `retired`'s five-row carry template."""
    return {
        ValueInputChannel.CONTINUATION_LEAF.value: MappingProxyType(
            {"retired": build_template_egm_carry(n_rows=5)}
        )
    }


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


def test_a_continuation_leaf_address_requires_an_artifact_key() -> None:
    """Without its key a leaf could belong to any stored payload schema."""
    with pytest.raises(TypeError, match="must name its ArtifactKey"):
        ValueArtifactAddress(
            kind=ValueArtifactKind.CONTINUATION_LEAF,
            period=4,
            regime="retired",
            leaf_path=("value",),
        )


def test_a_continuation_leaf_address_may_not_name_an_edge_target_regime() -> None:
    """Only a gated continuation is owned by a source regime and an edge target."""
    with pytest.raises(ValueError, match="cannot name an edge target regime"):
        ValueArtifactAddress(
            kind=ValueArtifactKind.CONTINUATION_LEAF,
            period=4,
            regime="retired",
            target_regime="single_f",
            artifact_key=EGM_CONTINUATION,
            leaf_path=("value",),
        )


def test_a_continuation_leaf_enters_only_through_its_own_channel() -> None:
    """A carry leaf reaches a core as `next_regime_to_continuation` or not at all."""
    sharding = _replicated_sharding()

    with pytest.raises(
        ValueError, match="only through the next_regime_to_continuation"
    ):
        ResolvedValueTransfer(
            target=_leaf_target(leaf="value"),
            source=ValueConsumerAddress(
                source_period=3,
                source_regime="working",
                core_key="main",
                channel=ValueInputChannel.NEXT_REGIME_VALUE,
                path=("retired", "value"),
            ),
            kind=ValueTransferKind.ALIGNED_LOCAL,
            stored_sharding=sharding,
            source_sharding=sharding,
            expected_shape=(5,),
            expected_dtype=jnp.float32,
        )


@pytest.mark.parametrize("leaf_name", ["endog_grid", "value", "marginal_utility"])
def test_a_mapping_channel_read_resolves_a_carry_attribute_leaf(
    *, leaf_name: str
) -> None:
    """`next_regime_to_continuation[target].<leaf>` is one addressable leaf."""
    carry = build_template_egm_carry(n_rows=5)
    expected = {
        "endog_grid": carry.endog_grid,
        "value": carry.value,
        "marginal_utility": carry.marginal_utility,
    }[leaf_name]
    read = _leaf_read(leaf=leaf_name, argument=None)
    program = _materialize(
        reads=(read,),
        arguments={
            ValueInputChannel.CONTINUATION_LEAF.value: MappingProxyType(
                {"retired": carry}
            )
        },
    )

    leaf = cast("jax.Array", _value_read_argument_leaf(program=program, read=read))

    aaae(leaf, expected, decimal=DECIMAL_PRECISION)


def test_a_leaf_path_naming_no_carry_field_is_refused() -> None:
    """A path step that names nothing on the carry addresses no leaf at all."""
    read = _leaf_read(leaf="absent_row", argument=None)
    program = _materialize(reads=(read,), arguments=_carry_arguments())

    with pytest.raises(ValueError, match="traverses a non-container value"):
        _value_read_argument_leaf(program=program, read=read)


def test_a_leaf_path_reaching_an_absent_carry_row_is_refused() -> None:
    """A carry field the template leaves unpublished resolves to no array leaf."""
    read = _leaf_read(leaf="breakpoints", argument=None)
    program = _materialize(reads=(read,), arguments=_carry_arguments())

    with pytest.raises(TypeError, match="array-like leaf with shape and dtype"):
        _value_read_argument_leaf(program=program, read=read)


def test_three_direct_argument_reads_share_no_locator() -> None:
    """Three named-argument reads of one carry are three distinct locators.

    A builder that flattens carry rows into program arguments addresses each row
    by its argument name, so channel and path coincide across the three reads and
    only the argument tells them apart.
    """
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

    assert tuple(
        (read.source.channel, read.source.path, read.source.argument)
        for read in resolved.requirements.value_reads
    ) == (
        (ValueInputChannel.CONTINUATION_LEAF, (), "next_liquid_grid"),
        (ValueInputChannel.CONTINUATION_LEAF, (), "next_value"),
        (ValueInputChannel.CONTINUATION_LEAF, (), "next_marginal"),
    )


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


def _egm_program() -> CoreProgram:
    """The plain-EGM `main` program of the first solved period."""
    model = _model(solver=EGM(savings_grid=_SAVINGS_GRID))
    kernel = model._regimes["saving"].solution.period_kernels[0]
    return core_program_graph(kernel=kernel)["main"]


@pytest.mark.parametrize(
    ("leaf", "argument"),
    [
        (("endog_grid",), "next_liquid_grid"),
        (("value",), "next_value"),
        (("marginal_utility",), "next_marginal"),
    ],
)
def test_plain_egm_declares_the_carry_row_each_argument_carries(
    *, leaf: tuple[str, ...], argument: str
) -> None:
    """The 1-D EGM program names the target carry leaf behind each argument."""
    reads = _egm_program().requirements.value_reads

    assert {(read.target.leaf_path, read.source.argument) for read in reads} >= {
        (leaf, argument)
    }


def test_plain_egm_declares_exactly_the_three_rows_it_reads() -> None:
    """The EGM argument builder supplies three carry rows and no more."""
    assert len(_egm_program().requirements.value_reads) == 3


@pytest.mark.parametrize(
    "predicate",
    [
        lambda read: read.target.kind is ValueArtifactKind.CONTINUATION_LEAF,
        lambda read: read.target.period == read.source.source_period + 1,
        lambda read: read.target.artifact_key is EGM_CONTINUATION,
    ],
)
def test_every_plain_egm_read_addresses_the_targets_next_period_carry(
    *, predicate: Callable[[ValueRead], bool]
) -> None:
    """Each declared read names the target's carry at the next period."""
    assert all(predicate(read) for read in _egm_program().requirements.value_reads)


@pytest.mark.parametrize(
    "predicate",
    [
        lambda read: read.source.source_period == 0,
        lambda read: read.source.source_regime == "saving",
        lambda read: read.source.core_key == "main",
    ],
)
def test_every_plain_egm_read_locates_the_consumer_that_runs_it(
    *, predicate: Callable[[ValueRead], bool]
) -> None:
    """A period-0 read names period 0, the `saving` regime and the `main` core.

    The consumer coordinates are absolute, so a read shifted whole — target and
    source together — is a different read even though the two still sit one
    period apart.
    """
    assert all(predicate(read) for read in _egm_program().requirements.value_reads)


def _dcegm_reads() -> tuple[ValueRead, ...]:
    """The DC-EGM twin's `main` reads at its first solved period."""
    model = build_dcegm_model()
    kernel = model._regimes["working_life"].solution.period_kernels[0]
    return core_program_graph(kernel=kernel)["main"].requirements.value_reads


def test_dcegm_declares_at_least_one_read() -> None:
    """A DC-EGM program reads its target's carry, so it declares it."""
    assert _dcegm_reads()


@pytest.mark.parametrize(
    "predicate",
    [
        lambda read: read.source.channel is ValueInputChannel.CONTINUATION_LEAF,
        lambda read: read.source.argument is None,
        lambda read: read.source.path == (read.target.regime, *read.target.leaf_path),
    ],
)
def test_dcegm_addresses_each_leaf_inside_the_rolling_mapping(
    *, predicate: Callable[[ValueRead], bool]
) -> None:
    """A DC-EGM program takes the whole payload, so it indexes the channel."""
    assert all(predicate(read) for read in _dcegm_reads())


def test_an_egm_node_pins_only_the_continuation_leaves_it_declares() -> None:
    """A dense EGM node that names its reads stops pinning every reachable value."""
    program = _egm_program()
    programs = {
        "main": _ProgramExecutionMetadata(
            requirements=program.requirements,
            disposition=program.disposition,
            scope=ProgramScope.ANY,
            input_transfer_plan=(),
        )
    }

    planned, unplanned_exact, has_unknown = _classify_dispatch_value_artifacts(
        programs=programs
    )

    assert (planned, has_unknown) == ((), False)
    assert {artifact.leaf_path for artifact in unplanned_exact} == {
        ("endog_grid",),
        ("value",),
        ("marginal_utility",),
    }
