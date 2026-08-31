"""The NB-EGM build probes fill each budget argument to its declared contract.

The affinity / interval-constancy probes differentiate the composed budget on
synthetic inputs. A budget DAG mixes 0-d scalar parameters (a rate multiplied
onto the liquid state) with array-valued schedule tables (a row indexed by a
discrete code). No single global fill satisfies both — a unit-1D fill violates a
scalar parameter's 0-d contract, a 0-d fill cannot be indexed as a table — so the
probe classifies each argument from the annotations its consumers declare.

An array annotation states that the argument is an array, not how many axes it
has: the rank-polymorphic aliases (`FloatND`, `IntND`) cover a schedule read with
one index and a table read with two alike. The fill rank is therefore not
declared anywhere, and the probe escalates it until the DAG evaluates.

The consumers whose annotations count are every function the probed DAG can
reach. The constancy probe differentiates laws of motion, so a parameter that
only ever appears in a state-transition law is classified from that law.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import pytest
from dags.signature import rename_arguments

from _lcm.params.mapping_leaf import MappingLeaf
from _lcm.solution.nbegm import (
    _annotated_bool_arg_names,
    _annotated_int_arg_names,
    _annotated_mapping_leaf_arg_names,
    _annotation_source_functions,
    _array_float_arg_names,
    _evaluate_on_first_workable_fill,
    _indexed_arg_ranks,
    _probe_fill,
)
from lcm.typing import (
    ContinuousState,
    Float1D,
    FloatND,
    IntND,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
)


def _array_fill(**kwargs: object) -> FloatND:
    """`_probe_fill`'s result where an array is what the classification asks for."""
    fill = _probe_fill(**kwargs)  # ty: ignore[invalid-argument-type]
    assert isinstance(fill, jax.Array)
    return fill


def _schedule_fill(**kwargs: object) -> MappingLeaf:
    """`_probe_fill`'s result where a grouped param is what it asks for."""
    fill = _probe_fill(**kwargs)  # ty: ignore[invalid-argument-type]
    assert isinstance(fill, MappingLeaf)
    return fill


def _schedule_entry(*, schedule: MappingLeaf, key: str) -> FloatND:
    """One entry of a grouped param, as the array its consumers read."""
    return cast("Mapping[str, FloatND]", schedule.data)[key]


@dataclass(frozen=True)
class _FakeRegime:
    """Stand-in carrying only the regime slot the classifiers read."""

    functions: Mapping[str, Callable[..., object]] = field(default_factory=dict)
    """The regime's functions, keyed by name."""


def _rate_term(*, liquid: ContinuousState, rate_of_return: ScalarFloat) -> FloatND:
    return liquid * rate_of_return


def _table_term(*, schedule: Float1D, code: int) -> FloatND:
    return schedule[code]


def _reads_rate_as_scalar(rate_of_return: ScalarFloat) -> FloatND:
    return jnp.asarray(rate_of_return)


def _reads_rate_as_array(rate_of_return: Float1D) -> FloatND:
    return rate_of_return


def _reads_insurance_code(insurance_status: IntND) -> FloatND:
    return jnp.asarray(insurance_status, dtype=jnp.float64)


def _reads_repeal_age(repeal_age: ScalarInt) -> FloatND:
    return jnp.asarray(repeal_age, dtype=jnp.float64)


def _reads_repeal_age_as_float(repeal_age: ScalarFloat) -> FloatND:
    return jnp.asarray(repeal_age)


def test_array_float_arg_names_includes_an_array_typed_param() -> None:
    """A leaf param annotated as a 1-D array is marked for unit-1D fill."""
    names = _array_float_arg_names(functions={"table_term": _table_term})
    assert "schedule" in names


def test_array_float_arg_names_excludes_a_scalar_typed_param() -> None:
    """A leaf param annotated as a 0-d scalar is never marked for array fill."""
    names = _array_float_arg_names(functions={"rate_term": _rate_term})
    assert "rate_of_return" not in names


def test_array_float_arg_names_lets_a_scalar_annotation_win_on_conflict() -> None:
    """A param any consumer annotates 0-d stays scalar (else its contract breaks)."""
    names = _array_float_arg_names(
        functions={"a": _reads_rate_as_scalar, "b": _reads_rate_as_array}
    )
    assert "rate_of_return" not in names


def test_probe_fill_gives_a_classified_array_arg_unit_1d() -> None:
    """An arg in the array set fills to shape `(1,)` so a scalar index clamps in."""
    table = _array_fill(
        name="schedule",
        fill=1.0,
        int_arg_names=frozenset(),
        array_float_arg_names=frozenset({"schedule"}),
    )
    assert jnp.shape(table) == (1,)


def test_probe_fill_keeps_an_unclassified_float_arg_scalar() -> None:
    """A float arg outside the array set stays 0-d, honouring its scalar contract."""
    scalar = _array_fill(
        name="rate_of_return",
        fill=1.0,
        int_arg_names=frozenset(),
        array_float_arg_names=frozenset({"schedule"}),
    )
    assert jnp.ndim(scalar) == 0


def test_annotated_int_arg_names_includes_a_rank_polymorphic_int_param() -> None:
    """A DAG intermediate annotated `IntND` is marked for an integer fill."""
    names = _annotated_int_arg_names(functions={"reads": _reads_insurance_code})
    assert "insurance_status" in names


def test_annotated_int_arg_names_includes_a_scalar_int_param() -> None:
    """A fixed parameter annotated `ScalarInt` is marked for an integer fill.

    Integer-valued parameters need not back a `DiscreteGrid`; an age threshold is
    an ordinary flat param whose only declaration of integer-ness is its
    annotation.
    """
    names = _annotated_int_arg_names(functions={"reads": _reads_repeal_age})
    assert "repeal_age" in names


def test_annotated_int_arg_names_excludes_a_float_param() -> None:
    """A float-annotated parameter is never marked for an integer fill."""
    names = _annotated_int_arg_names(functions={"rate_term": _rate_term})
    assert "rate_of_return" not in names


def test_annotated_int_arg_names_lets_a_float_annotation_win_on_conflict() -> None:
    """A param any consumer annotates float keeps its float fill.

    An integer fill would violate that consumer, so a name whose annotations
    disagree is left to the float default rather than guessed at.
    """
    names = _annotated_int_arg_names(
        functions={"a": _reads_repeal_age, "b": _reads_repeal_age_as_float}
    )
    assert "repeal_age" not in names


def test_probe_fill_gives_an_annotated_int_arg_an_integer_fill() -> None:
    """An arg classified integer by annotation fills as int32, not float."""
    code = _array_fill(
        name="insurance_status", fill=1.0, int_arg_names=frozenset({"insurance_status"})
    )
    assert code.dtype == jnp.int32


def test_probe_fill_gives_an_array_arg_the_requested_rank() -> None:
    """A table read with two indices fills to shape `(1, 1)` at array rank 2."""
    table = _array_fill(
        name="schedule",
        fill=1.0,
        int_arg_names=frozenset(),
        array_float_arg_names=frozenset({"schedule"}),
        array_rank=2,
    )
    assert jnp.shape(table) == (1, 1)


def test_probe_fill_keeps_a_scalar_arg_zero_d_at_a_higher_array_rank() -> None:
    """Escalating the array rank leaves a scalar parameter 0-d.

    The rank ladder exists for rank-polymorphic tables; a parameter its consumers
    annotate 0-d has a rank already, and raising it would violate that contract.
    """
    scalar = _array_fill(
        name="rate_of_return",
        fill=1.0,
        int_arg_names=frozenset(),
        array_float_arg_names=frozenset({"schedule"}),
        array_rank=2,
    )
    assert jnp.ndim(scalar) == 0


def _reads_a_schedule(*, income: FloatND, tax_schedule: MappingLeaf) -> FloatND:
    return income * _schedule_entry(schedule=tax_schedule, key="marginal_rates")[0, 0]


def test_annotated_mapping_leaf_arg_names_includes_a_schedule_param() -> None:
    """A parameter annotated `MappingLeaf` is marked for a mapping-leaf fill."""
    names = _annotated_mapping_leaf_arg_names(functions={"tax": _reads_a_schedule})
    assert "tax_schedule" in names


def test_annotated_mapping_leaf_arg_names_reads_a_stringified_annotation() -> None:
    """A composed function's stringified annotation classifies its argument.

    A DAG wrapper carries its parameters' types as names rather than objects, and
    those parameters are exactly the leaves the probe fills.
    """

    def _composed(tax_schedule: object) -> FloatND:  # noqa: ARG001
        return jnp.asarray(0.0)

    _composed.__annotations__ = {"tax_schedule": "MappingLeaf"}
    names = _annotated_mapping_leaf_arg_names(functions={"composed": _composed})
    assert "tax_schedule" in names


def test_annotated_mapping_leaf_arg_names_ignores_an_unresolvable_annotation() -> None:
    """An annotation naming nothing the probe knows decides nothing.

    Treating it as evidence of a non-grouped type would silently withdraw the
    classification every other consumer of that parameter agrees on.
    """

    def _composed(tax_schedule: object) -> FloatND:  # noqa: ARG001
        return jnp.asarray(0.0)

    _composed.__annotations__ = {"tax_schedule": "SomeTypeThePr obeCannotResolve"}
    names = _annotated_mapping_leaf_arg_names(
        functions={"declared": _reads_a_schedule, "composed": _composed}
    )
    assert "tax_schedule" in names


def test_annotated_mapping_leaf_arg_names_excludes_an_array_param() -> None:
    """An array-annotated parameter is never marked for a mapping-leaf fill."""
    names = _annotated_mapping_leaf_arg_names(functions={"table": _table_term})
    assert "schedule" not in names


def test_probe_fill_gives_a_schedule_arg_a_mapping_leaf() -> None:
    """A schedule argument fills to a `MappingLeaf`, satisfying its declared type."""
    schedule = _probe_fill(
        name="tax_schedule",
        fill=1.0,
        int_arg_names=frozenset(),
        mapping_leaf_arg_names=frozenset({"tax_schedule"}),
    )
    assert isinstance(schedule, MappingLeaf)


def test_probe_fill_answers_any_schedule_key_at_the_requested_rank() -> None:
    """Every key of a filled schedule answers with an array of the probed rank.

    Which keys a schedule carries is a property of the params, which arrive long
    after the kernels are built, so the probe cannot enumerate them and answers
    whatever the model's own code asks for.
    """
    schedule = _schedule_fill(
        name="tax_schedule",
        fill=1.0,
        int_arg_names=frozenset(),
        mapping_leaf_arg_names=frozenset({"tax_schedule"}),
        leaf_rank=2,
    )
    assert jnp.shape(
        _schedule_entry(schedule=schedule, key="never_declared_anywhere")
    ) == (1, 1)


def test_probe_fill_schedule_enters_a_compiled_call() -> None:
    """A filled schedule is an argument JAX can carry into a compiled function.

    A probe evaluates DAGs that are compiled by the time it reaches them, and JAX
    takes each argument apart against its registered structure. A fill it has to
    treat as one opaque value is rejected as not-an-array, stopping the probe for
    a reason that has nothing to do with the model it was called to check.
    """
    schedule = _schedule_fill(
        name="tax_schedule",
        fill=3.0,
        int_arg_names=frozenset(),
        mapping_leaf_arg_names=frozenset({"tax_schedule"}),
    )

    @jax.jit
    def _taxed(*, liquid: FloatND, tax_schedule: MappingLeaf) -> FloatND:
        return liquid * _schedule_entry(schedule=tax_schedule, key="marginal_rates")[0]

    assert float(_taxed(liquid=jnp.asarray(2.0), tax_schedule=schedule)) == 6.0


def test_probe_fill_ranks_schedule_contents_apart_from_plain_array_args() -> None:
    """A schedule's contents escalate without dragging plain array args along.

    An array parameter's own axis count is readable from its consumers, and some
    are strict about it — an interpolation table has to stay 1-D — while what a
    schedule holds is readable from nothing, so the two escalate separately.
    """
    row = _array_fill(
        name="interpolation_table",
        fill=1.0,
        int_arg_names=frozenset(),
        array_float_arg_names=frozenset({"interpolation_table"}),
        mapping_leaf_arg_names=frozenset({"tax_schedule"}),
        leaf_rank=3,
    )
    assert jnp.shape(row) == (1,)


def _reads_a_threshold_flag(crossed_threshold: ScalarBool) -> FloatND:
    return jnp.where(crossed_threshold, 1.0, 0.0)


def _reads_the_flag_as_float(crossed_threshold: ScalarFloat) -> FloatND:
    return jnp.asarray(crossed_threshold)


def test_annotated_bool_arg_names_includes_a_boolean_param() -> None:
    """A DAG intermediate annotated `ScalarBool` is marked for a boolean fill."""
    names = _annotated_bool_arg_names(functions={"eligible": _reads_a_threshold_flag})
    assert "crossed_threshold" in names


def test_annotated_bool_arg_names_excludes_a_float_param() -> None:
    """A float-annotated parameter is never marked for a boolean fill."""
    names = _annotated_bool_arg_names(functions={"rate_term": _rate_term})
    assert "rate_of_return" not in names


def test_annotated_bool_arg_names_lets_a_float_annotation_win_on_conflict() -> None:
    """A param any consumer annotates float keeps its float fill."""
    names = _annotated_bool_arg_names(
        functions={"a": _reads_a_threshold_flag, "b": _reads_the_flag_as_float}
    )
    assert "crossed_threshold" not in names


def test_annotated_int_arg_names_excludes_a_boolean_param() -> None:
    """A boolean parameter is not an integer one: its fill must stay boolean.

    JAX admits a boolean where an integer is expected, but the runtime type
    contract does not, and an integer fill violates the declared annotation.
    """
    names = _annotated_int_arg_names(functions={"eligible": _reads_a_threshold_flag})
    assert "crossed_threshold" not in names


def test_probe_fill_gives_an_annotated_bool_arg_a_boolean_fill() -> None:
    """An arg classified boolean by annotation fills as bool, not float."""
    flag = _array_fill(
        name="crossed_threshold",
        fill=3.0,
        int_arg_names=frozenset(),
        bool_arg_names=frozenset({"crossed_threshold"}),
    )
    assert flag.dtype == jnp.bool_


@pytest.mark.parametrize(("fill", "expected"), [(1.0, False), (3.0, True)])
def test_probe_fill_reaches_both_sides_of_a_boolean_arg(
    *,
    fill: float,
    expected: bool,
) -> None:
    """The probe's constant fills put a boolean argument on both branches.

    A flag pinned to one value would leave the other branch of every gate it
    controls unprobed, so the fill level the probes already sweep decides it.
    """
    flag = _array_fill(
        name="crossed_threshold",
        fill=fill,
        int_arg_names=frozenset(),
        bool_arg_names=frozenset({"crossed_threshold"}),
    )
    assert bool(flag) is expected


def _reads_a_two_index_table(
    *, period: ScalarInt, code: IntND, table: FloatND
) -> FloatND:
    return table[period, code]


def _reads_a_one_index_row(*, period: ScalarInt, row: FloatND) -> FloatND:
    return row[period]


def _reads_the_same_table_with_one_index(
    *, period: ScalarInt, table: FloatND
) -> FloatND:
    return table[period]


def test_indexed_arg_ranks_reads_a_two_index_table_as_two_axes() -> None:
    """A parameter its consumer subscripts with two indices is filled 2-D."""
    ranks = _indexed_arg_ranks(functions={"benefit": _reads_a_two_index_table})
    assert ranks["table"] == 2


def test_indexed_arg_ranks_reads_a_one_index_row_as_one_axis() -> None:
    """A parameter its consumer subscripts with one index is filled 1-D."""
    ranks = _indexed_arg_ranks(functions={"benefit": _reads_a_one_index_row})
    assert ranks["row"] == 1


def test_indexed_arg_ranks_takes_the_deepest_read_across_consumers() -> None:
    """A parameter read at two depths is filled for the deeper read.

    The shallower read still lands: indexing a 2-D fill once yields a row, and a
    one-axis fill could not carry the deeper read at all.
    """
    ranks = _indexed_arg_ranks(
        functions={
            "shallow": _reads_the_same_table_with_one_index,
            "deep": _reads_a_two_index_table,
        }
    )
    assert ranks["table"] == 2


def test_indexed_arg_ranks_keys_a_renamed_parameter_by_its_signature_name() -> None:
    """A parameter renamed for the DAG is keyed by the name the probe fills.

    Processing qualifies each function's parameters, while the body the rank is
    read from still spells them the way the model author wrote them.
    """
    qualified = rename_arguments(
        _reads_a_two_index_table, mapper={"table": "benefit__table"}
    )
    ranks = _indexed_arg_ranks(functions={"benefit": qualified})
    assert ranks["benefit__table"] == 2


def test_probe_fill_takes_an_array_arg_rank_from_the_inferred_ranks() -> None:
    """An argument read with two indices fills to `(1, 1)` on the first rung."""
    table = _array_fill(
        name="schedule",
        fill=1.0,
        int_arg_names=frozenset(),
        array_float_arg_names=frozenset({"schedule"}),
        array_arg_ranks={"schedule": 2},
    )
    assert jnp.shape(table) == (1, 1)


def test_probe_fill_leaves_a_scalar_arg_zero_d_despite_an_inferred_rank() -> None:
    """A 0-d parameter stays 0-d even when a same-named read was inferred.

    Rank inference reads subscripts, which say nothing about whether the argument
    is an array at all — that is the annotation's job, and it wins.
    """
    scalar = _array_fill(
        name="rate_of_return",
        fill=1.0,
        int_arg_names=frozenset(),
        array_float_arg_names=frozenset({"schedule"}),
        array_arg_ranks={"rate_of_return": 2},
    )
    assert jnp.ndim(scalar) == 0


def test_probe_fill_ladder_evaluates_on_the_first_workable_array_rank() -> None:
    """A DAG that indexes a table twice is probed at array rank 2.

    A one-axis fill cannot carry two indices, and no annotation states the rank,
    so the ladder raises it until the DAG evaluates.
    """

    def _evaluate(
        *,
        array_floats: bool,  # noqa: ARG001
        array_rank: int,  # noqa: ARG001
        leaf_rank: int,
    ) -> int:
        if leaf_rank < 2:
            msg = "Too many indices: array is 1-dimensional, but 2 were indexed"
            raise IndexError(msg)
        return leaf_rank

    assert _evaluate_on_first_workable_fill(_evaluate) == 2


def test_probe_fill_ladder_reports_the_classified_rung_when_none_works() -> None:
    """An unprobeable DAG is reported against its annotation-classified fill.

    The coarse final rung violates every 0-d parameter by construction, so its
    error names an argument whose contract the model states correctly; the first
    rung honours the declared contracts and its failure is the real one.
    """

    def _evaluate(*, array_floats: bool, array_rank: int, leaf_rank: int) -> int:  # noqa: ARG001
        msg = f"rank {array_rank}, coarse {array_floats}"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="rank 1, coarse False"):
        _evaluate_on_first_workable_fill(_evaluate)


def test_annotation_source_functions_includes_a_state_transition_law() -> None:
    """A law of motion contributes its parameter annotations to the classifiers.

    A parameter that appears only in a law of motion — an age at which a rule is
    repealed, read by the AIME accrual law — is otherwise unclassified, and its
    integer contract is violated by the float default.
    """
    sources = _annotation_source_functions(
        functions=MappingProxyType({}),
        transitions=MappingProxyType(
            {"retired": MappingProxyType({"next_aime": _reads_repeal_age})}
        ),
    )
    assert "repeal_age" in _annotated_int_arg_names(functions=sources)


def test_annotation_source_functions_includes_a_reachable_target_regime() -> None:
    """A function the continuation reaches in a target regime is classified too.

    The constancy probe differentiates a law composed against the child regime's
    own DAG, so a parameter only that regime declares still needs a fill matching
    its contract.
    """
    sources = _annotation_source_functions(
        functions=MappingProxyType({}),
        transitions=MappingProxyType({"retired": MappingProxyType({})}),
        user_regimes=MappingProxyType(
            {"retired": _FakeRegime(functions={"eligible": _reads_a_threshold_flag})}
        ),
    )
    assert "crossed_threshold" in _annotated_bool_arg_names(functions=sources)


def test_annotation_source_functions_skips_an_unreachable_regime() -> None:
    """A regime the transition cannot reach contributes no classification.

    Its declarations cannot appear in this regime's continuation, and folding
    them in would let an unrelated name collide with one that does.
    """
    sources = _annotation_source_functions(
        functions=MappingProxyType({}),
        transitions=MappingProxyType({"retired": MappingProxyType({})}),
        user_regimes=MappingProxyType(
            {
                "retired": _FakeRegime(functions={}),
                "working": _FakeRegime(functions={"eligible": _reads_a_threshold_flag}),
            }
        ),
    )
    assert "crossed_threshold" not in _annotated_bool_arg_names(functions=sources)


def test_annotated_arg_names_also_key_a_parameter_by_its_processed_name() -> None:
    """A classification is keyed by the name the probe will fill.

    Processing qualifies a function's parameters with the function's own name,
    and the probe fills the composed DAG's leaves, so the classification has to
    answer to the qualified spelling as well as the declared one.
    """
    names = _annotated_bool_arg_names(functions={"eligible": _reads_a_threshold_flag})
    assert "eligible__crossed_threshold" in names


def test_annotation_source_functions_includes_the_regime_transition() -> None:
    """The regime-transition function contributes its parameter annotations.

    The constancy probe differentiates it alongside the state laws, so a parameter
    only it reads needs the same classification.
    """
    sources = _annotation_source_functions(
        functions=MappingProxyType({}),
        transitions=MappingProxyType({}),
        compute_regime_transition_probs=_reads_insurance_code,
    )
    assert "insurance_status" in _annotated_int_arg_names(functions=sources)


def test_annotation_source_functions_keeps_every_econ_function() -> None:
    """Econ functions stay classified when transition laws are added alongside."""
    sources = _annotation_source_functions(
        functions=MappingProxyType({"table_term": _table_term}),
        transitions=MappingProxyType(
            {"retired": MappingProxyType({"next_aime": _reads_repeal_age})}
        ),
    )
    assert "schedule" in _array_float_arg_names(functions=sources)


def test_probe_fill_returns_the_models_own_value_for_a_declared_parameter() -> None:
    """A parameter the model declares answers with its value, not a synthetic fill."""
    value = _probe_fill(
        name="utility__crra",
        fill=1.0,
        int_arg_names=frozenset(),
        param_values=MappingProxyType({"utility__crra": jnp.asarray(2.5)}),
    )

    assert float(cast("FloatND", value)) == 2.5


def test_probe_fill_prefers_a_real_parameter_over_its_annotated_shape() -> None:
    """A real value wins over the shape the argument's annotation would synthesize."""
    table = jnp.asarray([[0.1, 0.2], [0.3, 0.4]])

    value = _probe_fill(
        name="taxes__marginal_rates",
        fill=1.0,
        int_arg_names=frozenset(),
        array_float_arg_names=frozenset({"taxes__marginal_rates"}),
        param_values=MappingProxyType({"taxes__marginal_rates": table}),
    )

    assert jnp.shape(cast("FloatND", value)) == (2, 2)


def test_probe_fill_hands_back_a_grouped_parameter_unchanged() -> None:
    """A declared tax schedule reaches the probe as the model's own schedule."""
    schedule = MappingLeaf(
        {"brackets_upper": jnp.asarray([10.0, 20.0]), "rates": jnp.asarray([0.1, 0.3])}
    )

    value = _probe_fill(
        name="taxes__income_tax_schedule",
        fill=1.0,
        int_arg_names=frozenset(),
        mapping_leaf_arg_names=frozenset({"taxes__income_tax_schedule"}),
        param_values=MappingProxyType({"taxes__income_tax_schedule": schedule}),
    )

    assert value is schedule


def test_probe_fill_synthesizes_an_argument_that_is_not_a_parameter() -> None:
    """A state, action, or unbound DAG intermediate still gets a synthetic fill."""
    value = _probe_fill(
        name="liquid",
        fill=3.0,
        int_arg_names=frozenset(),
        param_values=MappingProxyType({"utility__crra": jnp.asarray(2.5)}),
    )

    assert float(cast("FloatND", value)) == 3.0
