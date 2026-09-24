"""A negative reduce-fusion verdict needs a complete reading and a proved table origin.

The texts are synthetic edits of the trimmed GPU fixtures; nothing is compiled.
Refusing the module, or answering `UNKNOWN` or `MATERIALISED_GATHER`, is sound for
them. Answering `FUSED_GATHER`, `NO_GATHER` or nothing at all is not.
"""

import itertools
from collections.abc import Callable

import pytest

from _lcm.config import TEST_DATA
from _lcm.execution.hlo_fusions import (
    ReduceFusionVerdict,
    UnrecognisedHloError,
    classify_reduce_fusions,
)

# Every non-empty set of verdicts that claims no fused or gather-free table.
_SOUND_RESULTS = frozenset(
    {
        frozenset({ReduceFusionVerdict.MATERIALISED_GATHER}),
        frozenset({ReduceFusionVerdict.UNKNOWN}),
        frozenset(
            {ReduceFusionVerdict.MATERIALISED_GATHER, ReduceFusionVerdict.UNKNOWN}
        ),
    }
)


def _hlo(name: str) -> str:
    return TEST_DATA.joinpath("hlo_reduce_fusions", name).read_text()


def _verdicts_or_refusal(text: str) -> frozenset[ReduceFusionVerdict]:
    """The set of verdicts, with a refused module counted as `UNKNOWN`."""
    try:
        return frozenset(classify_reduce_fusions(text).values())
    except UnrecognisedHloError:
        return frozenset({ReduceFusionVerdict.UNKNOWN})


def _with_unreadable_entry_fusion() -> str:
    """The fused fixture with its reduce-fusion call spelled unreadably in ENTRY."""
    return _hlo("fused.hlo.txt").replace(
        "ROOT %input_reduce_fusion =", "ROOT %input_reduce_fusion :="
    )


def _with_tuple_projected_table(views: tuple[str, ...]) -> str:
    """The materialised fixture with its table passed into the reduce fusion as a tuple.

    The separate `%loop_gather_fusion` still writes the table. The reduce fusion
    projects it out of a tuple parameter and moves it through `views` and a final
    copy before gathering from it, which creates no new table.
    """
    shape = "f32[945,1,1,1,24,38]{5,4,3,2,1,0}"
    lines = [
        f"  %tuple_param = ({shape}) parameter(1)",
        f"  %projected = {shape} get-tuple-element(%tuple_param), index=0",
    ]
    previous = "projected"
    for position, opcode in enumerate(views):
        lines.append(f"  %view_{position} = {shape} {opcode}(%{previous})")
        previous = f"view_{position}"
    lines.append(f"  %param_4.3285 = {shape} copy(%{previous})")
    return (
        _hlo("materialised.hlo.txt")
        .replace(
            "param_4.3285: f32[945,1,1,1,24,38]) ->",
            "param_4.3285: (f32[945,1,1,1,24,38])) ->",
        )
        .replace(f"  %param_4.3285 = {shape} parameter(1)", "\n".join(lines))
        .replace(
            "fusion(%Arg_2.3, %get-tuple-element.3547), kind=kInput",
            "fusion(%Arg_2.3, %loop_gather_fusion), kind=kInput",
        )
    )


_VIEW_CHAINS = [
    chain
    for depth in range(5)
    for chain in itertools.product(("copy", "bitcast"), repeat=depth)
]


def test_classify_reduce_fusions_does_not_lose_an_unreadable_entry_fusion() -> None:
    """A reduce fusion whose ENTRY call is unreadable yields `UNKNOWN`, not nothing."""
    assert _verdicts_or_refusal(_with_unreadable_entry_fusion()) == {
        ReduceFusionVerdict.UNKNOWN
    }


def test_tuple_projected_table_fixture_edits_every_anchor() -> None:
    """The synthetic edit rewrites the header, the parameter and the ENTRY call."""
    text = _with_tuple_projected_table(("copy",))
    assert all(
        needle in text
        for needle in (
            "(f32[945,1,1,1,24,38])) ->",
            "get-tuple-element(%tuple_param), index=0",
            "fusion(%Arg_2.3, %loop_gather_fusion), kind=kInput",
        )
    )


@pytest.mark.parametrize("views", _VIEW_CHAINS)
def test_classify_reduce_fusions_needs_an_origin_for_a_tuple_projected_table(
    *, views: tuple[str, ...]
) -> None:
    """A table projected from a tuple parameter is never assumed fused."""
    verdicts = _verdicts_or_refusal(_with_tuple_projected_table(views))
    assert verdicts in _SOUND_RESULTS


_LOOP_STATE = (
    "(s64[], f32[1,3,5,24,38]{4,3,2,1,0}, s32[945,3]{1,0}, s32[13,7,2]{2,1,0}, "
    "f32[13,38,7]{2,1,0})"
)
_LOOP_TABLE = "f32[1,3,5,24,38]{4,3,2,1,0}"
_GATHER_PRODUCER = (
    "%fused_table (param_0.7: f32[1,3,5,24,38], param_1.7: s32[1,5]) "
    "-> f32[1,3,5,24,38] {\n"
    f"  %param_0.7 = {_LOOP_TABLE} parameter(0)\n"
    "  %param_1.7 = s32[1,5]{1,0} parameter(1)\n"
    f"  ROOT %gather.7 = {_LOOP_TABLE} gather(%param_0.7, %param_1.7), "
    "offset_dims={1,2,3,4}, collapsed_slice_dims={0}, start_index_map={0,1,2,3,4}, "
    "index_vector_dim=1, slice_sizes={1,3,5,24,38}\n"
    "}\n\n"
)


def _replace_once(*, text: str, old: str, new: str) -> str:
    """Replace the single occurrence of `old`, which must occur exactly once."""
    assert text.count(old) == 1, old
    return text.replace(old, new)


def _with_gathered_loop_input() -> str:
    """The loop fixture with the carried table written by a gather fusion in ENTRY."""
    text = _replace_once(
        text=_hlo("while_invariant_table.hlo.txt"),
        old="\nENTRY %main.241",
        new=f"\n{_GATHER_PRODUCER}ENTRY %main.241",
    )
    return _replace_once(
        text=text,
        old=f"  %tuple.593 = {_LOOP_STATE} tuple(%constant_0, %Arg_0.1,",
        new=f"  %loop_gather_fusion = {_LOOP_TABLE} fusion(%Arg_0.1, %Arg_1.2), "
        "kind=kLoop, calls=%fused_table\n"
        f"  %tuple.593 = {_LOOP_STATE} tuple(%constant_0, %loop_gather_fusion,",
    )


def _with_loop_carried_table() -> str:
    """The loop fixture with the body writing a new table into the carried slot."""
    text = _replace_once(
        text=_hlo("while_invariant_table.hlo.txt"),
        old="  %loop_add_fusion =",
        new=f"  %negate.1 = {_LOOP_TABLE} negate(%get-tuple-element.1980)\n"
        "  %loop_add_fusion =",
    )
    return _replace_once(
        text=text,
        old="tuple(%loop_add_fusion, %get-tuple-element.1980,",
        new="tuple(%loop_add_fusion, %negate.1,",
    )


def _with_shared_loop_body() -> str:
    """The loop fixture with a second loop running the same body on a gathered table."""
    text = _replace_once(
        text=_hlo("while_invariant_table.hlo.txt"),
        old="\nENTRY %main.241",
        new=f"\n{_GATHER_PRODUCER}ENTRY %main.241",
    )
    return _replace_once(
        text=text,
        old="  ROOT %get-tuple-element.1614 =",
        new=f"  %loop_gather_fusion = {_LOOP_TABLE} fusion(%Arg_0.1, %Arg_1.2), "
        "kind=kLoop, calls=%fused_table\n"
        f"  %tuple.594 = {_LOOP_STATE} tuple(%constant_0, %loop_gather_fusion, "
        "%Arg_1.2, %Arg_2.3, %broadcast.1)\n"
        f"  %while.66 = {_LOOP_STATE} while(%tuple.594), condition=%region_63.240, "
        "body=%region_0.239\n"
        "  ROOT %get-tuple-element.1614 =",
    )


def _with_called_reduce_fusion() -> str:
    """The materialised fixture with its reduce fusion inside a called computation.

    The table reaches the reduce fusion through the called computation's
    parameter, which the separate `%loop_gather_fusion` wrote.
    """
    text = _replace_once(
        text=_hlo("materialised.hlo.txt"),
        old="\nENTRY %main.1",
        new="\n%wrapped (table: f32[945,1,1,1,24,38], indices: s32[13,7,2]) "
        "-> f32[13,38,7] {\n"
        "  %table = f32[945,1,1,1,24,38]{5,4,3,2,1,0} parameter(0)\n"
        "  %indices = s32[13,7,2]{2,1,0} parameter(1)\n"
        "  ROOT %input_reduce_fusion = f32[13,38,7]{2,1,0} fusion(%indices, %table), "
        "kind=kInput, calls=%fused_reduce\n"
        "}\n\n"
        "ENTRY %main.1",
    )
    return _replace_once(
        text=text,
        old="  ROOT %input_reduce_fusion = f32[13,38,7]{2,1,0} "
        "fusion(%Arg_2.3, %get-tuple-element.3547), kind=kInput, calls=%fused_reduce",
        new="  ROOT %call.1 = f32[13,38,7]{2,1,0} "
        "call(%get-tuple-element.3547, %Arg_2.3), "
        "to_apply=%wrapped",
    )


def test_classify_reduce_fusions_follows_a_loop_carried_table_to_its_gather() -> None:
    """A table a loop carries unchanged from a gather fusion's output materialises."""
    assert classify_reduce_fusions(_with_gathered_loop_input()) == {
        "input_reduce_fusion": ReduceFusionVerdict.MATERIALISED_GATHER
    }


@pytest.mark.parametrize(
    "mutation",
    [_with_loop_carried_table, _with_shared_loop_body, _with_called_reduce_fusion],
)
def test_classify_reduce_fusions_needs_a_proved_origin_through_a_computation_parameter(
    *, mutation: Callable[[], str]
) -> None:
    """A table entering the reduce fusion's computation as a parameter needs its origin.

    - a loop updating the carried table has no single origin;
    - a loop body shared by two loops has one origin per loop;
    - a called computation's parameter is not an entry parameter.
    """
    assert _verdicts_or_refusal(mutation()) in _SOUND_RESULTS
