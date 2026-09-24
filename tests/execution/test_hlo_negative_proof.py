"""A negative reduce-fusion verdict needs a complete reading and a proved table origin.

The texts are synthetic edits of the trimmed GPU fixtures; nothing is compiled.
Refusing the module, or answering `UNKNOWN` or `MATERIALISED_GATHER`, is sound for
them. Answering `FUSED_GATHER`, `NO_GATHER` or nothing at all is not.
"""

import itertools

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
