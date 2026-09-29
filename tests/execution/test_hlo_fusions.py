"""Each reduce fusion of an optimized HLO module is classified by its gather tables."""

import pytest

from _lcm.config import TEST_DATA
from _lcm.execution.hlo_fusions import (
    ReduceFusionVerdict,
    UnrecognisedHloError,
    classify_reduce_fusions,
)


def _hlo(name: str) -> str:
    return TEST_DATA.joinpath("hlo_reduce_fusions", name).read_text()


# Each fixture is trimmed from an optimized GPU module:
# - `materialised`: an ACA GridSearch core whose dominant reduce fusion reads its
#   gather table from `%loop_gather_fusion`, which wrote it to device memory;
# - `fused`: the same core at a narrower cell chunk, where the first gather runs
#   inside the reduce fusion from the small source table;
# - `gather_summed`: a reduce fusion summing a separate gather fusion's output
#   elementwise, which reads a gather result but not as a gather table;
# - `materialised_root_copy` / `materialised_root_bitcast`: a gather fusion whose
#   root is a copy or a reshaping bitcast of its gather, read as a gather table;
# - `unrecognised_syntax`: a reduce fusion with a line the parser does not read.
@pytest.mark.parametrize(
    ("fixture", "verdict"),
    [
        ("materialised.hlo.txt", ReduceFusionVerdict.MATERIALISED_GATHER),
        ("fused.hlo.txt", ReduceFusionVerdict.FUSED_GATHER),
        ("gather_summed.hlo.txt", ReduceFusionVerdict.NO_GATHER),
        ("materialised_root_copy.hlo.txt", ReduceFusionVerdict.MATERIALISED_GATHER),
        (
            "materialised_root_bitcast.hlo.txt",
            ReduceFusionVerdict.MATERIALISED_GATHER,
        ),
        ("unrecognised_syntax.hlo.txt", ReduceFusionVerdict.UNKNOWN),
    ],
)
def test_classify_reduce_fusions_reads_the_table_source(
    *, fixture: str, verdict: ReduceFusionVerdict
) -> None:
    """The one reduce fusion of each fixture gets the verdict its table source sets."""
    assert classify_reduce_fusions(_hlo(fixture)) == {"input_reduce_fusion": verdict}


def test_classify_reduce_fusions_finds_nothing_in_a_module_without_fusions() -> None:
    text = (
        "HloModule empty\n\n"
        "ENTRY %main (x: f32[]) -> f32[] {\n"
        "  ROOT %x = f32[] parameter(0)\n"
        "}\n"
    )

    assert classify_reduce_fusions(text) == {}


@pytest.mark.parametrize(
    "text",
    [
        "HloModule parser_format_not_recognised\n",
        "",
        "HloModule stray\n  ROOT %x = f32[] parameter(0)\n",
    ],
)
def test_classify_reduce_fusions_refuses_text_it_does_not_recognise(
    *, text: str
) -> None:
    """Text without a recognised entry computation is refused, never read as empty."""
    with pytest.raises(UnrecognisedHloError, match=r"entry computation|outside"):
        classify_reduce_fusions(text)
