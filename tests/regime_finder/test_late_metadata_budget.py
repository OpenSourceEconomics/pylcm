"""Late metadata exhaustion must report incomplete before the supervisor kills it."""

from unittest.mock import patch

from _lcm_regime_finder.production import MetadataAnalysis, analyze_production
from .tiny_metadata_family import original


def test_interface_expansion_observes_the_metadata_deadline() -> None:
    clock = [0.0]
    expanded = []
    build = MetadataAnalysis.function_interfaces

    def overrun(analysis, *args):
        expanded.append(args[0]["id"])
        answer = build(analysis, *args)
        clock[0] = 2.0
        return answer

    with (
        patch(
            "_lcm_regime_finder.production.time.perf_counter",
            side_effect=lambda: clock[0],
        ),
        patch.object(MetadataAnalysis, "function_interfaces", overrun),
    ):
        result = analyze_production(original(), limits={"seconds": 1})
    assert (result["status"]) == ("INCOMPLETE")
    assert (result["semantic_check"]["status"]) == ("NOT_RUN")
    assert ("Metadata dependency analysis budget exhausted") in (result["error"])
    assert not (result["scenarios"])
    assert (len(expanded)) == (1)
