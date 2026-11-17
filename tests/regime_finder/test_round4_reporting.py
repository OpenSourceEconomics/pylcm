"""A public recheck that runs out of budget is not reported as a pass."""


def test_public_recheck_distinguishes_budget_exhaustion() -> None:
    from _lcm_regime_finder import find_regimes, verify_production_result
    from tests.regime_finder.tiny_metadata_family import original

    source = original()
    result = find_regimes(source, facts_backend="metadata")
    assert (result["status"]) == ("COMPLETE")
    checked = verify_production_result(source, result, limits={"seconds": 0})
    assert (checked["status"]) == ("UNRESOLVED")
