"""Fail fast when an installed benchmark dependency cannot use pylcm."""


def main() -> None:
    """Construct external fixtures and require folded ACA health-cost value axes."""
    from benchmarks.asv.bench_aca_baseline import _build

    model, _, _ = _build()
    living_regimes = tuple(
        regime_name for regime_name in model.user_regimes if regime_name != "dead"
    )
    if not living_regimes or any(
        "hcc_transitory" in model.state_names(regime_name=regime_name)
        for regime_name in living_regimes
    ):
        msg = "ACA benchmarks require hcc_transitory folding in every living regime."
        raise AssertionError(msg)


if __name__ == "__main__":
    main()
