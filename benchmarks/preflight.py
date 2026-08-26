"""Fail fast when an installed benchmark dependency cannot use pylcm."""


def main() -> None:
    """Construct every external benchmark fixture before ASV measures anything."""
    from benchmarks.asv.bench_aca_baseline import _build

    _build()


if __name__ == "__main__":
    main()
