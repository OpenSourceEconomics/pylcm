"""The historical-compatibility manifest (plan PR 13, pylcm side).

Merge-gate properties: canonical and legacy manifests cannot be confused
(distinct labels, refusal of canonical-looking outputs under historical
switches), the one-switch decomposition variants each differ from canonical
in exactly one field, and the model factory is honest per switch — the
buildable historical subset returns the brute model, an unbuildable switch
raises naming itself, and the paper implementation refuses historical
switches outright.
"""

from dataclasses import asdict, replace

import pytest

from lcm_examples.mahler_yum_2024 import legacy
from lcm_examples.mahler_yum_2024.paper import create_mahler_yum_model


def test_every_switch_is_exactly_solver_or_measurement() -> None:
    legacy.validate_switch_partition()
    all_fields = set(asdict(legacy.CANONICAL))
    assert all_fields == legacy.SOLVER_SWITCHES | legacy.MEASUREMENT_SWITCHES


def test_canonical_and_legacy_manifests_cannot_be_confused() -> None:
    canonical = legacy.manifest(legacy.CANONICAL)
    historical = legacy.manifest(legacy.LEGACY_FORTRAN)
    assert canonical["compatibility_label"] == "canonical"
    assert historical["compatibility_label"] == "legacy_fortran"
    partial = legacy.manifest(replace(legacy.CANONICAL, old_habit_continuation=True))
    assert partial["compatibility_label"] == "historical_partial:old_habit_continuation"
    labels = {
        canonical["compatibility_label"],
        historical["compatibility_label"],
        partial["compatibility_label"],
    }
    assert len(labels) == 3


def test_canonical_inference_refuses_historical_switches() -> None:
    legacy.assert_canonical_for_inference(legacy.CANONICAL, output_label="estimates")
    with pytest.raises(ValueError, match="old_habit_continuation"):
        legacy.assert_canonical_for_inference(
            replace(legacy.CANONICAL, old_habit_continuation=True),
            output_label="estimates",
        )
    # An explicitly historical label is the ONLY way to run non-canonical.
    legacy.assert_canonical_for_inference(
        legacy.LEGACY_FORTRAN, output_label="historical-reproduction-table2"
    )


def test_single_switch_variants_differ_in_exactly_one_field() -> None:
    variants = legacy.single_switch_variants()
    names = legacy.enabled_switches(legacy.LEGACY_FORTRAN)
    assert len(variants) == len(names) == len(asdict(legacy.CANONICAL))
    canonical = asdict(legacy.CANONICAL)
    target = asdict(legacy.LEGACY_FORTRAN)
    for name, variant in zip(names, variants, strict=True):
        departures = legacy.enabled_switches(variant)
        assert departures == (name,)
        assert asdict(variant)[name] == target[name]
        unchanged = {k: v for k, v in asdict(variant).items() if k != name}
        assert unchanged == {k: v for k, v in canonical.items() if k != name}


def test_factory_builds_the_buildable_historical_subset() -> None:
    """The historical finite-grid searches ARE the brute configuration."""
    from lcm_examples.mahler_yum_2024 import MAHLER_YUM_MODEL  # noqa: PLC0415

    grids_only = replace(
        legacy.CANONICAL,
        historical_chi_nodes=5,
        historical_effort_search=True,
        historical_saving_search=True,
    )
    assert legacy.unimplemented_solver_switches(grids_only) == ()
    model = create_mahler_yum_model(
        implementation="legacy_fortran", compatibility=grids_only
    )
    assert model is MAHLER_YUM_MODEL


def test_factory_refuses_unbuildable_solver_switches_by_name() -> None:
    with pytest.raises(NotImplementedError, match="old_habit_continuation"):
        create_mahler_yum_model(implementation="legacy_fortran")


def test_measurement_switches_do_not_block_the_model_build() -> None:
    """Moment-side history is not the model factory's to claim or refuse."""
    from lcm_examples.mahler_yum_2024 import MAHLER_YUM_MODEL  # noqa: PLC0415

    measurement_only = replace(
        legacy.CANONICAL,
        historical_vsly_moment=True,
        historical_income_normalizer=True,
        historical_chi_nodes=5,
        historical_effort_search=True,
        historical_saving_search=True,
    )
    model = create_mahler_yum_model(
        implementation="legacy_fortran", compatibility=measurement_only
    )
    assert model is MAHLER_YUM_MODEL


def test_paper_implementation_refuses_historical_switches() -> None:
    with pytest.raises(ValueError, match="legacy_fortran"):
        create_mahler_yum_model(
            implementation="paper",
            compatibility=replace(legacy.CANONICAL, old_habit_continuation=True),
        )
    # Passing the canonical object explicitly is legal and equivalent.
    model = create_mahler_yum_model(
        implementation="paper", compatibility=legacy.CANONICAL
    )
    assert "alive" in model.user_regimes
