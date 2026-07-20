"""Historical-compatibility manifest for the Mahler-Yum model (plan PR 13).

The published Fortran differs from the paper's own equations in identified,
enumerable ways (the continuation evaluated at the OLD effort habit contra
eq. (6) is the headline case). `MahlerLegacyCompatibility` names every such
switch in ONE manifest-visible object so that

- canonical estimation can never silently run with a historical switch
  enabled (`assert_canonical_for_inference`), and
- historical output differences are decomposed switch by switch
  (`single_switch_variants`), never attributed to an undifferentiated
  "legacy mode".

Switch semantics are split by where they act. SOLVER switches change the
Bellman recursion or the numerical search and are consumed by
`create_mahler_yum_model`; MEASUREMENT switches change parameter
preparation or moment construction and are consumed by the replication's
`legacy_fortran` module — a model factory that claimed them would be
asserting faithfulness it cannot see.

Implementation status is per switch and honest: the historical
finite-grid searches and the five-node cost grid ARE the existing brute
configuration and are buildable today; the remaining solver switches
(old-habit continuation above all) need solve-phase-only transition
support in the engine and raise `NotImplementedError` naming exactly the
unimplemented switches, rather than building a model that does not
implement what its manifest declares.
"""

from dataclasses import asdict, dataclass, fields, replace

# Solver-side switches: change the recursion or the numerical search.
SOLVER_SWITCHES = frozenset(
    {
        "old_habit_continuation",
        "historical_chi_nodes",
        "historical_effort_search",
        "historical_saving_search",
    }
)
# Measurement-side switches: change inputs, parameter preparation, or moment
# construction; consumed by the replication's legacy_fortran module.
MEASUREMENT_SWITCHES = frozenset(
    {
        "stale_survival_education",
        "swapped_type_indexing",
        "historical_income_normalizer",
        "historical_pension_scaling",
        "historical_employment_moment",
        "historical_vsly_moment",
    }
)

# Solver switches the current engine can actually build: the historical
# finite-grid effort/saving searches over the five-node adjustment-cost grid
# are exactly the brute configuration.
_BUILDABLE_SOLVER_SWITCHES = frozenset(
    {"historical_chi_nodes", "historical_effort_search", "historical_saving_search"}
)


@dataclass(frozen=True)
class MahlerLegacyCompatibility:
    """One manifest-visible object controlling every historical switch.

    All-off (and `historical_chi_nodes=0`, meaning the analytically
    integrated adjustment cost) is the canonical configuration. Canonical
    estimation and standard errors may never run with any switch enabled
    unless the output is explicitly labeled a historical reproduction.
    """

    old_habit_continuation: bool
    """Evaluate the continuation at the OLD effort habit (the Fortran's
    recursion, contra the paper's eq. (6))."""
    stale_survival_education: bool
    """Reproduce the stale education split in the survival-probability input."""
    swapped_type_indexing: bool
    """Reproduce the historical (productivity, health)-type index swap."""
    historical_income_normalizer: bool
    """Normalize incomes by the historical in-sample mean, not the model's."""
    historical_pension_scaling: bool
    """Reproduce the historical pension scaling convention."""
    historical_employment_moment: bool
    """Use the historical employment-moment definition."""
    historical_vsly_moment: bool
    """Use the historical VSLY-moment construction."""
    historical_chi_nodes: int
    """Adjustment-cost solve-state nodes (5 historically; 0 = analytical)."""
    historical_effort_search: bool
    """Search effort on the historical finite grid, not the continuous outer."""
    historical_saving_search: bool
    """Search saving by the historical golden-section-on-grid routine."""


CANONICAL = MahlerLegacyCompatibility(
    old_habit_continuation=False,
    stale_survival_education=False,
    swapped_type_indexing=False,
    historical_income_normalizer=False,
    historical_pension_scaling=False,
    historical_employment_moment=False,
    historical_vsly_moment=False,
    historical_chi_nodes=0,
    historical_effort_search=False,
    historical_saving_search=False,
)

LEGACY_FORTRAN = MahlerLegacyCompatibility(
    old_habit_continuation=True,
    stale_survival_education=True,
    swapped_type_indexing=True,
    historical_income_normalizer=True,
    historical_pension_scaling=True,
    historical_employment_moment=True,
    historical_vsly_moment=True,
    historical_chi_nodes=5,
    historical_effort_search=True,
    historical_saving_search=True,
)


def enabled_switches(compatibility: MahlerLegacyCompatibility) -> tuple[str, ...]:
    """Names of every switch that departs from canonical, in field order."""
    canonical = asdict(CANONICAL)
    return tuple(
        name
        for name, value in asdict(compatibility).items()
        if value != canonical[name]
    )


def manifest(compatibility: MahlerLegacyCompatibility) -> dict:
    """The run-manifest entry: label plus the full switch dictionary.

    The label alone distinguishes canonical from every non-canonical
    configuration, and a partial configuration from full legacy — a manifest
    reader can never confuse a historical reproduction with the canonical
    estimate by reading only the label.
    """
    if compatibility == CANONICAL:
        label = "canonical"
    elif compatibility == LEGACY_FORTRAN:
        label = "legacy_fortran"
    else:
        label = "historical_partial:" + ",".join(enabled_switches(compatibility))
    return {"compatibility_label": label, "switches": asdict(compatibility)}


def single_switch_variants(
    target: MahlerLegacyCompatibility = LEGACY_FORTRAN,
) -> tuple[MahlerLegacyCompatibility, ...]:
    """One variant per switch on which `target` departs from canonical.

    Each variant enables EXACTLY ONE of the target's departures on top of the
    canonical configuration — the evaluation set for a switch-by-switch
    decomposition of a historical output difference.
    """
    target_values = asdict(target)
    return tuple(
        replace(CANONICAL, **{name: target_values[name]})
        for name in enabled_switches(target)
    )


def assert_canonical_for_inference(
    compatibility: MahlerLegacyCompatibility, *, output_label: str
) -> None:
    """Refuse canonical-looking outputs from a non-canonical configuration.

    A non-canonical run is allowed only when its output label explicitly
    declares it (contains ``"historical"``); everything else raises. This is
    the plan's rule that canonical estimation and standard errors may never
    run with legacy switches enabled.
    """
    if compatibility == CANONICAL:
        return
    if "historical" in output_label:
        return
    switches = ", ".join(enabled_switches(compatibility))
    msg = (
        f"output {output_label!r} is not labeled a historical reproduction, "
        f"but historical switches are enabled: {switches}"
    )
    raise ValueError(msg)


def unimplemented_solver_switches(
    compatibility: MahlerLegacyCompatibility,
) -> tuple[str, ...]:
    """The enabled SOLVER switches the current engine cannot build.

    Measurement switches never appear here: they do not change the model
    object, and the replication layer that consumes them does its own
    per-switch accounting.
    """
    return tuple(
        name
        for name in enabled_switches(compatibility)
        if name in SOLVER_SWITCHES and name not in _BUILDABLE_SOLVER_SWITCHES
    )


def _all_switch_names() -> tuple[str, ...]:
    return tuple(f.name for f in fields(MahlerLegacyCompatibility))


def validate_switch_partition() -> None:
    """Every declared field is exactly one of solver- or measurement-side."""
    names = set(_all_switch_names())
    both = SOLVER_SWITCHES & MEASUREMENT_SWITCHES
    missing = names - SOLVER_SWITCHES - MEASUREMENT_SWITCHES
    extra = (SOLVER_SWITCHES | MEASUREMENT_SWITCHES) - names
    if both or missing or extra:
        msg = (
            f"switch partition broken: overlapping={sorted(both)}, "
            f"unassigned={sorted(missing)}, unknown={sorted(extra)}"
        )
        raise AssertionError(msg)


validate_switch_partition()
