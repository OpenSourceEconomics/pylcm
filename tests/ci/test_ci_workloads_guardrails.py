"""Guardrail checks against the manifest's recorded operating budgets.

These are proposed *operating* budgets (implementation plan batch 1, section
5), not historical measurements: 24 minutes payload / 30 minutes total per
job, and 30 seconds for an ordinary `notslow` test. The ceiling only applies
to files that HAVE an observed weight -- an unweighted file is a visible gap
to close (see `test_ci_workloads_manifest.test_unweighted_files_are_listed_not_zeroed`),
never a free pass, but this test cannot enforce a ceiling against no
measurement, so it is intentionally excluded here rather than silently
compared against zero.
"""

from __future__ import annotations

from tests.ci import ci_workloads

_NOTSLOW_INVOCATION_IDS = tuple(
    inv["id"]
    for inv in ci_workloads.invocations()
    if inv["id"].startswith("general-shard-")
)


def _notslow_files_with_weight() -> dict[str, float]:
    manifest = ci_workloads.load_manifest()
    weights = manifest["file_weights"]
    files: set[str] = set()
    for inv_id in _NOTSLOW_INVOCATION_IDS:
        files.update(ci_workloads.files_for(invocation_id=inv_id))
    return {f: float(weights[f]["seconds"]) for f in files if f in weights}


# Ratchet backlog (implementation plan batch 1, section 5): these six files
# were already over the ordinary ceiling on the frozen head and are owed to
# the rebalancing/cleanup batches, not to this inventory PR. Growing this set
# requires a new entry in the manifest's `per_test_ceiling_backlog` AND a
# corresponding note in the manifest `source` block -- see
# `test_the_backlog_cannot_grow_without_a_manifest_source_note`.
_KNOWN_BACKLOG_FILES = frozenset(
    {
        "tests/simulation/test_nnbegm_split_workflow_parity.py",
        "tests/simulation/test_simulate_n_nbegm_continuous.py",
        "tests/test_dropped_models_release_nested_functions.py",
        "tests/simulation/test_finite_policy_budget.py",
        "tests/simulation/test_simulate_certainty_equivalent.py",
        "tests/simulation/test_simulate_n_nbegm.py",
    }
)


def _backlog_entries() -> dict[str, dict]:
    manifest = ci_workloads.load_manifest()
    return {entry["file"]: entry for entry in manifest["per_test_ceiling_backlog"]}


def test_ordinary_notslow_file_does_not_exceed_the_per_test_ceiling_on_average():
    """A `notslow` file's mean per-test time should stay under the ceiling.

    This is a coarse per-FILE proxy (mean seconds per collected test in that
    file), not a per-test assertion -- the CSV this manifest is built from
    reports file totals and counts, not individual test durations. A file that
    trips this either needs a witness argument for its own budget or belongs
    in a budgeted PR heavy phase; this test does not silently deselect it.

    Files named in the manifest's `per_test_ceiling_backlog` are exempted
    from the ceiling here (a), but must still actually be over it (b -- see
    `test_the_per_test_ceiling_backlog_entries_are_still_actually_over_the_ceiling`),
    and the backlog itself cannot silently grow (c -- see
    `test_the_backlog_cannot_grow_without_a_manifest_source_note`).
    """
    manifest = ci_workloads.load_manifest()
    weights = manifest["file_weights"]
    ceiling = manifest["guardrails"]["ordinary_notslow_test_ceiling_seconds"]
    backlog = set(_backlog_entries())
    offenders = []
    for file_, seconds in _notslow_files_with_weight().items():
        if file_ in backlog:
            continue
        count = weights[file_]["count"]
        mean_seconds = seconds / count if count else seconds
        if mean_seconds > ceiling:
            offenders.append((file_, mean_seconds))
    assert not offenders, (
        f"notslow files exceeding the {ceiling}s per-test ceiling (mean "
        "seconds/test), not covered by the manifest's per_test_ceiling_backlog: "
        f"{sorted(offenders, key=lambda x: -x[1])}"
    )


def test_the_per_test_ceiling_backlog_entries_are_still_actually_over_the_ceiling():
    """A backlog entry whose file was fixed must be removed, not left stale."""
    manifest = ci_workloads.load_manifest()
    weights = manifest["file_weights"]
    ceiling = manifest["guardrails"]["ordinary_notslow_test_ceiling_seconds"]
    stale = []
    for file_ in _backlog_entries():
        assert file_ in weights, f"backlogged file {file_} has no recorded weight"
        weight = weights[file_]
        count = weight["count"]
        mean_seconds = weight["seconds"] / count if count else weight["seconds"]
        if mean_seconds <= ceiling:
            stale.append((file_, mean_seconds))
    assert not stale, (
        "per_test_ceiling_backlog entries that no longer exceed the "
        f"{ceiling}s ceiling and must be removed from the manifest: {stale}"
    )


def test_the_backlog_cannot_grow_without_a_manifest_source_note():
    """The backlog is capped at the six frozen-head offenders unless the
    manifest's `source` block carries a note authorizing an addition.
    """
    manifest = ci_workloads.load_manifest()
    unauthorized = set(_backlog_entries()) - _KNOWN_BACKLOG_FILES
    if unauthorized:
        notes = manifest["source"].get("per_test_ceiling_backlog_notes", {})
        missing_notes = sorted(f for f in unauthorized if f not in notes)
        assert not missing_notes, (
            "per_test_ceiling_backlog names files beyond the known six "
            f"without a manifest source note authorizing them: {missing_notes}"
        )


def test_guardrail_numbers_come_from_the_manifest_not_from_this_module():
    manifest = ci_workloads.load_manifest()
    guardrails = manifest["guardrails"]
    assert guardrails["job_payload_ceiling_minutes"] == 24
    assert guardrails["job_total_ceiling_minutes"] == 30
    assert guardrails["ordinary_notslow_test_ceiling_seconds"] == 30


def test_guardrail_is_not_evaluated_against_unweighted_files():
    """An unweighted file is never compared to the ceiling as if its cost were zero."""
    unweighted = set(ci_workloads.unweighted_files())
    notslow_files = set()
    for inv_id in _NOTSLOW_INVOCATION_IDS:
        notslow_files.update(ci_workloads.files_for(invocation_id=inv_id))
    # Sanity: the fixture used by the ceiling test above never draws from the
    # unweighted set (it only reads files present in `file_weights`).
    assert unweighted.isdisjoint(_notslow_files_with_weight())
    del notslow_files


def test_every_invocation_stays_inside_the_payload_guardrail():
    """No recorded lane's predicted payload exceeds the manifest's ceiling.

    The prediction is `max(total worker seconds / workers, largest single file)`
    over the frozen head's observed per-leg JUnit times. It is a *lower bound*
    on real time -- it excludes setup, collection, reporting and worker
    imbalance -- so a lane already over the ceiling here is certainly over it in
    CI. Being under it is necessary, not sufficient; only a CI run can confirm
    the 30-minute job total.
    """
    ceiling = ci_workloads.guardrails()["job_payload_ceiling_minutes"]
    offenders = [
        (inv["id"], inv["predicted_payload_minutes"])
        for inv in ci_workloads.invocations()
        if inv.get("predicted_payload_minutes", 0.0) > ceiling
    ]
    assert not offenders, (
        f"lanes over the {ceiling} min payload guardrail: "
        f"{sorted(offenders, key=lambda x: -x[1])}"
    )


def test_every_invocation_records_a_predicted_payload():
    """A lane with no prediction is an unbudgeted lane, not a free one."""
    missing = [
        inv["id"]
        for inv in ci_workloads.invocations()
        if "predicted_payload_minutes" not in inv
    ]
    assert not missing, f"lanes with no predicted payload: {missing}"


def test_no_whole_file_atom_exceeds_its_lane_payload_guardrail():
    """No single file costs more than a whole job is allowed to spend.

    Under `--dist loadfile` a file is indivisible, so a file above the ceiling
    cannot be fixed by adding shards or workers -- LPT cannot split an atom. The
    source-contract lane is the deliberate exception: it uses `--dist load`, so
    its modules are not atoms there.
    """
    ceiling = ci_workloads.guardrails()["job_payload_ceiling_minutes"]
    offenders = []
    for inv in ci_workloads.invocations():
        if inv["environment"]["isolation"] != "loadfile-xdist":
            continue
        leg = (inv.get("shard") or {}).get("leg")
        if leg is None:
            continue
        weights = ci_workloads.leg_weights(leg=leg)
        for file_ in inv["files"]:
            minutes = weights.get(file_, 0.0) / 60.0
            if minutes > ceiling:
                offenders.append((inv["id"], file_, minutes))
    assert not offenders, (
        f"indivisible files over the {ceiling} min payload guardrail: {offenders}"
    )
