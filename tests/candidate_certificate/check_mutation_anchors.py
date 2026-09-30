"""Check that every registered mutation anchor still resolves once in its source.

The certificate campaign rewrites one literal snippet per mutation into a copy of
the certified sources and expects the corridor to reject the result. A snippet
that no longer occurs, or occurs more than once, makes the whole campaign error
at fixture setup rather than fail one case, so this check runs at commit time
where the source edit happens. It reads the same registries the campaign reads
and never executes a corridor.

Exit status:
- 0: every anchor occurs exactly once
- 1: at least one anchor is missing or ambiguous (each is named)
"""

# ruff: noqa: E402  (the repository root must be importable before the registries are)

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests import test_simulation_candidate_program_certificate as campaign
from tests.candidate_certificate import direct_flow


def collect_anchors() -> dict[str, tuple[str, str]]:
    """Return `{mutation name: (repo-relative path, literal snippet)}`."""
    anchors: dict[str, tuple[str, str]] = {}
    for name, (old, _new) in campaign._RANDOM_HELPER_MUTATIONS.items():
        anchors[name] = ("src/_lcm/simulation/random.py", old)
    registries = (
        campaign._PROFILED_HELPER_MUTATIONS,
        campaign._FINITE_POLICY_MUTATIONS,
        campaign._COMBINED_INPUT_MUTATIONS,
        campaign._EAGER_PLACEMENT_MUTATIONS,
        campaign._FINITE_BUDGET_MUTATIONS,
        campaign._SOLVE_READINESS_MUTATIONS,
    )
    for registry in registries:
        for name, (relative, old, _new) in registry.items():
            anchors[name] = (relative, old)
    return anchors


def main() -> int:
    """Report every anchor whose occurrence count is not one."""
    defects: list[str] = []
    for name, (relative, old) in collect_anchors().items():
        path = ROOT / relative
        count = path.read_text(encoding="utf-8").count(old) if path.exists() else -1
        if count != 1:
            defects.append(
                f"{name}: {relative} holds the anchor {count} times (need 1)"
            )
    # The direct-flow registries build their specs from the live tree and raise
    # on a lost anchor themselves; building them is the check.
    try:
        direct_flow.direct_flow_mutation_specs(repo_root=ROOT)
        direct_flow.supplemental_direct_flow_mutation_specs(repo_root=ROOT)
    except AssertionError as error:
        defects.append(f"direct-flow registry: {error}")
    for line in defects:
        print(line, file=sys.stderr)
    return 1 if defects else 0


if __name__ == "__main__":
    raise SystemExit(main())
