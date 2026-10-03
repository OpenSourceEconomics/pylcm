# Approved: four macOS general shards plus macOS-only weight refresh

The user approved this change, including the measured-weight provenance, for
publication on the protected Stage 0–3 branch. It is prepared in an isolated
checkout at db1432ce361cc6091b5c7644211b40772d4e4ab3; the parent alone commits
and pushes. Publication has not yet occurred.
It preserves root frozen_head/file_weights, every other leg's weights/counts,
16 coverage contributors, 60-minute timeout, 24-minute payload and 30-minute
job-total guardrails, and all source/certificate/topology/timing gates.

Only fp64-macos leg measurements are refreshed from exactly reconciled canonical
#491 artifacts at tested merge fa1ba52864e032191df6620eecb08d6533b03964, run
37007461636 attempt 1, precision 64, not slow and not manual, four workers,
loadfile; existing deselections are retained. CSV SHA256:
6a542e2050f7190dff7adff7853b41ec62b77828914bf3f42861a765a3b5ec3e.
Existing per-leg junit_source records this provenance; all other maps explicitly
retain older measurements. Generic docs distinguish per-leg provenance from
the root CSV freeze. Independently tested shared JUnit delimiter repair is a
prerequisite, published separately in [PR #496](https://github.com/OpenSourceEconomics/pylcm/pull/496)
at commit `77d675c0498618a66f8dfdad3d4b82f24e402bcc`. The parser bytes and tests
included here match that published prerequisite. It does not handle collection
skips; those exact-population gaps remain unresolved.

Final exact file populations are 127/128/129/152. Each scheduling lower bound is
0:19:28.748; illustrative worker LPT estimate 0:19:28.750. Neither is a measured
native runtime or upper bound. The base covers 12242 observed nodes; one newer
observed-only module contributes 23 additional nodes. Twenty-four unmeasured
base files remain explicit, all in shard 4. Fresh native macOS completion,
24-minute payload and 30-minute total-job acceptance remain outstanding.

Local accommodated whole tests/ci passes 410, no failures/errors/skips,
0:00:31.664. Existing generator/check and normal hooks pass, including Ruff,
workflow schema and whole-project ty. Narrow evidence-only Pixi child routing
changes no assertions, selections or source/action paths; it is not tracked.
No numerical full suite or GPU/native performance acceptance is claimed.

Other observed #491 capacity issues remain untreated: exact existing lower
bounds Windows 29:25, fp32 Linux 27:33, solution64 33:06 and solution32 46:59.
Even ideal repartition at existing counts leaves solution64 29:07 and solution32
28:26. Native canonical audit 75/77 passes; both rest-slow collection-skip XML
population gaps remain unresolved. This is not whole-CI capacity success.

Full review artifacts, CSV/provenance/hash bindings, exact assignments, commands
and raw JUnit are in aca-dev .task-evidence/pylcm-handoff/ci-5b/proposal/.
The original handoff records a cloud-token workflow-scope limitation. No push
was attempted here, so current publication capability is unverified. If a push
is rejected for workflow scope, report that exact rejection and follow the
authorized publication route; do not bypass it by probing alternate credentials.

## Approved source and receipt checkpoint

The audited proposal patch is SHA256
7cc8576a4032cc91a7896812ce240eb3eb9e951045f7a7d2d5e744a6a10af160.
All nine prepared files matched its reviewed file hashes before updating only
this approval note and the handoff index. The live protected branch was
revalidated at db1432ce, so the existing 410-case CI helper, generator/check and
normal-hook receipts apply to the unchanged executable/config/test bytes.
No new test/build/hook was run while another worker owns the local compute slot.
The parent runs normal commit hooks for the documentation update before push.

The user also approved the blocked-vs-unblocked ULP contract and corrected the
authorized endpoint to Stage 8A. These decisions do not introduce numerical
source or Stage 8A changes in this macOS CI patch. Fresh macOS four-shard
payload/total-job acceptance remains required; numerical/HPC ownership and all
other pending gates remain with their existing owners.
