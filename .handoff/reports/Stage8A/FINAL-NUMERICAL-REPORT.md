# Stage 8A final local numerical acceptance

Candidate `/home/hmg/econ/aca-dev/.codex-worktrees/invariant-node-local-verified`,
branch `codex/invariant-node-local-verified`, HEAD
`cb62c3d9351b9d4aa2acdad885b6aaab93f8a16c`, with an uncommitted reviewed eight-file
port. Own frozen tests-cpu/type-checking prefixes were installed by the separate
CI worker; this worker performed no installation, commit, push, GPU or HPC action.
Parent alone publishes. Numerical compute is released: final cap Tasks0/Units0.

| Final frozen-source receipt | Passed | Failed | Errors | Skips | Duration | Exit |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| final-jobs-fp64.xml | 71 | 0 | 0 | 0 | 0:03:22.519 | 0 |
| final-jobs-fp32.xml | 71 | 0 | 0 | 0 | 0:03:29.709 | 0 |
| final-relevant-fp64.xml | 111 | 0 | 0 | 0 | 0:05:46.991 | 0 |
| final-relevant-fp32.xml | 111 | 0 | 0 | 0 | 0:05:32.051 | 0 |
| final-typed8cpu-fp64.xml | 13 | 0 | 0 | 0 | 0:00:50.069 | 0 |
| final-typed8cpu-fp32.xml | 13 | 0 | 0 | 0 | 0:00:48.266 | 0 |

All receipt/log/identity basenames in this report use absolute prefix
`/home/hmg/econ/aca-dev/.task-evidence/pylcm-handoff/stage8a-preparation/`.
`final-numerical-receipts.json` contains XML-derived counts, exact durations,
all selected nodeids and XML hashes, SHA256
`29ec5f1c577ebba602b95dcbd1e734a4ae400f3abf36cd1b0fb7858a422e78cf`.
The six final GREEN selections are 390 passed in total; this is bounded local
acceptance, not a full-suite, certificate or native production count.

## Last RED and minimal repair

`final-loaded-bundle-red.xml`: 0 passed, 3 failed, 0 errors/skips, 0:00:07.664,
exit1. All three actual public loader calls failed with DID NOT RAISE
SolutionIntegrityError: a nonnull simulation seedNone, overall population0, and
numeric initial-population SHA123. The minimum repair requires an exact integer
seed for a nonnull simulation bundle, uses existing persistence
`_require_positive_exact_int` for overall population and `_require_sha256` for
the population digest. Signed seeds remain supported. No schema framework.
Full71 includes all three refusals and the complete positive afterward.

The first own-prefix complete71 fp64 passed in 0:03:34.699, archived as
`prefinal-jobs-fp64.xml`. Parent then identified an introduced default allocation:
full-population grouped planning unnecessarily created np.arange(n_real) and a
copy of positions. Only after that run completed, the approved behavior-preserving
cleanup made internal SubjectGroupPlan.rows=None mean full population and returns
the existing positions unchanged. Selected-code row calculation/filtering remains
unchanged. Both final71 and both relevant matrices run after this cleanup. These
durations are not paired performance measurements and establish no speedup.

Before freeze, normal scoped Ruff/format and keyword-only hooks ran on exactly
eight source paths plus the new test. Observed style violations were corrected:
parameter container conventions, imports, raw regexes, keyword-only pytest
injection, exact active Pixi executable discovery and formatting. A small
parent-approved execution-equality helper extraction satisfies the existing
complexity limit while retaining worker-vs-worker and worker-vs-collector checks;
no C901 suppression or interface/framework was introduced. PD011 suppressions
are narrowly documented false positives on actual value mappings, not pandas.
RegimeName aliases describe fixed regime roles. These are not gate relaxations.

Initial restricted cap setup stopped before a hook ran; escalation then confirmed
zero neighbours and admitted the normal capped hooks. The first identity helper
invocation omitted candidate root PYTHONPATH and failed to import hatch_build,
before test execution; its dependent launch produced no feature RED. Both setup
logs remain `*-setup.log`. The corrected invocation explicitly binds candidate
root/src. Only the real three-case XML is claimed as RED.

## Covered public behavior

The 71-case module tests public plan/run/collect and public loader boundaries.
It includes complete solve-only values with no simulation, and full three-job
simulation under JIT/eager and all four logging levels. Its eleven-row population
contains unequal nonempty groups and an empty original type; every original
code's values survive. Values include ordered coordinates, dtype, shape and
bytes; raw outputs include tree/field/order and every leaf; panels include every
cell/column and index names, per-level dtype and order. Same blocked engine
single-process references are matched bitwise; no unblocked bitwise gate is added.

Refusal controls cover logical value/panel duplication and physical dataset alias,
exact address/header/shape/count/seed metadata, actual-model state/code order,
omitted empty-type values, mutually omitted required raw wealth, model-owned raw
schema/fixed fields, public prepublication Boolean counts/codes/jobs/seeds and
overall empty population, incomplete/failed/stale/foreign/checksum/partial
campaigns, actual job→original-row binding/row shape/dtype and runtime identity.
Actual publication OSError→failure record→successful retry restores complete
bytes. The required fixed raw fields use actual model authority; no blanket
grid-based raw-state dtype rule is added. RetainedComponentValues._keep already
checks value shapes against actual model layouts, and its coverage requires all
model coordinates/codes; no duplicate authority schedule is introduced.

Actual worker/collector execution records separately pin JAX/JAXlib, XLA_FLAGS,
native fingerprint, log level, model JIT/action choices, device/budget information,
effective ambient JIT, PRNG implementation, seed offset and Threefry partitioning.
No new hardware/compiler fields enter mathematical SolutionIdentity. Planning
does no numeric random execution and advertises no planner PRNG pin. The enforced
claim is matched actual workers/collector and the matched numerical reference,
not every possible JAX global setting. Checksums assume the owner's immutable
campaign plan; they do not authenticate a wholesale adversarial replacement.

The 111-case selection is the complete existing block-major lifetime, grouped
simulation and scalar invariant modules, plus the three existing summary cases
(legacy debug detail, logging-neutral output, budget report) and analytic
winning-action derivative. It covers changed params/seed, default period-major,
combined/split simulation, save/load, retention/release/outliving model, budget
admission/refusal, empty selected types and actual existing2CPU eager action/state
paths. The exact selectors are in FINAL-OWN-ENV-COMMANDS.md and nodeids in JSON.

The eleven named typed8CPU selectors expand to thirteen test nodes. One existing
report-backed child per precision really reports8 CPU devices, cpu backend and
matching x64. It tests type/state/action layouts, complete block-major values,
split/combined panels, retained component release, and exchanged-accumulator
workspace reserve/below/exact-budget/terminal controls. Raw reports copied before
pytest temp cleanup:

- final-typed8cpu-report-fp64.json, original
  `/tmp/pytest-of-hmg/pytest-236/action_partitions0/report_x64_1.json`, SHA256
  `12aa5bb8f6642cb8fa2edd568a36fb1e0cb331846f5d7171142d0193cf4c34df`.
- final-typed8cpu-report-fp32.json, original
  `/tmp/pytest-of-hmg/pytest-237/action_partitions0/report_x64_0.json`, SHA256
  `b72f2381c87f2530189587198033bf0f9edc6d27aa9a4d299ef02a0f512dce66`.

The typed report carries device/backend/precision and numerical/layout records,
not a separate child PID/import/native receipt. Its source binding is the guarded
parent, checked frozen source before/after, explicit existing subprocess cwd and
candidate PYTHONPATH plus own Pixi environment. Do not claim unrecorded telemetry.
CPU establishes semantics/placement, not GPU speed or three-node execution.

## Exact environment and source

Every numerical launch used zsh cap, own candidate pyproject.toml, frozen
`pixi run --as-is -e tests-cpu`, explicit candidate src/root PYTHONPATH and CPU
backend. The guarded existing runner verifies package hashes/strict native
readiness and supplies `--precision=32|64 -n0 -v` plus absolute JUnit paths. Only
one numerical process/battery was active at a time; waits were30seconds, no
unchanged-log polling. FINAL-OWN-ENV-COMMANDS.md records the selector/launch
forms. Actual postformat/postrepair/default-final identities are separate;
final-source-freeze.json is the one shared by all six final GREEN runs.

Final source freeze SHA256
`fc4a5bb8970b4ea6747e69ef597f8b85ff43c30924372bbca9d8392d51046da6`;
after execution all320 package Python files and all eight source files/test/
manifest/lock matched their frozen hashes. `final-source.sha256` lists every
owned source/test hash. Full eight-source diff including both untracked modules:
final-eight-source.patch SHA256
`65880ac137dcc20ae2be5322b9634649de8ce7c5b3932452f7374d096c7cdb41`.

Host hmg-office; Python3.14.7; JAX/JAXlib0.11.1; own tests-cpu prefix
`.pixi/envs/tests-cpu`; lock SHA256
`4548dd4405fb1810616c30d451c8eb272b9617d452528e90a2a23ff53474442e`.
Generated and installed metadata both identify dirty source
`0.0.2.dev255+gcb62c3d93.d20261003`; this does not pretend the port is committed.
Maintained native build inputs match actual candidate inputs. Strict CPU payload
READY, own prefix `_pylcm_native/libcertified_affine_ffi_cpu.so` SHA256
`471691b41926782c1da02705e7f51f1f2d5212bbefb29c18eae367c9d1d153b1`.

Final fresh public worker children: fp64 PID4086893/x64true, fp32
PID4092145/x64false. Both print candidate lcm import, own native payload/READY/
checksum, JAX/JAXlib, source version, cpu backend and empty XLA_FLAGS. Child
stderr is empty. Verbatim `final-jobs-fp*-child_stdout.txt`/`child_stderr.txt`
were extracted from actual JUnit properties. Both recover the removed job then
compare the full result against the independent matched single-process5B run.

Remaining gates: maintained CI generator/check/manifest tests; final narrow
simulate/model/result certificate reconciliation and all600 controls including29
actions; normal type/hooks, parent commit/push and actual CI. Certificate inventory,
anchors and allowlists were untouched in this chunk. Five port files remain
outside existing certificate inventory and have functional acceptance only.
Driver/Slurm integration is separately owned. Required simultaneous3nodes×8GPUs
on mlgpu, full canonicalACA inputs/seed/retention and explicit matched fp32
reference remains an ACA-owner native execution gate. Configured SSH unavailable;
no HPC call/submission here. Local390 does not close profiling/performance or
the remaining native hardware/precision matrix.
