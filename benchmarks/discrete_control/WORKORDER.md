# Missing-only B1 discrete baseline/head control

No existing matched B1 series was recovered. The bounded recovery report is at
sciebo:pro-audits/pylcm-architecture-performance-plan/vintage/2026-09-14-continuous-sharding-b1-preparation/recovery/REPORT.md.
HMG subsequently authorized queueing B1 from a separate Marvin directory without ACA.
This work order is not a completed measurement receipt. Current27534935 remains untouched.

## Fixed workload, source and quantities

Baseline620813df93b6a50db4383ffc43690dc70e4ae8c3 and
head72c333c82fab1cd1aa7322039b0301b4040cb589. Head production/test/manifest/lock bytes
remain identical to reviewedaaefa6ff86441906f50acbbf001dae64439ad2ba; later commits
add guidance and continuous-scaling tooling only. Compare baseline→head, not an
immediate-parent causal delta. No main pin or ACA workload is part of this packet.

SOURCE-MANIFEST.json seals both actual Git revisions and relevant source/environment
files. The entire tests/test_distributed_placement.py, pyproject.toml, pixi.lock and
hatch_build.py are byte-identical across arms. Extracted fixture definitions and the
existing eight-ULP comparator are AST-identical to both revisions; they retain their
original implementation, without the test module's forced CPU initialization.

Call the unchanged `_make_three_type_model(distributed=True, devices=(0,1,2,3),
budget_bytes=134217728)` with `_PARAMS` discount0.95. Twelve wealth nodes, ten
consumption nodes, three-valued type, five ages; GridSearch/JIT defaults, no economic
edits. The working regime occupies devices0/1/2; retired occupies device3. This is
four actual GPUs, not eight-way continuous sharding. No simulation timing is claimed.

For each fp64/fp32 and baseline/head: one untimed semantic process checks the original
nine-array roster, all finite values, distributed/canonical agreement within the
existing8ULP contract, and exact nonempty disjoint full shard coverage. Two fresh
measurement processes each perform one cold plus3synchronized warm public solves;
retain first solution throughout. Timed observations are separate from the extra
canonical-reference solves. Cross-process and baseline/head value comparisons use
that same8ULP comparator, without changing economics or tolerances. No new exact
simulation choice/RNG claim is made by this discrete control; continuous G4's
separate original semantic gates remain with27534935.

Record direct trace/lowering/backend events and inclusive compilation orchestration;
zero warm backend requests, positive cold event control. Per-selected-device allocator
live high-water and host ru_maxrss retain their different scope. Keep all four device
peaks and raw repeat samples; never replace them with nvidia-smi totals. Each process
has its own fresh compilation cache. Arm order reverses on repeat2. Reporting flags
5%warm/10%compile/peak trigger candidates, without auto-waiving or declaring causality
outside observed variation. All public timing includes planner work and synchronization;
I/O/formatting is outside timed calls. Compiler duration categories overlap.

## Proposed smallest topology and bounded allocation

One mlgpu_short node with4A40,8CPUs,64GiB and02:00:00; watchdog6900s+30s kill margin.
Four GPUs are the minimum for the unchanged3-device type mesh plus fourth retired
regime. No eight-GPU reservation or development companion is needed for B1.
Eight CPU threads and64GiB are a conservative serial-process setup allowance, not a
measured minimum. The two-hour ceiling is not an expected runtime: no B1 timing exists.

Bounded contents:4semantic processes (300s each),8timing processes (450s each), up to
one normal native rebuild per source only on probe exit2 (600s each),60s provenance/
probe/summary steps. These nominal bounds total under6900s before minor overhead;
the global watchdog always wins and partial results remain incomplete. Planned totals
are4semantic cases,8timing processes,32timed solves plus8untimed semantic solves.
No local numerical battery, full suite, parameter sweep, auto-retry or replacement.

## Installation and release prerequisites

The base/head lock and build sources are sealed, but **no fresh baseline installation
or four-GPU native payload has been observed in this preparation**. The existing
normal head installation receipt is a useful setup precedent, not permission to copy
its native files or claim a matched B1 environment. Installed library/compiler/driver
identities must be recorded and reviewed from the actual new source environments.

Before release, parent reviews and commits this packet at
`benchmarks/discrete_control/` (tools/ and manifest/contract files included), pushes,
and pulls the exact new tooling commit on Marvin into a dedicated Git checkout.
Do not pull/change the checkout serving27534935. Normal separate Git checkouts of
baseline and head must be clean at the full pins above and each installed via
`NVCCFLAGS=-arch=sm_86 pixi install --frozen -e tests-cuda13`. No production source,
version module or native overlay. Any change from checks/hooks requires resealing
SHA256SUMS and recording the new packet/tooling identity before release.

The root-designated verifier checks each normal install with the packet's
`tools/check_install.py --root <source-root> --output <receipt>` through that root's
own frozen tests-cuda13 environment, JAX_PLATFORMS=cpu on login, and
PYLCM_SCALING_EXPECTED_HEAD set to its exact base/head pin. Review installation paths,
generated version, actual package versions, build inputs and native/JAX hashes.
Both source environments must match the locked dependencies; normal source version
and target-native build differences are recorded, not hidden. Driver/topology/native
registration are rechecked inside the approved allocation. Only a native probe exit2
allows one ordinary same-source Pixi reinstall; other failures stop immediately.

Author release must name this packet/tooling hash, owner and the proposed resources.
B1 has its own subsequent HMG authorization; its approval is not inferred from27534935. HMG subsequently authorized this one B1 launch from separate checkouts, without ACA. Native execution and GPU collection remain untested here;
static review is not a passing hardware or runtime result.

## Exact commands after release only

Set PYLCM_B1_BASE_ROOT and PYLCM_B1_HEAD_ROOT to the two newly pulled source checkouts;
set B1_TOOLING_ROOT to the separate pulled tooling checkout and PYLCM_B1_TOOLING_SHA
to its verified full Git SHA. Set B1_CONTROLLER_MANIFEST to an already installed
pytask-slurm controller manifest. It orchestrates only this task; it does not supply
numerical imports or select its owning project's tasks. Then:

```bash
export PYLCM_B1_BASE_ROOT PYLCM_B1_HEAD_ROOT PYLCM_B1_TOOLING_SHA
export NVCCFLAGS=-arch=sm_86
export PYLCM_B1_RELEASE=approved
cd "$B1_TOOLING_ROOT"
pixi run --manifest-path "$B1_CONTROLLER_MANIFEST" --frozen --no-install -e default \
  python -m pytask collect --config benchmarks/discrete_control/tools/pytask.toml \
  benchmarks/discrete_control/tools/task_discrete_control.py
# Require exactly one task and explicit author release before the next command.
pixi run --manifest-path "$B1_CONTROLLER_MANIFEST" --frozen --no-install -e default \
  python -m pytask build --config benchmarks/discrete_control/tools/pytask.toml \
  benchmarks/discrete_control/tools/task_discrete_control.py --slurm
```

Exact underlying numerical commands and timeout/exit logging are in tools/run.py.
Tests use-v and absolute JUnit paths; the delegated verifier must reconcile all four
XMLs, exact counts/failing nodeids and all8timing records. Pytask controller/task exits,
generated batchscript/jobID and every phase's command/exit stay on disk. Do not restart
the controller to harvest a terminal failure: that can resubmit the task.

Stop first source/precision/topology/value/coverage/admission failure, skipped/empty
selection, warm backend request, timeout/OOM/fatal signal or unavailable instrument.
B1.json uses measured_pending_author_disposition, never clean/accepted automatically.
G4/U2, B2, original receipt qualifications and historical failures remain explicit.
This packet does not authorize production patches, ACA development or another audit.

The semantic test includes a nine-ULP rejection control under each requested dtype,
explicit shape/dtype checks and positive cold instrumentation controls. Summary verifies
matching JAX/JAXLIB/NumPy versions and identical installed JAX instrumentation sources.
