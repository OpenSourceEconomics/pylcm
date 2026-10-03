# General blocked/unblocked value contract

The user approved general blocked versus independently compiled unblocked
published values within **eight representable steps**, at both fp32 and fp64.
Policies, states, actions, regimes, subject identities/order, schema, dtype,
shape, and sharding remain exact. Same compiled-program and same-input-solution
byte gates retain signed zeros and NaN payloads. Stage 7's accepted one-ULP
exception and every certificate gate remain unchanged.

This isolated work started at `d5fb50e11fb86d03b4a9ed03717224bfb454b595` and
incorporated the parent-published mesh fix
`15defbb01186d49612da88185ef29927978777e1` with a no-commit merge. Its production
source tree is exactly `babc7fa21e75dccee1894fec46c6f7089b3a4810`; this chunk
changes tests and documentation, not production arithmetic or certificate pins.
Parent alone commits/publishes. Standards are pinned to
`4a44a464673e130a3d7fcb7a1af4b3cd2c26bcaf` in an independent local checkout.

## Defects in the comparison instrument

- The existing 5B panel fallback admitted a one-ULP float state or action change
  because it tolerated every float column. Initial fp64 RED: **2 failed,
  1 passed**, no errors/skips, **0:00:15.254**.
- The exact archived original comparator also admitted a structural signed-zero
  change. Control RED: **3 failed, 1 passed**, no errors/skips,
  **0:00:14.550**. All three failures were the expected missing refusal.
- Spacing at the larger magnitude admitted nine representable steps across a
  binade: seven steps below one, then nine toward positive infinity, reports
  5.5 units of the larger spacing. The actual spacing-based helper RED had
  **1 failed**, no errors/skips, **0:00:04.391**.

The general value helper reuses the existing ordered-bit `_array_ulp_gap` in
`lcm.tuning`, after exact coordinate/dtype/shape checks; its bound is eight.
The shared spacing-based helper and Stage 7 gates are unchanged. The singleton
panel comparator applies the value bound only to public `value` and raw `V_arr`.
Every other public/raw field remains exact, including float storage bytes.
The existing all-byte panel comparator remains intact for same-solution and
same-program callers. No blanket float-column or operand-magnitude tolerance
is introduced.

## Executed CPU acceptance

The final matrix contains **25 existing general comparisons plus seven detector
cases**, in fresh eight-host-device CPU processes. It covers blocked/unblocked
values and panels, changed parameters, shared typed-terminal reads, block-major
split/combined simulation, empty/unbalanced populations, sharded assets, and:

- eight steps accepted, both within a binade and across its boundary;
- nine steps refused in both geometries;
- one-ULP float state/action changes refused;
- a structural signed-zero change refused.

| Precision | Passed | Failed | Errors | Skipped | JUnit duration |
|---|---:|---:|---:|---:|---|
| fp64 | 32 | 0 | 0 | 0 | 0:03:12.105 |
| fp32 | 32 | 0 | 0 | 0 | 0:03:01.595 |

Widening the actual general helper's bound from eight to nine makes both nine-step
detectors fail while both eight-step controls pass: **2 failed, 2 passed**, no
errors/skips, at each precision. JUnit durations are **0:00:14.049** at fp64 and
**0:00:17.527** at fp32. This evidence-only module-local mutation never changes
tracked source or assertions.

A preceding spacing-helper diagnostic battery completed **537 passed, four
expected xfails**, no failures/errors, **0:06:59.729** at fp64. It freshly executed
the unchanged same-program value/panel and partitioned-kernel controls, but is
not the final ordered-distance general acceptance receipt. Its existing
nonfinite-spacing test emitted one invalid-subtract warning; that shared helper
was not edited. The complete new general matrices above supersede its general
value acceptance slice.

Raw XMLs/logs, exact selectors, tested patch/file hashes, archived baseline
comparator, mutation plugins, and environment proof are in the enclosing
workspace's `.task-evidence/pylcm-handoff/rounding-contract/`. Runtime imports
resolve to this candidate's `src/lcm` and `src/_lcm`; the explicitly reused
integration tests-cpu environment has byte-identical manifest/lock and matching
native source fingerprint. This worktree has its own frozen type environment.

Normal certificate verification passed (`verify.log`, result `pass`, no errors).
The manifest check passed without generated changes (`manifest-check.log`), and
all applicable normal hooks passed (`hooks-final.log`), including Ruff, type
checking, seal checks, and mutation-anchor resolution. Formatting and an
equivalent split of the helper's compound assertion followed the numerical
matrices; the final patch records these style-only changes. No full mutation
self-test rerun is claimed here: unchanged source/certificate populations permit reuse of the exact
parent mesh receipt at `15defbb`, including all 29 action controls and typed
supplemental controls. This CPU contract acceptance does not close native GPU,
four-A100 layout/performance, ACA applicability, or Stage 8A obligations. Optional
blueprint-cache work remains deferred; `main` is not merged.
