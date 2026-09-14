# Bounded task workflow

Read `CURRENT-TASK.md`, reconcile its source pin and dirty files, then read the adopted
contract and the routes required by `AGENTS.md`. A new task should have one verifiable
outcome. A fresh task at a meaningful boundary reduces repeated context; restarting
between individual edits or tests needlessly discards useful context.

## Execution and ownership

Use [the task contract](templates/task-contract.md) for a complex implementation or Pro
handoff. Fill its baseline, invariant, real fixtures, consumers, resource constraints,
ordered gates and completion criteria from verified source. A simple documentation edit
can record these directly in the checkpoint. The template does not create new approvals.

One implementation owner is the default. Test batteries remain delegated. A worker gets
its complete scope, baseline, actual fixture conventions, commands and return format at
launch, including `zsh -ic "cap pixi run ..."` on Linux. It does not commit. Preserve
existing ACA and cluster ownership, and never restart a solve to refresh a checkpoint.
Intervene for a changed decision, actual blocker or integration; avoid repeated status
requests. Follow the user's notification intent for any authorized monitoring.

For behavior changes, demonstrate the intended failing case first. Validate collection
and fixture types, then representative positive/negative cases, then the required
precision/device matrix and integration/certificate/resource gates. This orders required
checks; it does not remove any. Source-verify selectors and use the existing native test
rules in [build-and-test.md](build-and-test.md) and [testing.md](testing.md).

## Evidence and handoff

Keep raw logs on disk and summarize them using existing tools where available. Each
acceptance receipt identifies the source SHA and dirty diff or file hashes, command,
environment, owner, exit code, evidence path/hash and limits. A killed run lacking XML
is incomplete, not passing. Keep pending gates explicit. Do not rerun an accepted matrix
just because a session resumed; reuse requires the contract's permission and a check
that intervening changes cannot invalidate it.

Update `CURRENT-TASK.md` at completion or an external blocker with the exact checkout,
implemented outcome, accepted evidence, remaining gates, pending IDs and their owners
and observation times, plus the next bounded task. Keep it compact and link full plans.
Unknown or inherited status must be labeled. Never replace the numbered planning corpus
with this checkpoint, move it to an archive, or reopen closed findings without evidence.

After two equivalent failed approaches with no new evidence, preserve the diff and
prepare the smallest reproducer, invariant, observations, attempted approaches and
precise unresolved decision. Continue unaffected authorized work. Do not silently accept
a defect or turn a service outage into an implementation retry loop.

Use [stage prompts](templates/stage-prompts.md) one at a time. They do not supersede the
current audit protocol or its full reading requirements. High-risk designs still need
an independent oracle or reviewer; an author's own approval is not independent review.

## Evaluating this refactor

For the next two or three comparable accepted tasks, record parent and child usage,
model/effort, correction loops, escaped defects and unchanged acceptance gates. Compare
usage per accepted change. Entry-point character counts are only a size measurement,
not demonstrated billing or context savings. This refactor does not change model settings.
The supplied analyzer remains in the original efficiency kit; if used, inspect it first
and run it through `pixi run python`, including child exports where available.
