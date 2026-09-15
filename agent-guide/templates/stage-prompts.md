# Stage-specific prompts

Use one prompt at a time. These proposed prompts do not supersede adopted audit contracts, approval requirements, commit ownership, test delegation, or cluster rules. Replace angle-bracket fields with verified information.

## 1. One-time checkpoint from the existing Codex session

```text
Create a compact handoff for a fresh implementation session. Do not begin new
implementation, rerun passing tests, rebuild published bundles, or start/restart
cluster jobs for this handoff.

Use the current checkout and existing verified receipts to write CURRENT-TASK.md:
exact source head and dirty files; adopted decisions and authoritative references;
what is implemented; exact-head acceptance evidence and remaining gates; pending
review IDs and job IDs with owner and observation time; next bounded task and
its dependencies. Mark stale or unverified status explicitly. Preserve all full
numbered planning documents in their current authoritative location.

Resolve discrepancies from existing evidence where possible; otherwise record
the precise missing fact. Do not turn the handoff into a new broad audit.
Keep the checkpoint roughly 1,000 words, using paths/hashes instead of copied logs.
Return the checkpoint path and only material contradictions or blockers.
```

## 2. Pro: turn the next decision into an execution contract

```text
Use the supplied exact-head source, current checkpoint and authoritative decisions
to prepare the next bounded implementation contract, not another broad roadmap.
Resolve the design uncertainty here where the evidence permits it.

Specify: one outcome and its invariants; base SHA; relevant paths/symbols and
consumers; permitted changes and non-goals; failure modes and independent oracle;
real fixture/metadata conventions; ordered native acceptance gates; owner and
resource constraints; exact remaining external dependencies; completion criteria.
Use only source-verified test names/commands, or mark their resolution as required.
Do not invent native results.

Prefer an executable reference, reproducer or candidate patch over prose when
it removes an otherwise difficult ambiguity. Mark its applicability and what was
actually run. Leave ordinary implementation details to Codex. Order the next
few contracts, but make only the first unblocked one ready to execute. Preserve
all adopted quality gates and do not reopen closed findings without new evidence.
```

## 3. Codex: bounded execution

```text
Implement <TASK-CONTRACT> against its pinned baseline, reconciling CURRENT-TASK.md
with the checkout first. Complete the authorized task without asking to proceed
between routine steps. Do not add new workstreams or redesign settled contracts.

Read the required instructions, task source, affected consumers and tests. Expand
reading when a concrete dependency requires it; record a material scope expansion.
Keep raw logs on disk. Return paths, exact commands, exit codes and a short failure
summary rather than repeatedly pasting complete logs.

Use one implementation owner by default; retain mandated test/cluster delegation.
Use another implementation worker only for an independent, bounded scope. Parent
remains sole committer and existing ACA/cluster ownership remains unchanged.

Validate fixture collection and one representative case before running the full
required matrix. Fix change-owned failures; never loosen an invariant or claim
acceptance merely to meet a usage target. Reuse exact-head passing evidence only
where the contract permits it and intervening changes cannot invalidate it.

On two materially equivalent failed approaches with no new evidence, stop the
unproductive loop and return one precise escalation packet. For service outages,
do not repeat unchanged requests indefinitely or bypass approval controls.

Finish with the bounded diff, evidence manifest, exact head, unresolved gates and
updated checkpoint. Stop when complete or blocked on external evidence. Do not
continue into an open-ended next phase or conversational job monitoring.
```

## 4. Delegated worker contract

```text
Own only <FILES/SYMBOLS> for <TASK-ID> at <BASE-SHA>. Other agents may be working;
do not overwrite their changes. Implement <OUTCOME> under <INVARIANTS> and run
<AUTHORIZED CHECKS>, using real fixtures and the declared environment. Do not
commit; the parent integrates and commits. Do not expand into adjacent tasks.

Keep intermediate logs local. Report once when complete, or immediately for a
blocker that changes the parent's decision. Return changed paths, diff identity,
commands/exit codes, evidence paths, counterexamples and remaining limitations.
Avoid repeated status messages that add no new evidence.
```

## 5. Focused independent review

```text
Review <BASE..HEAD> against <AUTHORITATIVE-CONTRACT> for <SPECIFIED RISK CLASS>.
Follow the adopted audit protocol and its complete required-reading obligation.
Read the normative contract and source before relying on the implementation's
success narrative. Inspect unchanged consumers when the changed interface makes
them relevant. Do not reopen closed findings without a concrete new counterexample.

Retain numerical, ownership, admission, RNG, layout and provenance gates relevant
to this scope. Separate source-level conclusions, isolated reference checks,
supplied native evidence and independently executed native acceptance. No claim
may be upgraded merely because another level passed.

Return the protocol's complete required artifacts. For each remaining issue,
provide an exact counterexample or evidence gap, its owner, and one bounded next
action with an acceptance criterion. Do not append a general improvement backlog.
```

## 6. Escalation packet, instead of repeated redirection

```text
Prepare one diagnosis packet for <BLOCKER>. Include the exact base/diff, smallest
reproducer, expected invariant, observed result and raw evidence paths, approaches
already attempted and their outcomes, and the precise unresolved decision.
Distinguish project defects, invalid fixtures, stale receipts, missing permissions,
and external service failures. Do not retry or start unrelated work while packaging
this packet. Preserve the current work and all unpassed acceptance gates.
```
