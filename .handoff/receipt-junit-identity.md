# Shared JUnit identity repair

Bounded helper repair on #486 base `db1432ce361cc6091b5c7644211b40772d4e4ab3`.
`junit_identity` preserves complete parametrized names containing `::`, while
retaining class prefixes. Existing reconciliation populations and source checks
are unchanged. Seven new cases were observed RED before the minimal parser edit.

| Local accommodated gate | Passed | Failed | h:mm:ss | XML seconds |
|---|---:|---:|---|---:|
| Focused RED | 3 | 7 | 0:00:02.844 | 2.844 |
| Focused GREEN | 10 | 0 | 0:00:02.714 | 2.714 |
| Receipt module | 31 | 0 | 0:00:15.255 | 15.255 |
| Whole tests/ci | 410 | 0 | 0:00:32.413 | 32.413 |

All error/skip counts are zero. Own frozen type-checking install and normal hooks
passed, including whole-project ty. Candidate src tree is unchanged
`510b36091860caf978d722c068324e36a02fef35`; no certificate reseal is required.
Execution used capped Pixi with candidate-root imports and a narrowly scoped
evidence-only Pixi child launcher for inherited subprocess seams. It changes no
assertions or test selection and is not tracked in this patch. Native numerical
or performance acceptance is not claimed.

Native #491 audit reconciles 75/77 canonical receipts. Both rest-slow lanes still
fail exact population reconciliation due an extra zero-duration module-level
collection-skip XML outcome. These are retained unresolved gaps. All six
prescribed general/solution capacity legs reconcile exactly. This parser fix
does not handle collection skips or approve CI weight refresh. No full 77
completion/CSV acceptance claim, filtering or normalization of these gaps.

External evidence: `.task-evidence/pylcm-handoff/receipt-identity/REPORT.md`, raw
JUnit/logs, command traces, native audit and hashes in the aca-dev workspace.
The four-shard macOS proposal is separate and remains unadopted; its retained
weights give an observed shard 4 scheduling lower bound 31:42, above 24 minutes.
