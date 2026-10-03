# Receipt fixture source identity

The CI receipt fixtures scope their synthetic `GITHUB_SHA` with stdlib
`patch.dict`, restoring an inherited real source SHA when publication finishes.
Both `_payload` and the late-worker publication case use this context. The
receipt collector, identity requirements, generator and certificate bytes are
unchanged.

Two concrete inherited-SHA cases reproduced deletion of the source identity:
2 failed, 0 errors/skips, exact JUnit time **0:00:00.165**. After the minimal
restoration, both passed in **0:00:00.092**. The complete local CI-helper battery
passed **412 cases**, 0 errors/failures/skips, in **0:00:32.138**. These are
serialized CPU checks on the isolated `bdc19260` base, with candidate source
imports and matching frozen Pixi dependencies. The CI-helper battery uses the
reviewed evidence-only, module-local Pixi child accommodation; test assertions,
real child pytest/xdist and action scripts are unchanged.

The earlier native macOS run retains its qualification: shard 3's supporting
`gw0` receipt has a null source SHA. Its canonical receipt and selected/executed
JUnit population remain recorded separately. This fixture repair neither fills
that historical receipt nor declares its identity gap closed. Native GPU,
Stage 5B and Stage 8A acceptance remain separate.

Detailed logs, XML, dependency/source identities and patch hashes are under the
local evidence directory `.task-evidence/pylcm-handoff/receipt-env-fixture/`.
The parent owns commit/push and downstream propagation.
