# One measured macOS weight and maintained four-shard regeneration

Prepared against clean published Stage8A head
`3028a022e8156880d71c6733a47c43f20828283b`. The only tracked configuration
change is `tests/ci/ci-workloads.json`; parent alone commits/publishes. This
refresh changes no numerical source, workflow, sharder/generator, test,
certificate, lock/environment input, policy, timeout or acceptance budget.

## Measurement and mixed provenance

`tests/test_period_feasibility_fixed_bindings.py` macOS weight changes
**536.486→790.597 JUnit testcase seconds**; its count remains **150**. All150
actual cases passed. Millisecond inputs sum exactly790597milliseconds; the
floating-point JSON sum790.5970000000001 has only summation residue. This is
summed case time, not file/lane wall or causal regression evidence.

Matching one-row CSV: [mac-one-weight-refresh.csv](mac-one-weight-refresh.csv),
SHA256 `7656282729fff1adddd69be837fe3efd50fbf348cf61c2c5a658bcabea3a779a`.
Every other Mac weight/count retains its original run37007461636 attempt1,
tested mergefa1ba52864e032191df6620eecb08d6533b03964, macOS-general1/2 CSV
provenance. Existing `junit_source` and `source.leg_weights_note` remain verbatim
prefixes, followed by this explicit one-file exception. Other legs/root CSV
and `frozen_head` remain unchanged; this does not re-measure the whole leg.

Exception source: CPU run37092905063 attempt1, published head3028a022e8156880d71c6733a47c43f20828283b,
tested merge1717cb235b3f73740080c3ad7614f370c1d3e920, equal published/tested
tree1c48030396860fb016f19a2a9ce44b5b3817078a, artifact11262958408
`junit-cpu-fp64-macOS-general-3`. Actual fp64 CPU, Mac26.6.2 arm64,
Python3.14.7/pytest9.1.1, four workers, `dist=loadfile`, selection `not slow
and not manual`. Canonical/worker selected, collected, executed and outcome
unions and XML were exactly reconciled. ZIP SHA256
`722d0b5e96d576fef972544818117ce3a9185917f3dd41289f3e728140c9d2a3`;
XML SHA256 `0f92efb653147db34324b74f29d6ce06639cf1546c777e9ba2f4395e55c67494`;
canonical receipt SHA256 `c4058551ea5685a2fcacae7444a5763c4748a5ec26753bc458a389fec5601bc4`.

Raw source receipts/XML/artifact and complete per-node/per-file comparison:
`/home/hmg/econ/aca-dev/.task-evidence/pylcm-handoff/cascade-macos/497/snapshot-20261003T035110Z/`.
See its `REPORT.md`, `MAC3-TIMING.md`, `file-timings.json`,
`case-timings.json`, `selected-vs-495.json` and `artifact-hashes.txt`.

## Generated membership, prediction and preservation

The maintained weighted-LPT generator detected input-driven drift (exit1),
rewrote the manifest (exit0), and final `--check` passed (exit0), with no new
test registrations. The actual maintained result agrees with the earlier
algorithm preview: target module moves to **Mac2**,353 files change shard,
registered four-lane file counts become **127/128/158/130**, with543files
assigned exactly once. Four shards/four workers remain unchanged.

Only four Mac predictions were refreshed with existing
`ShardReport.payload_minutes`: `max(sum known seconds/workers, largest known
file seconds)/60`, rounded to two decimals as established in the manifest.

| Mac shard | Known summed seconds | Largest known file seconds | Unmeasured files | Rounded heuristic minutes |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4831.476 | 1084.353 | 0 | 20.13 |
| 2 | 4831.476 | 790.597 | 0 | 20.13 |
| 3 | 4831.475 | 639.232 | 30 | 20.13 |
| 4 | 4831.476 | 461.580 | 0 | 20.13 |

Historical stored19.48 predictions are replaced; unrounded refreshed lower
bounds are about20:07.869. Unweighted files remain explicitly unmeasured,
including the slow solution files whose registration is not macOS execution.
The heuristic excludes coordination/setup/imbalance and unknown costs. It is
not a measured wall-time guarantee. Prior actual Mac3 payload25:29 missed24
minutes while whole job26:53 fit30; **fresh naturally triggered macOS CI is
required before claiming all four24/30 budgets**. No CI rerun/cancel was issued.
Other-leg capacity and native ACA simultaneous3nodes×8A40 acceptance are unaffected.

Deep normalized before/after equality removes only this single seconds entry,
two appended provenance strings, four Mac `files` lists and their four
prediction fields. Every other field is identical, including counts, root
weights/freeze, other leg weights/memberships/predictions, shared universe,
unweighted/excluded sets, selection/deselections, environment/policy, topology,
all16 coverage contributors and24/30/30 guardrails. Original provenance strings
are verified prefixes. Complete tracked pathsets match;328 source files match
baseline byte checks. At the contract-run freeze, the sole tracked diff was
360insertions/360deletions in the generated manifest. Workflow/helpers/tests/source/certificate bytes are
unchanged. No test/gate relaxation or manual assignment override is introduced.

## Bounded local validation and identity

Three existing contract modules passed **129/0 failures/0 errors/0 skips**,
JUnit **0:00:07.783**, on hmg-office. Actual executable
`/home/hmg/econ/aca-dev/.codex-worktrees/invariant-node-local-verified/.pixi/envs/tests-cpu/bin/python3.14`,
Python3.14.7/pytest9.1.1; maintained accessor import was explicitly candidate-local.
Used own preinstalled locked prefix with `--as-is`, no install/build/symlink.
The run was serial under cap; afterwards cap.slice Units0/Tasks0 and no own
test/Python process remained. No new numerical/whole-CI/native acceptance is claimed.

From `/home/hmg/econ/aca-dev/.codex-worktrees/invariant-node-local-verified`:

```sh
zsh -ic 'PYTHONPATH=/home/hmg/econ/aca-dev/.codex-worktrees/invariant-node-local-verified:/home/hmg/econ/aca-dev/.codex-worktrees/invariant-node-local-verified/src cap pixi run --as-is --manifest-path /home/hmg/econ/aca-dev/.codex-worktrees/invariant-node-local-verified/pyproject.toml -e tests-cpu python -m tests.ci.generate_ci_workloads --repo-root /home/hmg/econ/aca-dev/.codex-worktrees/invariant-node-local-verified --check'
zsh -ic 'PYTHONPATH=/home/hmg/econ/aca-dev/.codex-worktrees/invariant-node-local-verified:/home/hmg/econ/aca-dev/.codex-worktrees/invariant-node-local-verified/src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 cap pixi run --as-is --manifest-path /home/hmg/econ/aca-dev/.codex-worktrees/invariant-node-local-verified/pyproject.toml -e tests-cpu pytest tests/ci/test_ci_workloads_manifest.py tests/ci/test_ci_workloads_guardrails.py tests/ci/test_cpu_workflow_contract.py --precision=64 -v -n 0 --junitxml=/home/hmg/econ/aca-dev/.task-evidence/pylcm-handoff/cascade-macos/497/mac-one-weight-refresh/ci-contracts.xml'
```

Full data application/generation/prediction commands, bounded scripts (stdlib+
maintained sharder only), expected-drift/generate/check logs, XML/log/summary,
baseline manifests/pathsets/source hashes and `manifest-proof.json` remain at
`/home/hmg/econ/aca-dev/.task-evidence/pylcm-handoff/cascade-macos/497/mac-one-weight-refresh/`.
This report and CSV are publication evidence; parent force-adds the ignored
report files. No whole-stage retest or normal-hook completion is claimed here.

Final manifest SHA256 `696f3fb61d39767672af3bea7bc04f706976c4d2e147f4a1b5fe48869a8a3a8a`;
tracked patch SHA256 `7c9ea31dd8eedb55a453f7f5f3ddb255b063490e9082bdbeb379b61f6b1814d5`;
XML SHA256 `121717a8361719afa1b4fd72dc2bb07a37599d4bf37e1f2c7f5754f49658eb94`;
manifest proof SHA256 `3d80d7aa5d76721c61a86f165f7b7fce25a76a2357351d9a41f8fb2985bd2976`.

After releasing the slot, parent requested a documentation-only live README
checkpoint correction. It records published#497/base#495, qualified local195
receipts, Mac subset/no new71 coverage, existing timing miss, this prepared
refresh and still-local driver/unsubmitted24GPU owner gate. Historical stage
table/decisions/plan remain intact; original cloud ownership is labeled historical.
README SHA256 `aa981b3c4fe9fe782bbe07c38e6debd96363ba75a8a2d1aaee81677341257cbd`;
complete tracked patch including checkpoint SHA256
`f516c6cae52826f334fff34153f1910fca65228db2c6cfa53b00945f1ea806a6`.
No source/config/test change followed the129-case acceptance and no further
Python/import/test/compute was run. Parent still owns normal commit/push hooks.
