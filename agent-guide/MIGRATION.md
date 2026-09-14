# Instruction architecture migration

The adopted source is the user-supplied efficiency proposal and kit in
`/home/hmg/sciebo/pro-audits/reduce-token-usage`. All 16 payload checksums and ZIP
integrity were verified before adoption. Historical usage evidence and the analyzer stay
with that kit; they are not injected into project instructions.

## Obligation preservation

Original project baseline: `aaefa6ff86441906f50acbbf001dae64439ad2ba`.
The old file is recoverable with `git show <baseline>:AGENTS.md`.
Every section from `## Build & Test` to EOF was preserved in order, with only
end-of-file blank lines normalized to one final newline:

| Original section | Destination | Mandatory trigger |
| --- | --- | --- |
| Build & Test | `build-and-test.md` | Tests, builds, dependencies, environments |
| Architecture | `architecture.md` | Model/DAG/API work |
| Model and Regime Interface | `model-interface.md` | Model/DAG/API work and fixtures |
| Testing | `testing.md` | Test and numerical work |
| Docstring Style | `style.md` | Python, documentation, comments and docstrings |
| Development Notes | `development.md` | Python, documentation, plots and notebooks |

The introductory package description is condensed in the new root. The original
`tier-a`, JAX and DAG imports have explicit mandatory triggers; the tier-a dependency
chain still includes shared standards, beartype and math. The shared submodule is
unchanged. Universal evidence, numerical, execution and ownership rules are explicit
in the root; their detailed obligations remain binding in the routed documents.

Commands in preserved examples must be read under the root's pixi/cap and test-delegation
rules, including old examples that show a bare `pytest`. Moving examples does not exempt
them from those rules.

## Routing adoption check

| Representative task | Required routes checked |
| --- | --- |
| Numerical kernel change | Shared standards; tier-a including beartype/math; development/style; JAX; testing/build; solver contract where applicable |
| Sharding change | All numerical/code routes plus execution/ownership/admission plans and relevant native acceptance |
| Fixture-only change | Shared standards; tier-a including real type/fixture contracts; development/style; build/testing; DAG/API routes if model fixtures |
| Audit receipt | Shared evidence rules; execution/audit route; applicable skill and its complete required reading, even if all 01–19 are required |
| Cluster request | Execution/owner route; applicable memory/cluster guidance; native test/delegation rules if running tests |

The documentation migration itself is the first bounded task: source-pinned scope,
section-content preservation check, route/link checks, raw evidence on disk, and a checkpoint
with inherited observations distinguished from live evidence. A subsequent numerical
or cluster task is still needed to assess real-world routing and usage; no savings
claim is made by this structural check.

Validation artifacts: `reports/instruction-refactor/preservation.json` records original
and extracted-section hashes; `reports/instruction-refactor/validation.json` records
structural checks and root character counts. These are local ignored working evidence.
The original plans, receipts, global instructions, client adapters and runtime source
were left in place. Only the parent owns this task's changes; no model configuration or
new monitor is installed by the templates.

Completed structural checks: six preserved sections, 13 Markdown files, six local
prose links and 14 authoritative plan paths. Root size changed from 56,048 to 5,725
characters (89.8% smaller). Markdown formatting and whitespace checks passed. This
is a root-file measurement; shared instructions and triggered modules add context.
