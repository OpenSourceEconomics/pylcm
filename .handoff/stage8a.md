# Stage 8A: node-local component jobs (WIP, interrupted)

The worker was stopped mid-implementation on 2026-10-02 because credits ran out.
**Nothing here has been validated.** The design is in `reports/stage8a-design.md`.

The tree was committed as found:
- `component_fragments.py`, `lcm/component_jobs.py` and `test_component_jobs.py`;
- edits to `simulate`, `block_major`, `model` and `result`;
- certificate repins that were in progress.

Pre-commit hooks were skipped (`--no-verify`).

Next: re-run the certificate procedure from scratch (repin, `check_seals`, `verify`,
`--self-test`), then `prek` and the tests, red first where they're missing. Treat every
pin in `direct_flow.py` and `sources.json` as suspect.
