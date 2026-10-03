# Handoff: #489 (keep shock states read only through their next-period draw)

This is a working directory for passing the work between sessions. Delete it before
anything merges into `main`. Owned by the cloud session
https://claude.ai/code/session_01Pxm4joxHzgeNTvT5vf4eN9; the original task came from
the aca session (`session_01Hs8mfXPYyEa24KaAp6BwCW`).

## State (2026-10-02)
- **Branch:** `claude/keep-states-feed-next-period-i6s9pd`, stacked on #488
  (`codex/phase-specific-model-graph`, f5294e45). Draft PR #489.
- **Contract:** in the PR description and in `CHANGES.md` ("A shock read only through
  its next-period draw stays in the regime").
- **Evidence:**
  - the closed-form test `tests/solution/test_transition_local_process_draw.py`
    (GridSearch, DC-EGM, NB-EGM);
  - the toy regression `tests/regime_building/test_draw_read_states_are_kept.py`;
  - NB-EGM direct-oracle routes `stochastic_node_draw` and `stochastic_node_draw_jump`.
- **CI on 00f87c6:** green except two.
  - The macOS general shard 1/2 timed out at 60 min; it took 56 min on #488.
    The solving tests in `test_draw_read_states_are_kept.py` now carry
    `@pytest.mark.slow`, as `tests/conftest.py` requires for solving tests outside
    `tests/solution/`.
  - "Run benchmarks on GPU" is red on #488 too: the pinned benchmark aca-model
    predates #488's API (PR comment 5950059143).

## Open
- **Downstream acceptance:** aca-model is being ported onto #489 by a Codex session.
  Then remove `medical_cost_shocks_carried` and lift its 16 strict xfails (15 NB-EGM,
  1 DC-EGM).
- **Before merge:** squash or reword the WIP commits `eb40719` and `2466dfc`.
