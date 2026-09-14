## Docstring Style

Docstrings and inline comments describe the code's *current* state in user-facing terms.
The 9-month-without-PR-context reader is the audience: a docstring that survives that
test stays useful; one that rehearses the diff or the prior implementation rots
immediately.

This applies to **all** docstrings and comments — source and tests. For tests
specifically, see also "Test docstrings — describe behavior, not history" above.

### Describe state, not history

State what is true now. Don't reference prior designs, removed code, or what was
changed. Words like "earlier", "previously", "now", "formerly", "the old", "before the
fix" are red flags.

```python
# Good — forward-looking constraint
class _DiagnosticRow:
    """Metadata captured during the backward-induction loop.

    Holds only Python-scalar metadata — no device-array references —
    so every (regime, period) row stays at a few bytes regardless of
    grid size.
    """


# Bad — rehearses prior design
class _DiagnosticRow:
    """Metadata captured during the backward-induction loop.

    Holds only Python-scalar metadata. The earlier design captured
    state_action_space and a closure directly on each row, which
    pinned every period's V template in device memory until the
    post-loop flush.
    """
```

### No PR numbers, no model-specific magic numbers

PR references (`#334 removed the host stalls`, `the bug was fixed in #42`) rot as the
codebase evolves and provide no useful signal to a reader who isn't already in context.
Magic numbers tied to a specific model size or hardware
(`~2 MB at production grid sizes`, `fits on a 16 GB device`) imply a fixed scale that's
only true on whichever model/box the comment was written against. State the qualitative
dependency instead.

```python
# Good — qualitative dependency
# Frees per-period intermediate buffers (V_arr-shaped, so
# model-dependent) so they don't stack up across the loop.

# Bad — PR reference + magic number
# Frees per-period intermediate buffers (~2 MB each at production
# grid sizes) so we don't re-introduce the host stalls that #334
# removed.
```

### Bulleted lists for enumerated cases

When describing a fixed set of cases (log levels, regime kinds, parameter types,
dispatch strategies), use one bullet per case rather than running prose. Bullets scan;
prose hides cases.

```python
# Good — scannable
# Gate falls out of the public log level:
# - `"off"` ⇒ nothing (skips even the NaN fail-fast)
# - `"warning"` / `"progress"` ⇒ NaN/Inf only
# - `"debug"` ⇒ adds the min/max/mean trio


# Bad — buried in prose
# Gate falls out of the public log level: `"off"` ⇒ nothing,
# `"warning"` / `"progress"` ⇒ NaN/Inf only, `"debug"` ⇒ adds the
# min/max/mean trio. `"off"` skips even the NaN fail-fast.
```
