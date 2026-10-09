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

## Precise Annotations

**Annotate with the narrowest type that has a name.** In order of preference: the
constructor union the value is built from (`Transition | Phased`), a Protocol from
`_lcm.typing` for a callable (`EconFunction`, `RegimeTransitionFunction`), a type
parameter when the output has the input's type, and a recursive alias for a tree
(`Params`, `UserParams`). A string that names a regime, state, action or function
carries its alias from `lcm.typing` (`RegimeName`, `StateName`, `ActionName`,
`FunctionName`), never a bare `str`.

```python
# Good — the constructor union and the label alias
def resolve_law(*, transition: Transition | Phased, regime_name: RegimeName) -> Law: ...


# Bad — `object` hides the union, `str` hides which label the string is
def resolve_law(*, transition: object, regime_name: str) -> Law: ...
```

`object` and `Any` are for slots that genuinely hold unrelated types. The
`precise-annotations` hook (`tests/ci/precise_annotations.py`) checks every annotation
under `src/`, string annotations and `cast` targets included:

- `PAN001` / `PAN002` ⇒ `object` / `Any` in an annotation
- `PAN003` ⇒ a bare `str` on a regime, state, action or function name

Two placements of `object` need no marker: a parameter of a comparison or containment
dunder (`__eq__`, `__contains__`, ...), and a parameter that its own function narrows
with `isinstance`, `issubclass` or `match`. Any other justified `object` or `Any` takes
a marker on its own line directly above it, naming one of three reasons:

- `heterogeneous=<slug>` ⇒ the slot holds unrelated types; the slug names the payload
- `library-signature=<dotted.name>` ⇒ an external signature fixes the type
- `import-cycle=<TypeName>` ⇒ the precise type, which cannot be imported here at runtime

```python
# annotation-exempt: heterogeneous=json-value
payload: object
```

Findings without a marker count against `tests/ci/precise-annotations-baseline.json`,
per file and rule. The hook lowers a count when it falls and fails when it rises, so
an edit can only remove imprecise annotations, never add them.
