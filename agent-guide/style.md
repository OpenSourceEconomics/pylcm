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
(`Params`, `UserParams`). A string that names a regime, state, action, function or
parameter carries its alias from `lcm.typing` (`RegimeName`, `StateName`, `ActionName`,
`FunctionName`, `ParameterName`), and a `__`-joined path through the params or function
namespace carries `QualifiedName` from `_lcm.typing`, never a bare `str`. Every alias is
a `type X = ...` statement.

```python
# Good — the constructor union and the label alias
def resolve_law(*, transition: Transition | Phased, regime_name: RegimeName) -> Law: ...


# Bad — `object` hides the union, `str` hides which label the string is
def resolve_law(*, transition: object, regime_name: str) -> Law: ...
```

`object` and `Any` are for slots that genuinely hold unrelated types. The
`precise-annotations` hook (`tests/ci/precise_annotations.py`) checks every annotation
under `src/` and in the `docs/` notebooks, string annotations and `cast` targets
included:

- `PAN001` / `PAN002` ⇒ `object` / `Any` in an annotation
- `PAN003` ⇒ a bare `str` on a regime, state, action, function, qualified or parameter
  name; on a target, source or argument name it asks for a domain alias picked by hand
- `PAN006` ⇒ a generic without its type arguments (`Callable`, `dict`, `type`, ...),
  which leaves them `Any`
- `PAN007` ⇒ an alias not written as a `type X = ...` statement

Two placements are exempt by rule:

- `object` in a parameter whose type the data model fixes: every parameter of a
  comparison or containment dunder (`__eq__`, `__contains__`, ...), `__setattr__`'s
  `value` and `__deepcopy__`'s `memo`
- `object` or `Any` in an alias of the `else:` branch of `if TYPE_CHECKING:`, the
  runtime fallback the beartype claw sees, when a comment directly above it, or above
  the run of fallbacks it belongs to, gives the reason

```python
if TYPE_CHECKING:
    from _lcm.solution.model_authority import SolutionAuthority
else:
    # The authority imports this module, so the claw sees a wide fallback.
    type SolutionAuthority = Any
```

Any other deliberate finding carries `# noqa: PANxxx - <reason>` on the line the hook
reports. A code without a reason suppresses nothing.

```python
def hash_user_object(*, value: object) -> str:  # noqa: PAN001 - any user object
    ...
```
