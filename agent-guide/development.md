## Development Notes

### JAX Integration

- All numerical computations use JAX arrays
- GPU support available via jax[cuda12] / jax[cuda13] (Linux). macOS runs on CPU: there
  is no `metal` pixi environment, so Apple-Silicon GPU acceleration is not installable
  from this project
- Functions are JIT-compiled during Model initialization for performance
- `MappingProxyType` is registered as a JAX pytree for use in JIT-compiled functions

### Immutability

- Internal data structures use `MappingProxyType` instead of `dict` for immutability
- Type annotations use `Mapping` for read-only dict-like interfaces
- User-provided dicts in `Regime` are automatically wrapped in `MappingProxyType`

### Type System

- Extensive use of typing with custom types: user-facing aliases in `src/lcm/typing.py`,
  engine-side aliases and protocols in `src/_lcm/typing.py`
- Type checking with ty (`prek run ty --all-files`)
- Use `# ty: ignore[error-code]` for type suppression, never `# type: ignore`
- JAX typing integration via jaxtyping

#### Domain string aliases

The following PEP 695 aliases (`type X = str`) live in `src/lcm/typing.py` (re-exported
from `src/_lcm/typing.py`, so `from _lcm.typing import RegimeName` keeps working) and
exist purely to make signatures self-documenting. They are runtime-equivalent to `str`;
ty erases them, so misuse never crashes — it just hides intent. Prefer the alias over
bare `str` whenever a string slot has a fixed semantic role.

| Alias                    | Use for                                                                                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RegimeName`             | Names of regimes — keys of `regimes`, `internal_regimes`, `regime_names_to_ids`, `state_action_spaces`, `period_to_regime_to_V_arr`, `regime_to_V_arr`. |
| `StateName`              | Names of states — entries of `state_names`, keys of `regime.states`, `states_per_regime` values.                                                        |
| `ActionName`             | Names of actions — entries of `action_names`, keys of `regime.actions`.                                                                                 |
| `StateOrActionName`      | Mixed flat keys covering both states and actions — `flat_grids`, `all_grids[regime]` values, `state_and_discrete_action_names`.                         |
| `ProcessName`            | Subset of `StateName` for stochastic processes — keys of `_ContinuousStochasticProcess`-typed mappings, process-transition helpers.                     |
| `FunctionName`           | User-supplied function names — `"utility"`, helpers; keys of `Regime.functions`, `derived_categoricals`.                                                |
| `TransitionFunctionName` | Names of transition callables — `next_<state>`, `weight_next_<state>`; keys of `state_transitions` and per-target dicts.                                |

When a string slot covers more than one of the categories above, prefer a union (e.g.
`dict[RegimeName | TransitionFunctionName, ...]`) over bare `str`. Plain `str` is the
right type only when the keys really are heterogeneous and don't map onto any of the
aliases — DataFrame column labels, free-form param-template leaf strings, and similar.
Use `NewType` only when an opaque ID is required; the project has not needed that so
far.

### Code Standards

- Ruff for linting and formatting (configured in pyproject.toml)
- Google-style docstrings
- All functions require type annotations
- Pre-commit hooks ensure code quality
- Never use `from __future__ import annotations` — this project requires Python 3.14+

### Keyword-Only Arguments

**A function takes its arguments keyword-only as soon as there are two.** This binds
everywhere — engine internals and model functions alike. Readability is not something
only the public surface earns, and a call reading `f(a, b)` says nothing about which
value is which.

```text
# Good
def compute_regret(*, published: FloatND, reference: FloatND) -> FloatND: ...


# Bad — the call site cannot say which is which
def compute_regret(published: FloatND, reference: FloatND) -> FloatND: ...
```

There are exactly three exemptions:

1. **A callback whose caller is a library that passes positionally** — `lax.scan` bodies
   `(carry, x)`, `custom_jvp` rules `(primals, tangents)`, pytree
   `unflatten(aux, children)`. Not a judgement call: Python raises otherwise. `dags` is
   *not* in this class — it binds by parameter name and accepts keyword-only functions,
   so a model function complies like anything else. Put a marker immediately above the
   callback (above its decorators, if any), naming the positional library caller:

   ```python
   # keyword-only-exempt: library-callback=jax.lax.scan
   def scan_body(carry: Carry, item: Item) -> tuple[Carry, Output]: ...
   ```

1. **A module that implements an arithmetic and nothing else** — operator surrogates
   keep the spelling their operation is known by (`two_sum(a, b)`,
   `dd_mul(left, right)`). State it once in the module docstring; such a module may
   contain *only* operators, so the exemption stays auditable rather than sprinkled per
   function. The declaration is `Keyword-only exemption: arithmetic-only module.`

1. **A public loader with one natural primary resource argument** — keep that resource
   positional while making every option keyword-only. Mark the function with the exact
   primary parameter so the exception remains auditable:

   ```python
   # keyword-only-exempt: primary-argument=path
   def load_snapshot(path: Path, *, exclude: Sequence[str] = ()) -> Snapshot: ...
   ```

### Module Layout

Write "deep" modules: important public function(s) at the top, private helpers below.
Readers should see the API first without scrolling past implementation details.

Never add decorative section-separator comments like:

```python
# ---------------------------------------------------------------------------
# Section name
# ---------------------------------------------------------------------------
```

Code structure should be self-evident from function names and ordering.

### Naming and Docstring Conventions

- **Noun vs verb for phase vocabulary — "the name tells you what you get."** Use the
  noun (`solution`, `simulation`) when the name yields a *thing* — an artifact or bundle
  of phase artifacts: `regime.solution`, `SolutionPhase`, `SimulationResult`,
  `PhasedRegimeSpec.solution`. Use the verb (`solve`, `simulate`) when the name denotes
  the *act* or selects a variant to be used when acting: `model.solve()`,
  `Phased(solve=f, simulate=g)`, `phase: Literal["solve", "simulate"]`, and all
  attributive compounds in identifiers and prose (`solve_transitions`, "the solve-phase
  consumer", "the solve grid"). Litmus: you get a bag → noun; you get behavior or name
  the act → verb.
- **No unnecessary parameter aliases.** When a function has a single (or very few) call
  site(s), the parameter name should match the variable name being passed. Don't shorten
  parameter names just for brevity — e.g., use
  `regime_transition_probs=regime_transition_probs` not `probs=regime_transition_probs`.
- **Docstrings must match type annotations.** Use the type name from the annotation:
  - `Mapping[...]` → "Mapping of ..." in docstrings
  - `MappingProxyType[...]` → "Immutable mapping of ..." in docstrings
  - `tuple[...]` → "Tuple of ..." in docstrings
  - `list[...]` → "List of ..." in docstrings
  - Never write "Dict" when the annotation is `Mapping` or `MappingProxyType`
- **Consistent naming across a file.** When multiple functions in the same file use the
  same concept (e.g., `arg_names`), use the same parameter name everywhere — don't
  introduce synonyms like `parameters`.
- **Helper function names follow `{verb}_{qualifier}_noun` patterns.** E.g.,
  `get_irreg_coordinate`, `find_irreg_coordinate`, `get_linspace_coordinate` — not
  `get_coordinate_irreg`.
- **Pick the single narrowest jaxtyping alias — never scalar/array `@overload` pairs,
  never `ScalarX | XND` unions.** ty erases jaxtyping shape annotations: `ScalarFloat`,
  `Float1D`, and `FloatND` all reveal as `Array`, so scalar/array `@overload` pairs and
  `ScalarFloat | FloatND`-style unions add zero static precision — they are pure noise.
  At runtime, beartype treats a 0-d float array as satisfying both `ScalarFloat` and
  `FloatND`, so `ScalarFloat ⊆ FloatND` and the union is redundant. Annotate each slot
  with the one alias that matches its genuine rank: `ScalarFloat`/`ScalarInt` for
  fixed-0-d, `Float1D`/`Int1D` for fixed-1-d, `FloatND`/`IntND` for genuinely rank-
  polymorphic. Never use a bare `Array` annotation — always reach for the narrowest
  `lcm.typing` alias.
- **`func` for callable abbreviations** — use `func`, `func_name`, `func_params` (never
  `fn`). Full word `function(s)` in dataclass field names and public method names.
- **Singular `state_names` / `action_names`** — not `states_names` / `actions_names`.
- **`arg_names`** — not `argument_names`.
- **Imperative mood for docstring summary lines.** Write "Return the value" not "Returns
  the value". The summary line uses bare imperative: "Create", "Get", "Compute",
  "Convert", etc.
- **Inline field docstrings (PEP 257) for dataclass attributes.** Place a `"""..."""` on
  the line after each field instead of listing fields in an `Attributes:` section in the
  class docstring.
- **MyST syntax in docstrings, not reStructuredText.** Use `` `code` `` (single
  backticks) for inline code, `$...$` for inline math, ```` ```{math} ```` fences for
  display math, and `[text](url)` for links. Never use rST-style ``` `` code `` ```,
  `:math:`, `:func:`, or `` `link <url>`_ ``.

### Plotting

- Always use **plotly** for visualizations, never matplotlib. Use `plotly.graph_objects`
  and `plotly.subplots.make_subplots`.

### Notebooks

Explanation notebooks live in `docs/explanations/*.ipynb`. After editing one, verify:

- Each cell's `source` is a JSON array of lines (one array element per line), never a
  single multi-line string — a one-string `source` produces an unreadable diff.
- Outputs and execution counts are stripped. `nbstripout` is a pre-commit hook, not a
  pixi task, so strip a file with `prek run nbstripout --files <file>`.
- Markdown and code use literal UTF-8 characters (`—`, `→`, `μ`), never `\u`-style
  escape sequences.

### Key Dependencies

- **jax**: Numerical computation
- **jaxtyping**: Array type annotations
- **pandas**: DataFrame output
- **dags**: Function composition and nested namespace flattening
- **plotly**: Plotting and visualization
