---
title: Conventions
---

# Conventions

## Code Style

- **Ruff** for linting and formatting (configured in `pyproject.toml`)
- All functions require type annotations
- Google-style docstrings in imperative mood ("Return", not "Returns")
- Use `# ty: ignore[error-code]` for type suppression (never `# type: ignore`)
- Never use `from __future__ import annotations`

## Testing

- Plain `pytest` functions — no test classes
- Use `@pytest.mark.parametrize` for test variations

## Naming

- `func` for callable abbreviations (never `fn`)
- `state_names` / `action_names` (not `states_names`)
- `arg_names` (not `argument_names`)

## Plotting

- Always use **plotly** (`plotly.graph_objects`, `plotly.subplots.make_subplots`), never
  matplotlib

## Docstrings

- MyST syntax (single backticks for code, `$...$` for math), not reStructuredText
- Inline field docstrings (PEP 257) for dataclass attributes
- Docstring types must match annotations (`Mapping` not "Dict" when annotation is
  `Mapping`)
