---
title: Development
---

# Development

How to set up a development environment, run tests, and contribute to pylcm.

## Setup

pylcm uses [pixi](https://pixi.sh/) for dependency management. Python 3.14+ is
required.

```bash
# Clone the repository
git clone https://github.com/OpenSourceEconomics/pylcm.git
cd pylcm
```

Install pre-commit hooks (requires [prek](https://github.com/hmgaudecker/prek)):

```bash
pixi global install prek
prek install
```

## Running Tests

```bash
# All tests
pixi run tests

# Tests with coverage
pixi run tests-with-cov

# Specific test file
pixi run tests tests/test_specific_module.py

# Specific test function
pixi run tests tests/test_specific_module.py::test_function_name
```

## Code Quality

```bash
# Type checking (ty, not mypy)
pixi run -e tests-cpu ty

# Run all pre-commit hooks
prek run --all-files
```

## Building Documentation

```bash
# Build docs
pixi run build-docs

# Live preview
pixi run view-docs
```

## Conventions

### Code Style

- **Ruff** for linting and formatting (configured in `pyproject.toml`)
- All functions require type annotations
- Google-style docstrings in imperative mood ("Return", not "Returns")
- Use `# ty: ignore[error-code]` for type suppression (never `# type: ignore`)
- Never use `from __future__ import annotations`

### Testing

- Plain `pytest` functions — no test classes
- Use `@pytest.mark.parametrize` for test variations

### Naming

- `func` for callable abbreviations (never `fn`)
- `state_names` / `action_names` (not `states_names`)
- `arg_names` (not `argument_names`)

### Plotting

- Always use **plotly** (`plotly.graph_objects`, `plotly.subplots.make_subplots`), never
  matplotlib

### Docstrings

- MyST syntax (single backticks for code, `$...$` for math), not reStructuredText
- Inline field docstrings (PEP 257) for dataclass attributes
- Docstring types must match annotations (`Mapping` not "Dict" when annotation is
  `Mapping`)

## Contributing

Report bugs and suggest features on the
[GitHub issue tracker](https://github.com/OpenSourceEconomics/pylcm/issues).
