---
title: Setup & Workflow
---

# Setup & Workflow

## Setup

pylcm uses [pixi](https://pixi.sh/) for dependency management. Python 3.14+ is
required.

```bash
# Clone the repository
git clone https://github.com/OpenSourceEconomics/pylcm.git
cd pylcm
```

The first `pixi run` command will install dependencies automatically, but you can run
`pixi install` explicitly if you prefer.

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
pixi run pytest tests/test_specific_module.py

# Specific test function
pixi run pytest tests/test_specific_module.py::test_function_name
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

# Live preview (watches for changes)
pixi run view-docs
```

## Contributing

Report bugs and suggest features on the
[GitHub issue tracker](https://github.com/OpenSourceEconomics/pylcm/issues).
