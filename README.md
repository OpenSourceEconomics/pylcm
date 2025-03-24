# Life Cycle Models

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/opensourceeconomics/pylcm/main.svg)](https://results.pre-commit.ci/latest/github/opensourceeconomics/pylcm/main)
[![image](https://codecov.io/gh/opensourceeconomics/pylcm/branch/main/graph/badge.svg)](https://codecov.io/gh/opensourceeconomics/pylcm)

This package aims to generalize and facilitate the specification, solution, and
simulation of finite-horizon discrete-continuous dynamic choice models.

## Installation

PyLCM can be installed via PyPI or via GitHub. To do so, type the following in a
terminal:

```console
$ pip install pylcm
```

or, for the latest development version, type:

```console
$ pip install git+https://github.com/OpenSourceEconomics/pylcm.git
```

### GPU Support

By default, the installation of PyLCM comes with the CPU version of `jax`. If you aim to
run PyLCM on a GPU, you need to install a `jaxlib` version with GPU support. For the
installation of `jaxlib`, please consult the `jax`
[docs](https://jax.readthedocs.io/en/latest/installation.html#supported-platforms).

> [!NOTE]
> GPU support is currently only tested on Linux with CUDA 12.

## Developing

We use [pixi](https://pixi.sh/latest/) for our local development environment. If you
want to work with or extend the PyLCM code base you can run the tests using

```console
$ git clone https://github.com/OpenSourceEconomics/pylcm.git
$ pixi run tests
```

This will install the development environment and run the tests. You can run
[mypy](https://mypy-lang.org/) using

```console
$ pixi run mypy
```

Before committing, install the pre-commit hooks using

```console
$ pixi global install pre-commit
$ pre-commit install
```

## Questions

If you have any questions, feel free to ask them on the PyLCM
[Zulip chat](https://ose.zulipchat.com/#narrow/channel/491562-PyLCM).

## License

This project is licensed under the Apache License, Version 2.0. See the
[LICENSE](LICENSE) file for details.

Copyright (c) 2023- The PyLCM Authors
