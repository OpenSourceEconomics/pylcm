# Life Cycle Models

[![PyPI version](https://img.shields.io/pypi/v/pylcm)](https://pypi.org/project/pylcm/)
[![conda-forge version](https://img.shields.io/conda/vn/conda-forge/pylcm)](https://anaconda.org/conda-forge/pylcm)
[![License](https://img.shields.io/github/license/OpenSourceEconomics/pylcm)](https://github.com/OpenSourceEconomics/pylcm/blob/main/LICENSE)
[![Documentation](https://readthedocs.org/projects/pylcm/badge/?version=latest)](https://pylcm.readthedocs.io/en/latest/)
[![CPU tests](https://github.com/OpenSourceEconomics/pylcm/actions/workflows/cpu.yml/badge.svg?branch=main)](https://github.com/OpenSourceEconomics/pylcm/actions/workflows/cpu.yml)
[![GPU tests](https://github.com/OpenSourceEconomics/pylcm/actions/workflows/gpu32.yml/badge.svg?branch=main)](https://github.com/OpenSourceEconomics/pylcm/actions/workflows/gpu32.yml)
[![Notebooks](https://github.com/OpenSourceEconomics/pylcm/actions/workflows/notebooks.yml/badge.svg?branch=main)](https://github.com/OpenSourceEconomics/pylcm/actions/workflows/notebooks.yml)
[![ty](https://github.com/OpenSourceEconomics/pylcm/actions/workflows/ty.yml/badge.svg?branch=main)](https://github.com/OpenSourceEconomics/pylcm/actions/workflows/ty.yml)
[![Benchmarks](https://img.shields.io/badge/benchmarked%20by-asv-blue)](https://open-econ.org/pylcm-benchmarks/)
[![codecov](https://codecov.io/gh/OpenSourceEconomics/pylcm/branch/main/graph/badge.svg)](https://codecov.io/gh/OpenSourceEconomics/pylcm)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/OpenSourceEconomics/pylcm/main.svg)](https://results.pre-commit.ci/latest/github/OpenSourceEconomics/pylcm/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This package aims to generalize and facilitate the specification, solution, and
simulation of finite-horizon discrete-continuous dynamic choice models.

## Installation

PyLCM can be installed via PyPI or via GitHub. To do so, type the following in a
terminal (or install via uv):

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
> GPU support is currently only tested on Linux with CUDA 12 and 13.

## Developing

We use [pixi](https://pixi.sh/latest/) for our local development environment. If you
want to work with or extend the PyLCM code base you can run the tests using

```console
$ git clone https://github.com/OpenSourceEconomics/pylcm.git
$ pixi run tests
```

This will install the development environment and run the tests. You can run
[ty](https://docs.astral.sh/ty) using

```console
$ prek run ty --all-files
```

Before committing, install the pre-commit hooks using

```console
$ pixi global install prek
$ prek install
```

## Questions

If you have any questions, feel free to ask them on the PyLCM
[Zulip chat](https://ose.zulipchat.com/#narrow/channel/491562-PyLCM).

## Acknowledgments

PyLCM builds on the endogenous grid method (Carroll, 2006), its discrete-continuous
extension (Iskhakov, Jørgensen, Rust & Schjerning, 2017), the Fast Upper-Envelope Scan
(Dobrescu & Shanker, 2022), and the broader open-source ecosystem for dynamic
programming — including OpenSourceEconomics, NumEconCopenhagen, and QuantEcon. See the
[Credits & Acknowledgments](docs/credits.md) page for the full list of methods,
replicated models, and software we are grateful to.

## License

This project is licensed under the Apache License, Version 2.0. See the
[LICENSE](LICENSE) file for details.

Copyright (c) 2023- The PyLCM Authors
