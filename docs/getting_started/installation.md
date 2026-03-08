---
title: Installation
---

# Installation

## Prerequisites

pylcm requires **Python 3.14+**. You can download the latest Python release from
[python.org](https://www.python.org/downloads/).

We recommend [pixi](https://pixi.sh/) for environment management, as it handles Python
version pinning and dependency resolution automatically.

## Install with pixi (recommended)

Add pylcm to your `pixi.toml`:

```toml
[dependencies]
pylcm = "*"
```

Then install:

```bash
pixi install
```

## Install with pip

```bash
pip install pylcm
```

:::{note}
pylcm depends on `dags`, which may be installed from GitHub. Ensure you have `git`
installed on your system.
:::

## GPU Acceleration (optional)

pylcm uses [JAX](https://jax.readthedocs.io/) for numerical computation. By default,
JAX runs on CPU. For GPU acceleration, install the appropriate JAX variant:

### Linux (CUDA)

```bash
pip install jax[cuda13]
```

For CUDA 12, use `jax[cuda12]` instead. When using pixi, the `cuda13` and `cuda12`
environments handle this automatically.

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)
for details on CUDA toolkit requirements.

### macOS (Metal)

```bash
pip install jax-metal
```

This requires Apple Silicon (M1 or later).

## Verify Installation

```python
import lcm
import jax

print(jax.devices())  # Should show GPU if configured
```

If GPU acceleration is set up correctly, you will see a `GpuDevice` or `MetalDevice` in
the output. Otherwise, you will see `CpuDevice`, which is fine for development and
smaller models.

## Troubleshooting

- **Python version too old**: pylcm requires Python 3.14+. Check with `python --version`.
- **JAX GPU not detected**: Ensure the CUDA toolkit (Linux) or jax-metal (macOS) is
  properly installed. See the
  [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).
- **Import errors from `dags`**: Ensure `git` is installed and accessible from your
  terminal, as `dags` may be fetched from GitHub during installation.
