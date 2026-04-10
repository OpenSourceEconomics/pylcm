---
title: Installation
---

# Installation

## Prerequisites

pylcm requires **Python 3.14+**. We recommend [pixi](https://pixi.sh/) or
[uv](https://docs.astral.sh/uv/) for environment management.

## Install with pixi

```bash
pixi add pylcm
```

## Install with uv

```bash
uv add pylcm
```

## Install from GitHub

If you require features not yet in a released version, install from GitHub:

```bash
# pixi
pixi add pylcm --pypi --git https://github.com/OpenSourceEconomics/pylcm.git --rev main

# uv
uv add pylcm --git https://github.com/OpenSourceEconomics/pylcm.git --rev main
```

## GPU Acceleration (optional, but then this is the whole point of it)

pylcm uses [JAX](https://jax.readthedocs.io/) for numerical computation. By default, JAX
runs on CPU. For GPU acceleration, install the appropriate JAX variant.

### Linux (CUDA)

If you use pixi, add a CUDA feature to your `pyproject.toml`:

```toml
[tool.pixi.feature.cuda13]
platforms = ["linux-64"]
system-requirements = {cuda = "13"}

[tool.pixi.feature.cuda13.target.linux-64.dependencies]
cuda-nvcc = "~=13.0"

[tool.pixi.feature.cuda13.target.linux-64.pypi-dependencies]
jax = {version = ">=0.8", extras = ["cuda13"]}
```

For CUDA 12, replace `cuda13` with `cuda12` throughout.

If you use uv:

```bash
uv add "jax[cuda13]"
```

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)
for details on CUDA toolkit requirements.

### macOS (Metal)

```bash
# pixi
pixi add jax-metal --pypi

# uv
uv add jax-metal
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

## JAX Settings

pylcm sets two JAX environment variables on import:

- **`XLA_PYTHON_CLIENT_PREALLOCATE=false`** — disables JAX's default of reserving 75% of
  GPU memory upfront. This lets `nvidia-smi` reflect actual usage and plays nicely with
  other GPU processes.
- **`JAX_COMPILATION_CACHE_DIR=~/.cache/jax`** — enables persistent JIT compilation
  caching. Large models (many regimes and states) can take minutes to compile on first
  run; the cache makes subsequent runs near-instant.

Both use `os.environ.setdefault`, so they only apply if the variable is not already set.

On HPC systems where the home directory is on a slow network filesystem, you may want to
point the compilation cache at a fast local disk. Set the environment variable before
importing pylcm:

```python
import os

os.environ["JAX_COMPILATION_CACHE_DIR"] = os.path.expandvars(
    "/scratch/$USER/.cache/jax"
)

import lcm
```

## Troubleshooting

- **Python version too old**: pylcm requires Python 3.14+. Check with
  `python --version`.
- **JAX GPU not detected**: Ensure the CUDA toolkit (Linux) or jax-metal (macOS) is
  properly installed. See the
  [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).
