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

## The compiled kernel, and installing without a C++ compiler

The certified upper envelopes use pylcm's *exact-affine kernel*: a small native payload
built from source maintained in this repository. It is part of the pylcm installation,
not an arbitrary shared library downloaded or discovered at runtime.

How the payload arrives depends on how pylcm is installed:

- a compatible binary wheel already contains the CPU and, where that wheel was built
  with CUDA support, CUDA libraries for its platform and toolchain;
- a source or editable install runs pylcm's build hook locally. It requires `c++` or
  `g++` on Linux/macOS, an activated MSVC `cl.exe` or `clang-cl` environment on Windows,
  or an appropriate `CXX`; it adds a CUDA library only when `nvcc` is available at build
  time.

A CPU library does not satisfy a solve whose current JAX backend is a GPU. Likewise, a
payload built for another platform, Python ABI, toolchain, or JAX installation is not a
portable substitute. If no compatible wheel exists, the package manager performs the
source build and therefore needs the local toolchain.

If the required compiler is missing, a source install fails with a message naming it. To
install without the certified capability on purpose, set the explicit build flag:

```bash
LCM_SKIP_EXACT_AFFINE=1 pip install pylcm
```

The build then compiles no exact-affine payload and reports the omitted capability. Any
other value, including `0`, requests the normal build. A skipped installation can use
`GridSearch`, a typed approximate DCEGM envelope, or
`NBEGM(envelope_arithmetic="ordinary")`—including as the inner solver of `NNBEGM`—under
that mode's approximation contract.

Defaults that request certified arithmetic require a compatible payload and never
silently fall back: DCEGM's `ExactEnvelope` checks the active backend during
`Model(...)`, while NBEGM's default `envelope_arithmetic="certified"` requests the
payload at its first certified envelope evaluation, normally during solve or tracing.
Either path raises `ExactAffineKernelUnavailableError` if the payload is absent or
unloadable. The same NBEGM requirement applies when it is the inner solver of `NNBEGM`.

To restore the capability after changing the toolchain or native sources, reinstall
pylcm in the target environment; for a pixi development checkout use:

```bash
pixi reinstall pylcm
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

pylcm sets three JAX configuration defaults on import:

- **`XLA_PYTHON_CLIENT_PREALLOCATE=false`** — disables JAX's default of reserving 75% of
  GPU memory upfront. This lets `nvidia-smi` reflect actual usage and plays nicely with
  other GPU processes.
- **`JAX_COMPILATION_CACHE_DIR=~/.cache/jax`** — enables persistent JIT compilation
  caching. Large models (many regimes and states) can take minutes to compile on first
  run; the cache makes subsequent runs near-instant.
- **`JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`** — writes every compiled program to
  the persistent cache. JAX's default only caches programs that take longer than a
  second to compile, which excludes most of the many small programs a pylcm model
  compiles — leaving the cache empty and every fresh process recompiling everything.

All three only apply if you have not already set the variable yourself.

### Import order does not matter

JAX reads its environment variables once, while it defines its configuration — so a
value exported after `import jax` never reaches it. pylcm therefore applies both
compilation-cache settings through `jax.config` as well, and they hold whether `lcm` is
imported before or after `jax`. This matters in practice: test suites, notebooks, and
other libraries routinely import `jax` first, and a cache that is switched off reports
nothing at all — it simply recompiles, which on a large model costs minutes per process.

`XLA_PYTHON_CLIENT_PREALLOCATE` is read by XLA when the backend is first initialised
rather than at import, so it is enough to import `lcm` before running any computation.

To confirm caching is live in a given process:

```python
import jax

import lcm

print(jax.config.jax_compilation_cache_dir)  # a path
print(jax.config.jax_persistent_cache_min_compile_time_secs)  # 0.0
```

A directory of `None` means the cache is off and every fresh process is recompiling the
whole model.

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
- **`No C++ compiler found` during install** (`No MSVC C++ compiler found` on Windows):
  install one — on Windows, activate an MSVC developer environment so `cl.exe` is on the
  path — or install without the certified upper envelope using `LCM_SKIP_EXACT_AFFINE=1`
  (see above), accepting that DCEGM and NBEGM will not run on their certified defaults.
- **An `ExactEnvelope` availability error during `Model(...)`**: the install skipped the
  kernel, or carries one built by a different toolchain. Rebuild in the current
  environment with `pixi reinstall pylcm`, or explicitly select another typed envelope
  under its approximation contract.
- **An `ExactAffineKernelUnavailableError` during an NBEGM solve**: certified NBEGM has
  reached its first exact-affine verdict without a compatible payload for the active JAX
  backend. Reinstall pylcm in that environment, or select
  `NBEGM(envelope_arithmetic="ordinary")` only after validating its working-format
  ownership near model-specific crossings.
- **JAX GPU not detected**: Ensure the CUDA toolkit (Linux) or jax-metal (macOS) is
  properly installed. See the
  [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).
