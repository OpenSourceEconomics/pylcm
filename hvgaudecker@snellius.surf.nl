---
# Conda/mamba environment for the skillmodels-applications workspace.
#
# Target: Linux x86-64 with NVIDIA GPUs. Covers both Snellius and the reserved
# OSSC node (same OS and module stack).
#
#   module load 2024 && module load 2025
#   module load Mamba/24.9.0-0
#   module load CUDA/12.9.1
#   mamba env create -p "$PROJECT/envs/skillmodels-applications" \
#       -f environment-linux-cuda12.yml
#
# This file is Linux-only -- it will not solve on win-64 (no alsa-lib, libgbm,
# nss, memray, ...). The CBS/Windows route is pip-based: see
# CLUSTER-ENVIRONMENTS.md.
#
# Contents mirror the union of the pixi environments of the three sub-projects
# (skillmodels, health-cognition, skane-struct-bw), minus the two application
# packages themselves -- those are run from a source checkout, not installed.
# See CLUSTER-ENVIRONMENTS.md for the full install recipes, the offline
# (OSSC) transfer, and the CBS Windows variant.
#
# Two rules that this file cannot enforce, and that broke the earlier OSSC
# attempts -- read CLUSTER-ENVIRONMENTS.md before creating this environment:
#
#   1. Create it with `-p` at its FINAL path. Conda environments are not
#      relocatable; copying one afterwards leaves every console script
#      (ipython, pytask, pytest, jupyter) with a shebang pointing at the old
#      path. To get an environment onto a machine with no internet, use
#      conda-pack, which rewrites those paths on unpack.
#   2. Do not create a venv inside this environment. Pick one or the other.
#
name: skillmodels-applications
channels:
  - conda-forge
  - nodefaults
dependencies:
  # The interpreter must be pinned here. Without it conda is free to pick an
  # older Python (it chose 3.13.5 on OSSC), and skillmodels requires
  # >=3.14,<3.15 -- hence "Package 'skillmodels' requires a different Python".
  - python ~=3.14.0
  - pip
  # Core numerical stack.
  # numpy is capped below 2.5 because numba (pulled in by tranquilo) requires
  # numpy<2.5. Without the cap a pip resolver silently backtracks numba to
  # 0.53.1 (2021), which has no cp314 wheels. 2.4.x is what the pixi lock uses.
  - numpy >=2.4,<2.5
  - pandas >=3
  - scipy >=1.16.0
  - scikit-learn >=1.5
  - h5py >=3.16,<4
  - statsmodels >=0.14.5
  - networkx *
  - filterpy *
  # CUDA toolchain for the pip-installed jax[cuda12] below. The upper bound
  # matters: conda-forge otherwise resolves nvcc 13, which does not match the
  # nvidia-*-cu12 wheels that jax[cuda12] pulls in. Pixi avoids this via its
  # `linux-64-cuda12` platform; plain conda has no such constraint.
  # Matches `module load CUDA/12.9.1` on Snellius.
  - cuda-nvcc >=12,<13
  # Estimation stack
  - beartype >=0.22
  - dags >=0.5.1
  - jaxtyping *
  - optimagic >=0.5.4
  - pybaum >=0.1.3
  # Optimizers reachable from optimagic
  - dfo-ls *
  - fides >=0.7.8
  - tranquilo >=0.0.4
  # Workflow
  - pytask >=0.5.8
  - pytask-parallel >=0.5.2
  # Plotting and reporting
  - plotly >=6.6
  - python-kaleido >=1.2
  - seaborn *
  - matplotlib-base *
  - pygraphviz *
  - tabulate >=0.9.0
  - deepdiff >=8.5.0
  # Notebooks and Jupyter Book. `mystmd` supplies the `myst` Node CLI that
  # jupyter-book 2 shells out to; without it a docs build tries to download
  # Node at runtime, which fails on a machine with no internet.
  - ipykernel >=6.29.5
  - jupyterlab *
  - jupyter-book >=2.0
  - mystmd *
  - nbformat >=5.10.4
  # Shared libraries that the headless Chrome behind kaleido 1.x needs for
  # static figure export. Chrome is not a conda binary, so conda only sets
  # RPATH for its own libraries -- see the LD_LIBRARY_PATH note in
  # CLUSTER-ENVIRONMENTS.md, which has to be set by hand here (an
  # environment-linux-cuda12.yml has no equivalent of pixi's activation.env).
  # Taken from health-cognition @ add-tests-cuda12.
  - alsa-lib >=1.2.16.1,<2
  - at-spi2-atk >=2.38.0,<3
  - cairo >=1.18.4,<2
  - libcups >=2.3.3,<3
  - libgbm >=1.0.7,<2
  - libxkbcommon >=1.13.2,<2
  - nss >=3.118,<4
  - pango >=1.56.4,<2
  - xorg-libxcomposite >=0.4.7,<0.5
  - xorg-libxdamage >=1.1.6,<2
  - xorg-libxfixes >=6.0.2,<7
  - xorg-libxrandr >=1.5.5,<2
  # Data I/O
  - xlrd >=2
  # Type checking
  - ty *
  - pandas-stubs *
  - types-pyyaml *
  - types-pytz *
  # Test and dev tooling
  - pytest >=8.4.1
  - pytest-cov >=6.2.1
  - pytest-xdist >=3.8.0
  - pytest-memray *
  - memray >=1.17.2
  - snakeviz *
  - prek *
  - pip:
      # jax with the CUDA 12 plugin: conda-forge lags PyPI and its CUDA builds
      # are awkward, so this comes from PyPI exactly as in the pixi lock.
      # On a login node without a GPU, run smoke tests with JAX_PLATFORMS=cpu.
      - jax[cuda12]>=0.9
      # Not packaged on conda-forge.
      - skillmodels>=0.1.2
      - statadict>=1.1.0
      - pdbp
