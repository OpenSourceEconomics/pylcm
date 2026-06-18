# Credits & Acknowledgments

pylcm stands on a large body of methodological and applied research, and on a vibrant
open-source ecosystem for solving discrete-continuous dynamic models. This page credits
the methods, models, and software pylcm draws on. We are grateful to all of these
authors and maintainers.

## Authors & maintainers

pylcm is developed under the
[OpenSourceEconomics](https://github.com/OpenSourceEconomics) organization by Tim
Mensinger, Maximilian Jahn, Janos Gabler, and Hans-Martin von Gaudecker.

## Numerical methods

### The endogenous grid method (EGM)

The endogenous grid method — inverting the Euler equation on an exogenous end-of-period
grid instead of root-finding on a dense grid — originates with Carroll (2006), "The
Method of Endogenous Gridpoints for Solving Dynamic Stochastic Optimization Problems,"
*Economics Letters* 91(3), 312–320,
[doi:10.1016/j.econlet.2005.09.013](https://doi.org/10.1016/j.econlet.2005.09.013).

### Discrete-continuous EGM (DC-EGM)

pylcm's `DCEGM` solver implements the discrete-continuous endogenous grid method of
Iskhakov, Jørgensen, Rust & Schjerning (2017), "The endogenous grid method for
discrete-continuous dynamic choice models with (or without) taste shocks," *Quantitative
Economics* 8(2), 317–365, [doi:10.3982/QE643](https://doi.org/10.3982/QE643). The
deterministic retirement model from that paper is shipped as a closed-form test oracle
(see below).

- Original MATLAB implementation:
  [fediskhakov/dcegm](https://github.com/fediskhakov/dcegm).
- A fully JAX-compatible Python implementation in the same organization, whose design
  pylcm draws on:
  [OpenSourceEconomics/dcegm](https://github.com/OpenSourceEconomics/dcegm).

### Upper-envelope refinement

pylcm refines the (non-monotone) EGM candidate correspondence into the upper envelope
with the Fast Upper-Envelope Scan (FUES) of Dobrescu & Shanker (2022), "Fast
Upper-Envelope Scan for Discrete-Continuous Dynamic Programming," SSRN working paper
[4181302](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4181302). The treatment of
non-smooth, non-concave value functions traces to Fella (2014), "A Generalized
Endogenous Grid Method for Non-Smooth and Non-Concave Problems," *Review of Economic
Dynamics* 17(2), 329–344,
[doi:10.1016/j.red.2013.07.001](https://doi.org/10.1016/j.red.2013.07.001).

- Reference implementations: [akshayshanker/FUES](https://github.com/akshayshanker/FUES)
  and
  [OpenSourceEconomics/upper-envelope](https://github.com/OpenSourceEconomics/upper-envelope).

### Multidimensional & nested EGM (reserved / planned backends)

pylcm reserves an upper-envelope backend slot for multidimensional envelopes and is
designed to grow toward solving models with more than one continuous (Euler) state. That
roadmap builds on:

- Druedahl (2021), "A Guide on Solving Non-convex Consumption-Saving Models,"
  *Computational Economics* 58(3), 747–775,
  [doi:10.1007/s10614-020-10045-x](https://doi.org/10.1007/s10614-020-10045-x) — nested
  EGM (NEGM). Code:
  [NumEconCopenhagen/ConsumptionSaving](https://github.com/NumEconCopenhagen/ConsumptionSaving).
- Druedahl & Jørgensen (2017), "A General Endogenous Grid Method for Multi-Dimensional
  Models with Non-Convexities and Constraints," *Journal of Economic Dynamics and
  Control* 74, 87–107,
  [doi:10.1016/j.jedc.2016.11.005](https://doi.org/10.1016/j.jedc.2016.11.005) — G2EGM.
  Code: [JeppeDruedahl/G2EGM](https://github.com/JeppeDruedahl/G2EGM).
- Dobrescu & Shanker (2024), "Using Inverse Euler Equations to Solve Multidimensional
  Discrete-Continuous Dynamic Models: A General Method," SSRN working paper
  [4850746](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4850746) — the
  inverse-Euler / Rooftop-Cut (RFC) multidimensional envelope. Code:
  [akshayshanker/InverseDCDP](https://github.com/akshayshanker/InverseDCDP).

## Stochastic-process discretization

pylcm's AR(1) / Markov discretization follows Tauchen (1986), Rouwenhorst (1995), and
Kopecky & Suen (2010), with the Tauchen implementation following
[QuantEcon](https://quanteconpy.readthedocs.io/). The choice of Rouwenhorst as the
default for non-stationary lifecycle processes follows Fella, Gallipoli & Pan (2019),
"Markov-Chain Approximations for Life-Cycle Models," *Review of Economic Dynamics* 34,
183–201, [doi:10.1016/j.red.2019.03.013](https://doi.org/10.1016/j.red.2019.03.013),
against whose results pylcm's discretization accuracy is validated. Code & data:
[RePEc red/ccodes/17-149](https://ideas.repec.org/c/red/ccodes/17-149.html).

## Replicated example models

Models shipped with pylcm (`src/lcm_examples/`) that replicate published work:

- **Iskhakov, Jørgensen, Rust & Schjerning (2017)** — the deterministic retirement model
  (`lcm_examples.iskhakov_et_al_2017`), used as a closed-form test oracle.
- **Mahler & Yum (2024)**, "Lifestyle Behaviors and Wealth-Health Gaps in Germany,"
  *Econometrica* 92(5), 1697–1733,
  [doi:10.3982/ECTA20603](https://doi.org/10.3982/ECTA20603) — a finite-horizon
  discrete-continuous lifecycle model (`lcm_examples.mahler_yum_2024`). The replication
  package is available via the Econometrica supplemental materials.

## Pedagogy

pylcm's design and documentation are informed by Sargent & Stachurski's
[*Dynamic Programming*](https://dp.quantecon.org/) (QuantEcon, 2024).

## The open-source ecosystem

pylcm interoperates with, learns from, and is grateful to the broader ecosystem for
solving discrete-continuous dynamic models:

- [OpenSourceEconomics](https://github.com/OpenSourceEconomics) — `dcegm`,
  `upper-envelope`.
- [NumEconCopenhagen](https://github.com/NumEconCopenhagen) — `ConsumptionSaving`
  (ConSav), `EconModel`.
- [akshayshanker](https://github.com/akshayshanker) — `FUES`, `InverseDCDP`.
- [JeppeDruedahl](https://github.com/JeppeDruedahl) — `G2EGM`.
- [fediskhakov](https://github.com/fediskhakov) — `dcegm` (original DC-EGM).
- [QuantEcon](https://quantecon.org/) — dynamic-programming pedagogy and reference code.
