# Validity Report: `gyeol_bench/retraction_dag` (`0.1.0`)

- **Registry Standard**: `UK AISI Inspect Evals` (`2026-05-08`)
- **Governance Tier**: `[주의]` (Pre-launch staged template; populated upon Stage 1 `N = 800` completion)

## 1. Construct Validity

`gyeol_bench/retraction_dag` isolates multi-turn non-destructive premise retraction (`fix->`, `err->`, `sup->`) coupled with SI unit (`ug`, `mg`, `g`, `kg`, `mm`, `cm`, `m`, `km`, `mL`, `L`, `ms`, `s`, `min`, `hr`) and financial scale (`USD`, `K_USD`, `M_USD`, `B_USD`, `%`, `bps`) re-propagation.

## 2. Ground-Truth Oracle Soundness

Ground truth is computed exclusively via Python `fractions.Fraction` exact rational arithmetic in `gyeol_bench/oracle.py`, eliminating IEEE-754 intermediate rounding drift. Candidate answers are scored after dimensional alignment under `abs_tol = 1e-4` and `rel_tol = 1e-5`.

## 3. Pre-Registered Statistical Thresholds

- **Sample Balance**: `N = 800` (`200` retraction + `200` unit/scale + `400` unperturbed control).
- **Paired Superiority (`K2`)**: `(b - c) / 800 >= +0.05` (`b - c >= 40`), `b + c < 416`, two-sided exact `McNemar p < 0.05`.
- **Human Audit Bound**: `0` oracle errors in `N >= 150` stratified audit yields `1.9773%` one-sided 95% upper bound.
- **Control Specificity (`K4`)**: `<= 5 / 400` false rejections (`2.8928%` two-sided 95% Clopper-Pearson upper bound `< 5%`).
