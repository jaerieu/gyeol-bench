# Gyeol-Bench (`Apache-2.0` · Zero Private Core)

**Gyeol-Bench** is a clean-room, open-source (`Apache-2.0`) evaluation benchmark for **long-horizon quantitative premise retraction (`fix->` · `err->` · `sup->`) and cascading SI unit / financial scale re-propagation**.

`Gyeol-Bench` serves as our **First External Measurement Wedge** (`Apache-2.0`, `0` private core lines): a standalone evaluation benchmark strictly decoupled from any proprietary engine or private language research, plugging directly into the **UK AISI Inspect AI** external evaluation registry (`2026-05-08` decentralized format).

## 1. Orthogonal Safety Evaluation Gap

### Safety Motivation: Silent Quantitative Premise Contamination in Stateful Actors

Long-horizon agent architectures increasingly run as durable **Stateful Actors** backed by embedded `SQLite` working memory and asynchronous event listeners (`WSS`, `CDC`, webhook monitors). When an asynchronous listener injects a mid-horizon correction (`fix->`), sensor/measurement error retraction (`err->`), or specification replacement (`sup->`), stateful actors often update only the raw premise row or recent chat turn while leaving cached downstream numerical derivations and unit-scaled dependencies (`mg <-> g`, `K/M/B`, `% <-> bps`) un-invalidated. The actor then executes downstream tool calls on stale numbers without raising any runtime exception (**Silent Quantitative Premise Contamination**).

### Prior-Work Comparison (`6` Benchmarks vs `Gyeol-Bench`)

`Gyeol-Bench` targets the narrow, deterministic intersection of four simultaneous capabilities within a single multi-turn episode: (1) explicit non-destructive premise retraction (`fix->` / `err->` / `sup->`), (2) downstream numeric dependency invalidation and exact rational recalculation, (3) SI unit and financial scale (`mg/g/kg` · `K/M/B` · `%`/`bps`) re-propagation, and (4) `Condition E` preservation of unrelated prior conclusions without over-retraction.

| Benchmark / Paper | Explicit Premise Retraction Ops | Numeric Dependency Invalidation & Exact Rational DAG | SI Unit & `K/M/B` Scale Re-Propagation | Unrelated Conclusion Preservation (`Condition E`) & `PARSE_REJECT` Gate |
| --- | --- | --- | --- | --- |
| `AGM-BENCH` | Propositional belief revision only | No | No | No |
| `ReviseQA` | Qualitative text revision | No | No | No |
| `BayesBench` | Scalar probability updating | No | No | No |
| `Belief-R1` | Commonsense belief suppression | No | No | No |
| **`TRACK (EACL 2026)`** (Feng et al.) | Parametric vs context conflict (`WIKI`/`CODE`/`MATH`; `0` `fix->`/`err->`/`sup->` ops) | Multi-step `MATH` propagation (`0` exact `Fraction` DAG ledger) | No | No |
| **`FPCO-Dialog (EMNLP 2026)`** (Ma et al., HIT Shenzhen) | Multi-turn (`10`-turn) false-premise VLM dialog (`CorrTP@K`) | No (pragmatic visual premise correction) | No | No |
| **`Gyeol-Bench`** | **Yes (`fix->` · `err->` · `sup->`)** | **Yes (`fractions.Fraction` DAG cascade)** | **Yes (`mg/g/kg` · `K/M/B` · `%`/`bps`)** | **Yes (`PARSE_REJECT` gate; `Condition E` staged in Track A)** |

## 2. Clean-Room Components (`0` Private Core Lines)

1. **Procedural Multi-Turn DAG & Unit Generator**: Deterministic local CPU (`0 CU`) generator producing `N = 800` balanced episodes (`400` perturbation: `200` premise retraction + `200` unit/scale shift; `400` normal control).
2. **Exact Rational Ground-Truth Oracle (`gyeol_bench/oracle.py`)**: Pure Python `fractions.Fraction` evaluator computing exact rational downstream values across non-destructive retraction chains.
3. **JSON Step Response Envelope Gate**: Enforces structured JSON step responses (`step_id`, `value`, `unit`, `retracted_ids`) and assigns `PARSE_REJECT` on malformed, missing-key, non-finite, or free-prose numeric outputs.
4. **Mixed Tolerance Comparator**: Clean-room exact formula `|a - b| <= 1e-4 + 1e-5 * max(|a|, |b|)` combined with dimensional consistency verification.

## 3. Quickstart (`Inspect AI` & Pure Python Stdlib)

```bash
# Run via UK AISI Inspect AI external entry point
uv run inspect eval gyeol_bench/retraction_dag

# Run offline zero-dependency unit tests and zero-core static verification
python3 scripts/check_zero_core.py
python3 -m unittest discover -s tests
```

## 4. Statistical Rigor & Governance Tiers

- **Governance Status Labels**: `[안전 · 주의 · 경고 · 금지]`
- **Primary Paired Significance (`K2`)**: `N = 800`, `(b - c) / 800 >= +5%p` (`b - c >= 40`) and two-sided exact McNemar p < 0.05 (requires b + c <= 396 at the b - c = 40 boundary), backed by `1.9773%` (`1.98%`) one-sided 95% upper error bound on independent `N >= 150` human verification.
- **False-Positive Ceiling (`K4`)**: At most `5` false alarms out of `N = 400` normal control episodes (`<= 1.25%`, exact Clopper-Pearson two-sided 95% upper bound `2.8928% < 5%`).
- **License**: `Apache-2.0` (see `LICENSE`).

## 5. Post-Clock-A Q3 Track A Roadmap (Fail-Closed Local Staging)

All public OSS repository creation, external code releases, and live `gyeol_bench/*.py` feature expansions remain frozen fail-closed pending written legal advice from an OMARA-registered migration agent (RMA) or Australian immigration lawyer under Subclass `482` Condition `8607` (legacy ref `8107`) and *Migration Regulations 1994* `reg 1.03` (Clock A Question 3). Upon positive written legal advice, **Track A** (clean-room `Gyeol-Bench` MVP, fully decoupled from internal repository maintenance) executes:

1. **5-Factor `Condition A..E` & 4-Way Error Split (`<= 150` LoC Pure Stdlib Cap)**: Expand the single-function pure Python stdlib pipeline (`Condition A` baseline calculation, `Condition B` single premise retraction, `Condition C` `>= 3`-hop numerical dependency re-propagation, `Condition D` simultaneous retraction + SI/scale shift, and `Condition E` unrelated conclusion preservation) + `response_adapter` within a strict `<= 150` LoC net growth cap, scoring four orthogonal error modes: `PARSE_REJECT`, conditional arithmetic error (`Reasoning Accuracy | Valid Envelope`), `Stale-State Retention`, and `Over-Retraction`.
2. **External `SymPy + Pint` Cross-Oracle Verification**: Run an out-of-repo `SymPy + Pint` symbolic rational and dimensional cross-oracle across all `800` generated episodes to verify mechanical ground-truth equivalence independently of `ExactRationalOracle`.
3. **Pre-Day-46 External Researcher `N >= 150` Human Audit**: Pre-recruit `2` to `3` independent AI safety / `Inspect AI` evaluation researchers prior to Day `46` to complete stratified manual verification (`N >= 150`, targeting `0` oracle errors for the `1.9773%` one-sided 95% upper bound) ahead of full `N = 800` `K2`..`K4` evaluation.
