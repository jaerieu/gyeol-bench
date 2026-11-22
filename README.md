# Gyeol-Bench (`Apache-2.0` · Zero Private Core)

**Gyeol-Bench** is a clean-room, open-source (`Apache-2.0`) evaluation benchmark for **long-horizon quantitative premise retraction (`fix->` · `err->` · `sup->`) and cascading SI unit / financial scale re-propagation**.

It plugs directly into the **UK AISI Inspect AI** external evaluation registry (`2026-05-08` decentralized format) while maintaining **zero lines (`0` lines)** of private engine code.

## 1. Orthogonal Safety Evaluation Gap

| Benchmark / Tool | Multi-Turn Premise Retraction | Exact Rational DAG Cascade | SI Unit & `K/M/B` Scale Conversion | Strict JSON `PARSE_REJECT` Gate |
| --- | --- | --- | --- | --- |
| `AGM-BENCH` | Propositional only | No | No | No |
| `ReviseQA` | Qualitative text | No | No | No |
| `BayesBench` | Scalar probability | No | No | No |
| `Belief-R1` | Commonsense suppression | No | No | No |
| **`Gyeol-Bench`** | **Yes (`fix->` · `err->` · `sup->`)** | **Yes (`fractions.Fraction`)** | **Yes (`mg/g/kg` · `K/M/B`)** | **Yes (`PARSE_REJECT`)** |

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
- **Primary Paired Significance (`K2`)**: `N = 800`, `(b - c) / 800 >= +5%p` (`b - c >= 40`) and two-sided exact `McNemar p < 0.05` (`b + c < 416` discordant-pair power condition), backed by `1.9773%` (`1.98%`) one-sided 95% upper error bound on human verification.
- **False-Positive Ceiling (`K4`)**: At most `5` false alarms out of `N = 400` normal control episodes (`<= 1.25%`, exact Clopper-Pearson two-sided 95% upper bound `2.8928% < 5%`).
- **License**: `Apache-2.0` (see `LICENSE`).
