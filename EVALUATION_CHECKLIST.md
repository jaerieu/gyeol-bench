# Inspect Evals Decentralized Registry Checklist (`2026-05-08` Format)

- **Package**: `gyeol-bench` (`0.1.0`, `Apache-2.0`)
- **Entry Point**: `[project.entry-points."inspect_ai"]` -> `gyeol_bench = "gyeol_bench._registry"`
- **Task Identifier**: `gyeol_bench/retraction_dag`
- **Registry Policy Target**: `2026-05-08` decentralized external package layout

## 1. Required Artefact Layout

| Path | Purpose | Status |
| --- | --- | --- |
| `pyproject.toml` | Declares `inspect_ai >= 0.3.0` and `inspect_ai` entry point | `[안전]` Verified |
| `eval.yaml` | Structured task metadata, sample counts (`N = 800`), tolerance parameters | `[안전]` Verified |
| `EVALUATION_CHECKLIST.md` | Pre-submission reproducibility, contamination, and statistical audit checklist | `[안전]` Verified |
| `agent_artefacts/retraction_dag_0.1.0/validity/VALIDITY_REPORT.md` | Construct validity, `Fraction` oracle soundness, `McNemar` / `Clopper-Pearson` bounds | `[주의]` Pre-launch staged |
| `agent_artefacts/retraction_dag_0.1.0/review/SUMMARY.md` | Independent code & zero-core boundary review record | `[주의]` Pre-launch staged |

## 2. Technical & Governance Gates

- [x] **Zero Private Core Boundary**: Static AST + regex check (`python3 scripts/check_zero_core.py`) confirms `0` private engine imports or symbols and `0` visual-spectrum terms.
- [x] **Offline Determinism (`0 CU`)**: `ExactRationalOracle` uses pure Python `fractions.Fraction` with deterministic seed generation and zero live API calls during unit testing.
- [x] **Strict Envelope Hygiene**: Non-JSON prose, missing fields, booleans, and non-finite floats deterministically trigger `PARSE_REJECT`.
- [x] **Mixed Tolerance Equivalence**: `|candidate - truth| <= 1e-4 + 1e-5 * max(|candidate|, |truth|)` enforced after exact base-unit conversion.
- [ ] **Stage 1 Live Model Matrix (`K2` · `K3` · `K4`)**: Frozen baseline hash publication at least 14 days prior to Day 60 (`b - c >= 40`, `b + c < 416`, `McNemar p < 0.05`; normal control false-positive `<= 5 / 400`).
