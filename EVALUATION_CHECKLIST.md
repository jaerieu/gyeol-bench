# Inspect Evals Decentralized Registry Checklist (`2026-05-08` Format · Capability-Stratified & 4-Tier Verifier Synthesis)

- **Package**: `gyeol-bench` (`0.1.0`, `Apache-2.0`, `0` private core lines)
- **Entry Point**: `[project.entry-points."inspect_ai"]` -> `gyeol_bench = "gyeol_bench._registry"`
- **Primary Task**: `gyeol_bench/retraction_dag_multiturn` (`v3.0` 2-live-turn `Shared-Turn-1` stateful handoff)
- **Ablation 0 Task**: `gyeol_bench/retraction_dag` (`v2.0` single-prompt control)
- **Explicit Scope Boundary (`DSL-Schema Scope`)**: Evaluates deterministic revision over structured tool/state-ledger DSL workflows (`Turn k.i [id]:` + `fix->`/`err->`/`sup->` + JSON envelopes), **NOT** unconstrained free-form natural-language multi-turn chat.
- **Cumulative Resource Ledger**: `AI Studio $2.909901 / $100.00 USD` (`<= $5.00` gate) · `Colab 118.85 / 3,251.14 CU` (`<= 162.56 CU` gate, `Active assignments: 0`).

## 1. Required Artefact Layout

| Path | Purpose | Status |
| --- | --- | --- |
| `pyproject.toml` | Declares `inspect_ai >= 0.3.0` and `inspect_ai` entry point | `[안전]` Verified |
| `eval.yaml` | Declares `retraction_dag_multiturn` (`N = 800`) & `retraction_dag` (`Ablation 0`) | `[안전]` Verified |
| `manifests/v3-multiturn-prepilot-lock.json` | Unmodified pre-pilot SHA-256 lock (`eac8d83e62ceb784f9096d5aab8540ee70a07f57d10950cb77fea57202502e53`) | `[안전]` Verified |
| `manifests/day46-baseline-freeze.json` | Untouched pre-Phase-3 `23`-leaf `RFC 6962` freeze (`a542492ed604bd2b3b2b0e02e0fde3c6c8ee856082f3e27d952f55a74ada68d4`) | `[안전]` Verified |
| `manifests/day60-empirical-results.json` | Sealed Phase-3 `Arm 1` vs `Arm 2` results (`d46e4f21...`), bundle SHA-256 (`785beb4c...`), spend (`$1.130102`, `38.23 CU`) | `[안전]` Verified |
| `manifests/arm3-empirical-k2-results.json` | Sealed Track 1A `Arm 3` 4-Tier results + `quarantinedPreFixProvenance` (body SHA-256 `615d24bb...`, file `7fef3728...`, `Holm m = 30`) | `[안전]` Verified |
| `manifests/p2-qwen38-27b-control-results.json` | Consolidated `Qwen3.8-27B BF16` (`47/50`) + `L4 NF4` (`105.291 s/item`) + `deferred_colab_80gb_503_stockout` (body `4b300bb7...`) | `[안전]` Verified |
| `manifests/frontier-eviction-v02-results.json` & `...-hydrogen-results.json` | Sealed Track 1C `v0.2` `contents[0]` eviction (`C1`/`C2`, non-pooled per-cell `0/50` stale, `95% CP` upper `7.11% < 10.0%`) | `[안전]` Verified |
| `manifests/p1-frontier-arm1-control-results.json` | `P1.1` Frontier same-harness standard `Arm 1` control (`seed = 20261122`, `50/50` on `carbon` & `hydrogen`, file `f08bf8dc...`) | `[안전]` Verified |
| `manifests/p1-v03-aliasing-prelock.json` & `p1-frontier-v03-aliasing-results.json` | `P1.2` `v0.3` value-aliasing prelock (`d937ff16...`) & 3-way `C1` vs `C2_v1` vs `C2_v2` proof (`64/64` `p = 3.73e-09`, `32/32` `p = 6.104e-05`) | `[안전]` Verified |
| `manifests/ARM3_PR_PACKAGING.md` | Complete companion PR packaging dossier (`ITT`-first tables, 4-Tier hierarchy, 5 footnotes, verified `P0`/`P1`/`P2` readouts) | `[안전]` Verified |
| `agent_artefacts/retraction_dag_0.1.0/validity/SATURATION_REPORT.md` | Frozen Branch 1 (`Option C`) empirical `v1 -> v2.0 -> v3.0` frontier KV re-reference saturation report | `[안전]` Verified |
| `agent_artefacts/retraction_dag_0.1.0/validity/VALIDITY_REPORT.md` | Capability-Stratified Synthesis (`ITT`-first Phase 3 `Arm 2` + Path 1 `Arm 3` 4-Tier + `P0`/`P1.1`/`P1.2`/`P2` synthesis) | `[안전]` Verified |
| `agent_artefacts/retraction_dag_0.1.0/review/SUMMARY.md` | Static zero-core, `SymPy + Pint`, 4-Tier `Arm 3`, `v0.3` value-aliasing confirmation, and `K6` claim-bound review | `[안전]` Verified |

## 2. Technical, Empirical & Governance Gates

- [x] **Zero Private Core & Hygiene Boundary**: `python3 scripts/check_zero_core.py` -> `[안전]` (`0` private symbols, `0` spectrum terms, `0` tilde/box defects).
- [x] **Pre-Experiment Cryptographic Seals Preserved**: `v3-multiturn-prepilot-lock.json` (`eac8d83e...`), `day46-baseline-freeze.json` (`a542492e...`), `day60-empirical-results.json` (`d46e4f21...`), and `p1-v03-aliasing-prelock.json` (`d937ff16...`) verified intact.
- [x] **Capability & Active-Capacity Stratified Regime + `P1.1` / `P1.2` Controls Complete (`$2.909901 / $100.00 USD <= $5.00`)**: Contrasts dense `Qwen3.8-27B BF16` (`47/50 = 94.0%`, `STALE 0`) and frontier standard `Arm 1` (`P1.1`: `50/50 = 100.0%` on `carbon` & `hydrogen`, `seed = 20261122`) + `v0.2` eviction (`0/50` stale per cell, `95% CP` upper `7.11% < 10.0%`) against sparse MoE `GLM-4.7-Flash 30B-A3B BF16` (`102/200 = 51.0%`, `STALE 73`) and `14B..24B` open-weight cohorts; `P1.2` (`v0.3`, `seed = 20261153`) confirms value-aliasing mechanism (`C1` `35/64` & `17/32` vs values-only `C2_v1` `39/64` vs full-rule `C2_v2` `64/64` `p = 3.73e-09` & `32/32` `p = 6.104e-05`, with transparent `v0.2` `"derived"` vs `"derive"` typo disclosure).
- [x] **Prompt-Only `Arm 2` Falsified, 4-Tier `Arm 3` Hierarchy & `P0` / `P2` Closeout Verified (`118.85 / 3,251.14 CU <= 162.56 CU`, `Active assignments: 0`)**: Preserves zero positive `K2` prompt claim (`-91.8%` `STALE` vs `113 -> 517` `OVER` seesaw), enforces `ITT`-first 4-Tier `Arm 3` reporting (`Qwen2.5-14B K4 ITT 6/400 FAIL` vs `0/394 Clean-T1 PASS`, `Mistral-24B` `205/800` exclusions, all 5 footnotes), verifies `P0` `0.00 CU` patched `apply_structural_gate_only` replay (`2,400` episodes, `+0` pass delta, `gb-0026` `PARSE -> STALE`), and records `P2` `Case 2B` `0.00 CU` Colab `A100/H100/G4` `503` stockout closeout (`quarantinedPreFixProvenance`, `t1only` cache `ffdc749b...`, `deferred_colab_80gb_503_stockout`).
- [ ] **External Blinded Human Audit Return (`N >= 150`) & 14-Day Public PR Window**: `audit_packet/` (`SEALED_KEY_COMMITMENT.json` `3a85aca2...`) ready for external reviewer reconciliation (`[PENDING_EXTERNAL_HUMAN_AUDIT_N160]`).
