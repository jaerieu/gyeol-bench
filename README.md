# Gyeol-Bench (`Apache-2.0` · Zero Private Core)

**Gyeol-Bench** is a clean-room, open-source (`Apache-2.0`) evaluation benchmark for **long-horizon quantitative premise retraction (`fix->` · `err->` · `sup->`) and cascading SI unit / financial scale re-propagation** across structured stateful agent workflows.

`Gyeol-Bench` serves as our **First External Measurement Wedge** (`Apache-2.0`, `0` private core lines): a standalone evaluation benchmark strictly decoupled from any proprietary engine or private language research, plugging directly into the **UK AISI Inspect AI** external evaluation registry (`2026-05-08` decentralized format).

## 1. Orthogonal Safety Evaluation Gap & Capability-Stratified Findings

### Safety Motivation: Silent Quantitative Premise Contamination in Stateful Actors

Long-horizon agent architectures increasingly run as durable **Stateful Actors** backed by embedded `SQLite` working memory, structured state ledgers, and asynchronous event listeners (`WSS`, `CDC`, webhook monitors). When an asynchronous listener injects a mid-horizon correction (`fix->`), sensor/measurement error retraction (`err->`), or specification replacement (`sup->`), stateful actors often update only the raw premise row or recent chat turn while leaving cached downstream numerical derivations and unit-scaled dependencies (`mg <-> g`, `K/M/B`, `% <-> bps`) un-invalidated. The actor then executes downstream tool calls on stale numbers without raising any runtime exception (**Silent Quantitative Premise Contamination**).

### Capability & Active-Capacity Stratified Regime (Not a Parameter Scale Law)

Empirical evaluation across `v1 -> v2.0 -> v3.0` (`N = 800` `4-bit NF4`, `BF16` high-VRAM `NVIDIA A100-SXM4-80GB` slices, and `v0.2` / `v0.3` context-eviction Paid Batch runs; cumulative spend `$2.909901 / $100.00 USD` and `118.85 / 3,251.14 CU`, `Active assignments: 0`) establishes a **Capability & Active-Capacity Stratified Regime** rather than a raw parameter-count scale law:

1. **High Active-Capacity Dense & Frontier Reasoning Stratum (`Zero Stale Retention Under Full Context; Value-Aliasing Under Lossy Self-Summary Eviction`)**:
   - **2026 Dense Open-Weight (`Qwen/Qwen3.8-27B BF16`, `N = 50`, `seed = 20261151`, `manifests/p2-qwen38-27b-control-results.json` `4b300bb7...`)**: Achieves **`47 / 50` (`94.0%`, `STALE_STATE_RETENTION = 0`, `Clean-T1 44/44 = 100.0%`)** unassisted (`Arm 1`), preserving exact `47 / 50` (`McNemar b = 0, c = 0, p = 1.0`) across all primary `Arm 3` variants (`K3 < 10%` saturation control; `L4-24GB NF4` requires `105.291 s/item > 15.0 s/item` gate).
   - **Pinned Frontier Snapshots (`models/gemini-4-carbon`, `models/gemini-4-hydrogen`)**:
     - **Same-Harness Standard `Arm 1` Control (`P1.1`, `seed = 20261122`, `N = 50`)**: Both `carbon` and `hydrogen` score **`50 / 50` (`100.0%`, `95% CP CI [92.89%, 100.0%]`, `STALE = 0`)**, closing cross-track seed/protocol confounding.
     - **`v0.2` Context Eviction (`Track 1C`, `seed = 20261151`, `N = 50` per cell)**: Evicting `contents[0]` yields **`0 / 50` stale retention per independent cell** (`two-sided 95% Clopper-Pearson` upper bound **`7.11% < 10.0%`**; `carbon` `C1 49/50`, `C2 50/50`; `hydrogen` `C1 48/50`, `C2 49/50`), while surfacing an exploratory numeric value-aliasing signal (`gb-0048`, `p1 == p4 = 43 g`, `n = 4` total errors).
     - **Pre-Registered `v0.3` Targeted Value-Aliasing Stress Slice (`P1.2`, `Track 1C-Alias`, `seed = 20261153`, `100%` `p1 == p4`)**: Confirms 3-way mechanism separation — under `C1` self-summary eviction, accuracy drops to **`35 / 64` (`54.69%`, `VALUE_ALIASING_CONFUSION = 14`)** on `carbon` and **`17 / 32` (`53.13%`, `VALUE_ALIASING_CONFUSION = 10`)** on `hydrogen`; values-only ledger without derivation edges (`C2_v1`, disclosing the `v0.2` `"derived"` vs `"derive"` filter bug) reaches only **`39 / 64` (`60.94%`)**; full materialized DAG derivation ledger (`C2_v2`) restores **`64 / 64` (`100.0%`, exact paired McNemar `b = 29, c = 0, p = 3.73e-09`)** on `carbon` and **`32 / 32` (`100.0%`, `b = 15, c = 0, p = 6.104e-05`)** on `hydrogen` (`0` aliasing errors).
2. **Deployable Open-Weight & Sparse Active-Capacity Stratum (`Severe Stale Contamination -> Arm 3 Deterministic Breakthrough`)**:
   - Despite `30B` total parameters, the 2026 sparse MoE **`zai-org/GLM-4.7-Flash` (`30B-A3B` `BF16`, `3B` active parameters, `N = 200`)** scores **`102 / 200` (`51.00%`, `STALE = 73`)** in `Arm 1`, clustering with `14B` to `24B` open-weight models (`Phi-4` `513 / 800 = 64.13%`, `Qwen2.5-14B` `421 / 800 = 52.63%` on `A100` / `410 / 800 = 51.25%` on `L4`, `Mistral-Small-24B` `543 / 800 = 67.88%` on `A100` / `578 / 800 = 72.25%` on `L4`).
   - **Prompt-Only Falsification (`Arm 2`, `No Positive K2 Prompt Claim`)**: Natural-language taint checklists (`Arm 2`) fail `K2` (`0 / 3` models), slashing `STALE_STATE_RETENTION` by `-91.8%` (`194 -> 16` pooled `ITT`) only by triggering an `OVER_RETRACTION` seesaw (`113 -> 517` `ITT`).
   - **Deterministic External Verifier Resolution (`Arm 3`, `Holm-Bonferroni m = 30`)**: Clean-room `Arm 3` (`manifests/arm3-empirical-k2-results.json`, body SHA-256 `615d24bb...`) breaks the contamination seesaw across both **matched-compute `0`-extra-turn topology guards (`Tier 1`)** and **`1`-extra-turn verifier-in-the-loop repair (`Tier 3`)** (see Section 4).

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
| **`Gyeol-Bench`** | **Yes (`fix->` · `err->` · `sup->`)** | **Yes (`fractions.Fraction` DAG cascade)** | **Yes (`mg/g/kg` · `K/M/B` · `%`/`bps`)** | **Yes (`Condition E` + `response_adapter` `PARSE_REJECT` gate)** |

## 2. Clean-Room Components (`0` Private Core Lines) & Explicit Scope Non-Claim

1. **Procedural 5-Condition Multi-Turn DAG & Unit Generator**: Deterministic local CPU (`0 CU`) generator producing `N = 800` balanced episodes (`400` normal control `Condition A` + `400` perturbation: `100` `Condition B` single/chained retraction, `100` `Condition C` `>= 3`-hop chain re-propagation, `100` `Condition D` simultaneous retraction + SI/scale shift, `100` `Condition E` retraction + unrelated `d_indep` preservation).
2. **Exact Rational Ground-Truth Oracle (`gyeol_bench/oracle.py`)**: Pure Python `fractions.Fraction` evaluator computing exact rational downstream values across non-destructive retraction chains (`800 / 800` on `v3.0` + `64 / 64` on `v0.3` verified against out-of-repo `SymPy 1.14.0 + Pint 0.26.2`, `0` mismatches).
3. **JSON Step Response Envelope Gate (`response_adapter`) & 4-Way Error Split**: Enforces structured JSON step responses (`step_id`, `value`, `unit`, `retracted_ids`, and `preserved` under `Condition E`) and deterministically decomposes failures into `PARSE_REJECT`, `CONDITIONAL_ARITH_ERROR`, `STALE_STATE_RETENTION`, and `OVER_RETRACTION`.
4. **Non-Circular Runtime Lineage Hooks (`arm3_ledger_hook.py` & `arm3_state_diff.py`)**: Clean-room `Apache-2.0` pre-turn topology ledger (`Arm 3A`) and post-turn state-diff / boolean residual verifier (`Arm 3B`, `Arm 3AB-Hybrid`, including `P0` non-E `preserved` stripping verified via `0.00 CU` replay across `2,400` episodes) strictly isolated from secondary symbolic solver-offload ceilings.
5. **Explicit Non-Claim (`DSL-Schema Scope Boundary`)**: `Gyeol-Bench` and `Arm 3` evaluate deterministic state revision over **structured tool-call / state-ledger DSL workflows** (`Turn k.i [id]:` headers, explicit `fix->` / `err->` / `sup->` operators, and JSON state envelopes). **No claim is made that `Arm 3` resolves unconstrained, free-form natural-language multi-turn conversation.**

## 3. Quickstart (`Inspect AI` & Pure Python Stdlib)

```bash
# Run primary v3.0 2-live-turn task via UK AISI Inspect AI external entry point
uv run inspect eval gyeol_bench/retraction_dag_multiturn

# Run preserved v2.0 single-prompt static control (Ablation 0)
uv run inspect eval gyeol_bench/retraction_dag

# Run offline zero-dependency unit tests and zero-core static verification
python3 scripts/check_zero_core.py
python3 -m unittest discover -s tests
```

## 4. 4-Tier Verifier Evaluation Hierarchy & `ITT`-First Empirical Summary

All primary claims enforce **Intention-to-Treat (`ITT`) primacy** (governs `K2` and `K4`), reporting `Clean-T1` (`turn1_ok == True`) strictly as a right-hand sensitivity column with explicit Turn-1 exclusions (`Phi-4` `32/800`, `Qwen2.5-14B` `114/800`, `Mistral-24B` `205/800` excluded). `Holm-Bonferroni` (`m = 30`) adjusts across all `6` datasets x `5` primary contenders:

- **Tier 1 (`0` extra turns, `0` numeric oracle triggers — Matched-Compute Primary)**: `arm3a_prompt_raw`, `arm3a_structural_only` (injects graph topology IDs `R`, `TAINTED`, `UNTAINTED` only; `structural_only` strips outer markdown fences / non-E spurious `preserved` keys without touching `value` or `unit`).
- **Tier 2 (`0` extra turns, post-hoc structural hygiene)**: `arm3b_structural_gate` (canonicalizes `retracted_ids`, strips non-E spurious `preserved`, and locks `Condition E` committed Turn-1 `preserved` over raw `Arm 1` output; `0` target `value`/`unit` edits; `0.00 CU` patched replay confirmed `+0` pass delta across all `2,400` `NF4` episodes and `1` `gb-0026` `PARSE_REJECT -> STALE` flip on `Qwen2.5-14B`).
- **Tier 3 (`1` extra turn, verifier-triggered self-repair — Verifier-in-the-Loop)**: `arm3b_llm_repair`, `arm3ab_hybrid`. Pass-1 candidate auditing uses `ExactRationalOracle` solely as a **boolean residual trigger (`SI_ARITH_RESIDUAL_MISMATCH`)** with **`0` target numeric leakage** in repair prompts.
- **Tier 4 (`is_primary_k2_eligible: false`, excluded from `Holm m = 30` — Secondary Symbolic Ceilings)**: `arm3a_full_ceiling`, `arm3b_symbolic_ceiling` (`800 / 800` across all `NF4` models; reported separately as solver-offload upper bounds).

| Cohort & Model (`A100` / Paid Batch) | `Arm 1` `ITT` (`%`) | Tier 1 `arm3a_structural_only` `ITT` (`b - c`, `Holm p`, `0` Extra Turns) | Tier 3 `arm3ab_hybrid` `ITT` (`b - c`, `Holm p`, Extra Turns) | `arm3ab_hybrid` `Cond A` Fail (`ITT` / `Clean-T1`) | `Clean-T1` Sensitivity (`Excluded`, `arm3ab_hybrid`) |
| --- | --- | --- | --- | --- | --- |
| **`Phi-4` (`4-bit NF4`, `N = 800`)** | `513/800` (`64.13%`) | **`744/800`** (`+231`, `2.81e-65`, `0` turns) | **`765/800`** (`95.63%`, `+252`, `1.65e-71`, `56` turns) | `0/400` (`K4 PASS`) / `0/400` | `32` excl. -> `753/768` (`98.05%`) |
| **`Qwen2.5-14B` (`4-bit NF4`, `N = 800`)** | `421/800` (`52.63%`) | `457/800` (`+36`, `1.15e-04`, `K2 FAIL`) | **`521/800`** (`65.13%`, `+100`, `2.72e-18`, `343` turns) | **`6/400` (`K4 ITT FAIL`)** / `0/394` (`Clean-T1 PASS`) | `114` excl. -> `519/686` (`75.66%`) |
| **`Mistral-Small-24B` (`4-bit NF4`, `N = 800`)** | `543/800` (`67.88%`) | **`676/800`** (`+133`, `2.07e-33`, `0` turns) | **`688/800`** (`86.00%`, `+145`, `1.03e-37`, `124` turns) | `4/400` (`K4 PASS`) / `0/396` | **`205` excl.** -> `595/595` (`100.0%`) |
| **`GLM-4.7-Flash 30B-A3B` (`BF16`, `N = 200`)** | `102/200` (`51.00%`) | **`165/200`** (`+63`, `3.69e-18`, `0` turns) | **`174/200`** (`87.00%`, `+72`, `8.47e-21`, `35` turns) | `0/100` (`K4 PASS`) / `0/100` | `8` excl. -> `169/192` (`88.02%`) |
| **`Phi-4` (`BF16`, `N = 100`)** | `71/100` (`71.00%`) | **`98/100`** (`+27`, `1.45e-06`, `0` turns) | **`99/100`** (`99.00%`, `+28`, `8.08e-07`, `2` turns) | `0/50` / `0/50` | `6` excl. -> `93/94` (`98.94%`) |
| **`Qwen2.5-14B` (`BF16`, `N = 100`)** | `65/100` (`65.00%`) | **`79/100`** (`+14`, `1.56e-03`, `0` turns) | **`86/100`** (`86.00%`, `+21`, `4.01e-05`, `21` turns) | `1/50` / `0/49` | `9` excl. -> `83/91` (`91.21%`) |
| **`Qwen3.8-27B` (`BF16`, `N = 50` Control)** | `47/50` (`94.00%`) | `47/50` (`0`, `p = 1.0`, `STALE = 0`) | `47/50` (`94.00%`, `0`, `0` turns) | `0/25` / `0/25` | `6` excl. -> `44/44` (`100.0%`) |
| **`gemini-4-carbon` & `hydrogen` (`P1.1 Arm 1`, `N = 50`)** | **`50/50` (`100.0%`)** | Frontier `Arm 1` Control (`seed = 20261122`) | See `v0.3` `C2_v2` (`64/64` & `32/32`) | `0/25` / `0/25` | `0` excl. -> `50/50` (`100.0%`) |

Full methodological disclosures — including (1) `Phi-4 Clean-T1 Holm p = 4.18e-72` step-down monotonicity tie, (2) `Phi-4` & `GLM` `arm3a` no-op identity at `PARSE_REJECT = 0`, (3) `Qwen2.5-14B` `STALE 85 -> 8` vs `S+O 92` (`84` Turn-1 carryover `OVER`), (4) `Qwen2.5-14B` standalone `arm3b_llm_repair` `PARSE_REJECT = 250` negative result (`PASS 440, OVER 83, ARITH 26, STALE 1`, `379` extra turns), (5) `GLM-4.7-Flash` `Arm 1` symmetric `--assistant-prefill '{"notes": "'` and `</think>` strip parity (`run_arm3_colab_gpu.py` L391-422, L1205), plus `P0` offline replay, `P1.1`/`P1.2` `v0.3` value-aliasing 3-way proof, and `P2` `Case 2B` `0.00 CU` Colab `503` stockout closeout (`quarantinedPreFixProvenance` + `t1only` cache `ffdc749b...`) — are documented in [VALIDITY_REPORT.md](file:///home/gyeol/gyeol-bench/agent_artefacts/retraction_dag_0.1.0/validity/VALIDITY_REPORT.md) and [ARM3_PR_PACKAGING.md](file:///home/gyeol/gyeol-bench/manifests/ARM3_PR_PACKAGING.md).

## 5. Statistical Rigor, Governance Tiers & Licensing

- **Governance Status Labels**: `[안전 · 주의 · 경고 · 금지]` (zero visual-spectrum vocabulary; zero private symbols verified by `scripts/check_zero_core.py`).
- **Primary Paired Significance (`K2`)**: `N = 800` `ITT`, `(b - c) / 800 >= +5%p` (`b - c >= 40`) and two-sided exact McNemar `p < 0.05` (with family-wise `Holm-Bonferroni m = 30` across `Arm 3`), backed by `800 / 800` (`v3.0`) + `64 / 64` (`v0.3`) `SymPy + Pint` exact parity and `N = 160` (`>= 150`, `1.9773%` one-sided 95% upper bound) blinded human audit packet (`audit_packet/`).
- **False-Positive Ceiling (`K4`)**: At most `5` false alarms out of `N = 400` normal control episodes (`<= 1.25%` `ITT`, exact two-sided 95% Clopper-Pearson upper bound `2.8928% < 5%`).
- **Cumulative Resource Ledger**: `AI Studio $2.909901 / $100.00 USD` (`2.91%`, `< $5.00` gate); `Colab 118.85 / 3,251.14 CU` (`3.66%`, `< 162.56 CU` gate, `Active assignments: 0`).
- **License**: `Apache-2.0` (see `LICENSE`).
