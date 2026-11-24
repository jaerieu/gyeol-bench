# Companion PR Packaging & Reviewer Defense Dossier: `Arm 3` 4-Tier Verifier, Capability-Stratified Synthesis & `v0.3` Value-Aliasing Proof (`v3.0`)

- **Repository & License**: `jaerieu/gyeol-bench` (`Apache-2.0`, Clean-Room Benchmark; `0` private core lines)
- **Registry Target**: `UK AISI Inspect Evals` (`2026-05-08` Decentralized External Package Specification)
- **Primary Task**: `gyeol_bench/retraction_dag_multiturn` (`v3.0` 2-Live-Turn `Shared-Turn-1` Stateful Handoff; `gyeol_bench/retraction_dag` preserved intact as `Ablation 0`)
- **Governance Verdict**: `[안전]` (All `13` frozen files in `manifests/day46-baseline-freeze.json` and `manifests/day60-empirical-results.json` preserved untouched; `scripts/check_zero_core.py` `0` violations)
- **Final Cumulative Resource Ledger**:
  - **AI Studio Paid Batch**: **`$2.909901 / $100.00 USD`** (`2.91%`, strictly inside `5%` `$5.00 USD` gate; `P1` delta `+$1.274331 USD`)
  - **Colab Pro+**: **`118.85 / 3,251.14 CU`** (`3.66%`, balance `3,132.29 CU`, strictly inside `5%` `162.56 CU` gate; `P2` delta `0.00 CU`, **`Active assignments: 0`**, `Usage rate: 0.00/hr`)
- **Canonical Manifest Seals**:
  - `manifests/v3-multiturn-prepilot-lock.json`: `eac8d83e62ceb784f9096d5aab8540ee70a07f57d10950cb77fea57202502e53` (file SHA-256 `54aa2354...`)
  - `manifests/day46-baseline-freeze.json`: `a542492ed604bd2b3b2b0e02e0fde3c6c8ee856082f3e27d952f55a74ada68d4` (`RFC 6962` root `d54cc303f3ab11521356e563610da53b799d6b9d9619017d51de43a08ee6a634`)
  - `manifests/day60-empirical-results.json`: `d46e4f217c1d47d85c08f35810efc8bb95303fa6845b184a884ccd81e0e78d25` (raw Phase-3 bundle `785beb4c3a67fc85fadd985261a484253e2e686c483fb5a130c8113703f0ba8e`)
  - `manifests/arm3-empirical-k2-results.json`: body SHA-256 `615d24bb69b0ede1e72a47acbec57e7b19d6ed8f5e8ab92330f9f5ce81b1829f` (file SHA-256 `7fef3728384185279d6068ee9f9915fd90719574daedf2301df31feab8133487`, `holm_bonferroni_family_size = 30`)
  - `manifests/p2-qwen38-27b-control-results.json`: body SHA-256 `4b300bb7859c34e4a614cd150ec2dd78e85b4027b2c42a6d617391bd95746d6c` (file SHA-256 `29ee0cfb635ba5b59856a11b4f20d6cf44a41a198e8dc4f472b32b4d2ae0da94`)
  - `manifests/p1-frontier-arm1-control-results.json`: file SHA-256 `f08bf8dc456b01f985c8da7c999adc7c3ba7b034699fe4ad05caddb7dff91aac`
  - `manifests/p1-v03-aliasing-prelock.json`: dataset SHA-256 `d937ff16cb3dc1f9186a29301c1b36b6c3897670fb293ae2890e6c057e11dd9e` (file SHA-256 `517982093b127ef81c9288d9cf5f2c13088b875f17f02125024b6bc4e0e82f8b`)
  - `manifests/p1-frontier-v03-aliasing-results.json`: file SHA-256 `1584db9211ad8424e99c28c60f55e62ec955d3fb35caf91979bd3cd6dd910bd5`
  - `manifests/frontier-eviction-v02-results.json` (`4c1e6f88...`) & `manifests/frontier-eviction-v02-hydrogen-results.json` (`19d84f1f...`)

---

## 1. Capability & Active-Capacity Stratified Regime (Replacing Parameter Scale Law Framing)

This companion dossier synthesizes the complete empirical record across `Phase 3` (`L4`, prompt-only `Arm 2` falsification), `Path 1` (`A100-SXM4-80GB` `Arm 3` runtime verifiers + `v0.2` context eviction), and Wave 1 to Wave 3 follow-up controls (`P0` offline replay, `P1.1` frontier same-harness `Arm 1`, `P1.2` `v0.3` value-aliasing 3-way proof, and `P2` `0.00 CU` clean closeout) without modifying any pre-registered freeze file.

Empirical results across dense, sparse-MoE, quantized, and frontier models demonstrate a **Capability & Active-Capacity Stratified Regime** rather than a raw parameter-count scale law:

1. **High Active-Capacity Dense & Frontier Stratum (`Zero Stale Contamination in Full Context; Value-Aliasing Collapse Under Lossy Eviction`)**:
   - **Dense 2026 Open-Weight (`Qwen/Qwen3.8-27B BF16`, `N = 50`, `seed = 20261151`)**: Achieves **`47 / 50` (`94.00%`, `STALE_STATE_RETENTION = 0`, `CONDITIONAL_ARITH_ERROR = 3`, `Clean-T1 44/44 = 100.0%`)** in unassisted `Arm 1`, preserving exact `47 / 50` (`McNemar b = 0, c = 0, p = 1.0`) across all primary `Arm 3` variants (`K3 < 10%` saturation control).
   - **Pinned Frontier Reasoning (`models/gemini-4-carbon`, `models/gemini-4-hydrogen`)**:
     - Achieve **`50 / 50` (`100.0%`, `95% CP CI [92.89%, 100.0%]`)** on standard 2-live-turn `Arm 1` (`P1.1`, `seed = 20261122`) and `50 / 50` in `v3.0` full-context pilot (`Option C`).
     - Achieve **`0 / 50` `STALE_STATE_RETENTION` in every independent `v0.2` context-eviction cell** (`two-sided 95% Clopper-Pearson` upper bound **`7.11% < 10.0%`**; `carbon` `C1 49/50`, `C2 50/50`; `hydrogen` `C1 48/50`, `C2 49/50`), while surfacing an exploratory `gb-0048` (`p1 == p4 = 43 g`, `n = 4` errors) value-aliasing signal.
     - Under pre-registered `P1.2 v0.3` (`seed = 20261153`, `100%` `p1 == p4`), prose self-summaries (`C1`) collapse to **`35 / 64` (`54.69%`, `VALUE_ALIASING_CONFUSION = 14`)** on `carbon` and **`17 / 32` (`53.13%`, `VALUE_ALIASING_CONFUSION = 10`)** on `hydrogen`, values-only `C2_v1` reaches only **`39 / 64` (`60.94%`)**, and full materialized derivation-rule ledger `C2_v2` restores **`64 / 64` (`100.0%`, `p = 3.73e-09`)** and **`32 / 32` (`100.0%`, `p = 6.104e-05`)**.
2. **Deployable Open-Weight & Sparse Active-Capacity Stratum (`Severe Stale Contamination -> Arm 3 Deterministic Breakthrough`)**:
   - **Active-Capacity Contrast**: Despite `30B` total parameters, the 2026 sparse MoE **`zai-org/GLM-4.7-Flash` (`30B-A3B` `BF16`, `3B` active parameters, `N = 200`)** scores **`102 / 200` (`51.00%`, `STALE = 73`)** in `Arm 1`, clustering with `14B..24B` open-weight deployments (`Phi-4` `513/800 = 64.13%`, `Qwen2.5-14B` `421/800 = 52.63%`, `Mistral-Small-24B` `543/800 = 67.88%` on `A100`) rather than with dense `Qwen3.8-27B BF16` (`94.0%`).
   - **Prompt-Only `Arm 2` Falsification Preserved**: No positive `K2` prompt-only superiority claim is made (`Arm 2` failed `0 / 3` models in `day60-empirical-results.json`, cutting `STALE` by `-91.8%` only via a `113 -> 517` `ITT` `OVER_RETRACTION` seesaw).
   - **`Arm 3` Deterministic Lineage Resolution**: External deterministic graph-topology hooks (`Tier 1`, `0` extra turns) and non-leaking residual verifier loops (`Tier 3`, `<= 1` extra turn) eliminate the contamination seesaw (`Holm p <= 2.72e-18` across all `4` non-saturated models).

---

## 2. The 4-Tier Verifier Evaluation Hierarchy & Non-Circularity Audit

To address peer-review scrutiny regarding compute matching and verifier oracle access, `Arm 3` (`gyeol_bench/arm3_ledger_hook.py`, `gyeol_bench/arm3_state_diff.py`) strictly separates interventions into four evaluation tiers:

| Tier | Evaluation Arm(s) | Extra LLM Turns | Numeric Oracle Trigger | Primary `K2` / `Holm m = 30` Eligible? | Exact Mechanism & Non-Circularity Boundary |
| --- | --- | --- | --- | --- | --- |
| **Tier 1** | `arm3a_prompt_raw`, `arm3a_structural_only` | **`0`** | **`0` (`None`)** | **Yes (Matched-Compute Primary)** | Pre-execution deterministic DAG reachability header (`R`, `TAINTED`, `UNTAINTED` node IDs only; `0` numeric computations or target hints). `arm3a_structural_only` strips outer markdown fences and non-E spurious `preserved` keys without altering candidate `value` or `unit`. |
| **Tier 2** | `arm3b_structural_gate` | **`0`** | **`0` (`None`)** | **Yes (Post-Hoc Structural Hygiene)** | Post-hoc deterministic state-diff canonicalization of `retracted_ids`, stripping of non-E spurious `preserved` (`P0` patch), and `Condition E` committed Turn-1 `preserved` lock over raw `Arm 1` output; candidate `value` and `unit` are untouched. |
| **Tier 3** | `arm3b_llm_repair`, `arm3ab_hybrid` | **`<= 1`** (Selective) | **Boolean Pass-1 Residual (`SI_ARITH_RESIDUAL_MISMATCH`)** | **Yes (Verifier-in-the-Loop)** | Pass-1 candidate audit (`pass1_audit_candidate`) checks structural state-diff rules and queries `ExactRationalOracle` **strictly as a boolean residual trigger (`SI_ARITH_RESIDUAL_MISMATCH`)** to gate a single repair turn. **Zero ground-truth target numbers (`d6`, `d_indep`) ever appear in repair prompts** (verified by unit tests). `arm3ab_hybrid` pairs `Arm 3A` pre-anchoring with `Arm 3B` Pass-1 audit + post-repair envelope hygiene. |
| **Tier 4** | `arm3a_full_ceiling`, `arm3b_symbolic_ceiling` | `0` / `<= 1` | Direct `Fraction` + `SI` Substitution | **`No` (`is_primary_k2_eligible: false`)** | **Excluded from `Holm-Bonferroni m = 30`.** Reported exclusively in a separate reference table (`800/800`, `100/100`, `200/200`, `50/50`) as neuro-symbolic solver-offload upper bounds. |

---

## 3. Canonical `ITT`-First Empirical Tables (`Holm-Bonferroni m = 30`)

Every empirical table places **Intention-to-Treat (`ITT`) first** as the primary column governing `K2` and `K4`, reporting `Clean-T1` strictly as a right-hand sensitivity column with explicit Turn-1 exclusion counts (`Phi-4` `32/800`, `Qwen2.5-14B` `114/800`, **`Mistral-24B` `205/800` excluded**) and explicitly contrasting **`Qwen2.5-14B K4 ITT 6/400 FAIL` vs `0/394 Clean-T1 PASS`**.

### 3.1 Primary `N = 800` `4-bit NF4` Cohort (`seed = 20261122`, Symmetric `A100` Regeneration Over `Shared-Turn-1`)

| Model (`NF4`, `N = 800`) & Tier / Arm | `ITT` `PASS` (`%`) | `ITT` `b, c` (`b - c`) | `ITT` Exact `p` (`Holm p`) | `K2` (`ITT b - c >= 40`) | `Cond A` Fail (`ITT` / `Clean-T1`) | `K4` (`ITT <= 5/400`) | `ITT` `STALE / OVER` (`S+O`, `Delta`) | `Clean-T1` Sensitivity (`Excl.`, `PASS`, `b - c`, `Holm p`) | Extra Turns (`ITT` / `Clean-T1`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`Phi-4`** — `Arm 1` Baseline | `513/800` (`64.13%`) | Ref | Ref | Ref | `1/400` / `1/400` | `PASS` `[안전]` | `106 / 16` (`122`, Ref) | `32` excl. -> `504/768` (`65.63%`) | `0` / `0` |
| `Phi-4` — Tier 1 `arm3a_prompt_raw` | `744/800` (`93.00%`) | `233, 2` (`+231`) | `1.00e-66` (`2.81e-65`) | **`PASS`** | `0/400` / `0/400` | **`PASS`** `[안전]` | `0 / 15` (`15`, `-107`) | `32` excl. -> `733/768` (`+229`, `3.76e-66`) | `0` / `0` |
| `Phi-4` — Tier 1 `arm3a_structural_only` | `744/800` (`93.00%`) | `233, 2` (`+231`) | `1.00e-66` (`2.81e-65`) | **`PASS`** | `0/400` / `0/400` | **`PASS`** `[안전]` | `0 / 15` (`15`, `-107`) | `32` excl. -> `733/768` (`+229`, `3.76e-66`) | `0` / `0` |
| `Phi-4` — Tier 2 `arm3b_structural_gate` | `626/800` (`78.25%`) | `113, 0` (`+113`) | `1.93e-34` (`4.43e-33`) | **`PASS`** | `0/400` / `0/400` | **`PASS`** `[안전]` | `106 / 15` (`121`, `-1`) | `32` excl. -> `617/768` (`+113`, `5.01e-33`) | `0` / `0` |
| `Phi-4` — Tier 3 `arm3b_llm_repair` | `759/800` (`94.88%`) | `246, 0` (`+246`) | `1.77e-74` (`5.31e-73`) | **`PASS`** | `0/400` / `0/400` | **`PASS`** `[안전]` | `0 / 15` (`15`, `-107`) | `32` excl. -> `747/768` (`+243`, `4.18e-72`) | `287` / `264` |
| `Phi-4` — **Tier 3 `arm3ab_hybrid`** | **`765/800` (`95.63%`)** | `254, 2` (**`+252`**) | `5.68e-73` (`1.65e-71`) | **`PASS`** | **`0/400`** / **`0/400`** | **`PASS`** `[안전]` | **`0 / 15` (`15`, `-107`)** | `32` excl. -> **`753/768` (`+249`, `4.18e-72`)** | **`56` / `35`** |
| **`Qwen2.5-14B`** — `Arm 1` Baseline | `421/800` (`52.63%`) | Ref | Ref | Ref | `19/400` / `13/394` | `FAIL` `[주의]` | `85 / 92` (`177`, Ref) | `114` excl. -> `419/686` (`61.08%`) | `0` / `0` |
| `Qwen2.5-14B` — Tier 1 `arm3a_prompt_raw` | `457/800` (`57.13%`) | `53, 17` (`+36`) | `1.92e-05` (`1.15e-04`) | `FAIL` | **`6/400`** / `0/394` | **`FAIL`** (`Clean-T1 PASS`) | `70 / 84` (`154`, `-23`) | `114` excl. -> `456/686` (`+37`, `6.37e-05`) | `0` / `0` |
| `Qwen2.5-14B` — Tier 1 `arm3a_structural_only` | `457/800` (`57.13%`) | `53, 17` (`+36`) | `1.92e-05` (`1.15e-04`) | `FAIL` | **`6/400`** / `0/394` | **`FAIL`** (`Clean-T1 PASS`) | `70 / 84` (`154`, `-23`) | `114` excl. -> `456/686` (`+37`, `6.37e-05`) | `0` / `0` |
| `Qwen2.5-14B` — Tier 2 `arm3b_structural_gate` | `486/800` (`60.75%`) | `65, 0` (`+65`) | `5.42e-20` (`1.03e-18`) | **`PASS`** | `9/400` / `3/394` | **`FAIL`** (`Clean-T1 PASS`) | `85 / 80` (`165`, `-12`) | `114` excl. -> `484/686` (`+65`, `1.03e-18`) | `0` / `0` |
| `Qwen2.5-14B` — Tier 3 `arm3b_llm_repair` (`PARSE 250`) | `440/800` (`55.00%`) | `19, 0` (`+19`) | `3.81e-06` (`3.05e-05`) | `FAIL` | `9/400` / `3/394` | **`FAIL`** (`Clean-T1 PASS`) | `1 / 83` (`84`, `-93`) | `114` excl. -> `438/686` (`+19`, `3.05e-05`) | `379` / `267` |
| `Qwen2.5-14B` — **Tier 3 `arm3ab_hybrid`** | **`521/800` (`65.13%`)** | `117, 17` (**`+100`**) | `1.51e-19` (`2.72e-18`) | **`PASS`** | **`6/400`** / **`0/394`** | **`FAIL`** (`Clean-T1 PASS`) | **`8 / 84` (`92`, `-85`)** | `114` excl. -> **`519/686` (`+100`, `1.20e-18`)** | **`343` / `230`** |
| **`Mistral-24B`** — `Arm 1` Baseline | `543/800` (`67.88%`) | Ref | Ref | Ref | `10/400` / `6/396` | `FAIL` `[주의]` | `4 / 1` (`5`, `PARSE 100`) | **`205` excl.** -> `502/595` (`84.37%`) | `0` / `0` |
| `Mistral-24B` — Tier 1 `arm3a_prompt_raw` | `655/800` (`81.88%`) | `117, 5` (`+112`) | `8.14e-29` (`1.71e-27`) | **`PASS`** | `5/400` / `1/396` | **`PASS`** `[안전]` | `0 / 0` (`0`, `PARSE 100`) | **`205` excl.** -> `594/595` (`+92`, `9.69e-27`) | `0` / `0` |
| `Mistral-24B` — Tier 1 `arm3a_structural_only` | `676/800` (`84.50%`) | `138, 5` (`+133`) | `8.64e-35` (`2.07e-33`) | **`PASS`** | `5/400` / `1/396` | **`PASS`** `[안전]` | `2 / 73` (`75`, `PARSE 3`) | **`205` excl.** -> `594/595` (`+92`, `9.69e-27`) | `0` / `0` |
| `Mistral-24B` — Tier 2 `arm3b_structural_gate` | `650/800` (`81.25%`) | `107, 0` (`+107`) | `1.23e-32` (`2.71e-31`) | **`PASS`** | `10/400` / `6/396` | `FAIL` | `8 / 72` (`80`, `PARSE 1`) | **`205` excl.** -> `585/595` (`+83`, `4.34e-24`) | `0` / `0` |
| `Mistral-24B` — Tier 3 `arm3b_llm_repair` | `666/800` (`83.25%`) | `123, 0` (`+123`) | `1.88e-37` (`4.70e-36`) | **`PASS`** | `8/400` / `4/396` | `FAIL` (`Clean-T1 PASS`) | `1 / 61` (`62`, `PARSE 13`) | **`205` excl.** -> `587/595` (`+85`, `1.14e-24`) | `257` / `93` |
| `Mistral-24B` — **Tier 3 `arm3ab_hybrid`** | **`688/800` (`86.00%`)** | `149, 4` (**`+145`**) | `3.95e-39` (`1.03e-37`) | **`PASS`** | **`4/400`** / **`0/396`** | **`PASS`** `[안전]` | **`0 / 72` (`72`, `PARSE 2`)** | **`205` excl.** -> **`595/595` (`+93`, `5.05e-27`)** | **`124` / `1`** |

### 3.2 `BF16` Full-Precision Slices & `2026` Dense / Sparse-MoE Generalization (`A100-SXM4-80GB`)

| Model (`BF16`) & Tier / Arm | `ITT` `PASS` (`%`) | `ITT` `b, c` (`b - c`) | `ITT` Exact `p` (`Holm p`) | `Cond A` Fail (`ITT` / `Clean-T1`) | `ITT` `STALE / PARSE` | `Clean-T1` Sensitivity (`Excl.`, `PASS`, `b - c`) | Extra Turns (`ITT` / `Clean-T1`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`Phi-4 BF16` (`N = 100`)** — `Arm 1` Baseline | `71/100` (`71.00%`) | Ref | Ref | `0/50` / `0/50` | `0 / 0` | `6` excl. -> `67/94` (`71.28%`, Ref) | `0` / `0` |
| `Phi-4 BF16` — Tier 1 `arm3a_prompt_raw` & `structural_only` | `98/100` (`98.00%`) | `28, 1` (`+27`) | `1.12e-07` (`1.45e-06`) | `0/50` / `0/50` | `0 / 0` | `6` excl. -> `92/94` (`97.87%`, `+25`) | `0` / `0` |
| `Phi-4 BF16` — Tier 2 `arm3b_structural_gate` | `95/100` (`95.00%`) | `24, 0` (`+24`) | `1.19e-07` (`1.45e-06`) | `0/50` / `0/50` | `0 / 0` | `6` excl. -> `91/94` (`96.81%`, `+24`) | `0` / `0` |
| `Phi-4 BF16` — Tier 3 `arm3b_llm_repair` | **`100/100` (`100.00%`)** | `29, 0` (`+29`) | `3.73e-09` (`5.59e-08`) | `0/50` / `0/50` | `0 / 0` | `6` excl. -> `94/94` (`100.00%`, `+27`) | `29` / `27` |
| `Phi-4 BF16` — **Tier 3 `arm3ab_hybrid`** | **`99/100` (`99.00%`)** | `29, 1` (`+28`) | `5.77e-08` (`8.08e-07`) | `0/50` / `0/50` | `0 / 0` | `6` excl. -> `93/94` (`98.94%`, `+26`) | `2` / `2` |
| **`Qwen2.5-14B BF16` (`N = 100`)** — `Arm 1` Baseline | `65/100` (`65.00%`) | Ref | Ref | `1/50` / `0/49` | `8 / 0` | `9` excl. -> `63/91` (`69.23%`, Ref) | `0` / `0` |
| `Qwen2.5-14B BF16` — Tier 1 `arm3a_prompt_raw` (`quarantinedPreFixProvenance`) | `28/100` (`28.00%`) | `13, 50` (`-37`) | `3.02e-06` (`2.71e-05`) | `50/50` / `49/49` | `3 / 59` | `9` excl. -> `25/91` (`27.47%`, `-38`) | `0` / `0` |
| `Qwen2.5-14B BF16` — Tier 1 `arm3a_structural_only` | `79/100` (`79.00%`) | `15, 1` (`+14`) | `5.19e-04` (`1.56e-03`) | `1/50` / `0/49` | `6 / 0` | `9` excl. -> `76/91` (`83.52%`, `+13`) | `0` / `0` |
| `Qwen2.5-14B BF16` — Tier 2 `arm3b_structural_gate` | `78/100` (`78.00%`) | `13, 0` (`+13`) | `2.44e-04` (`9.77e-04`) | `1/50` / `0/49` | `8 / 0` | `9` excl. -> `75/91` (`82.42%`, `+12`) | `0` / `0` |
| `Qwen2.5-14B BF16` — Tier 3 `arm3b_llm_repair` | `72/100` (`72.00%`) | `7, 0` (`+7`) | `1.56e-02` (`1.56e-02`) | `1/50` / `0/49` | `2 / 22` | `9` excl. -> `70/91` (`76.92%`, `+7`) | `35` / `28` |
| `Qwen2.5-14B BF16` — **Tier 3 `arm3ab_hybrid`** | **`86/100` (`86.00%`)** | `22, 1` (`+21`) | `5.72e-06` (`4.01e-05`) | `1/50` / `0/49` | `0 / 0` | `9` excl. -> `83/91` (`91.21%`, `+20`) | `21` / `15` |
| **`GLM-4.7-Flash 30B-A3B BF16` (`N = 200` Generalization)** — `Arm 1` | `102/200` (`51.00%`) | Ref | Ref | `0/100` / `0/100` | `73 / 2` | `8` excl. -> `102/192` (`53.13%`, Ref) | `0` / `0` |
| `GLM-4.7-Flash BF16` — Tier 1 `arm3a_prompt_raw` & `structural_only` | **`165/200` (`82.50%`)** | `63, 0` (**`+63`**) | `2.17e-19` (`3.69e-18`) | `0/100` / `0/100` | `25 / 0` | `8` excl. -> `162/192` (`84.38%`, `+60`) | `0` / `0` |
| `GLM-4.7-Flash BF16` — Tier 2 `arm3b_structural_gate` | `112/200` (`56.00%`) | `10, 0` (`+10`) | `1.95e-03` (`3.91e-03`) | `0/100` / `0/100` | `73 / 2` | `8` excl. -> `112/192` (`58.33%`, `+10`) | `0` / `0` |
| `GLM-4.7-Flash BF16` — Tier 3 `arm3b_llm_repair` | `126/200` (`63.00%`) | `24, 0` (`+24`) | `1.19e-07` (`1.45e-06`) | `0/100` / `0/100` | `54 / 1` | `8` excl. -> `124/192` (`64.58%`, `+22`) | `98` / `90` |
| `GLM-4.7-Flash BF16` — **Tier 3 `arm3ab_hybrid`** | **`174/200` (`87.00%`)** | `72, 0` (**`+72`**) | `4.24e-22` (`8.47e-21`) | **`0/100`** / **`0/100`** | **`15 / 0`** | `8` excl. -> **`169/192` (`88.02%`, `+67`)** | **`35` / `30`** |
| **`Qwen/Qwen3.8-27B BF16` (`N = 50`, `seed = 20261151` Control)** | `47/50` (`94.00%`) | `0, 0` (`0`) | Saturation Control (`p = 1.0`, `K3 < 10%`) | `0/25` / `0/25` | `0 / 0` (`ARITH 3`) | `6` excl. -> `44/44` (`100.0%`; `47/50` across all `arm3` primary arms) | `0` (`llm_repair 3`) |

### 3.3 Tier 4 Secondary Symbolic Solver-Offload Ceilings (`is_primary_k2_eligible: false`)

| Secondary Reference Arm (`is_primary_k2_eligible: false`) | Symbolic Offload Mechanism | `N = 800 NF4` (`Phi-4` · `Qwen14B` · `Mistral24B`) | `BF16 N = 100` (`Phi-4` · `Qwen14B`) | `GLM-4.7-Flash BF16` (`N = 200`) | `Qwen3.8-27B BF16` (`N = 50`) |
| --- | --- | --- | --- | --- | --- |
| `arm3a_full_ceiling` | Overwrites Turn-2 target value with exact `Fraction` + `SI` DAG recomputation | `800 / 800` (`100.00%`) on all `3` models | `100 / 100` (`100.00%`) on both | `200 / 200` (`100.00%`) | `50 / 50` (`100.00%`) |
| `arm3b_symbolic_ceiling` | Substitutes exact symbolic target and `preserved` subgraph values post-Pass-2 | `800 / 800` (`100.00%`) on all `3` models | `100 / 100` (`100.00%`) on both | `200 / 200` (`100.00%`) | `50 / 50` (`100.00%`) |

### 3.4 Track 1C Non-Pooled Frontier `v0.2` Eviction & `P1.2` Confirmatory `v0.3` Value-Aliasing Proof

#### Table 3.4A — Exploratory `v0.2` Context Eviction (`seed = 20261151`, `N = 50` Per Cell, Non-Pooled Inference)

| Pinned Frontier Snapshot (`Paid Batch`, `seed = 20261151`) | Eviction Branch (`contents[0]` Evicted) | `ITT` `PASS` (`%`) | `Clean-T1` `PASS` (`T1 = 50/50`) | Independent Per-Cell `STALE_STATE_RETENTION` (`2s 95% CP` Upper) | `CONDITIONAL_ARITH_ERROR` (Episode IDs) | `Cond A` Fail (`25`) |
| --- | --- | --- | --- | --- | --- | --- |
| `models/gemini-4-carbon` | `C1` (`self_summary_eviction`) | **`49 / 50` (`98.0%`)** | `49 / 50` (`98.0%`) | **`0 / 50` (`0.0%`, upper `7.11% < 10.0%`)** | `1 / 50` (`gb-0048`, Cond E, err `1460`) | `0 / 25` |
| `models/gemini-4-carbon` | `C2` (`materialized_ledger_eviction`) | **`50 / 50` (`100.0%`)** | `50 / 50` (`100.0%`) | **`0 / 50` (`0.0%`, upper `7.11% < 10.0%`)** | `0 / 50` | `0 / 25` |
| `models/gemini-4-hydrogen` | `C1` (`self_summary_eviction`) | **`48 / 50` (`96.0%`)** | `48 / 50` (`96.0%`) | **`0 / 50` (`0.0%`, upper `7.11% < 10.0%`)** | `2 / 50` (`gb-0032` err `1808`, `gb-0048` err `1460`, Cond E) | `0 / 25` |
| `models/gemini-4-hydrogen` | `C2` (`materialized_ledger_eviction`) | **`49 / 50` (`98.0%`)** | `49 / 50` (`98.0%`) | **`0 / 50` (`0.0%`, upper `7.11% < 10.0%`)** | `1 / 50` (`gb-0034`, Cond B, err `816`) | `0 / 25` |

- **Non-Pooled Statistical Rule**: Because the `4` cells share the same `50` episodes and per-model Turn-1 completions, inferential rejection of `STALE_STATE_RETENTION >= 10.0%` rests strictly on each independent cell's **`0 / 50` (`95% Clopper-Pearson` upper bound `7.11% < 10.0%`)** (`0` across `200` total cell evaluations reported solely as a descriptive summary). Episode `gb-0048` (`p1 == p4 = 43 g`, `n = 4` total errors in `v0.2`) served as an exploratory hypothesis-generating observation tested directly in `P1.2` below.

#### Table 3.4B — Confirmatory `P1.2` Pre-Registered `v0.3` Value-Aliasing 3-Way Proof (`Track 1C-Alias`, `seed = 20261153`, Prelock `d937ff16...`)

| Frontier Model & `v0.3` Branch (`100%` `p1 == p4`) | `ITT` `PASS` (`%`, `95% CP CI`) | `Clean-T1` `PASS` (`T1 = 100%`) | `VALUE_ALIASING_CONFUSION` (`Mode`) | 4-Way (`PARSE / ARITH / STALE / OVER`) | `Cond A` Fail | Exact Paired McNemar (`C2_v2` vs `C1`) |
| --- | --- | --- | --- | --- | --- | --- |
| `models/gemini-4-carbon` (`N = 64`) — `C1` (`self_summary_eviction`) | **`35 / 64` (`54.69%`, `[41.75%, 67.18%]`)** | `35 / 64` (`54.69%`) | **`14 / 64`** (`aliased_both_retracted`) | `3 / 25 / 1 / 0` (`MAX_TOKENS = 3`) | `0 / 16` | Reference (`C1`) |
| `models/gemini-4-carbon` (`N = 64`) — `C2_v1` (Archived Values-Only, Empty Rules) | **`39 / 64` (`60.94%`, `[47.93%, 72.90%]`)** | `39 / 64` (`60.94%`) | **`4 / 64`** (`aliased_both_retracted`) | `11 / 13 / 1 / 0` (`MAX_TOKENS = 11`) | `0 / 16` | Ablation Control (Empty Rules) |
| `models/gemini-4-carbon` (`N = 64`) — **`C2_v2` (`materialized_ledger_eviction`)** | **`64 / 64` (`100.0%`, `[94.40%, 100.0%]`)** | `64 / 64` (`100.0%`) | **`0 / 64`** | `0 / 0 / 0 / 0` (`MAX_TOKENS = 0`) | `0 / 16` | **`b = 29, c = 0`, `p = 3.73e-09`** (`ITT` & `B..E`) |
| `models/gemini-4-hydrogen` (`N = 32` Stratified) — `C1` (`self_summary_eviction`) | **`17 / 32` (`53.13%`, `[34.74%, 70.91%]`)** | `17 / 32` (`53.13%`) | **`10 / 32`** (`aliased_both_retracted`) | `0 / 15 / 0 / 0` (`MAX_TOKENS = 0`) | `0 / 8` | Reference (`C1`) |
| `models/gemini-4-hydrogen` (`N = 32` Stratified) — **`C2_v2` (`materialized_ledger_eviction`)** | **`32 / 32` (`100.0%`, `[89.11%, 100.0%]`)** | `32 / 32` (`100.0%`) | **`0 / 32`** | `0 / 0 / 0 / 0` (`MAX_TOKENS = 0`) | `0 / 8` | **`b = 15, c = 0`, `p = 6.104e-05`** (`ITT` & `B..E`) |

- **Transparent `v0.2` `"derived"` vs `"derive"` Typo Disclosure & Symbolic-Rule Ablation**: `scripts/run_frontier_eviction_v02.py` originally checked `t.kind == "derived"` instead of `"derive"`, leaving `turn1_derivation_rules` empty in `v0.2` `C2` and initial `carbon` `P1.2` `C2_v1` (sealed `manifests/frontier-eviction-v02-*.json` remain untouched). Fixing to `t.kind in ("derive", "derived")` in `C2_v2` populated explicit DAG derivation rules and yielded a decisive 3-way symbolic-rule ablation: primitive values alone (`C2_v1`: `39/64`, `MAX_TOKENS = 11`) fail under value aliasing almost as badly as prose self-summaries (`C1`: `35/64`), whereas explicit symbolic derivation edges (`C2_v2`) achieve **`100.0%` (`64 / 64` and `32 / 32`, `0` aliasing confusion, `FINISH_REASON_MAX_TOKENS = 0`)**.
- **Non-Pooled Stratified Reporting (`hydrogen` `N = 32` vs `carbon` `N = 64`)**: `models/gemini-4-hydrogen` ran on a deterministic `1:2` proportional condition-stratified subset (`N = 32`: `8 A, 6 B, 6 C, 6 D, 6 E`) inside the `$5.00 USD` cap (`$2.909901 USD` final spend). Per-model metrics, wider `hydrogen` `95% Clopper-Pearson` lower bound (`0.891119` vs `carbon` `0.943991`), and exact paired McNemar `p`-values (`6.104e-05` and `3.73e-09`) are reported separately and never cross-pooled with equal weight.

---

## 4. Five Reconciled Technical Footnotes & Explicit Non-Claims

1. **`Phi-4 Clean-T1 Holm p = 4.18e-72` Step-Down Monotonicity Tie**: Exact mathematical clamping under Holm-Bonferroni (`m = 30`). Rank `1` is `arm3ab_hybrid` (`b = 250, c = 1, b - c = +249`, `raw p = 1.39284126e-73`, `30 * raw p = 4.17852379e-72`); rank `2` is `arm3b_llm_repair` (`b = 243, c = 0, b - c = +243`, `raw p = 1.41494986e-73`, `29 * raw p = 4.10335458e-72`), clamped via `max(4.17852379e-72, 4.10335458e-72) = 4.17852379e-72` (`4.18e-72`).
2. **`Phi-4` & `GLM-4.7-Flash` `arm3a` No-Op Identity at `PARSE_REJECT = 0`**: `arm3a_structural_only` strips only markdown fences and out-of-schema keys without touching candidate `value` or `unit`. At `PARSE_REJECT = 0` (`Phi-4 NF4 0/800`, `Phi-4 BF16 0/100`, `GLM-4.7-Flash BF16 0/200`, W2-patched `Qwen2.5-14B NF4 0/800`), `arm3a_prompt_raw` and `arm3a_structural_only` are item-for-item identical, whereas fence-heavy `Mistral-24B NF4` (`PARSE_REJECT 100 -> 3`, `655 -> 676`) and pre-patch `Qwen2.5-14B BF16` (`PARSE_REJECT 59 -> 0`, `28 -> 79`) diverge as designed.
3. **`Qwen2.5-14B NF4` `STALE 85 -> 8` vs `S+O 92` (`84` Turn-1 Carryover `OVER`)**: `Arm 1 ITT` has `STALE 85 + OVER 92 = S+O 177`; `arm3ab_hybrid ITT` has `STALE 8 + OVER 84 = S+O 92` (`Delta = -85`). On `Clean-T1` (`n = 686`), `arm3ab_hybrid` achieves `STALE 8 + OVER 0 = S+O 8` (`Cond A 0/394`), confirming all `84` residual `ITT` `OVER_RETRACTION` episodes are Turn-1 carryover failures.
4. **`Qwen2.5-14B NF4` Standalone `arm3b_llm_repair` `PARSE_REJECT = 250` Negative Result**: Without `Arm 3A` pre-turn topology anchoring or post-repair structural hygiene, `Qwen2.5-14B` `arm3b_llm_repair` (`ITT`: `PASS 440, PARSE_REJECT 250, OVER_RETRACTION 83, CONDITIONAL_ARITH_ERROR 26, STALE_STATE_RETENTION 1`, `379` extra turns; `Clean-T1`: `PASS 438, PARSE_REJECT 235, ARITH 13, STALE 0`, `267` extra turns) eliminates stale retention (`85 -> 1`) but breaks standalone JSON formatting via verbose self-repair prose.
5. **`GLM-4.7-Flash Arm 1` Symmetric `--assistant-prefill '{"notes": "'` & `</think>` Strip Parity**: Verified in `_run_subbatch` (`scripts/run_arm3_colab_gpu.py` L391-422, L1205) across `Turn 1`, `Arm 1`, `Arm 3A`, `Arm 3B`, and `Arm 3AB-Hybrid` (`Arm 1 PARSE_REJECT = 2/200`, `STALE = 73/200`), ruling out prefill asymmetry. Pre-patch `qwen14b-bf16-n100` `arm3a_prompt_raw` (`28/100`) is documented in `quarantinedPreFixProvenance`, and `GLM-4.7-Flash` (`N = 200`) is reported as a `2026` sparse-MoE `BF16` generalization cohort.
6. **Explicit Non-Claim (`DSL-Schema Scope`)**: `Arm 3` proves deterministic revision over structured tool/state-ledger DSL workflows (`Turn k.i [id]:` headers + `fix->`/`err->`/`sup->` + JSON state envelopes), **NOT** unconstrained free-form natural-language multi-turn chat.

---

## 5. Completed Wave 1 to Wave 3 Verification Ledger (`P0` / `P1` / `P2`)

All follow-up controls proposed by the 6-model blind cross-review have been executed and sealed within the `5%` budget gates (`AI Studio $2.909901 / $100.00 USD <= $5.00 USD`, `Colab 118.85 / 3,251.14 CU <= 162.56 CU`, `Active assignments: 0`):

| Verification Item | Target Script / Manifest | Scope & Hypothesis | Verified Final Empirical Readout |
| --- | --- | --- | --- |
| **`P0` Offline `apply_structural_gate_only` Non-E `"preserved"` Strip Replay (`0.00 CU`)** | `gyeol_bench/arm3_state_diff.py` & `tests/test_arm3_state_diff.py` (`31/31` unit tests `OK`) | Strip spurious non-`Condition E` `"preserved"` keys in `Tier 2` `arm3b_structural_gate` without touching `value`/`unit`; replay all `2,400` saved `NF4` + `100` `BF16` episodes. | **`+0` pass-count change across all `2,500` episodes** (`qwen14b-nf4` `ITT 486/800`, `Clean-T1 484/686`, `Clean-T1 Cond A` `3/394` confirmed as arithmetic errors `gb-0113`, `gb-0639`, `gb-0641`; `phi4-nf4` `626/800`; `mistral24b-nf4` `650/800`; `qwen14b-bf16` `78/100`), with **`1` non-pass taxonomy flip on `qwen14b-nf4` `Condition B` (`gb-0026`: `PARSE_REJECT` -> `STALE_STATE_RETENTION`, `PARSE_REJECT 3 -> 2`)**. |
| **`P1.1` Frontier Same-Harness Standard `Arm 1` Control (`seed = 20261122`, `N = 50`)** | `scripts/run_p1_frontier_controls.py` & `manifests/p1-frontier-arm1-control-results.json` (`f08bf8dc...`) | Run exact 2-live-turn `Arm 1` (`V3_MULTITURN_BASELINE_SYSTEM_PROMPT`, `25 A, 7 B, 6 C, 6 D, 6 E`) on `models/gemini-4-carbon` and `models/gemini-4-hydrogen` to close cross-track seed/harness confounding. | **`50 / 50` (`100.0%` `ITT` & `Clean-T1`, `95% CP CI [92.89%, 100.0%]`, `STALE = 0`, `MAX_TOKENS = 0`) on BOTH `carbon` (`$0.031592`) and `hydrogen` (`$0.181147`)**, confirming frontier full-context invariance across `seed = 20261151` and `seed = 20261122`. |
| **`P1.2` Pre-Registered `v0.3` Value-Aliasing Stress Slice (`Track 1C-Alias`, `seed = 20261153`)** | `manifests/p1-v03-aliasing-prelock.json` (`d937ff16...`) & `manifests/p1-frontier-v03-aliasing-results.json` (`1584db92...`, `64/64` `SymPy + Pint` `0` mismatches) | Test `100%` aliased `p4.value == p1.value` & `p4.unit == p1.unit` under `C1` (`self_summary`) vs `C2_v1` (values-only) vs `C2_v2` (explicit derivation rules). | **`carbon` (`N = 64`)**: `C1` **`35/64` (`54.69%`, `VALUE_ALIASING_CONFUSION = 14`)** vs `C2_v1` **`39/64` (`60.94%`, `MAX_TOKENS = 11`)** vs `C2_v2` **`64/64` (`100.0%`, `b = 29, c = 0, exact p = 3.73e-09`)**. **`hydrogen` (`N = 32` stratified)**: `C1` **`17/32` (`53.13%`, `VALUE_ALIASING_CONFUSION = 10`)** vs `C2_v2` **`32/32` (`100.0%`, `b = 15, c = 0, exact p = 6.104e-05`)**. Synced `21` files to `r2:gyeol-bench-checkpoints/stage1/p1-frontier-controls/`. |
| **`P2` `Case 2B` Clean `0.00 CU` Offline Closeout (`2026-11-24` Colab `503` Stockout)** | `manifests/arm3-empirical-k2-results.json` (`615d24bb...`) & `manifests/p2-qwen38-27b-control-results.json` (`4b300bb7...`) | Document `qwen14b-bf16-n100` pre-fix quarantine + ready `t1only` cache and `Qwen3.8-27B` `BF16`/`L4` controls after Colab `A100/H100/G4` `HTTP 503` stockout. | **`0.00 CU` burned (`118.85 / 3,251.14 CU`, `Active assignments: 0`)**. Attached `quarantinedPreFixProvenance` (`28/100` `arm3a_prompt_raw`, `0`-flip patched gate `78/100`, `arm3a_structural_only` `79/100`, `arm3ab_hybrid` `86/100`, `t1only` cache `ffdc749b...`) and consolidated `Qwen3.8-27B BF16` (`47/50 = 94.0%`, `STALE 0`, `p = 1.0`) + `L4 NF4` (`105.291 s/item > 15.0 s/item`) with `status: "deferred_colab_80gb_503_stockout"`. Synced to R2 (`22` + `5` files). |

---

## 6. Concise Reviewer Q&A Summary (`NeurIPS / ICLR`, `UK AISI Inspect Evals`, `Safety Grants`)

- **Q1 (Oracle Circularity)**: *Does Pass-1 residual gating leak answers?* — **No.** Primary matched-compute `Tier 1` (`arm3a_prompt_raw`, `arm3a_structural_only`) uses `0` extra turns and `0` numeric oracle calls (`+231` on `Phi-4`, `+133` on `Mistral-24B`, `+63` on `GLM-4.7-Flash`). `Tier 3` (`arm3ab_hybrid`) uses `ExactRationalOracle` solely as a boolean residual trigger (`SI_ARITH_RESIDUAL_MISMATCH`) with `0` target numeric leakage, while `Tier 4` symbolic ceilings (`800/800`) are isolated (`is_primary_k2_eligible: false`).
- **Q2 (Frontier Saturation Utility)**: *Why include a benchmark where frontier models score `100%` (`50/50`) in full context?* — First, calibrated diagnostics pinpoint the `32%..49%` baseline failure cliff on deployable `14B..24B` and sparse `30B-A3B` open-weight agents (`0` LLM-judge variance via exact `Fraction` + `SymPy + Pint` parity). Second, under mandatory long-horizon context eviction (`Track 1C-Alias` `v0.3`), frontier prose self-summaries (`C1`) drop to **`53.1%..54.7%`** due to numeric value aliasing (`p1 == p4`), whereas materialized symbolic derivation ledgers (`C2_v2`) restore **`100.0%`** (`p = 3.73e-09` and `p = 6.104e-05`).
- **Q3 (Selection Bias & Scope)**: *Does `Clean-T1` or synthetic DSL overstate generality?* — All claims enforce `ITT` primacy (`Qwen2.5-14B K4 ITT 6/400 FAIL` and `Mistral-24B` `205/800` exclusions disclosed upfront) and explicitly bound `Arm 3` to structured tool/state-ledger DSL workflows (`DSL-Schema Scope`).
