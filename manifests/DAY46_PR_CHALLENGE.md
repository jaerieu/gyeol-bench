# Day-46 Cryptographic Baseline Hash Freeze & 14-Day Public GitHub PR Challenge Protocol (`v3.0` Multi-Turn Stateful Handoff)

- **Repository**: `jaerieu/gyeol-bench` (`Apache-2.0`, Track A Clean-Room Benchmark)
- **Primary Task**: `gyeol_bench/retraction_dag_multiturn` (`0.1.0`, `UK AISI Inspect Evals` `2026-05-08` spec, `v3.0` 2-Live-Turn `Shared-Turn-1` Stateful Handoff)
- **Ablation 0 Task (Preserved)**: `gyeol_bench/retraction_dag` (`v2.0` Single-Prompt Static DAG)
- **Pre-Pilot Lock**: `manifests/v3-multiturn-prepilot-lock.json` (`seed = 20261151, N = 50` pilot & `seed = 20261122, N = 800` main evaluation locked prior to live `v3.0` execution)
- **Freeze Gate**: `Day 46` (`14` calendar days prior to `Day 60` `K2 / K3 / K4` evaluation lock)
- **Governance Tier**: `[안전]`

---

## 1. Transparent Pre-Freeze Disclosure: `v1` / `v2.0` Single-Prompt Saturation (`Ablation 0`) & `K3` Redraw Consumption

During pre-freeze `N = 50` pilot diagnostics (`seed = 20261150`), single-prompt presentation (`v1` and `v2.0` `gyeol_bench/retraction_dag`, where all premise turns and later `fix-> / err-> / sup->` retractions appeared simultaneously inside a single user message) exhibited **single-turn scratchpad saturation** on frontier reasoning models (`0 / 50` failure on `gemini-4-carbon`) and arithmetic-dominated errors (`20 / 50` `CONDITIONAL_ARITH_ERROR` vs `1 / 50` `STALE_STATE_RETENTION` on `microsoft/phi-4`).

Root-cause analysis across four independent reviewers confirmed a construct-validity boundary: **silent quantitative premise contamination requires the agent to first commit intermediate derived values (`d1..d4`) into its own assistant conversation state (Turn 1) before an asynchronous follow-up retraction arrives (Turn 2)**. Without a committed Turn-1 assistant state, no stale working-memory commitment exists to contaminate Turn 2.

Accordingly, prior to the Day-46 freeze:
1. **`v2.0` Preserved Intact as `Ablation 0`**: `generate_procedural_dataset` and `gyeol_bench/retraction_dag` remain unmodified in the repository as an explicit `1`-turn control proving that single-prompt presentation eliminates stateful premise retention on frontier models.
2. **Permanent Consumption of `K3` Single Redraw & `Option C` Frontier Saturation Resolution (`v3.0` `seed = 20261151`, `N = 50`)**: Transitioning the primary task to `gyeol_bench/retraction_dag_multiturn` (`v3.0` 2-live-turn `Shared-Turn-1` stateful handoff) **permanently consumed the single pre-registered `K3` redraw allowance**. On `v3.0` (`seed = 20261151`, `N = 50`), pinned frontier snapshots (`gemini-4-carbon` and `gemini-4-hydrogen` under `thinkingLevel="medium"`) achieved **`0 / 50` (`0.00%`) Turn-2 failure** by re-reading `contents[0]` directly in KV attention (`validity/SATURATION_REPORT.md`), while all three held-out `14B..24B` `4-bit NF4` open-weight families (`microsoft/phi-4`, `Qwen/Qwen2.5-14B-Instruct`, `Mistral-Small-24B-Instruct-2501`) passed `Condition A = 25 / 25` (`0.0%` false alarms) and met `K3 >= 10.0%` (`34.0%`, `44.0%`, `20.0%` Turn-2 failure; `12.0%` `STALE_STATE_RETENTION` on `Phi-4` and `Qwen2.5-14B`). Per `Option C`, frontier `2x800` `K2` execution is terminated at `$0.00` incremental spend (reported as a KV Re-Reference Saturation Control; `contents[0]` eviction separated as a post-hoc `v0.2` ablation), and Phase 3 `N = 800` `K2` evaluation is locked onto the `14B..24B` open-weight cohort.

---

## 2. Official Pre-Locked `v3.0` Multi-Turn Prompts (`manifests/v3-multiturn-prepilot-lock.json`)

### 2.1 Unassisted Multi-Turn Baseline System Prompt (`v3.0-multiturn-baseline`)
Unless superseded by an eligible community Pull Request during the 14-day challenge window, every unassisted baseline run uses the exact system prompt below across Turn 1 and Turn 2:

```text
You are a precise multi-turn quantitative reasoning assistant. Across sequential user turns:
1. Track all base premises and their exact units (SI mass ug/mg/g/kg, length mm/cm/m/km, volume uL/mL/L, time ms/s/min/h, USD financial scales USD/K_USD/M_USD/B_USD, percentages %, and basis points bps), ignoring inert context or distractor notes unrelated to the target node.
2. Whenever a retraction step (fix->, err->, or sup->) supersedes an earlier node (including chained retractions such as sup-> p1_r1), record ONLY unique node IDs appearing directly on the left of ':=' in "retracted_ids" sorted alphabetically (never include un-superseded replacement turn IDs or derived d* nodes; output [] if zero retractions occur), and recompute every downstream derived node from earlier turns that depends on it using exact dimensional conversion.
3. Keep any independent derived subgraph (such as d_indep) that has no directed dependency on a retracted node strictly unchanged in "preserved" when requested.
4. On every turn, output ONLY a single standalone valid JSON object starting with "notes" (containing a concise step-by-step calculation and retraction trace) followed by "step_id", "value", "unit", "retracted_ids" (and "preserved" when requested), with zero surrounding prose or markdown fences.
```

### 2.2 Paired Lineage-Guided System Prompt (`v3.0-multiturn-lineage-k2`)
Locked simultaneously in `manifests/v3-multiturn-prepilot-lock.json` under identical model snapshot, token/thinking budget, and single-pass-per-turn `response_adapter` JSON gate:

```text
You are a lineage-verified multi-turn quantitative reasoning engine. Enforce strict dependency-graph taint invalidation across sequential user turns:
1. In Turn 1 "notes", record an explicit ancestor ledger for every derived node (e.g., d1<-{p1,p2}; d2<-{p1,p2,p3}; d3<-{...}; d4<-{...}; d_indep<-{u1,u2,u3}) alongside canonical base-unit values (g, m, L, s, USD, scalar).
2. In Turn 2 "notes", BEFORE evaluating d5 or d6: (a) extract the exact sorted set R of superseded target IDs appearing directly on the left of ':=' in any fix->, err->, or sup-> step (output R=[] if none, never include un-superseded replacement IDs or derived d* IDs); (b) mark as TAINTED every Turn-1 derived node whose ancestor set intersects R, and NEVER reuse a committed Turn-1 numeric value for any TAINTED node; (c) rebuild every TAINTED Turn-1 node bottom-up from the latest active root premises in canonical base units before computing d5 and d6 and converting to the requested output unit; (d) copy any UNTAINTED disjoint node (d_indep) verbatim from Turn 1 into "preserved".
3. On every turn, output ONLY a single standalone valid JSON object starting with "notes" followed by "step_id", "value", "unit", "retracted_ids" (and "preserved" when requested), with zero surrounding prose or markdown fences.
```

---

## 3. Rules & Invariants for Community Baseline PR Submissions (`Days 46..60`)

| Invariant | Requirement | Disallowed (`[금지]`) |
| --- | --- | --- |
| **1. Model & 2-Turn Parity** | Identical pinned model snapshot, `temperature`, `max_output_tokens`, `thinkingBudget`, and 2-live-turn `Shared-Turn-1` handoff (`1` inference call on Turn 1, `1` inference call on Turn 2). | External Python/calculator execution, multi-sample voting, or re-injecting Turn-1 user premises inside Turn 2. |
| **2. Identical Envelope Gate** | Standalone JSON validated by unmodified `response_adapter` (`"notes"` string optional, `PARSE_REJECT` rules unchanged). | Post-processing regex repair or markdown-fence stripping wrappers. |
| **3. Zero Seed Leakage** | Few-shot exemplars must come from disjoint seeds (`seed not in {20261122, 20261150, 20261151, 20261199}`). | Hardcoding values or lookup tables from pilot, validation, or evaluation seeds. |
| **4. Zero Private-Core Contamination** | Passes `python3 scripts/check_zero_core.py` (`[안전]`, `0` violations) and `python3 -m unittest discover -s tests`. | Non-`Apache-2.0` code, runtime dependencies, or forbidden vocabulary. |
| **5. Held-Out Validation Gain** | Strictly improves Turn-2 accuracy (`b_val - c_val >= 1`) on `generate_multiturn_dataset(seed=20261199, count=160)` without increasing `PARSE_REJECT` or `Condition A` false rejections. | Degrading `Condition A` control specificity or Turn-1 checkpoint accuracy. |

---

## 4. Strict Separation Between Day-46 PR Validation (`N = 160`) and Day-60 Final `K2` Lock (`N = 800`)

### 4.1 Stage A: Day-46 to Day-60 PR Validation Regime (`seed = 20261199`, `N = 160`)
- Total episodes `N_val = 160` (`A=80, B=20, C=20, D=20, E=20`), so discordant pairs satisfy `0 <= b_val + c_val <= 160`.
- Highest-accuracy eligible PR with `b_val - c_val >= 1` over `v3.0-multiturn-baseline` becomes the **Locked Unassisted Baseline** (ties broken by lower `PARSE_REJECT`, then two-sided exact McNemar `p` over `b_val + c_val <= 160` logged as diagnostic, then earliest PR timestamp).

### 4.2 Stage B: Day-60 Final `K2 / K3 / K4` Evaluation Lock (`seed = 20261122`, `N = 800` · `14B..24B` Open-Weight Primary Cohort)
- **Locked Population Hierarchy (Stratified Per Model Family, Unmodified `v3.0` `oracle.py` Scorer)**:
  1. **Primary (`ITT`, `N = 800`: `A=400, B..E=400`)**: Governs binary `K2`, `K3`, and `K4` pass/fail per model under `Shared-Turn-1` (`raw1` shared identically across Baseline Arm 1 and Lineage-Guided Arm 2).
  2. **Sensitivity (`Clean-T1`, `turn1_ok == True`)**: Co-reported alongside `ITT` to isolate Turn-2 belief-revision and stale-state retention free of Turn-1 baseline arithmetic or format noise.
- **Co-Primary `K2` Outcomes**:
  1. Overall Turn-2 accuracy gain `(b - c) / 800 >= +0.05` (`b - c >= 40`) **AND** two-sided exact binomial McNemar `p < 0.05` (`b + c <= 396` at `b - c = 40`), backed by `100%` discordant pair manual audit + `N = 50` concordant sample.
  2. Co-primary `STALE_STATE_RETENTION + OVER_RETRACTION` episode count change between Arm 1 (`v3.0-multiturn-baseline`) and Arm 2 (`v3.0-multiturn-lineage-k2`), stratified per model family.
- **`K3` (`Non-Saturation`, `0` remaining redraws)**: Top held-out open-weight model failure rate `>= 10%` (frontier reported separately as `Option C` KV Re-Reference Saturation Control).
- **`K4` (`Condition A`, `N = 400`)**: Unperturbed normal-control false alarms `<= 5 / 400` (`<= 1.25%`, two-sided 95% Clopper-Pearson upper bound `2.8928% < 5.0%`).
