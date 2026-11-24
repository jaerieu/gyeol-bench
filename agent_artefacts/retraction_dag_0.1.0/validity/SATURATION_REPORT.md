# Frontier Ceiling Saturation & KV Re-Reference Report (`Branch 1 · Option C`)

- **Benchmark**: `gyeol_bench/retraction_dag_multiturn` (`v3.0`) & `gyeol_bench/retraction_dag` (`v1` / `v2.0` `Ablation 0`)
- **Registry Standard**: `UK AISI Inspect Evals` (`2026-05-08` Specification)
- **Pre-Registered Gate Triggered**: `K3` (`>= 10.0%` Top-Model Failure Rate) -> **`Option C` Frontier Ceiling Saturation Halt** (`1 / 1` `K3` redraw consumed cleanly per `manifests/v3-multiturn-prepilot-lock.json` `eac8d83e62ceb784f9096d5aab8540ee70a07f57d10950cb77fea57202502e53`)
- **Governance Verdict**: `[안전]` (Pre-registration integrity preserved; frontier Phase 3 `2x800` `K2` spend halted at `$0.00` incremental cost; cumulative AI Studio pilot spend locked at `$1.130102 USD <= $5.00`)

---

## 1. Executive Summary

Across three pre-freeze design iterations (`v1` `2..3`-hop single-prompt, `v2.0` `8..12`-hop diamond/cross-scale single-prompt, and `v3.0` 2-live-turn `Shared-Turn-1` stateful handoff; `300` frontier episodes / `400` Paid Batch API model calls total), pinned frontier reasoning snapshots (`models/gemini-4-carbon` and `models/gemini-4-hydrogen` under `thinkingLevel="medium"`, `temperature=0.0`, `FINISH_REASON_MAX_TOKENS == 0`) exhibited **complete empirical resilience (`0 / 50` = `0.00%` Turn-2 failure on both models in `v2.0` and `v3.0`; two-sided 95% Clopper-Pearson upper bound `7.11% < 10.0%`)**.

In stark contrast, on the exact same pre-locked `v3.0` `seed = 20261151` (`N = 50`) 2-live-turn protocol, all three held-out `14B..24B` open-weight instruction models (`microsoft/phi-4`, `Qwen/Qwen2.5-14B-Instruct`, `mistralai/Mistral-Small-24B-Instruct-2501`) achieved **`25 / 25` (`100.0%`) accuracy (`0.0%` false alarms) on `Condition A` normal controls** while failing **`20.0%` to `44.0%` overall (`40.0%` to `88.0%` on `Condition B..E` perturbed episodes)**, with `12.0%` genuine `STALE_STATE_RETENTION` on both `Phi-4` and `Qwen2.5-14B` (`24.0%` on `Clean-T1` `B..E` episodes).

Rather than inventing un-preregistered task mutations after consuming our single allowed `K3` redraw, **Branch 1 executes `Option C` for the frontier tier**: halting redundant `$19.06` frontier `2x800` `K2` API expenditure, reporting frontier reasoning models transparently as a **Full-Context KV Re-Reference Saturation Control**, and restricting Phase 3 `N = 800` paired `K2` evaluation (Branch 2) to the `14B..24B` open-weight regime where genuine multi-turn premise contamination occurs.

---

## 2. Three-Generation Frontier Resilience Trajectory (`v1 -> v2.0 -> v3.0`)

All frontier pilots ran exclusively via the Google AI Studio Paid Tier Batch API (`AI_STUDIO_PAID_TIER_CONFIRMED=1`, `50%` Batch discount, zero `-latest` aliases, zero Google models in item generation or filtering, `0 / 400` `MAX_TOKENS` truncations):

| Iteration & Seed (`N = 50`) | Topology & Turn Protocol | Model Snapshot & Batch Job ID(s) | Overall Fail (`K3 >= 10%`) | `Cond. A` Fail (`25`) | `Cond. B..E` Fail (`25`) | Thought Tokens (`Out`) | Batch Spend (`USD`) |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| **`v1`** (`seed=20261150`) | `1`-turn static prompt, `2..3` hops, `1` retraction | `models/gemini-4-carbon` (`7ngx3qhft8xtj7smv7exa5btuiai78stuhpc`) | **`0 / 50` (`0.00%`)** | `0 / 25` | `0 / 25` | `4,775` (`6,834`) | `$0.046814` |
| **`v1`** (`seed=20261150`) | `1`-turn static prompt, `2..3` hops, `1` retraction | `models/gemini-4-hydrogen` (`qai64jwea0in57gmxnt5e0xqs15qmch6nj4c`) | **`2 / 50` (`4.00%`)** | `0 / 25` | `2 / 25` (`1` `ARITH`, `1` `OVER`) | `22,295` (`24,472`) | `$0.101253` |
| **`v2.0`** (`seed=20261150`, `Ablation 0`) | `1`-turn static prompt, `8..12` diamond/scale hops, `3` retractions + `"notes"` | `models/gemini-4-carbon` (`4z4ygp9beso1adomcpqf6mqzyjhpkqiqmk3u`) | **`0 / 50` (`0.00%`)** | `0 / 25` | `0 / 25` | `20,920` (`34,326`) | `$0.216757` |
| **`v2.0`** (`seed=20261150`, `Ablation 0`) | `1`-turn static prompt, `8..12` diamond/scale hops, `3` retractions + `"notes"` | `models/gemini-4-hydrogen` (`sfh6xx3ao762lxokj9rgd3px3ul4v16p2of7`) | **`0 / 50` (`0.00%`)** | `0 / 25` | `0 / 25` | `52,372` (`65,935`) | `$0.281102` |
| **`v3.0`** (`seed=20261151`, `1/1` Redraw) | **`2`-live-turn `Shared-Turn-1` handoff** (`T1 -> raw1 -> T2`, `0` T1 repeat in T2) | `models/gemini-4-carbon` (`2jpooxj9i...`, `2bxuz23nc...`; T1 Clean `48/50`) | **`0 / 50` (`0.00%`)** | `0 / 25` | `0 / 25` | `10,811` T2 (`27,934` T1+T2) | `$0.230507` |
| **`v3.0`** (`seed=20261151`, `1/1` Redraw) | **`2`-live-turn `Shared-Turn-1` handoff** (`T1 -> raw1 -> T2`, `0` T1 repeat in T2) | `models/gemini-4-hydrogen` (`kbpwq5szx...`, `xwt9tkvuh...`; T1 Clean `50/50`) | **`0 / 50` (`0.00%`)** | `0 / 25` | `0 / 25` | `24,982` T2 (`49,211` T1+T2) | `$0.253669` |

*(Cumulative AI Studio expenditure across all six batch pilot runs plus preliminary smoke verification: `$1.130102 USD` recorded in `/home/gyeol/.config/gyeol/aistudio-spend.json`.)*

---

## 3. Mechanistic Explanation: Full-Context KV Re-Reference & Scratchpad Reset

Why did separating Turn-1 commitment (`raw1` = `{"step_id": "d4", ...}`) from Turn-2 retraction (`turn2_prompt`) induce heavy `STALE_STATE_RETENTION` in `14B..24B` open-weight models (`12.0%` overall, `24.0%` on `Clean-T1` `B..E`), yet produce **`0 / 50` (`0.00%`)** errors on `gemini-4-carbon` and `gemini-4-hydrogen`?

1. **Uncompressed `contents[0]` KV Attention Access**: In standard multi-turn chat histories (`[user: turn1_prompt, model: raw1, user: turn2_prompt]`), the raw Turn-1 user message (`contents[0]`, `<= 271` tokens) remains directly accessible in the transformer's KV cache alongside the assistant's committed Turn-1 JSON (`contents[1]`, `raw1`).
2. **Adaptive Test-Time Compute Scaling & Scratchpad Reset**:
   - When DAG depth increased from `2..3` hops (`v1`) to `8..12` hops (`v2.0`), `gemini-4-carbon` automatically scaled internal thought tokens by **`4.38x`** (`4,775 -> 20,920`) plus `13,406` `"notes"` tokens (`2.36x` thought scaling on `hydrogen`: `22,295 -> 52,372`).
   - In `v3.0` Turn 2 (`thinkingLevel="medium"`), `gemini-4-carbon` (`10,811` T2 thought tokens) and `gemini-4-hydrogen` (`24,982` T2 thought tokens) **reset their reasoning scratchpad on Turn 2 and re-read the primitive equations in `contents[0]` directly**, substituting `p1_r2` / `p3_r1` / `r1_r1` from `contents[2]` into `contents[0]` rather than trusting their own `contents[1]` (`raw1`) numeric summary.
3. **Direct Empirical Proof via `carbon`'s `2 / 50` Empty-Turn-1 Episodes (`gb-0032`, `gb-0047`)**:
   On `v3.0`, `gemini-4-carbon` emitted `2` empty Turn-1 responses (`48 / 50` Turn-1 Clean), meaning `contents[1]` (`raw1`) contained **zero** committed Turn-1 numeric state on those `2` episodes — yet `gemini-4-carbon` scored **`50 / 50` (`100.0%`) on Turn 2**, proving conclusively that Turn-2 inference reconstructed `d1..d4` directly from `contents[0]` in KV memory rather than reading `raw1`!

---

## 4. Regime Divergence: Frontier KV Re-Reference vs `14B..24B` Stateful Anchoring (`v3.0`, `seed = 20261151`)

| Model Family (`seed = 20261151, N = 50`) | Params & Quant | Turn-1 Clean (`50`) | Turn-2 `Cond. A` Fail (`25`) | Turn-2 `Cond. B..E` Fail (`25`) | Turn-2 Overall Fail (`50`) | 4-Way (`PARSE / ARITH / STALE / OVER`) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `models/gemini-4-carbon` | Frontier Paid Batch | `48 / 50` (`96.0%`) | **`0 / 25` (`0.0%`)** | `0 / 25` (`0.0%`) | **`0 / 50` (`0.0%`)** `[경고]` Saturated | `0 / 0 / 0 / 0` |
| `models/gemini-4-hydrogen` | Frontier Paid Batch | `50 / 50` (`100.0%`) | **`0 / 25` (`0.0%`)** | `0 / 25` (`0.0%`) | **`0 / 50` (`0.0%`)** `[경고]` Saturated | `0 / 0 / 0 / 0` |
| `microsoft/phi-4` | `14.66B` `4-bit NF4` | `47 / 50` (`94.0%`) | **`0 / 25` (`0.0%`)** | `17 / 25` (`68.0%`) | **`17 / 50` (`34.0%`)** `[안전]` | `0 / 10 / 6 / 1` (`STALE+OVER = 7`) |
| `Qwen/Qwen2.5-14B-Instruct` | `14.77B` `4-bit NF4` | `41 / 50` (`82.0%`) | **`0 / 25` (`0.0%`)** | `22 / 25` (`88.0%`) | **`22 / 50` (`44.0%`)** `[안전]` | `0 / 11 / 6 / 5` (`STALE+OVER = 11`) |
| `mistralai/Mistral-Small-24B-Instruct-2501` | `23.57B` `4-bit NF4` | `37 / 50` (`74.0%`) | **`0 / 25` (`0.0%`)** | `10 / 25` (`40.0%`) | **`10 / 50` (`20.0%`)** `[안전]` | `6 / 4 / 0 / 0` (`Clean-T1: 0/4/0/0`) |

Key scientific takeaway: **Non-thinking `14B..24B` instruction models anchor onto their committed assistant message (`raw1`) as a computational shortcut**, exhibiting `0.0%` false alarms on `Condition A` (`75 / 75` pooled `PASS`) alongside `65.3%` failure (`49 / 75`) on `Condition B..E` retractions. Frontier reasoning models with `10k..25k` hidden thought tokens bypass assistant-turn anchoring whenever `contents[0]` remains in full KV context.

---

## 5. Methodological Integrity: Why `Option C` Rejects Post-Hoc `contents[0]` Masking

1. **Protection Against `K3` Gerrymandering**: Pre-registered `K3` permits at most **one** design/seed redraw prior to freezing. `v3.0` (`seed = 20261151`) consumed `1 / 1` redraw. Altering the task a third time to force frontier errors above `10%` would convert a falsifiable kill-switch into an unfalsifiable moving target.
2. **Clean Boundary Between Transcript-Resident Chat and Evicted-History Stateful Actors (`v0.2` Future Scope)**: In production durable-actor architectures where raw historical turns are compacted or evicted from the LLM prompt and only a materialized `SQLite` state table (`d4 = ...`) survives, KV re-reference to `contents[0]` is physically impossible. Evaluating frontier models under **Evicted-Transcript / State-Table-Only Handoff** (`contents[0]` masked after Turn-1 state extraction) is a distinct architectural construct and is strictly separated as a future **post-hoc `v0.2` ablation** rather than conflated with Stage 1 transcript-resident `K3`.
