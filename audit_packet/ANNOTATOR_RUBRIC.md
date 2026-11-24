# External Blinded Human Ground-Truth Audit Rubric (`v2.0` · `N = 160`, `>= 150`)

- **Benchmark**: `gyeol_bench/retraction_dag` (`0.1.0`, `Apache-2.0`, `v2.0` Escalated `8..12`-Hop Multi-Retraction Diamond & Cross-Scale DAGs)
- **Registry Standard**: `UK AISI Inspect Evals` (`2026-05-08`)
- **Audit Sample Size**: `N = 160` stratified episodes (`40` unperturbed `17`-turn controls + `120` perturbed episodes across `8..12`-hop diamond, cross-branch, `3`-scale SI/USD, and `3`-hop `d_indep` preservation topologies, shuffled in blinded order; exceeds the `N >= 150` pre-registered gate)
- **Pre-Registered Error Bound**: `0 / 150` verified oracle defects yields a `1.9773%` (`1.98%`) one-sided 95% upper bound (`1 - 0.05^(1/150)`); `0 / 160` yields a `1.8557%` (`1.86%`) one-sided 95% upper bound
- **Governance Tier**: `[안전]`

> **Why Human Audit Is Strictly Separated from `SymPy + Pint` Validation**:
> Automated cross-oracle verification (`SymPy 1.14.0 + Pint 0.26.2` across `800 / 800` `v2.0` episodes, `0` discrepancies) proves mechanical algebraic and unit-conversion parity between two independent codebases. It does **not** prove natural-language prompt clarity or construct validity for human readers. External human auditors evaluate the exact rendered text prompts **blinded** to oracle answers and condition labels.

## 1. Blinding & Non-Repudiation Protocol

1. **Zero Ground-Truth Leakage in Worksheet**: `blinded_annotator_sheet_n160.jsonl` and `blinded_annotator_sheet_n160.csv` contain only `audit_id`, `episode_id`, `target_node_id`, `expected_unit`, `required_preserved_nodes`, and the exact `prompt` string shown to candidate models. All target values, rational fractions, `retracted_ids`, full/partial stale values, and factorial condition tags (`A..E`) are stripped.
2. **Pre-Committed Sealed Answer Key**: Before distributing worksheets to external reviewers, the exact SHA-256 digest of the canonical out-of-repository answer key (`sealed_answer_key_sha256`) is published in `audit_packet/SEALED_KEY_COMMITMENT.json` and included in the Day-46 RFC 6962 Merkle freeze. This prevents post-hoc answer alteration while preserving strict annotator blinding.
3. **Independent Multi-Reviewer Assignment**: `2` to `3` independent AI safety researchers or `UK AISI Inspect AI` contributors divide the `160` items with at least `30` overlapping items (`audit-001` through `audit-030`) double-annotated to measure inter-annotator agreement (exact numeric + retraction match rate and Cohen's kappa on construct clarity).

## 2. Step-by-Step Solving Rules for Annotators (`v2.0` Multi-Retraction & Diamond DAGs)

Each prompt lists sequential turns (`Turn 1`, `Turn 2`, ..., `Turn 17..28`) defining a directed acyclic calculation graph with inert distractor chains (`x*`, `z*`, `u_near`) interleaved alongside the target chain:

1. **Premise Turn (`[node_id]: value = <number> <unit>`)**:
   Introduces a base numeric quantity in an explicit SI, time, dimensionless, or USD financial scale unit. Ignore inert distractor chains (`x*`, `z*`, `u_near`) that have no directed path to `target_node_id` or `required_preserved_nodes`.
2. **Derivation Turn (`[node_id]: <op>(<left_id>, <right_id>)`)**:
   Defines an arithmetic node (`add`, `sub`, `mul`, `div`) over two prior node IDs (including diamond reconvergence and cross-branch reuse):
   - `add(a, b)` -> `a + b` (operands share the same dimension; convert both to a common unit first).
   - `sub(a, b)` -> `a - b` (operands share the same dimension).
   - `mul(a, b)` -> `a * b` (at least one operand is dimensionless: `scalar`, `%`, or `bps`).
   - `div(a, b)` -> `a / b` (either equal dimensions yielding `scalar`, or division by a clean dimensionless divisor `{2, 4, 5, 8, 10}` yielding an exact terminating decimal).
3. **Retraction Turn (`[new_id]: <fix-> | err-> | sup-> > <target_id> := <value> <unit>`) — Exact `retracted_ids` Convention**:
   Supersedes `<target_id>` with the updated `<value> <unit>` (including chained retractions such as `p1_r1: fix-> p1 := ...` followed later by `p1_r2: sup-> p1_r1 := ...`):
   - **Exact Rule for `annotator_retracted_ids`**: Record **ONLY unique node IDs appearing directly as the superseded target on the left of `:=` in any retraction turn**, sorted alphabetically (for example, `["p1", "p1_r1", "p3"]` or `["p1", "p1_r1", "r1"]`). Do **NOT** include the final un-superseded replacement turn ID (`p1_r2`, `p3_r1`, `r1_r1`) and do **NOT** include derived nodes (`d*`). Record `[]` when zero retraction turns appear.
   - Recompute **every** downstream derivation node along all diamond and cross-branch paths using the final active replacement values and units.
   - Keep any independent subgraph (`u1..u4 -> u_d1 -> u_d2 -> d_indep`) that has **no directed path** from any retracted node completely unchanged (`preserved`).

## 3. Exact Unit & Scale Reference Table

All arithmetic uses exact decimal/rational scaling without intermediate rounding:

| Dimension Family | Supported Tokens & Exact Base-Unit Factor |
| --- | --- |
| **Mass** (base `g`) | `ug` / `mcg` = `10^-6 g` · `mg` = `10^-3 g` · `g` = `1 g` · `kg` = `10^3 g` |
| **Length** (base `m`) | `mm` = `10^-3 m` · `cm` = `10^-2 m` · `m` = `1 m` · `km` = `10^3 m` |
| **Volume** (base `L`) | `uL` = `10^-6 L` · `mL` = `10^-3 L` · `L` = `1 L` |
| **Time** (base `s`) | `ms` = `10^-3 s` · `s` = `1 s` · `min` = `60 s` · `h` / `hr` = `3600 s` |
| **Financial USD** (base `USD`) | `USD` = `1 USD` · `K_USD` / `$K` = `10^3 USD` · `M_USD` / `$M` = `10^6 USD` · `B_USD` / `$B` = `10^9 USD` |
| **Dimensionless** (base `scalar`) | `scalar` / `1` = `1` · `%` / `pct` = `10^-2` · `bps` = `10^-4` |

> **Important**: Financial `M_USD` (`10^6 USD`) and length `m` (`meter`) belong to strictly disjoint dimension families. Percentage (`%` = `0.01`) and basis points (`100 bps` = `1%` = `0.01`) act as exact dimensionless multipliers when combined via `mul`.

## 4. Completing Each Row & Clarity Verdicts

For each row (`audit-001` through `audit-160`), fill in:

- **`annotator_id`**: Reviewer handle or initials (e.g., `rev-1`).
- **`annotator_value`**: Exact terminating decimal or rational string for `target_node_id` expressed in `expected_unit`. Mixed equivalence tolerance during reconciliation is `|a - b| <= 1e-4 + 1e-5 * max(|a|, |b|)`.
- **`annotator_unit`**: Copy `expected_unit` after confirming dimensional validity.
- **`annotator_retracted_ids`**: JSON array of sorted superseded target node IDs appearing directly on the left of `:=` (e.g., `[]` or `["p1", "p1_r1", "p3"]`).
- **`annotator_preserved_json`**: Leave empty (`""`) when `required_preserved_nodes` is empty; when `required_preserved_nodes` lists `d_indep`, record `{"d_indep": {"value": <number>, "unit": "<expected_unit>"}}`.
- **`construct_clarity_verdict`**: Select exactly one of `SOUND`, `PROMPT_AMBIGUOUS`, `UNIT_AMBIGUOUS`, or `ARITH_DEFECT`.
- **`annotator_notes`**: Mandatory explanation whenever `construct_clarity_verdict != SOUND`.

## 5. Reconciliation & Pre-Registered Gate Resolution

Once all `160` blinded rows are returned and timestamped prior to opening the sealed key:

1. Verify `sha256sum` of the sealed answer key against `SEALED_KEY_COMMITMENT.json` (`sealed_answer_key_sha256`).
2. Compare human-derived values, `retracted_ids`, and `preserved` values against the sealed answer key.
3. If even `1` genuine oracle or prompt-ambiguity defect is confirmed (`defects >= 1`), the generator defect MUST be fixed, re-audited on a fresh seed, and re-anchored before `Phase 3` (`N = 800`) claims `K2` validity.
