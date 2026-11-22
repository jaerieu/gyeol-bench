# Validity Report: `gyeol_bench/retraction_dag` (`0.1.0`)

- **Registry Standard**: `UK AISI Inspect Evals` (`2026-05-08`)
- **Governance Tier**: `[안전]` (Clean-room `Apache-2.0` Track A MVP; `Condition A..E` + `response_adapter` + 4-way error split + `800/800` external `SymPy + Pint` cross-oracle verified; `0` private core symbols)

## 1. Construct Validity, `Condition A..E` Design & 4-Way Error Taxonomy

`gyeol_bench/retraction_dag` isolates multi-turn non-destructive premise retraction (`fix->`, `err->`, `sup->`) coupled with SI unit (`ug`, `mg`, `g`, `kg`, `mm`, `cm`, `m`, `km`, `mL`, `L`, `ms`, `s`, `min`, `hr`) and financial scale (`USD`, `K_USD`, `M_USD`, `B_USD`, `%`, `bps`) re-propagation, motivated by **Silent Quantitative Premise Contamination in Stateful Actors** (durable actors retaining stale derived numbers in `SQLite` memory after asynchronous premise corrections).

### Factorial Episode Conditions (`Condition A..E`)

`gyeol_bench/oracle.py` implements the five balanced factorial conditions (`N = 800`: `400` `Condition A` normal controls + `100` each for `Condition B..E`, paired deterministic seeds):

- **`Condition A` (Baseline Calculation Control)**: Multi-step exact arithmetic with zero premise retractions and zero cross-scale shifts (serves simultaneously as the `K4` `N = 400` normal control specificity benchmark).
- **`Condition B` (Single Premise Retraction)**: Mid-horizon `fix->`, `err->`, or `sup->` value override within a fixed unit scale.
- **`Condition C` (Multi-Hop Numerical Dependency Re-Propagation)**: Downstream cascade across `>= 3` derivation hops (`p1 -> d1 -> d2 -> d3`) testing complete invalidation of stale intermediate nodes.
- **`Condition D` (Simultaneous Retraction & SI Unit / Scale Shift)**: Coupled premise override and dimensional conversion (`mg <-> g`, `K_USD <-> M_USD`, `% <-> bps`).
- **`Condition E` (Unrelated Conclusion Preservation)**: Embeds a retracted dependency chain (`p1 -> d1 -> d2`) alongside a disjoint, independent subgraph (`u1, u2 -> d_indep`) in the same dialogue, testing that the model updates retracted dependents while preserving unrelated prior conclusions (`d_indep`) intact.

### 4-Way Error Taxonomy

To prevent format compliance failures from confounding quantitative reasoning diagnostics, candidate outputs are decomposed into four mutually informative categories:

1. **`PARSE_REJECT` (Envelope Format Non-Compliance)**: Non-JSON prose, missing required keys, boolean values, or non-finite floats rejected deterministically at the schema gate before numeric scoring.
2. **Conditional Arithmetic Error (Value / Dimension Mismatch | Valid Envelope)**: Valid JSON step envelopes that fail dimensional alignment or exceed mixed tolerance `|candidate - truth| <= 1e-4 + 1e-5 * max(|candidate|, |truth|)` without matching stale pre-retraction values (reported alongside `Reasoning Accuracy | Valid Envelope` and unconditional `N = 800` accuracy).
3. **`Stale-State Retention` (Silent Premise Contamination)**: Valid JSON responses whose numeric output still matches the superseded pre-retraction truth (`pre-retraction truth`) within tolerance after an explicit `fix->`, `err->`, or `sup->` event.
4. **`Over-Retraction` (Spurious Invalidation)**: Under `Condition E`, altering the value of an unrelated independent conclusion (`d_indep`) or erroneously marking independent node IDs in `retracted_ids`.

## 2. Ground-Truth Oracle Soundness & Dual Verification Layers

Ground truth is computed exclusively via Python `fractions.Fraction` exact rational arithmetic in `gyeol_bench/oracle.py`, eliminating IEEE-754 intermediate rounding drift. Candidate answers are scored after dimensional alignment under `abs_tol = 1e-4` and `rel_tol = 1e-5`.

Oracle verification strictly separates **automated symbolic cross-checking** from **independent human construct audit**:

1. **Automated `SymPy + Pint` Cross-Oracle Validation**: An out-of-repo `SymPy + Pint` symbolic rational and physical-unit script independently recomputes all `800` episode answers to verify `0` algebraic or unit-scaling discrepancies against `ExactRationalOracle`. Automated cross-validation proves mechanical code-to-code equivalence only and **never substitutes for human ground-truth audit**.
2. **Independent `N >= 150` Human Ground-Truth Audit**: Prior to Day `46`, external evaluation researchers manually audit a stratified sample of `N >= 150` rendered prompts and target derivations to verify natural-language clarity, unambiguous scope, and ground-truth soundness (`0 / 150` errors required for the `1.9773%` one-sided 95% upper error bound).

## 3. Pre-Registered Statistical Thresholds

- **Sample Balance & Pairing**: `N = 800` (`400` `Condition A` unperturbed control + `100` `Condition B` single retraction + `100` `Condition C` `>= 3`-hop cascade + `100` `Condition D` retraction + scale shift + `100` `Condition E` retraction + `d_indep` preservation) evaluated on identical seeds and identical JSON `response_adapter` envelopes comparing Standard Prompting Baseline against Lineage-Guided / Verifier-Assisted execution.
- **Paired Superiority (`K2`)**: `(b - c) / 800 >= +0.05` (`b - c >= 40`) and two-sided exact McNemar p < 0.05 (requires b + c <= 396 at the b - c = 40 boundary).
- **Human Audit Bound**: `0` oracle errors in independent `N >= 150` stratified human audit yields `1.9773%` (`1 - 0.05^(1/150)`, `1.98%`) one-sided 95% upper bound, reported separately from automated `SymPy + Pint` `800 / 800` cross-oracle agreement.
- **Control Specificity (`K4`)**: `<= 5 / 400` false rejections (`2.8928%` two-sided 95% Clopper-Pearson upper bound `< 5%`).
