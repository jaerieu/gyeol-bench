# External Independent Auditor Outreach Drafts (`2` to `3` Reviewers · Pre-Day-46)

- **Purpose**: Recruit `2` to `3` independent AI safety researchers and `UK AISI Inspect AI` / `inspect_evals` contributors prior to the Day-46 hash freeze to complete the blinded `N = 160` (`>= 150`) human ground-truth audit and review the 14-day public baseline PR challenge.
- **Governance Tier**: `[안전]` (Zero private core references; compensation/honorarium strictly conditional on `Clock A` legal clearance or `Model A` vendor-direct payment; unpaid open-science peer review supported at `$0` transfer).

---

## Draft 1 · `UK AISI Inspect Evals` Contributor / Maintainer Profile

**Subject**: `[Inspect Evals 2026-05-08] Blinded N=160 Construct Audit Invite — gyeol_bench/retraction_dag (Apache-2.0)`

Hi `[Name / GitHub Handle]`,

I am preparing an external `Apache-2.0` evaluation package (`gyeol_bench/retraction_dag`, adhering to the `2026-05-08` decentralized `inspect_evals` registry spec) that measures **Silent Quantitative Premise Contamination** in multi-turn stateful AI agents: specifically whether models retract superseded premises (`fix->`, `err->`, `sup->`), re-propagate exact rational dependencies across `3+` hops and SI/financial unit scales (`mg <-> g`, `K_USD <-> M_USD`, `% <-> bps`), and preserve unrelated conclusions (`d_indep`) without over-retraction.

While all `800` procedural items are already cross-verified (`0 / 800` discrepancies) between a pure-Python `fractions.Fraction` oracle and an independent `SymPy + Pint` symbolic check, automated code-to-code parity does not substitute for human construct validation. Before locking our Day-46 baseline hash freeze, we are inviting `2` to `3` independent `Inspect AI` contributors and evaluation researchers to audit a **blinded stratified packet (`N = 160` total, approx. `55` to `80` items per reviewer including a `30`-item overlap slice, est. `90` to `120` minutes)**.

What is already prepared for fast, friction-free review:
- Single-file worksheet (`CSV` or `JSONL`) with all ground-truth answers and condition labels stripped, paired with a SHA-256 pre-committed sealed answer key (`audit_packet/SEALED_KEY_COMMITMENT.json`) and a 2-page rubric (`audit_packet/ANNOTATOR_RUBRIC.md`).
- One-command offline task verification (`uv run inspect eval gyeol_bench/retraction_dag`) plus our `VALIDITY_REPORT.md` and 14-day public baseline PR challenge specification (`manifests/DAY46_PR_CHALLENGE.md`).
- Full acknowledgment in `VALIDITY_REPORT.md` and `eval.yaml` review records (or anonymous attribution if preferred).

Would you have `90` to `120` minutes over the next `10` days to take a `65`-item slice of the blinded audit packet (and optionally sanity-check our `2026-05-08` `inspect_evals` layout)? If yes, I will send over your assigned slice (`audit-001..065`) right away.

Best regards,
Jae Rieu
Independent AI Safety & Epistemic Verification Researcher (Brisbane, Australia)
Repository staging: `jaerieu/gyeol-bench` (`Apache-2.0`)

---

## Draft 2 · Epistemic / Multi-Turn Belief-Revision Evaluation Researcher Profile (`TRACK` / `AGM-BENCH` / `Apart` Community)

**Subject**: `Independent Blinded Ground-Truth Audit Invite (N=160): Multi-Turn Quantitative Premise Retraction Benchmark`

Dear `[Dr. / Researcher Name]`,

Following your work on `[belief revision / multi-turn knowledge updating / epistemic evaluations]`, I am reaching out to invite you as one of `2` to `3` external independent auditors for **`Gyeol-Bench` (`gyeol_bench/retraction_dag`, `Apache-2.0`)**, a clean-room `UK AISI Inspect AI` benchmark targeting multi-turn quantitative premise retraction and cascading SI/financial scale recalculation.

Existing multi-turn benchmarks (`AGM-BENCH`, `ReviseQA`, `TRACK [EACL 2026]`, `FPCO-Dialog [EMNLP 2026]`) evaluate qualitative propositional belief revision or contextual QA, leaving a measurement gap when stateful actors cache derived numbers in working memory and later receive an explicit premise retraction (`fix->`, `err->`, `sup->`) coupled with unit shifts (`500 mg` updated to `2 g`) alongside an unrelated conclusion (`d_indep`) that must remain untouched. `Gyeol-Bench` decomposes model failures into a 4-way split: `PARSE_REJECT`, conditional arithmetic error, `Stale-State Retention`, and `Over-Retraction`.

To establish a pre-registered one-sided 95% upper oracle-error bound below `1.98%` (`0 / 150` defects; `1.86%` at `0 / 160`) before our Day-46 cryptographic hash freeze (`Sigstore Rekor v1` + `FreeTSA`), we have prepared a **blinded `N = 160` human audit worksheet** (`audit_packet/ANNOTATOR_RUBRIC.md`). Each reviewer audits a `65`-item slice (`30` shared overlap items + `35` unique items, requiring approx. `90` to `120` minutes of straightforward rational/unit checking).

We also welcome critical adversarial PRs during our **14-day public baseline PR challenge** (`Days 46..60`) to strengthen the unassisted standard prompting baseline prior to paired exact McNemar (`K2`) evaluation.

Please let me know if you are available to audit a `65`-item blinded slice before `[Target Date UTC]`. We are happy to credit your independent audit directly in `VALIDITY_REPORT.md` and resulting technical documentation.

Warm regards,
Jae Rieu
Independent AI Safety & Epistemic Verification Researcher

---

## Draft 3 · Stateful Agent Reliability & Quantitative Safety Researcher Profile

**Subject**: `Pre-Freeze Reviewer Invitation: Blinded Audit + 14-Day Baseline Challenge for Gyeol-Bench (Inspect AI)`

Hi `[Name]`,

I am building `Gyeol-Bench` (`Apache-2.0`), an offline-reproducible `UK AISI Inspect AI` benchmark testing whether long-horizon tool/memory agents suffer from **Silent Quantitative Premise Contamination** when earlier numeric inputs are amended mid-session.

Before freezing our baseline prompts and dataset hashes on `Sigstore Rekor v1` (`RFC 6962 §2.1`) for our `N = 800` evaluation across held-out open-weight models and frontier APIs, we are recruiting `2` to `3` external safety researchers for two lightweight, high-leverage checks:
1. **Blinded Human Ground-Truth Audit (`60` to `80` items out of `N = 160`, approx. `90` min)**: Solving short `5`-to-`9`-turn arithmetic/SI-unit episodes from `blinded_annotator_sheet_n160.csv` (with all oracle labels stripped and pre-committed via SHA-256 in `SEALED_KEY_COMMITMENT.json`) and flagging any prompt or unit ambiguity (`SOUND` vs `PROMPT_AMBIGUOUS` / `UNIT_AMBIGUOUS` / `ARITH_DEFECT`).
2. **Optional Unassisted Baseline Review (`manifests/DAY46_PR_CHALLENGE.md`)**: Verifying that our unassisted JSON baseline prompt is fair, strong, and uncrippled prior to paired evaluation (`K2`: `(b - c) / 800 >= +5.0%p` and two-sided exact McNemar `p < 0.05`).

Everything runs in pure Python standard library + `inspect_ai >= 0.3.0` with zero proprietary dependencies. Would you be open to taking slice `C` (`audit-001..030` overlap + `audit-101..160`, `90` items total or a `65`-item subset) this week?

Best,
Jae Rieu
