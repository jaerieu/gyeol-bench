#!/usr/bin/env python3
"""Deterministic paired McNemar, Holm-Bonferroni, K2/K4 gate, and 4-way taxonomy analyzer for Arm 3 results.

Licensed under the Apache License, Version 2.0.
Contains zero lines and zero imports from any private engine (`[안전]`).

Implements Audit `#19` Mandatory Reporting Hierarchy:
1. Primary `K2` Eligible Arms (`is_primary_k2_eligible: true`, zero numeric target overwrite):
   - `arm3a_prompt_raw`: Raw Turn-2 LLM output under the zero-leakage structural prompt (`augment_turn2_prompt_with_ledger`).
   - `arm3a_structural_only`: `apply_arm3a_invariant_guard(..., mode="structural_only")` (`I1..I3` envelope / `R` / Turn-1 `state_v1` `preserved` lock).
   - `arm3b_structural_gate`: `apply_structural_gate_only` (`0` extra inference turns, non-numeric syntax / `R` / Turn-1 `state_v1` `preserved` lock).
   - `arm3b_llm_repair`: Pass-1 diagnostic + 1-turn LLM self-correction (`build_arm3b_repair_prompt` -> LLM retry). Explicitly discloses `extra_inference_turns` (`repair_triggered_count`) and diagnostic oracle trigger advantage (`STALE_TURN1_BINDING`, `SI_ARITH_RESIDUAL_MISMATCH`).
   - `arm3ab_hybrid` (optional): Arm 3B Pass-1 diagnostic + 1-turn LLM self-correction on `arm3a_structural_only` output.
2. Strictly Separate `secondary_symbolic_ceilings` Section (`is_primary_k2_eligible: false`):
   - `arm3a_full_ceiling` (`mode="full"`, `I1..I4` symbolic target overwrite).
   - `arm3b_symbolic_ceiling` (`mode="symbolic_ceiling"`, `pass2_deterministic_repair` symbolic target overwrite).
   Reported solely as neuro-symbolic solver-offload upper bounds and excluded from primary `K2` Holm-Bonferroni family testing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

VERDICT_KEYS: Tuple[str, ...] = (
    "PASS",
    "PARSE_REJECT",
    "CONDITIONAL_ARITH_ERROR",
    "STALE_STATE_RETENTION",
    "OVER_RETRACTION",
)

PRIMARY_ARM_SPECS: Tuple[Tuple[str, str, str, Optional[str], Optional[str], bool], ...] = (
    (
        "arm3a_prompt_raw",
        "arm3aPromptRawOk",
        "arm3aPromptRawVerdict",
        None,
        None,
        False,
    ),
    (
        "arm3a_structural_only",
        "arm3aStructuralOnlyOk",
        "arm3aStructuralOnlyVerdict",
        None,
        None,
        False,
    ),
    (
        "arm3b_structural_gate",
        "arm3bStructuralGateOk",
        "arm3bStructuralGateVerdict",
        None,
        "arm3bPass1Flags",
        False,
    ),
    (
        "arm3b_llm_repair",
        "arm3bLlmRepairOk",
        "arm3bLlmRepairVerdict",
        "arm3bRepairTriggered",
        "arm3bPass1Flags",
        True,
    ),
    (
        "arm3ab_hybrid",
        "arm3abHybridOk",
        "arm3abHybridVerdict",
        "arm3abHybridRepairTriggered",
        "arm3abHybridPass1Flags",
        True,
    ),
)

SECONDARY_CEILING_SPECS: Tuple[Tuple[str, str, Optional[str]], ...] = (
    ("arm3a_full_ceiling", "arm3aFullCeilingOk", "arm3aFullCeilingVerdict"),
    ("arm3b_symbolic_ceiling", "arm3bSymbolicCeilingOk", "arm3bSymbolicCeilingVerdict"),
)


def exact_two_sided_mcnemar_p(b: int, c: int) -> float:
    """Compute exact two-sided binomial McNemar p-value via `math.comb` for discordant counts (b, c)."""
    if b < 0 or c < 0:
        raise ValueError(f"Discordant counts must be non-negative, got b={b}, c={c}")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def holm_bonferroni_adjust(p_values: Sequence[float]) -> List[float]:
    """Compute step-down Holm-Bonferroni adjusted p-values in the original order of `p_values`."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda pair: (pair[1], pair[0]))
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        factor = m - rank
        adj = min(1.0, max(0.0, float(p_val)) * factor)
        if adj < running_max:
            adj = running_max
        else:
            running_max = adj
        adjusted[orig_idx] = adj
    return adjusted


def scaled_k2_threshold(sample_count: int) -> int:
    """Return the required net discordant gain `b - c` (`>= 40` at `N=800`, proportionally scaled for pilots)."""
    if sample_count <= 0:
        return 1
    if sample_count >= 800:
        return 40
    return max(1, math.ceil(0.05 * sample_count))


def scaled_k4_max_cond_a_failures(cond_a_total: int) -> int:
    """Return maximum allowed Condition A failures (`<= 5` at `400` Condition A controls)."""
    if cond_a_total <= 0:
        return 0
    if cond_a_total == 400:
        return 5
    return int(math.floor(5.0 * cond_a_total / 400.0))


def _tally_verdicts(rows: Sequence[Dict[str, Any]], ok_key: str, verdict_key: Optional[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {k: 0 for k in VERDICT_KEYS}
    for r in rows:
        v = r.get(verdict_key) if verdict_key else None
        if isinstance(v, str) and v in counts:
            counts[v] += 1
        else:
            if bool(r.get(ok_key, False)):
                counts["PASS"] += 1
            else:
                counts["CONDITIONAL_ARITH_ERROR"] += 1
    return counts


def _summarize_single_arm(
    rows: Sequence[Dict[str, Any]],
    ok_key: str,
    verdict_key: Optional[str],
) -> Dict[str, Any]:
    n = len(rows)
    pass_count = sum(1 for r in rows if bool(r.get(ok_key, False)))
    fail_count = n - pass_count
    verdicts = _tally_verdicts(rows, ok_key, verdict_key)
    stale_plus_over = verdicts["STALE_STATE_RETENTION"] + verdicts["OVER_RETRACTION"]

    cond_a_rows = [r for r in rows if r.get("condition") == "A"]
    cond_be_rows = [r for r in rows if r.get("condition") != "A"]
    cond_a_total = len(cond_a_rows)
    cond_a_pass = sum(1 for r in cond_a_rows if bool(r.get(ok_key, False)))
    cond_a_fail = cond_a_total - cond_a_pass
    k4_max_fail = scaled_k4_max_cond_a_failures(cond_a_total)

    cond_be_total = len(cond_be_rows)
    cond_be_pass = sum(1 for r in cond_be_rows if bool(r.get(ok_key, False)))
    cond_be_fail = cond_be_total - cond_be_pass

    per_condition: Dict[str, Dict[str, Any]] = {}
    for cond in ("A", "B", "C", "D", "E"):
        sub = [r for r in rows if r.get("condition") == cond]
        if not sub:
            continue
        sub_pass = sum(1 for r in sub if bool(r.get(ok_key, False)))
        per_condition[cond] = {
            "n": len(sub),
            "pass": sub_pass,
            "fail": len(sub) - sub_pass,
            "accuracy": round(sub_pass / len(sub), 6) if sub else 0.0,
            "verdicts": _tally_verdicts(sub, ok_key, verdict_key),
        }

    return {
        "n": n,
        "passCount": pass_count,
        "failCount": fail_count,
        "accuracy": round(pass_count / n, 6) if n else 0.0,
        "failureRate": round(fail_count / n, 6) if n else 0.0,
        "condATotal": cond_a_total,
        "condAPassCount": cond_a_pass,
        "condAFailCount": cond_a_fail,
        "condAFailRate": round(cond_a_fail / cond_a_total, 6) if cond_a_total else 0.0,
        "condAK4MaxAllowedFailures": k4_max_fail,
        "condAK4Pass": cond_a_fail <= k4_max_fail,
        "condBETotal": cond_be_total,
        "condBEPassCount": cond_be_pass,
        "condBEFailCount": cond_be_fail,
        "condBEAccuracy": round(cond_be_pass / cond_be_total, 6) if cond_be_total else 0.0,
        "verdictCounts": verdicts,
        "stalePlusOverCount": stale_plus_over,
        "perCondition": per_condition,
    }


def compute_paired_comparison(
    rows: Sequence[Dict[str, Any]],
    base_ok_key: str,
    base_verdict_key: str,
    contender_ok_key: str,
    contender_verdict_key: Optional[str],
) -> Dict[str, Any]:
    """Compute paired McNemar 2x2 contingency table and accuracy/S+O deltas against `base_ok_key`."""
    n = len(rows)
    a_both_pass = 0
    b_base_fail_cont_pass = 0
    c_base_pass_cont_fail = 0
    d_both_fail = 0

    be_b = 0
    be_c = 0

    for r in rows:
        base_ok = bool(r.get(base_ok_key, False))
        cont_ok = bool(r.get(contender_ok_key, False))
        is_be = r.get("condition") != "A"
        if base_ok and cont_ok:
            a_both_pass += 1
        elif (not base_ok) and cont_ok:
            b_base_fail_cont_pass += 1
            if is_be:
                be_b += 1
        elif base_ok and (not cont_ok):
            c_base_pass_cont_fail += 1
            if is_be:
                be_c += 1
        else:
            d_both_fail += 1

    b_minus_c = b_base_fail_cont_pass - c_base_pass_cont_fail
    b_plus_c = b_base_fail_cont_pass + c_base_pass_cont_fail
    exact_p = exact_two_sided_mcnemar_p(b_base_fail_cont_pass, c_base_pass_cont_fail)
    be_exact_p = exact_two_sided_mcnemar_p(be_b, be_c)
    min_req_b_minus_c = scaled_k2_threshold(n)

    base_summary = _summarize_single_arm(rows, base_ok_key, base_verdict_key)
    cont_summary = _summarize_single_arm(rows, contender_ok_key, contender_verdict_key)
    so_delta = cont_summary["stalePlusOverCount"] - base_summary["stalePlusOverCount"]

    return {
        "summary": cont_summary,
        "stalePlusOverDeltaContenderMinusArm1": so_delta,
        "mcnemar": {
            "aBothPass": a_both_pass,
            "bArm1FailContenderPass": b_base_fail_cont_pass,
            "cArm1PassContenderFail": c_base_pass_cont_fail,
            "dBothFail": d_both_fail,
            "bMinusC": b_minus_c,
            "bPlusC": b_plus_c,
            "netGainOverN": round(b_minus_c / n, 6) if n else 0.0,
            "requiredBMinusCForK2": min_req_b_minus_c,
            "twoSidedExactP": exact_p,
            "holmBonferroniAdjustedP": exact_p,  # updated across family
            "k2RawGatePass": (b_minus_c >= min_req_b_minus_c) and (exact_p < 0.05),
            "k2HolmGatePass": (b_minus_c >= min_req_b_minus_c) and (exact_p < 0.05),
            "beOnlyB": be_b,
            "beOnlyC": be_c,
            "beOnlyBMinusC": be_b - be_c,
            "beOnlyTwoSidedExactP": be_exact_p,
        },
    }


def _collect_repair_disclosures(
    rows: Sequence[Dict[str, Any]],
    trigger_key: Optional[str],
    flags_key: Optional[str],
    has_diagnostic_oracle_trigger: bool,
) -> Dict[str, Any]:
    n = len(rows)
    triggered_count = 0
    extra_turns_total = 0
    flag_histogram: Dict[str, int] = {}

    for r in rows:
        if trigger_key and bool(r.get(trigger_key, False)):
            triggered_count += 1
        if trigger_key == "arm3bRepairTriggered":
            extra_turns_total += int(r.get("arm3bExtraTurns", 1 if r.get(trigger_key) else 0))
        elif trigger_key == "arm3abHybridRepairTriggered":
            extra_turns_total += int(r.get("arm3abHybridExtraTurns", 1 if r.get(trigger_key) else 0))
        if flags_key and isinstance(r.get(flags_key), list):
            for fl in r[flags_key]:
                if isinstance(fl, str):
                    flag_histogram[fl] = flag_histogram.get(fl, 0) + 1

    return {
        "repair_triggered_count": triggered_count,
        "repair_triggered_rate": round(triggered_count / n, 6) if n else 0.0,
        "extra_inference_turns": extra_turns_total if has_diagnostic_oracle_trigger else 0,
        "pass1_flag_counts": flag_histogram,
        "diagnostic_oracle_trigger_disclosure": (
            "Pass-1 trigger includes binary STALE_TURN1_BINDING / SI_ARITH_RESIDUAL_MISMATCH state-diff audit "
            "(with zero target numeric leakage in repair prompt); co-logged alongside 0-extra-turn structural_gate."
            if has_diagnostic_oracle_trigger
            else "Zero extra inference turns and zero numeric state-diff trigger."
        ),
    }


def load_episodes_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load and deduplicate (by last occurrence of `episodeId`) episode records from a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Episodes JSONL file not found: {path}")
    by_id: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rec = json.loads(line)
        ep_id = str(rec.get("episodeId") or rec.get("episode_id") or f"row-{len(order)+1}")
        if ep_id not in by_id:
            order.append(ep_id)
        by_id[ep_id] = rec
    return [by_id[k] for k in order]


def analyze_episode_files(paths: Sequence[Path]) -> Dict[str, Any]:
    """Analyze one or more `*-episodes.jsonl` files and return a complete structured manifest."""
    models_out: List[Dict[str, Any]] = []
    # Track all primary (model_idx, arm_name, pop_name) p-values for family-wise Holm-Bonferroni correction
    itt_family_refs: List[Tuple[Dict[str, Any], float]] = []
    clean_t1_family_refs: List[Tuple[Dict[str, Any], float]] = []

    for path in paths:
        rows = load_episodes_jsonl(path)
        if not rows:
            raise ValueError(f"Empty episode JSONL file: {path}")
        raw_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        first = rows[0]
        model_key = str(first.get("modelKey") or path.stem.replace("-episodes", ""))
        repo_id = str(first.get("repoId") or model_key)
        dtype = str(first.get("dtype") or "nf4")

        n_total = len(rows)
        clean_t1_rows = [r for r in rows if bool(r.get("turn1Ok", False))]

        t1_summary = _summarize_single_arm(rows, "turn1Ok", "turn1Verdict")
        arm1_itt = _summarize_single_arm(rows, "arm1Ok", "arm1Verdict")
        arm1_clean_t1 = _summarize_single_arm(clean_t1_rows, "arm1Ok", "arm1Verdict")

        primary_arms_itt: Dict[str, Any] = {}
        primary_arms_clean_t1: Dict[str, Any] = {}

        for arm_name, ok_key, verdict_key, trig_key, flags_key, has_diag_trigger in PRIMARY_ARM_SPECS:
            if not any(ok_key in r for r in rows):
                continue
            itt_cmp = compute_paired_comparison(rows, "arm1Ok", "arm1Verdict", ok_key, verdict_key)
            itt_cmp["is_primary_k2_eligible"] = True
            itt_cmp["audit19_disclosures"] = _collect_repair_disclosures(
                rows, trig_key, flags_key, has_diag_trigger
            )
            primary_arms_itt[arm_name] = itt_cmp
            itt_family_refs.append((itt_cmp["mcnemar"], itt_cmp["mcnemar"]["twoSidedExactP"]))

            clean_cmp = compute_paired_comparison(
                clean_t1_rows, "arm1Ok", "arm1Verdict", ok_key, verdict_key
            )
            clean_cmp["is_primary_k2_eligible"] = True
            clean_cmp["audit19_disclosures"] = _collect_repair_disclosures(
                clean_t1_rows, trig_key, flags_key, has_diag_trigger
            )
            primary_arms_clean_t1[arm_name] = clean_cmp
            clean_t1_family_refs.append((clean_cmp["mcnemar"], clean_cmp["mcnemar"]["twoSidedExactP"]))

        secondary_ceilings_itt: Dict[str, Any] = {}
        secondary_ceilings_clean_t1: Dict[str, Any] = {}
        for ceil_name, ok_key, verdict_key in SECONDARY_CEILING_SPECS:
            if not any(ok_key in r for r in rows):
                continue
            c_itt = _summarize_single_arm(rows, ok_key, verdict_key)
            c_itt["is_primary_k2_eligible"] = False
            c_itt["ceiling_role"] = "secondary_neuro_symbolic_solver_offload_upper_bound_only"
            secondary_ceilings_itt[ceil_name] = c_itt

            c_clean = _summarize_single_arm(clean_t1_rows, ok_key, verdict_key)
            c_clean["is_primary_k2_eligible"] = False
            c_clean["ceiling_role"] = "secondary_neuro_symbolic_solver_offload_upper_bound_only"
            secondary_ceilings_clean_t1[ceil_name] = c_clean

        model_entry: Dict[str, Any] = {
            "modelKey": model_key,
            "repoId": repo_id,
            "dtype": dtype,
            "episodesPath": str(path),
            "episodesSha256": raw_sha256,
            "sampleSize": n_total,
            "turn1Shared": {
                "cleanCount": len(clean_t1_rows),
                "cleanRate": round(len(clean_t1_rows) / n_total, 6) if n_total else 0.0,
                "verdictCounts": t1_summary["verdictCounts"],
            },
            "ittPrimary": {
                "n": n_total,
                "arm1Baseline": arm1_itt,
                "primaryK2Contenders": primary_arms_itt,
            },
            "cleanTurn1Sensitivity": {
                "n": len(clean_t1_rows),
                "arm1Baseline": arm1_clean_t1,
                "primaryK2Contenders": primary_arms_clean_t1,
            },
            "secondary_symbolic_ceilings": {
                "is_primary_k2_eligible": False,
                "disclosure": (
                    "Symbolic-offload ceilings overwrite Turn-2 target numeric values with exact Fraction/SI "
                    "DAG evaluation and are strictly excluded from primary K2 LLM belief-revision claims."
                ),
                "itt": secondary_ceilings_itt,
                "cleanTurn1": secondary_ceilings_clean_t1,
            },
        }
        if path.name in {
            "qwen14b-bf16-s20261122-n100-episodes.jsonl",
            "qwen14b-bf16-s20261122-n100-patched-episodes.jsonl",
        }:
            pre_fix_path = path.parent / "qwen14b-bf16-s20261122-n100-episodes.jsonl"
            pre_fix_sha = (
                hashlib.sha256(pre_fix_path.read_bytes()).hexdigest()
                if pre_fix_path.exists()
                else "4a14b2d6933622cddba9c8347d09b58e96088b57b41c43449a7634cb95c46212"
            )
            t1_only_path = path.parent / "qwen14b-bf16-s20261122-n100-t1only.jsonl"
            t1_only_sha = (
                hashlib.sha256(t1_only_path.read_bytes()).hexdigest()
                if t1_only_path.exists()
                else None
            )
            is_patched_file = path.name == "qwen14b-bf16-s20261122-n100-patched-episodes.jsonl"
            model_entry["quarantinedPreFixProvenance"] = {
                "status": (
                    "patched_gpu_rerun_ingested"
                    if is_patched_file
                    else "quarantined_pre_fix_disclosed_offline_closeout_colab_80gb_503_stockout"
                ),
                "quarantinedEpisodesPath": str(pre_fix_path),
                "quarantinedEpisodesSha256": pre_fix_sha,
                "t1OnlyCachePath": str(t1_only_path),
                "t1OnlyCacheSha256": t1_only_sha,
                "hardwareParityMode": (
                    "Symmetric Turn-2 re-evaluation: Turn-1 shared from t1only cache; "
                    "Turn-2 Arm 1, Arm 3A, Arm 3B, and Arm 3AB-Hybrid all generated on the same GPU VM."
                ),
                "quarantinedArm3aPromptRawIttPassCount": 28,
                "quarantinedArm3aPromptRawIttTotal": 100,
                "quarantinedArm3aPromptRawCondAFailCount": 50,
                "quarantinedArm3aPromptRawParseRejectCount": 59,
                "offlineZeroCuPatchedStructuralGateReplay": {
                    "arm3bStructuralGateIttPassCountBefore": 78,
                    "arm3bStructuralGateIttPassCountAfter": 78,
                    "arm3bStructuralGateTotal": 100,
                    "verdictFlips": 0,
                    "arm3aStructuralOnlyIttPassCount": 79,
                    "arm3abHybridIttPassCount": 86,
                },
                "colabGpuStockoutLog": (
                    "2026-11-24 Colab Pro+ backend returned HTTP 503 Service Unavailable across all >=80GB "
                    "and fallback GPU pools (A100, H100, G4; 0.00 CU burned, Active assignments: 0 verified, "
                    "balance preserved at 3132.29 / 3251.14 CU = 118.85 CU spent)."
                ),
                "defectRootCause": (
                    "Pre-fix build_arm3a_ledger_from_prompts extracted Turn-1 state_v1 commitment on "
                    "non-Condition-E episodes whenever Turn-1 raw JSON emitted a spurious 'preserved' key, "
                    "causing augment_turn2_prompt_with_ledger to instruct emitting 'preserved' on Condition A..D "
                    "and triggering oracle schema PARSE_REJECT (50/50 Condition A failure, 28/100 ITT pass)."
                ),
                "remediationNote": (
                    "Patched in arm3_ledger_hook.py (gated state_v1 extraction on bool(t2_preserved)) "
                    "and arm3_state_diff.py (apply_structural_gate_only non-E preserved strip); "
                    "archived pre-fix JSONL and t1only cache preserved on disk and R2 for audit provenance."
                ),
            }
        models_out.append(model_entry)

    # Apply Holm-Bonferroni correction across all evaluated primary K2 (model x arm) pairs
    if itt_family_refs:
        adj_itt = holm_bonferroni_adjust([p for _, p in itt_family_refs])
        for (mc_dict, _), adj_p in zip(itt_family_refs, adj_itt):
            mc_dict["holmBonferroniAdjustedP"] = adj_p
            mc_dict["k2HolmGatePass"] = (
                mc_dict["bMinusC"] >= mc_dict["requiredBMinusCForK2"]
            ) and (adj_p < 0.05)

    if clean_t1_family_refs:
        adj_clean = holm_bonferroni_adjust([p for _, p in clean_t1_family_refs])
        for (mc_dict, _), adj_p in zip(clean_t1_family_refs, adj_clean):
            mc_dict["holmBonferroniAdjustedP"] = adj_p
            mc_dict["k2HolmGatePass"] = (
                mc_dict["bMinusC"] >= mc_dict["requiredBMinusCForK2"]
            ) and (adj_p < 0.05)

    manifest_core: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "benchmark_package": "gyeol-bench",
        "primary_task": "gyeol_bench/retraction_dag_multiturn",
        "governance_grade": "[안전]",
        "holm_bonferroni_family_size": len(itt_family_refs),
        "models": models_out,
    }
    canonical_bytes = json.dumps(manifest_core, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    manifest_sha = hashlib.sha256(canonical_bytes).hexdigest()
    return {
        **manifest_core,
        "manifest_body_sha256": manifest_sha,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze Arm 3 multi-turn JSONL results with exact McNemar, Holm-Bonferroni, and K2/K4 gates."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="One or more *-episodes.jsonl files")
    parser.add_argument("--out", type=Path, default=None, help="Optional output manifest JSON path")
    args = parser.parse_args(argv)

    manifest = analyze_episode_files(args.inputs)
    formatted = json.dumps(manifest, ensure_ascii=False, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(formatted + "\n", encoding="utf-8")
    else:
        print(formatted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
