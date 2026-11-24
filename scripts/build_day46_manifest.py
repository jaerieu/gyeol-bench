#!/usr/bin/env python3
"""Deterministic builder and verifier for the Day-46 v2.0 cryptographic baseline freeze manifest.

Computes canonical SHA-256 digests across:
1. Procedural datasets: v2.0 main evaluation (seed=20261122, N=800 + per-condition A..E),
   Phase 1 pilot slice (seed=20261150, N=50 + seed=20261122, N=50), and PR-challenge validation split (seed=20261199, N=160).
2. Scorer, exact rational oracle, downward-closed stale-state detector, response_adapter, and Inspect AI task/registry files.
3. Blinded N=160 human audit worksheet digests and out-of-repo sealed answer key SHA-256 commitment.
4. Frozen v2.0-day46 unassisted baseline system prompt ("notes" trace + retracted_ids convention) and 14-day PR challenge contract.
5. RFC 6962 Section 2.1 Merkle Tree Hash (0x00 leaf / 0x01 interior) matching ops/runners/rekor-merkle-anchor.mjs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gyeol_bench import ExactRationalOracle, RetractionEpisode, generate_procedural_dataset
from gyeol_bench.oracle import convert_from_base_fraction

EVAL_SEED = 20261122
EVAL_COUNT = 800
PILOT_SEED = 20261150
PILOT_COUNT = 50
CHALLENGE_VAL_SEED = 20261199
CHALLENGE_VAL_COUNT = 160

FROZEN_BASELINE_SYSTEM_PROMPT = (
    "You are a precise quantitative reasoning assistant. Read every turn carefully in sequential order.\n"
    "1. Track all base premises and their exact units (SI mass ug/mg/g/kg, length mm/cm/m/km, volume uL/mL/L, "
    "time ms/s/min/h, USD financial scales USD/K_USD/M_USD/B_USD, percentages %, and basis points bps), "
    "ignoring inert distractor chains unrelated to the target node.\n"
    "2. Whenever a retraction turn (fix->, err->, or sup->) supersedes an earlier node (including chained "
    "retractions such as sup-> p1_r1), record ONLY unique node IDs appearing directly on the left of ':=' in "
    "\"retracted_ids\" sorted alphabetically (never include un-superseded replacement turn IDs or derived d* "
    "nodes; output [] if zero retractions occur), and recompute every downstream derived node along all "
    "diamond and cross-branch paths using exact dimensional conversion.\n"
    "3. Keep any independent derived subgraph (such as d_indep) that has no directed dependency on a retracted "
    "node strictly unchanged in \"preserved\" when requested.\n"
    "4. Output ONLY a single standalone valid JSON object starting with \"notes\" (containing a concise "
    "step-by-step calculation and retraction trace) followed by \"step_id\", \"value\", \"unit\", "
    "\"retracted_ids\" (and \"preserved\" when requested), with zero surrounding prose or markdown fences."
)

HASHED_REPO_FILES: Tuple[Tuple[str, str], ...] = (
    ("package_metadata", "pyproject.toml"),
    ("inspect_eval_spec", "eval.yaml"),
    ("oracle_and_adapter", "gyeol_bench/oracle.py"),
    ("inspect_registry_and_scorer", "gyeol_bench/_registry.py"),
    ("package_init", "gyeol_bench/__init__.py"),
    ("zero_core_static_guard", "scripts/check_zero_core.py"),
    ("human_audit_rubric", "audit_packet/ANNOTATOR_RUBRIC.md"),
    ("human_audit_commitment", "audit_packet/SEALED_KEY_COMMITMENT.json"),
    ("human_audit_blinded_jsonl", "audit_packet/blinded_annotator_sheet_n160.jsonl"),
    ("human_audit_blinded_csv", "audit_packet/blinded_annotator_sheet_n160.csv"),
    ("pr_challenge_protocol", "manifests/DAY46_PR_CHALLENGE.md"),
    ("v3_multiturn_prepilot_lock", "manifests/v3-multiturn-prepilot-lock.json"),
    ("frontier_saturation_report", "agent_artefacts/retraction_dag_0.1.0/validity/SATURATION_REPORT.md"),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def largest_power_of_two_less_than(n: int) -> int:
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    k = 1
    while k * 2 < n:
        k *= 2
    return k


def compute_rfc6962_merkle_root(leaves_utf8: Sequence[str]) -> Tuple[str, List[str]]:
    """Compute RFC 6962 Section 2.1 Merkle Tree Hash (0x00 leaf, 0x01 interior)."""
    raw_leaves = [s.encode("utf-8") for s in leaves_utf8]
    leaf_hashes = [hashlib.sha256(b"\x00" + b).hexdigest() for b in raw_leaves]

    def mth(slice_bytes: Sequence[bytes]) -> bytes:
        count = len(slice_bytes)
        if count == 0:
            return hashlib.sha256(b"").digest()
        if count == 1:
            return hashlib.sha256(b"\x00" + slice_bytes[0]).digest()
        k = largest_power_of_two_less_than(count)
        left = mth(slice_bytes[:k])
        right = mth(slice_bytes[k:])
        return hashlib.sha256(b"\x01" + left + right).digest()

    return mth(raw_leaves).hex(), leaf_hashes


def canonicalize_episodes_jsonl(episodes: Sequence[RetractionEpisode]) -> bytes:
    lines: List[str] = []
    for ep in episodes:
        oracle = ExactRationalOracle()
        for t in ep.turns:
            oracle.append_turn(t)
        snap = oracle.evaluate()
        base_val, dim, _ = snap.active_values_base[ep.target_node_id]
        truth_frac = convert_from_base_fraction(base_val, dim, ep.expected_unit)
        pres_truth = {}
        for pid, punit in ep.preserved_targets:
            pb, pdim, _ = snap.active_values_base[pid]
            pf = convert_from_base_fraction(pb, pdim, punit)
            pres_truth[pid] = {"unit": punit, "num": pf.numerator, "den": pf.denominator}

        rec = {
            "episode_id": ep.episode_id,
            "condition": ep.condition,
            "category": ep.category,
            "target_node_id": ep.target_node_id,
            "expected_unit": ep.expected_unit,
            "truth_num": truth_frac.numerator,
            "truth_den": truth_frac.denominator,
            "expected_value_float": ep.expected_value_float,
            "expected_retracted_ids": list(ep.expected_retracted_ids),
            "pre_retraction_value_float": ep.pre_retraction_value_float,
            "preserved_truth": pres_truth,
            "prompt": ep.prompt,
            "turns": [
                {
                    "node_id": t.node_id,
                    "kind": t.kind,
                    "value": t.value,
                    "unit": t.unit,
                    "retraction_op": t.retraction_op,
                    "target_id": t.target_id,
                    "arith_op": t.arith_op,
                    "left_id": t.left_id,
                    "right_id": t.right_id,
                }
                for t in ep.turns
            ],
        }
        lines.append(json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_day46_artifacts(repo_root: Path = REPO_ROOT) -> Tuple[Dict[str, Any], List[str]]:
    ds_800 = generate_procedural_dataset(seed=EVAL_SEED, count=EVAL_COUNT)
    ds_50 = generate_procedural_dataset(seed=PILOT_SEED, count=PILOT_COUNT)
    ds_val_160 = generate_procedural_dataset(seed=CHALLENGE_VAL_SEED, count=CHALLENGE_VAL_COUNT)

    ds_800_sha = sha256_bytes(canonicalize_episodes_jsonl(ds_800))
    ds_50_sha = sha256_bytes(canonicalize_episodes_jsonl(ds_50))
    ds_val_160_sha = sha256_bytes(canonicalize_episodes_jsonl(ds_val_160))

    condition_digests: Dict[str, Dict[str, Any]] = {}
    for cond in ("A", "B", "C", "D", "E"):
        sub = [ep for ep in ds_800 if ep.condition == cond]
        condition_digests[cond] = {
            "count": len(sub),
            "sha256": sha256_bytes(canonicalize_episodes_jsonl(sub)),
        }

    file_digests: Dict[str, Dict[str, str]] = {}
    for role, rel_path in HASHED_REPO_FILES:
        full = repo_root / rel_path
        if not full.exists():
            raise FileNotFoundError(f"Required freeze file missing: {full}")
        file_digests[rel_path] = {
            "role": role,
            "sha256": sha256_bytes(full.read_bytes()),
        }

    prompt_sha = sha256_bytes(FROZEN_BASELINE_SYSTEM_PROMPT.encode("utf-8"))

    pr_challenge_rules = {
        "freeze_milestone": "Day-46",
        "evaluation_lock_milestone": "Day-60",
        "challenge_window_days": 14,
        "challenge_window_hours": 336,
        "target_repository": "jaerieu/gyeol-bench",
        "dataset_generator_version": "v2.0",
        "baseline_system_prompt_id": "v2.0-day46",
        "baseline_system_prompt_sha256": prompt_sha,
        "invariants": [
            "identical_pinned_model_and_token_thinking_budget",
            "single_pass_inference_without_external_code_execution_in_unassisted_arm",
            "unmodified_response_adapter_standalone_json_gate_with_optional_string_notes",
            "zero_seed_20261122_or_20261199_leakage_in_prompts_or_exemplars",
            "strict_accuracy_gain_on_seed_20261199_without_parse_reject_or_condition_a_regression",
        ],
        "day46_pr_validation_gate_n160": {
            "held_out_validation_seed": CHALLENGE_VAL_SEED,
            "held_out_validation_count": CHALLENGE_VAL_COUNT,
            "held_out_validation_sha256": ds_val_160_sha,
            "max_b_plus_c": CHALLENGE_VAL_COUNT,
            "min_b_minus_c_for_adoption": 1,
            "two_sided_exact_mcnemar_role": "diagnostic_tie_breaker_on_n160",
        },
        "day60_final_evaluation_gates_n800": {
            "evaluation_seed": EVAL_SEED,
            "evaluation_count": EVAL_COUNT,
            "K2_min_net_gain_proportion": 0.05,
            "K2_min_b_minus_c_at_n800": 40,
            "K2_mcnemar_two_sided_p_lt": 0.05,
            "K2_max_b_plus_c_at_b_minus_c_40": 396,
            "K2_discordant_manual_audit_ratio": 1.0,
            "K2_concordant_stratified_audit_n": 50,
            "K2_external_human_audit_min_n": 150,
            "K2_external_human_audit_prepared_n": 160,
            "K3_min_top_model_failure_rate": 0.10,
            "K4_control_n": 400,
            "K4_max_false_alarms": 5,
            "K4_max_false_alarm_rate": 0.0125,
            "K4_clopper_pearson_95_upper": 0.028928,
            "frontier_tier_status": "Option_C_Halted_At_Zero_Phase3_Spend_KV_ReReference_Saturation_Control",
            "primary_k2_cohort": "open_weight_14b_to_24b_nf4_three_families",
            "population_hierarchy": ["ITT_N800_primary", "Clean_T1_sensitivity"],
            "co_primary_outcomes": ["turn2_overall_accuracy_delta_b_minus_c", "stale_plus_over_retraction_count_delta"],
        },
    }
    rules_canonical_sha = sha256_bytes(
        json.dumps(pr_challenge_rules, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    )

    leaves: List[str] = [
        f"dataset:v2_n800:seed={EVAL_SEED}:{ds_800_sha}",
        f"dataset:v2_pilot_n50:seed={PILOT_SEED}:{ds_50_sha}",
        f"dataset:v2_challenge_val_n160:seed={CHALLENGE_VAL_SEED}:{ds_val_160_sha}",
    ]
    for cond in ("A", "B", "C", "D", "E"):
        leaves.append(f"dataset:v2_condition_{cond}:n={condition_digests[cond]['count']}:{condition_digests[cond]['sha256']}")
    for _, rel_path in HASHED_REPO_FILES:
        leaves.append(f"file:{rel_path}:{file_digests[rel_path]['sha256']}")
    leaves.append(f"contract:baseline_prompt_v2.0-day46:{prompt_sha}")
    leaves.append(f"contract:pr_challenge_rules_v2:{rules_canonical_sha}")

    merkle_root_hex, leaf_hashes_hex = compute_rfc6962_merkle_root(leaves)

    commitment_doc = json.loads((repo_root / "audit_packet" / "SEALED_KEY_COMMITMENT.json").read_text(encoding="utf-8"))

    manifest_core: Dict[str, Any] = {
        "schema_version": "2.0.0",
        "benchmark_package": "gyeol-bench",
        "benchmark_version": "0.1.0",
        "dataset_generator_version": "v2.0",
        "task_identifier": "gyeol_bench/retraction_dag",
        "inspect_evals_spec_date": "2026-05-08",
        "license": "Apache-2.0",
        "governance_grade": "[안전]",
        "datasets": {
            "main_eval_n800": {
                "seed": EVAL_SEED,
                "sample_count": EVAL_COUNT,
                "canonical_jsonl_sha256": ds_800_sha,
                "conditions": condition_digests,
            },
            "pilot_k3_n50": {
                "seed": PILOT_SEED,
                "sample_count": PILOT_COUNT,
                "canonical_jsonl_sha256": ds_50_sha,
            },
            "pr_challenge_validation_n160": {
                "seed": CHALLENGE_VAL_SEED,
                "sample_count": CHALLENGE_VAL_COUNT,
                "canonical_jsonl_sha256": ds_val_160_sha,
            },
        },
        "repository_files": file_digests,
        "external_human_audit_commitment": {
            "total_audit_items": commitment_doc["total_audit_items"],
            "blinded_jsonl_sha256": commitment_doc["blinded_jsonl_sha256"],
            "sealed_answer_key_sha256": commitment_doc["sealed_answer_key_sha256"],
        },
        "pr_challenge": pr_challenge_rules,
        "rfc6962_merkle": {
            "hash_algorithm": "SHA-256",
            "leaf_prefix_hex": "00",
            "interior_prefix_hex": "01",
            "leaf_count": len(leaves),
            "merkle_root_hex": merkle_root_hex,
            "leaf_hashes_hex": leaf_hashes_hex,
        },
    }

    manifest_canonical_bytes = json.dumps(
        manifest_core, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    manifest_sha256 = sha256_bytes(manifest_canonical_bytes)
    manifest_doc = {
        **manifest_core,
        "manifest_body_sha256": manifest_sha256,
    }
    return manifest_doc, leaves


def verify_day46_on_disk(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    expected_manifest, expected_leaves = build_day46_artifacts(repo_root)
    manifest_path = repo_root / "manifests" / "day46-baseline-freeze.json"
    leaves_path = repo_root / "manifests" / "day46-leaves.json"
    violations: List[str] = []

    if not manifest_path.exists():
        violations.append(f"Missing {manifest_path}")
    if not leaves_path.exists():
        violations.append(f"Missing {leaves_path}")

    if not violations:
        disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        disk_leaves = json.loads(leaves_path.read_text(encoding="utf-8"))
        if disk_manifest != expected_manifest:
            violations.append("manifests/day46-baseline-freeze.json out of sync with deterministic recomputation.")
        if disk_leaves != expected_leaves:
            violations.append("manifests/day46-leaves.json out of sync with deterministic recomputation.")

    ok = len(violations) == 0
    return {
        "ok": ok,
        "grade": "[안전]" if ok else "[금지]",
        "manifestBodySha256": expected_manifest["manifest_body_sha256"],
        "rfc6962MerkleRootHex": expected_manifest["rfc6962_merkle"]["merkle_root_hex"],
        "leafCount": len(expected_leaves),
        "datasetN800Sha256": expected_manifest["datasets"]["main_eval_n800"]["canonical_jsonl_sha256"],
        "violations": violations,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify Day-46 v2.0 cryptographic baseline freeze manifest.")
    parser.add_argument("--verify", action="store_true", help="Verify existing manifest files without writing")
    args = parser.parse_args(argv)

    if args.verify:
        rep = verify_day46_on_disk(REPO_ROOT)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0 if rep["ok"] else 1

    manifest_doc, leaves = build_day46_artifacts(REPO_ROOT)
    manifests_dir = REPO_ROOT / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / "day46-baseline-freeze.json").write_text(
        json.dumps(manifest_doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (manifests_dir / "day46-leaves.json").write_text(
        json.dumps(leaves, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rep = verify_day46_on_disk(REPO_ROOT)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
