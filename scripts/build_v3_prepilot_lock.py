#!/usr/bin/env python3
"""Deterministic builder and verifier for manifests/v3-multiturn-prepilot-lock.json.

Locks SHA-256 digests prior to running the v3.0 2-live-turn stateful handoff pilot (seed=20261151, N=50)
and main evaluation (seed=20261122, N=800):
- gyeol_bench/oracle.py, gyeol_bench/_registry.py, gyeol_bench/__init__.py, eval.yaml
- V3_MULTITURN_BASELINE_SYSTEM_PROMPT ("v3.0-multiturn-baseline")
- V3_MULTITURN_LINEAGE_K2_SYSTEM_PROMPT ("v3.0-multiturn-lineage-k2")
- Canonical v3.0 multi-turn datasets: seed=20261151 (N=50), seed=20261122 (N=800 + A..E), seed=20261199 (N=160)
- Preserved Ablation 0 v2.0 single-prompt dataset: seed=20261122 (N=800)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gyeol_bench import (
    ExactRationalOracle,
    MultiTurnEpisode,
    V3_MULTITURN_BASELINE_SYSTEM_PROMPT,
    V3_MULTITURN_LINEAGE_K2_SYSTEM_PROMPT,
    generate_multiturn_dataset,
    generate_procedural_dataset,
)
from gyeol_bench.oracle import convert_from_base_fraction

V3_PILOT_SEED = 20261151
V3_PILOT_COUNT = 50
V3_EVAL_SEED = 20261122
V3_EVAL_COUNT = 800
V3_VAL_SEED = 20261199
V3_VAL_COUNT = 160


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _serialize_turn_list(turns: Any) -> List[Dict[str, Any]]:
    return [
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
        for t in turns
    ]


def canonicalize_multiturn_jsonl(episodes: Sequence[MultiTurnEpisode]) -> bytes:
    lines: List[str] = []
    for ep in episodes:
        t1_oracle = ExactRationalOracle()
        for t in ep.turn1_turns:
            t1_oracle.append_turn(t)
        t1_snap = t1_oracle.evaluate()
        t1_b, t1_dim, _ = t1_snap.active_values_base[ep.turn1_target_id]
        t1_frac = convert_from_base_fraction(t1_b, t1_dim, ep.turn1_expected_unit)

        t2_oracle = ExactRationalOracle()
        for t in ep.all_turns:
            t2_oracle.append_turn(t)
        t2_snap = t2_oracle.evaluate()
        t2_b, t2_dim, _ = t2_snap.active_values_base[ep.target_node_id]
        t2_frac = convert_from_base_fraction(t2_b, t2_dim, ep.expected_unit)

        pres_truth: Dict[str, Dict[str, Any]] = {}
        for pid, punit in ep.preserved_targets:
            pb, pdim, _ = t2_snap.active_values_base[pid]
            pf = convert_from_base_fraction(pb, pdim, punit)
            pres_truth[pid] = {"unit": punit, "num": pf.numerator, "den": pf.denominator}

        rec = {
            "episode_id": ep.episode_id,
            "condition": ep.condition,
            "category": ep.category,
            "turn1_target_id": ep.turn1_target_id,
            "turn1_expected_unit": ep.turn1_expected_unit,
            "turn1_truth_num": t1_frac.numerator,
            "turn1_truth_den": t1_frac.denominator,
            "turn1_expected_value_float": ep.turn1_expected_value_float,
            "turn1_prompt": ep.turn1_prompt,
            "turn1_turns": _serialize_turn_list(ep.turn1_turns),
            "target_node_id": ep.target_node_id,
            "expected_unit": ep.expected_unit,
            "truth_num": t2_frac.numerator,
            "truth_den": t2_frac.denominator,
            "expected_value_float": ep.expected_value_float,
            "expected_retracted_ids": list(ep.expected_retracted_ids),
            "pre_retraction_value_float": ep.pre_retraction_value_float,
            "stale_target_values_float": list(ep.stale_target_values_float),
            "preserved_truth": pres_truth,
            "turn2_prompt": ep.turn2_prompt,
            "turn2_turns": _serialize_turn_list(ep.turn2_turns),
        }
        lines.append(json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_v3_lock_doc(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    ds_pilot_50 = generate_multiturn_dataset(seed=V3_PILOT_SEED, count=V3_PILOT_COUNT)
    ds_eval_800 = generate_multiturn_dataset(seed=V3_EVAL_SEED, count=V3_EVAL_COUNT)
    ds_val_160 = generate_multiturn_dataset(seed=V3_VAL_SEED, count=V3_VAL_COUNT)
    ds_ablation0_800 = generate_procedural_dataset(seed=V3_EVAL_SEED, count=V3_EVAL_COUNT)

    pilot_sha = sha256_bytes(canonicalize_multiturn_jsonl(ds_pilot_50))
    eval_sha = sha256_bytes(canonicalize_multiturn_jsonl(ds_eval_800))
    val_sha = sha256_bytes(canonicalize_multiturn_jsonl(ds_val_160))

    cond_shas: Dict[str, Dict[str, Any]] = {}
    for cond in ("A", "B", "C", "D", "E"):
        sub = [e for e in ds_eval_800 if e.condition == cond]
        cond_shas[cond] = {
            "count": len(sub),
            "sha256": sha256_bytes(canonicalize_multiturn_jsonl(sub)),
        }

    ablation0_lines = [
        json.dumps(
            {
                "episode_id": e.episode_id,
                "condition": e.condition,
                "target_node_id": e.target_node_id,
                "expected_unit": e.expected_unit,
                "expected_value_float": e.expected_value_float,
                "expected_retracted_ids": list(e.expected_retracted_ids),
                "prompt": e.prompt,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        for e in ds_ablation0_800
    ]
    ablation0_sha = sha256_bytes(("\n".join(ablation0_lines) + "\n").encode("utf-8"))

    tracked_files = (
        "gyeol_bench/oracle.py",
        "gyeol_bench/_registry.py",
        "gyeol_bench/__init__.py",
        "eval.yaml",
        "scripts/check_zero_core.py",
    )
    file_hashes = {
        rel: sha256_bytes((repo_root / rel).read_bytes())
        for rel in tracked_files
    }

    baseline_prompt_sha = sha256_bytes(V3_MULTITURN_BASELINE_SYSTEM_PROMPT.encode("utf-8"))
    lineage_prompt_sha = sha256_bytes(V3_MULTITURN_LINEAGE_K2_SYSTEM_PROMPT.encode("utf-8"))

    core: Dict[str, Any] = {
        "schema_version": "3.0.0",
        "lock_stage": "pre_v3_live_pilot",
        "governance_grade": "[안전]",
        "primary_task": "gyeol_bench/retraction_dag_multiturn",
        "ablation_zero_task": "gyeol_bench/retraction_dag",
        "k3_redraw_accounting": {
            "total_allowed_redraws": 1,
            "consumed_by_v3_transition": 1,
            "remaining_post_freeze_redraws": 0,
            "fallback_on_v3_pilot_saturation": "Option_C_Clean_Archive_Halt",
        },
        "source_files_sha256": file_hashes,
        "locked_system_prompts": {
            "v3.0-multiturn-baseline": {
                "sha256": baseline_prompt_sha,
                "text": V3_MULTITURN_BASELINE_SYSTEM_PROMPT,
            },
            "v3.0-multiturn-lineage-k2": {
                "sha256": lineage_prompt_sha,
                "text": V3_MULTITURN_LINEAGE_K2_SYSTEM_PROMPT,
            },
        },
        "datasets": {
            "v3_pilot_n50": {
                "seed": V3_PILOT_SEED,
                "sample_count": V3_PILOT_COUNT,
                "canonical_jsonl_sha256": pilot_sha,
            },
            "v3_main_eval_n800": {
                "seed": V3_EVAL_SEED,
                "sample_count": V3_EVAL_COUNT,
                "canonical_jsonl_sha256": eval_sha,
                "conditions": cond_shas,
            },
            "v3_pr_validation_n160": {
                "seed": V3_VAL_SEED,
                "sample_count": V3_VAL_COUNT,
                "canonical_jsonl_sha256": val_sha,
            },
            "ablation0_v2_single_prompt_n800": {
                "seed": V3_EVAL_SEED,
                "sample_count": V3_EVAL_COUNT,
                "summary_jsonl_sha256": ablation0_sha,
            },
        },
    }
    lock_sha = sha256_bytes(
        json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    return {**core, "lock_manifest_sha256": lock_sha}


def verify_v3_lock_on_disk(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    expected = build_v3_lock_doc(repo_root)
    lock_path = repo_root / "manifests" / "v3-multiturn-prepilot-lock.json"
    violations: List[str] = []
    if not lock_path.exists():
        violations.append(f"Missing {lock_path}")
    else:
        disk = json.loads(lock_path.read_text(encoding="utf-8"))
        if disk != expected:
            violations.append("manifests/v3-multiturn-prepilot-lock.json out of sync with deterministic recomputation.")
    ok = len(violations) == 0
    return {
        "ok": ok,
        "grade": "[안전]" if ok else "[금지]",
        "lockManifestSha256": expected["lock_manifest_sha256"],
        "v3PilotN50Sha256": expected["datasets"]["v3_pilot_n50"]["canonical_jsonl_sha256"],
        "v3MainN800Sha256": expected["datasets"]["v3_main_eval_n800"]["canonical_jsonl_sha256"],
        "baselinePromptSha256": expected["locked_system_prompts"]["v3.0-multiturn-baseline"]["sha256"],
        "lineageK2PromptSha256": expected["locked_system_prompts"]["v3.0-multiturn-lineage-k2"]["sha256"],
        "violations": violations,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify manifests/v3-multiturn-prepilot-lock.json.")
    parser.add_argument("--verify", action="store_true", help="Verify existing lock manifest without writing")
    args = parser.parse_args(argv)

    if args.verify:
        rep = verify_v3_lock_on_disk(REPO_ROOT)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0 if rep["ok"] else 1

    doc = build_v3_lock_doc(REPO_ROOT)
    out_path = REPO_ROOT / "manifests" / "v3-multiturn-prepilot-lock.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rep = verify_v3_lock_on_disk(REPO_ROOT)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
