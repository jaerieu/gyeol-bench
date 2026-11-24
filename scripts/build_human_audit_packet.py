#!/usr/bin/env python3
"""Deterministic builder and verifier for the blinded N=160 (>= 150) external human audit packet.

Generates inside gyeol-bench/audit_packet/:
- blinded_annotator_sheet_n160.jsonl (ground-truth answers and condition labels stripped)
- blinded_annotator_sheet_n160.csv   (spreadsheet-ready blinded annotator worksheet)
- SEALED_KEY_COMMITMENT.json         (SHA-256 pre-commitment to the out-of-repo sealed answer key)

Writes the sealed answer key strictly OUTSIDE the public repository (via --sealed-key-out)
to prevent accidental annotator unblinding while guaranteeing cryptographic non-repudiation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gyeol_bench import ExactRationalOracle, RetractionEpisode, generate_procedural_dataset
from gyeol_bench.oracle import convert_from_base_fraction

DATASET_SEED = 20261122
DATASET_COUNT = 800
AUDIT_SAMPLING_SEED = 20261146
STRATIFIED_QUOTAS: Tuple[Tuple[str, int], ...] = (
    ("A", 40),
    ("B", 30),
    ("C", 30),
    ("D", 30),
    ("E", 30),
)

FORBIDDEN_BLINDED_KEYS = {
    "expected_value_float",
    "truth_numerator",
    "truth_denominator",
    "expected_retracted_ids",
    "pre_retraction_value_float",
    "preserved_truth",
    "condition",
    "category",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def select_stratified_sample(
    episodes: List[RetractionEpisode],
    sampling_seed: int = AUDIT_SAMPLING_SEED,
) -> List[RetractionEpisode]:
    rng = random.Random(sampling_seed)
    by_cond: Dict[str, List[RetractionEpisode]] = {c: [] for c, _ in STRATIFIED_QUOTAS}
    for ep in episodes:
        by_cond[ep.condition].append(ep)

    selected: List[RetractionEpisode] = []
    for cond, quota in STRATIFIED_QUOTAS:
        pool = list(by_cond[cond])
        if len(pool) < quota:
            raise ValueError(f"Insufficient pool for condition {cond}: {len(pool)} < {quota}")
        picked = rng.sample(pool, quota)
        selected.extend(picked)

    rng.shuffle(selected)
    return selected


def build_packet_payloads(
    dataset_seed: int = DATASET_SEED,
    sampling_seed: int = AUDIT_SAMPLING_SEED,
) -> Tuple[str, str, str, Dict[str, Any]]:
    episodes = generate_procedural_dataset(seed=dataset_seed, count=DATASET_COUNT)
    sampled = select_stratified_sample(episodes, sampling_seed=sampling_seed)

    blinded_rows: List[Dict[str, Any]] = []
    sealed_entries: List[Dict[str, Any]] = []

    for idx, ep in enumerate(sampled, start=1):
        audit_id = f"audit-{idx:03d}"
        oracle = ExactRationalOracle()
        for t in ep.turns:
            oracle.append_turn(t)
        snap = oracle.evaluate()
        base_val, dim, _ = snap.active_values_base[ep.target_node_id]
        truth_frac = convert_from_base_fraction(base_val, dim, ep.expected_unit)

        preserved_req: List[Dict[str, str]] = []
        preserved_truth: Dict[str, Dict[str, Any]] = {}
        for pid, punit in ep.preserved_targets:
            preserved_req.append({"node_id": pid, "expected_unit": punit})
            pbase, pdim, _ = snap.active_values_base[pid]
            pfrac = convert_from_base_fraction(pbase, pdim, punit)
            preserved_truth[pid] = {
                "unit": punit,
                "numerator": pfrac.numerator,
                "denominator": pfrac.denominator,
                "value_float": float(pfrac),
            }

        blinded_rows.append(
            {
                "audit_id": audit_id,
                "episode_id": ep.episode_id,
                "target_node_id": ep.target_node_id,
                "expected_unit": ep.expected_unit,
                "required_preserved_nodes": preserved_req,
                "prompt": ep.prompt,
                "annotator_id": "",
                "annotator_value": "",
                "annotator_unit": "",
                "annotator_retracted_ids": "",
                "annotator_preserved_json": "",
                "construct_clarity_verdict": "",
                "annotator_notes": "",
            }
        )

        sealed_entries.append(
            {
                "audit_id": audit_id,
                "episode_id": ep.episode_id,
                "condition": ep.condition,
                "category": ep.category,
                "target_node_id": ep.target_node_id,
                "expected_unit": ep.expected_unit,
                "truth_numerator": truth_frac.numerator,
                "truth_denominator": truth_frac.denominator,
                "expected_value_float": ep.expected_value_float,
                "expected_retracted_ids": list(ep.expected_retracted_ids),
                "pre_retraction_value_float": ep.pre_retraction_value_float,
                "preserved_truth": preserved_truth,
            }
        )

    jsonl_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n" for row in blinded_rows
    )

    csv_buf = io.StringIO(newline="")
    fieldnames = [
        "audit_id",
        "episode_id",
        "target_node_id",
        "expected_unit",
        "required_preserved_nodes_json",
        "prompt",
        "annotator_id",
        "annotator_value",
        "annotator_unit",
        "annotator_retracted_ids",
        "annotator_preserved_json",
        "construct_clarity_verdict",
        "annotator_notes",
    ]
    writer = csv.DictWriter(csv_buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in blinded_rows:
        writer.writerow(
            {
                "audit_id": row["audit_id"],
                "episode_id": row["episode_id"],
                "target_node_id": row["target_node_id"],
                "expected_unit": row["expected_unit"],
                "required_preserved_nodes_json": json.dumps(row["required_preserved_nodes"]),
                "prompt": row["prompt"],
                "annotator_id": "",
                "annotator_value": "",
                "annotator_unit": "",
                "annotator_retracted_ids": "",
                "annotator_preserved_json": "",
                "construct_clarity_verdict": "",
                "annotator_notes": "",
            }
        )
    csv_text = csv_buf.getvalue()

    sealed_doc = {
        "schema_version": "1.0.0",
        "notice": "SEALED ANSWER KEY - DO NOT COMMIT TO PUBLIC REPOSITORY BEFORE BLINDED HUMAN AUDIT COMPLETION.",
        "dataset_seed": dataset_seed,
        "audit_sampling_seed": sampling_seed,
        "total_audit_items": len(sealed_entries),
        "stratification_counts": {c: q for c, q in STRATIFIED_QUOTAS},
        "entries": sealed_entries,
    }
    sealed_json_text = json.dumps(sealed_doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    n_total = len(sealed_entries)
    bound_150 = round(1.0 - (0.05 ** (1.0 / 150.0)), 6)
    bound_n = round(1.0 - (0.05 ** (1.0 / float(n_total))), 6)

    commitment_doc: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "task_identifier": "gyeol_bench/retraction_dag",
        "governance_grade": "[안전]",
        "dataset_seed": dataset_seed,
        "dataset_total_episodes": DATASET_COUNT,
        "audit_sampling_seed": sampling_seed,
        "total_audit_items": n_total,
        "minimum_required_items": 150,
        "stratification_counts": {c: q for c, q in STRATIFIED_QUOTAS},
        "zero_error_one_sided_95pct_upper_bound_at_150": bound_150,
        "zero_error_one_sided_95pct_upper_bound_at_160": bound_n,
        "blinded_jsonl_path": "audit_packet/blinded_annotator_sheet_n160.jsonl",
        "blinded_jsonl_sha256": _sha256_bytes(jsonl_text.encode("utf-8")),
        "blinded_csv_path": "audit_packet/blinded_annotator_sheet_n160.csv",
        "blinded_csv_sha256": _sha256_bytes(csv_text.encode("utf-8")),
        "sealed_answer_key_sha256": _sha256_bytes(sealed_json_text.encode("utf-8")),
        "blinding_guarantees": [
            "All ground-truth target values, rational numerators/denominators, and pre-retraction values are stripped.",
            "All ground-truth retracted_ids and preserved subgraph numeric answers are stripped.",
            "Factorial condition labels (A..E) and normal/perturbed category labels are stripped and shuffled.",
            "Sealed answer key is stored out-of-repository and bound via sealed_answer_key_sha256.",
        ],
    }
    return jsonl_text, csv_text, sealed_json_text, commitment_doc


def verify_packet_on_disk(audit_dir: Path, sealed_key_path: Path | None = None) -> Dict[str, Any]:
    jsonl_text, csv_text, sealed_json_text, expected_commit = build_packet_payloads()
    violations: List[str] = []

    jsonl_path = audit_dir / "blinded_annotator_sheet_n160.jsonl"
    csv_path = audit_dir / "blinded_annotator_sheet_n160.csv"
    commit_path = audit_dir / "SEALED_KEY_COMMITMENT.json"

    for p in (jsonl_path, csv_path, commit_path):
        if not p.exists():
            violations.append(f"Missing required audit packet file: {p}")

    if not violations:
        disk_jsonl = jsonl_path.read_text(encoding="utf-8")
        disk_csv = csv_path.read_text(encoding="utf-8")
        disk_commit = json.loads(commit_path.read_text(encoding="utf-8"))

        if _sha256_bytes(disk_jsonl.encode("utf-8")) != expected_commit["blinded_jsonl_sha256"]:
            violations.append("blinded_annotator_sheet_n160.jsonl SHA-256 mismatch.")
        if _sha256_bytes(disk_csv.encode("utf-8")) != expected_commit["blinded_csv_sha256"]:
            violations.append("blinded_annotator_sheet_n160.csv SHA-256 mismatch.")
        if disk_commit != expected_commit:
            violations.append("SEALED_KEY_COMMITMENT.json content mismatch.")

        lines = [ln for ln in disk_jsonl.splitlines() if ln.strip()]
        if len(lines) < 150:
            violations.append(f"Audit sample size {len(lines)} < 150 minimum.")
        for idx, ln in enumerate(lines, start=1):
            obj = json.loads(ln)
            leaked = FORBIDDEN_BLINDED_KEYS & set(obj.keys())
            if leaked:
                violations.append(f"Line {idx} leaked forbidden ground-truth keys: {sorted(leaked)}")

        # Ensure no sealed answer key accidentally sits inside audit_dir
        for child in audit_dir.iterdir():
            if "sealed" in child.name.lower() and child.name != "SEALED_KEY_COMMITMENT.json":
                violations.append(f"Unblinding hazard: sealed key file present in public audit_dir: {child.name}")

        if sealed_key_path is not None:
            if not sealed_key_path.exists():
                violations.append(f"Sealed answer key missing at {sealed_key_path}")
            else:
                disk_sealed = sealed_key_path.read_text(encoding="utf-8")
                if _sha256_bytes(disk_sealed.encode("utf-8")) != expected_commit["sealed_answer_key_sha256"]:
                    violations.append("Out-of-repo sealed answer key SHA-256 does not match commitment.")

    ok = len(violations) == 0
    return {
        "ok": ok,
        "grade": "[안전]" if ok else "[금지]",
        "totalAuditItems": expected_commit["total_audit_items"],
        "sealedAnswerKeySha256": expected_commit["sealed_answer_key_sha256"],
        "blindedJsonlSha256": expected_commit["blinded_jsonl_sha256"],
        "violations": violations,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify the blinded N=160 human audit packet.")
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=REPO_ROOT / "audit_packet",
        help="Target public audit_packet directory inside gyeol-bench",
    )
    parser.add_argument(
        "--sealed-key-out",
        type=Path,
        default=None,
        help="Out-of-repo path where the sealed answer key JSON is written or verified",
    )
    parser.add_argument("--verify", action="store_true", help="Verify existing files without writing")
    args = parser.parse_args(argv)

    if args.sealed_key_out is not None:
        resolved_out = args.sealed_key_out.resolve()
        if REPO_ROOT.resolve() in resolved_out.parents or resolved_out == REPO_ROOT.resolve():
            print(
                json.dumps(
                    {
                        "ok": False,
                        "grade": "[금지]",
                        "error": "Sealed answer key MUST be stored outside the gyeol-bench repository.",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 1

    if args.verify:
        rep = verify_packet_on_disk(args.audit_dir, sealed_key_path=args.sealed_key_out)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0 if rep["ok"] else 1

    jsonl_text, csv_text, sealed_json_text, commitment_doc = build_packet_payloads()
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    (args.audit_dir / "blinded_annotator_sheet_n160.jsonl").write_text(jsonl_text, encoding="utf-8")
    (args.audit_dir / "blinded_annotator_sheet_n160.csv").write_text(csv_text, encoding="utf-8")
    (args.audit_dir / "SEALED_KEY_COMMITMENT.json").write_text(
        json.dumps(commitment_doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.sealed_key_out is not None:
        args.sealed_key_out.parent.mkdir(parents=True, exist_ok=True)
        args.sealed_key_out.write_text(sealed_json_text, encoding="utf-8")

    rep = verify_packet_on_disk(args.audit_dir, sealed_key_path=args.sealed_key_out)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
