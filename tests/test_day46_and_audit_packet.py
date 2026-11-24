"""Unit tests for Day-46 cryptographic baseline freeze, Day-60 empirical results manifest, blinded N=160 audit packet, and inspect_evals artefacts."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestDay46AndHumanAuditPacket(unittest.TestCase):
    def test_day46_manifest_untouched_and_rfc6962_merkle_root_deterministic(self) -> None:
        mod = _load_script_module("build_day46_manifest", REPO_ROOT / "scripts" / "build_day46_manifest.py")
        rep = mod.verify_day46_on_disk(REPO_ROOT)
        self.assertTrue(rep["ok"], msg=f"Day-46 manifest verification failed: {rep['violations']}")
        self.assertEqual(rep["grade"], "[안전]")
        self.assertEqual(rep["leafCount"], 23)
        self.assertEqual(
            rep["manifestBodySha256"],
            "a542492ed604bd2b3b2b0e02e0fde3c6c8ee856082f3e27d952f55a74ada68d4",
        )
        self.assertEqual(
            rep["rfc6962MerkleRootHex"],
            "d54cc303f3ab11521356e563610da53b799d6b9d9619017d51de43a08ee6a634",
        )

    def test_day60_empirical_results_manifest_complete_and_bounded(self) -> None:
        day60_path = REPO_ROOT / "manifests" / "day60-empirical-results.json"
        self.assertTrue(day60_path.exists())
        doc = json.loads(day60_path.read_text(encoding="utf-8"))
        self.assertEqual(doc["governance_grade"], "[안전]")
        self.assertFalse(doc["k6_claim_bounds_and_gate_verdicts"]["positive_k2_prompt_claim_permitted"])
        self.assertEqual(
            doc["pre_experiment_seals_preserved"]["day46_manifest_body_sha256"],
            "a542492ed604bd2b3b2b0e02e0fde3c6c8ee856082f3e27d952f55a74ada68d4",
        )
        self.assertEqual(
            doc["phase3_raw_bundle_provenance"]["bundle_sha256"],
            "785beb4c3a67fc85fadd985261a484253e2e686c483fb5a130c8113703f0ba8e",
        )
        self.assertEqual(doc["cumulative_resource_ledger"]["aistudio_paid_batch_pilot_usd"], 1.130102)
        self.assertEqual(doc["cumulative_resource_ledger"]["colab_cumulative_used_cu"], 38.23)
        self.assertEqual(len(doc["branch2_open_weight_n800_models"]), 3)

    def test_blinded_human_audit_packet_zero_ground_truth_leakage(self) -> None:
        mod = _load_script_module("build_human_audit_packet", REPO_ROOT / "scripts" / "build_human_audit_packet.py")
        audit_dir = REPO_ROOT / "audit_packet"
        rep = mod.verify_packet_on_disk(audit_dir)
        self.assertTrue(rep["ok"], msg=f"Human audit packet verification failed: {rep['violations']}")
        self.assertEqual(rep["grade"], "[안전]")
        self.assertGreaterEqual(rep["totalAuditItems"], 150)
        self.assertEqual(rep["totalAuditItems"], 160)

        csv_raw = (audit_dir / "blinded_annotator_sheet_n160.csv").read_text(encoding="utf-8")
        reader = list(csv.DictReader(io.StringIO(csv_raw)))
        self.assertEqual(len(reader), 160)
        for row in reader:
            self.assertEqual(row["annotator_value"], "")
            self.assertEqual(row["annotator_retracted_ids"], "")
            self.assertNotIn("expected_value_float", row)
            self.assertNotIn("condition", row)
            self.assertIn('"notes":', row["prompt"])

        outreach_path = audit_dir / "OUTREACH_DRAFTS.md"
        self.assertTrue(outreach_path.exists())
        outreach_text = outreach_path.read_text(encoding="utf-8")
        self.assertIn("Draft 1", outreach_text)
        self.assertIn("Draft 2", outreach_text)
        self.assertIn("Draft 3", outreach_text)

    def test_inspect_evals_validity_and_summary_artefacts_complete(self) -> None:
        validity_path = REPO_ROOT / "agent_artefacts" / "retraction_dag_0.1.0" / "validity" / "VALIDITY_REPORT.md"
        saturation_path = REPO_ROOT / "agent_artefacts" / "retraction_dag_0.1.0" / "validity" / "SATURATION_REPORT.md"
        summary_path = REPO_ROOT / "agent_artefacts" / "retraction_dag_0.1.0" / "review" / "SUMMARY.md"
        checklist_path = REPO_ROOT / "EVALUATION_CHECKLIST.md"
        self.assertTrue(validity_path.exists())
        self.assertTrue(saturation_path.exists())
        self.assertTrue(summary_path.exists())

        v_text = validity_path.read_text(encoding="utf-8")
        sat_text = saturation_path.read_text(encoding="utf-8")
        s_text = summary_path.read_text(encoding="utf-8")
        c_text = checklist_path.read_text(encoding="utf-8")

        for text in (v_text, s_text, c_text):
            self.assertNotIn("[PENDING_PHASE3_N800_OPEN_K2]", text)
            self.assertNotIn("[PENDING_PHASE3_N800_RUN]", text)

        self.assertIn("2026-05-08", v_text)
        self.assertIn("SymPy 1.14.0", v_text)
        self.assertIn("No positive `K2` superiority claim is made.", v_text)
        self.assertIn("-91.8%", v_text)
        self.assertIn("513 / 800", v_text)
        self.assertIn("410 / 800", v_text)
        self.assertIn("578 / 800", v_text)
        self.assertIn("No Positive `K2` Prompt Claim", s_text)
        self.assertIn("Ablation 0", v_text)
        self.assertIn("Clean-T1", v_text)
        self.assertIn("Option C", sat_text)
        self.assertIn("contents[0]", sat_text)

        challenge_text = (REPO_ROOT / "manifests" / "DAY46_PR_CHALLENGE.md").read_text(encoding="utf-8")
        self.assertIn("Ablation 0", challenge_text)
        self.assertIn("v3.0-multiturn-baseline", challenge_text)

    def test_v3_multiturn_prepilot_lock_deterministic(self) -> None:
        mod = _load_script_module("build_v3_prepilot_lock", REPO_ROOT / "scripts" / "build_v3_prepilot_lock.py")
        rep = mod.verify_v3_lock_on_disk(REPO_ROOT)
        self.assertTrue(rep["ok"], msg=f"v3 pre-pilot lock verification failed: {rep['violations']}")
        self.assertEqual(rep["grade"], "[안전]")
        lock = json.loads((REPO_ROOT / "manifests" / "v3-multiturn-prepilot-lock.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["datasets"]["v3_pilot_n50"]["seed"], 20261151)
        self.assertEqual(lock["datasets"]["v3_pilot_n50"]["sample_count"], 50)
        self.assertEqual(lock["datasets"]["v3_main_eval_n800"]["seed"], 20261122)
        self.assertEqual(lock["datasets"]["v3_main_eval_n800"]["sample_count"], 800)
        self.assertIn("v3.0-multiturn-baseline", lock["locked_system_prompts"])
        self.assertIn("v3.0-multiturn-lineage-k2", lock["locked_system_prompts"])


if __name__ == "__main__":
    unittest.main()
