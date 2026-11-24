#!/usr/bin/env python3
"""Unit tests for Track 1C (v0.2 Frontier Context-Eviction Ablation)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gyeol_bench import generate_multiturn_dataset
import run_frontier_eviction_v02 as fe


class TestFrontierEvictionV02(unittest.TestCase):
    def setUp(self) -> None:
        self.ep = generate_multiturn_dataset(seed=20261151, count=5)[0]
        self.long_notes = "A" * 600
        self.raw1 = json.dumps({
            "notes": self.long_notes,
            "step_id": self.ep.turn1_target_id,
            "value": 42.0,
            "unit": self.ep.turn1_expected_unit,
            "retracted_ids": [],
        })

    def test_c1_untruncated_notes_and_roles(self) -> None:
        c1 = fe.build_c1_self_summary_contents(self.ep, self.raw1)
        self.assertEqual([m["role"] for m in c1], ["user", "model", "user"])
        self.assertIn(self.long_notes, c1[0]["parts"][0]["text"])
        self.assertNotIn(self.ep.turn1_prompt, c1[0]["parts"][0]["text"])

    def test_c2_materialized_ledger_solvable_and_evicted(self) -> None:
        c2 = fe.build_c2_materialized_ledger_contents(self.ep, self.raw1)
        self.assertEqual([m["role"] for m in c2], ["user", "model", "user"])
        text = c2[0]["parts"][0]["text"]
        self.assertIn("turn1_primitive_bindings", text)
        self.assertIn("turn1_derivation_rules", text)
        self.assertIn("stale_cached_turn1_intermediates", text)
        self.assertNotIn(self.ep.turn1_prompt, text)

    def test_extract_response_fields_strips_thoughts(self) -> None:
        item = {
            "response": {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "parts": [
                            {"thought": True, "text": "internal reasoning"},
                            {"text": "{\"step_id\":\"d6\"}"},
                        ]
                    },
                }],
                "usageMetadata": {
                    "promptTokenCount": 100,
                    "candidatesTokenCount": 20,
                    "thoughtsTokenCount": 50,
                },
            }
        }
        txt, mtok, usage, err = fe._extract_response_fields(item)
        self.assertEqual(txt, "{\"step_id\":\"d6\"}")
        self.assertFalse(mtok)
        self.assertIsNone(err)
        self.assertEqual(usage["thoughtsTokenCount"], 50)

    def test_safety_guards_reject_latest_and_require_paid_env(self) -> None:
        with self.assertRaises(ValueError):
            fe._assert_safety_guards("models/gemini-4-carbon-latest", live=False)
        old = os.environ.pop("AI_STUDIO_PAID_TIER_CONFIRMED", None)
        try:
            with self.assertRaises(RuntimeError):
                fe._assert_safety_guards("models/gemini-4-carbon", live=True)
        finally:
            if old is not None:
                os.environ["AI_STUDIO_PAID_TIER_CONFIRMED"] = old

    def test_live_end_to_end_mocked_batch_exercises_real_oracle(self) -> None:
        def fake_batch(model, display_name, reqs, api_key):
            items = []
            for key, _ in reqs:
                items.append({
                    "metadata": {"key": key},
                    "response": {
                        "candidates": [{
                            "finishReason": "STOP",
                            "content": {"parts": [{"text": json.dumps({
                                "notes": "mock",
                                "step_id": "d4" if "t1" in display_name else "d6",
                                "value": 1.0,
                                "unit": "M_USD",
                                "retracted_ids": [],
                            })}]},
                        }],
                        "usageMetadata": {
                            "promptTokenCount": 10,
                            "candidatesTokenCount": 5,
                            "thoughtsTokenCount": 2,
                        },
                    },
                })
            return f"batches/mock-{display_name}", items

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            spend_path = td_path / "spend.json"
            spend_path.write_text(json.dumps({"spentUsd": 1.130102}), encoding="utf-8")
            with patch.dict(os.environ, {"AI_STUDIO_PAID_TIER_CONFIRMED": "1", "GEMINI_API_KEY": "dummy"}):
                with patch.object(fe, "_batch_generate_v1beta", side_effect=fake_batch):
                    res = fe.run_ablation(
                        "models/gemini-4-carbon",
                        live=True,
                        out_path=td_path / "out.json",
                        checkpoint_dir=td_path / "ckpt",
                        spend_path=spend_path,
                    )
            self.assertEqual(res["mode"], "live")
            self.assertEqual(res["count"], 50)
            self.assertEqual(res["finish_reason_max_tokens"], 0)
            self.assertIn("PARSE_REJECT", res["C1_self_summary_eviction"]["itt_verdicts"])
            self.assertIn("CONDITIONAL_ARITH_ERROR", res["C2_materialized_ledger_eviction"]["itt_verdicts"])


if __name__ == "__main__":
    unittest.main()
