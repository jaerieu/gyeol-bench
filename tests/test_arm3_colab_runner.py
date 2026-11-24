"""Unit and integration tests for `scripts/analyze_arm3_results.py` and `scripts/run_arm3_colab_gpu.py`.

Covers:
1. Exact two-sided binomial McNemar (`math.comb`), step-down Holm-Bonferroni correction across evaluated
   (model x arm) pairs, and K2 (`b - c >= 40` at N=800) / K4 (`Condition A` failure count `<= 5 / 400`) gates.
2. Audit `#19` mandatory disclosures (`extra_inference_turns`, `repair_triggered_count`, diagnostic oracle note
   for `arm3b_llm_repair`, and strictly separated `secondary_symbolic_ceilings` with `is_primary_k2_eligible: false`).
3. End-to-end `N=10` mock run via `scripts/run_arm3_colab_gpu.py --mock-backend` testing `arm1,arm3a,arm3b,arm3ab_hybrid`.
4. Chunked checkpoint `--resume` (stop after 5 episodes, resume to 10 with zero duplicate `episodeId`s).
5. `--turn1-cache` reuse avoiding duplicate Turn-1 / Arm-1 inference.
6. Zero numeric target (`d6`) leakage inside generated Turn-2 `arm3a` directives and `arm3b` repair prompts.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Dict, List
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gyeol_bench.arm3_ledger_hook import (
    augment_turn2_prompt_with_ledger,
    build_arm3a_ledger_from_prompts,
)
from gyeol_bench.arm3_state_diff import (
    build_arm3b_repair_prompt,
    compute_state_diff_snapshot,
    pass1_audit_candidate,
)
from gyeol_bench.oracle import generate_multiturn_dataset

ANALYZE_SCRIPT_PATH = REPO_ROOT / "scripts" / "analyze_arm3_results.py"
RUNNER_SCRIPT_PATH = REPO_ROOT / "scripts" / "run_arm3_colab_gpu.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestAnalyzeArm3ResultsAndColabRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analyzer = _load_module("analyze_arm3_results", ANALYZE_SCRIPT_PATH)
        cls.episodes_n10 = generate_multiturn_dataset(seed=20261151, count=10)

    def test_exact_mcnemar_holm_bonferroni_and_k2_k4_gate_math(self) -> None:
        # 1. Exact McNemar binomial p-value verification
        self.assertEqual(self.analyzer.exact_two_sided_mcnemar_p(0, 0), 1.0)
        self.assertEqual(self.analyzer.exact_two_sided_mcnemar_p(10, 10), 1.0)
        # For b=45, c=5 (n=50, k=5): 2 * sum_{i=0..5} C(50, i) / 2^50
        expected_p_45_5 = 2.0 * sum(math.comb(50, i) for i in range(6)) / (2**50)
        self.assertAlmostEqual(
            self.analyzer.exact_two_sided_mcnemar_p(45, 5),
            expected_p_45_5,
            places=15,
        )

        # 2. Step-down Holm-Bonferroni monotonicity and order preservation
        raw_ps = [0.04, 0.005, 0.015, 0.20]
        adj_ps = self.analyzer.holm_bonferroni_adjust(raw_ps)
        # Sorted order:
        # rank 0: p=0.005 * 4 = 0.020 (idx 1)
        # rank 1: p=0.015 * 3 = 0.045 (idx 2)
        # rank 2: p=0.040 * 2 = 0.080 (idx 0)
        # rank 3: p=0.200 * 1 = 0.200 (idx 3)
        self.assertAlmostEqual(adj_ps[1], 0.020, places=12)
        self.assertAlmostEqual(adj_ps[2], 0.045, places=12)
        self.assertAlmostEqual(adj_ps[0], 0.080, places=12)
        self.assertAlmostEqual(adj_ps[3], 0.200, places=12)

        # 3. Synthetic N=800 manifest analysis check (400 Condition A, 400 Condition B..E)
        synth_rows: List[Dict[str, Any]] = []
        for idx in range(800):
            cond = "A" if idx % 2 == 0 else ("B", "C", "D", "E")[(idx // 2) % 4]
            t1_ok = idx % 10 != 9
            # Arm 1 passes 398/400 on A, 120/400 on B..E (total 518/800)
            arm1_ok = (idx < 796) if cond == "A" else (idx < 240)
            arm1_verdict = "PASS" if arm1_ok else ("OVER_RETRACTION" if cond == "A" else "STALE_STATE_RETENTION")

            # Arm 3A structural_only passes 400/400 on A (0/400 K4 false alarms), and 190/400 on B..E (total 590/800, b-c = +72 >= 40)
            arm3a_struct_ok = True if cond == "A" else (idx < 380)
            arm3a_struct_verdict = "PASS" if arm3a_struct_ok else "CONDITIONAL_ARITH_ERROR"

            # Arm 3B llm_repair passes 397/400 on A (3/400 <= 5/400 K4 pass), and 220/400 on B..E (total 617/800, b-c = +99 >= 40)
            arm3b_repair_ok = (idx < 794) if cond == "A" else (idx < 440)
            arm3b_repair_verdict = "PASS" if arm3b_repair_ok else "CONDITIONAL_ARITH_ERROR"
            repair_trig = not arm1_ok

            synth_rows.append(
                {
                    "episodeId": f"gb-{idx+1:04d}",
                    "condition": cond,
                    "category": "normal" if cond == "A" else "perturbed",
                    "modelKey": "synth-phi4",
                    "repoId": "microsoft/phi-4",
                    "dtype": "nf4",
                    "turn1Ok": t1_ok,
                    "turn1Verdict": "PASS" if t1_ok else "CONDITIONAL_ARITH_ERROR",
                    "arm1Ok": arm1_ok,
                    "arm1Verdict": arm1_verdict,
                    "arm3aPromptRawOk": arm3a_struct_ok,
                    "arm3aPromptRawVerdict": arm3a_struct_verdict,
                    "arm3aStructuralOnlyOk": arm3a_struct_ok,
                    "arm3aStructuralOnlyVerdict": arm3a_struct_verdict,
                    "arm3aFullCeilingOk": True,
                    "arm3aFullCeilingVerdict": "PASS",
                    "arm3bStructuralGateOk": arm3a_struct_ok,
                    "arm3bStructuralGateVerdict": arm3a_struct_verdict,
                    "arm3bLlmRepairOk": arm3b_repair_ok,
                    "arm3bLlmRepairVerdict": arm3b_repair_verdict,
                    "arm3bRepairTriggered": repair_trig,
                    "arm3bPass1Flags": ["STALE_TURN1_BINDING"] if repair_trig else [],
                    "arm3bExtraTurns": 1 if repair_trig else 0,
                    "arm3bSymbolicCeilingOk": True,
                    "arm3bSymbolicCeilingVerdict": "PASS",
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            ep_path = tmp_dir / "synth-phi4-episodes.jsonl"
            out_path = tmp_dir / "arm3-manifest.json"
            ep_path.write_text("\n".join(json.dumps(r) for r in synth_rows) + "\n", encoding="utf-8")

            rc = self.analyzer.main([str(ep_path), "--out", str(out_path)])
            self.assertEqual(rc, 0)
            self.assertTrue(out_path.exists())
            doc = json.loads(out_path.read_text(encoding="utf-8"))

            self.assertEqual(doc["governance_grade"], "[안전]")
            m0 = doc["models"][0]
            itt_contenders = m0["ittPrimary"]["primaryK2Contenders"]
            self.assertIn("arm3a_structural_only", itt_contenders)
            self.assertIn("arm3b_llm_repair", itt_contenders)

            c3a = itt_contenders["arm3a_structural_only"]
            self.assertTrue(c3a["is_primary_k2_eligible"])
            self.assertEqual(c3a["mcnemar"]["bMinusC"], 72)
            self.assertTrue(c3a["mcnemar"]["k2HolmGatePass"])
            self.assertEqual(c3a["summary"]["condAFailCount"], 0)
            self.assertTrue(c3a["summary"]["condAK4Pass"])

            c3b = itt_contenders["arm3b_llm_repair"]
            self.assertTrue(c3b["is_primary_k2_eligible"])
            self.assertEqual(c3b["mcnemar"]["bMinusC"], 99)
            self.assertTrue(c3b["mcnemar"]["k2HolmGatePass"])
            self.assertEqual(c3b["summary"]["condAFailCount"], 3)
            self.assertTrue(c3b["summary"]["condAK4Pass"])
            self.assertEqual(c3b["audit19_disclosures"]["repair_triggered_count"], 282)
            self.assertEqual(c3b["audit19_disclosures"]["extra_inference_turns"], 282)
            self.assertIn("STALE_TURN1_BINDING", c3b["audit19_disclosures"]["diagnostic_oracle_trigger_disclosure"])

            ceilings = m0["secondary_symbolic_ceilings"]
            self.assertFalse(ceilings["is_primary_k2_eligible"])
            self.assertEqual(ceilings["itt"]["arm3a_full_ceiling"]["passCount"], 800)
            self.assertFalse(ceilings["itt"]["arm3a_full_ceiling"]["is_primary_k2_eligible"])
            self.assertEqual(ceilings["itt"]["arm3b_symbolic_ceiling"]["passCount"], 800)
            self.assertFalse(ceilings["itt"]["arm3b_symbolic_ceiling"]["is_primary_k2_eligible"])

    def test_zero_target_numeric_leakage_across_arm3a_and_arm3b_turn2_prompts(self) -> None:
        for ep in self.episodes_n10:
            # 1. Verify Arm 3A augmented Turn-2 prompt has zero Turn-2 target numeric answer leakage
            ledger = build_arm3a_ledger_from_prompts(ep.turn1_prompt, ep.turn2_prompt, turn1_raw_output="")
            aug_p2 = augment_turn2_prompt_with_ledger(ep.turn2_prompt, ledger)
            directive_only = aug_p2[len(ep.turn2_prompt.rstrip()) :]

            t2_truth = ledger.target_truth_fraction
            t2_tokens = {f"{float(t2_truth):.6g}", str(float(t2_truth))}
            if t2_truth.denominator == 1:
                t2_tokens.add(str(int(t2_truth.numerator)))
            for tok in t2_tokens:
                if len(tok) >= 3:
                    self.assertNotIn(tok, directive_only, msg=f"Arm 3A directive leaked target {tok} on {ep.episode_id}")

            # 2. Verify Arm 3B repair prompt has zero Turn-2 target numeric answer leakage
            diff_snap = compute_state_diff_snapshot(
                ep.turn1_turns,
                ep.turn2_turns,
                target_node_id=ep.target_node_id,
                expected_unit=ep.expected_unit,
                preserved_targets=dict(ep.preserved_targets) or None,
            )
            bad_candidate = json.dumps(
                {
                    "notes": "stale or wrong value",
                    "step_id": "d6",
                    "value": ep.stale_target_values_float[0] if ep.stale_target_values_float else -777.0,
                    "unit": ep.expected_unit,
                    "retracted_ids": ["d1", "spurious"],
                }
            )
            p1 = pass1_audit_candidate(diff_snap, bad_candidate, turn1_raw_output=None)
            rep_prompt = build_arm3b_repair_prompt(diff_snap, p1, turn1_raw_output=None)
            for tok in t2_tokens:
                if len(tok) >= 3:
                    self.assertNotIn(tok, rep_prompt, msg=f"Arm 3B repair prompt leaked target {tok} on {ep.episode_id}")

    def test_end_to_end_mock_runner_resume_turn1_cache_and_analyzer_integration(self) -> None:
        if not RUNNER_SCRIPT_PATH.exists():
            self.skipTest("scripts/run_arm3_colab_gpu.py not yet present")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            run_dir = tmp_dir / "mock_run"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Run first 5 episodes (--limit 5, --chunk-size 2) with --mock-backend across all arms
            cmd_part1 = [
                sys.executable,
                str(RUNNER_SCRIPT_PATH),
                "--model-key",
                "mock-phi4",
                "--repo-id",
                "microsoft/phi-4",
                "--seed",
                "20261151",
                "--count",
                "10",
                "--limit",
                "5",
                "--chunk-size",
                "2",
                "--arms",
                "arm1,arm3a,arm3b,arm3ab_hybrid",
                "--out-dir",
                str(run_dir),
                "--mock-backend",
            ]
            proc1 = subprocess.run(cmd_part1, capture_output=True, text=True, check=False)
            self.assertEqual(proc1.returncode, 0, msg=f"Part 1 failed: {proc1.stderr}\n{proc1.stdout}")

            jsonl_files = sorted(run_dir.glob("*-episodes.jsonl"))
            self.assertEqual(len(jsonl_files), 1)
            ep_file = jsonl_files[0]
            rows_part1 = [json.loads(line) for line in ep_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows_part1), 5)

            # Step 2: Resume from 5 to 10 (--limit 10 --resume) and verify zero duplicate episodeIds
            cmd_part2 = [
                sys.executable,
                str(RUNNER_SCRIPT_PATH),
                "--model-key",
                "mock-phi4",
                "--repo-id",
                "microsoft/phi-4",
                "--seed",
                "20261151",
                "--count",
                "10",
                "--limit",
                "10",
                "--chunk-size",
                "3",
                "--arms",
                "arm1,arm3a,arm3b,arm3ab_hybrid",
                "--out-dir",
                str(run_dir),
                "--resume",
                "--mock-backend",
            ]
            proc2 = subprocess.run(cmd_part2, capture_output=True, text=True, check=False)
            self.assertEqual(proc2.returncode, 0, msg=f"Part 2 resume failed: {proc2.stderr}\n{proc2.stdout}")

            rows_part2 = [json.loads(line) for line in ep_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            ep_ids = [r["episodeId"] for r in rows_part2]
            self.assertEqual(len(ep_ids), 10)
            self.assertEqual(len(set(ep_ids)), 10, msg="Duplicate episodeIds found after --resume")

            # Check required schema fields on every row
            required_keys = {
                "episodeId",
                "condition",
                "turn1Ok",
                "arm1Ok",
                "arm1Verdict",
                "arm3aPromptRawOk",
                "arm3aPromptRawVerdict",
                "arm3aStructuralOnlyOk",
                "arm3aStructuralOnlyVerdict",
                "arm3aFullCeilingOk",
                "arm3bStructuralGateOk",
                "arm3bStructuralGateVerdict",
                "arm3bLlmRepairOk",
                "arm3bLlmRepairVerdict",
                "arm3bRepairTriggered",
                "arm3bPass1Flags",
                "arm3bSymbolicCeilingOk",
                "arm3abHybridOk",
                "arm3abHybridVerdict",
            }
            for row in rows_part2:
                self.assertTrue(required_keys.issubset(row.keys()), msg=f"Missing keys: {required_keys - set(row.keys())}")

            # Step 3: Run with --turn1-cache pointing to ep_file into a second output directory
            cache_run_dir = tmp_dir / "cache_run"
            cache_run_dir.mkdir(parents=True, exist_ok=True)
            cmd_cache = [
                sys.executable,
                str(RUNNER_SCRIPT_PATH),
                "--model-key",
                "mock-phi4-cached",
                "--repo-id",
                "microsoft/phi-4",
                "--seed",
                "20261151",
                "--count",
                "10",
                "--limit",
                "10",
                "--turn1-cache",
                str(ep_file),
                "--arms",
                "arm3a,arm3b,arm3ab_hybrid",
                "--out-dir",
                str(cache_run_dir),
                "--mock-backend",
            ]
            proc3 = subprocess.run(cmd_cache, capture_output=True, text=True, check=False)
            self.assertEqual(proc3.returncode, 0, msg=f"Cache run failed: {proc3.stderr}\n{proc3.stdout}")
            cached_jsonl = sorted(cache_run_dir.glob("*-episodes.jsonl"))[0]
            rows_cached = [json.loads(line) for line in cached_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows_cached), 10)
            for r_orig, r_cac in zip(rows_part2, rows_cached):
                self.assertEqual(r_orig["turn1RawOutput"], r_cac["turn1RawOutput"])
                self.assertEqual(r_orig["arm1RawOutput"], r_cac["arm1RawOutput"])

            # Step 4: Run analyze_arm3_results.py on both output files
            manifest_out = tmp_dir / "analyzed-manifest.json"
            rc = self.analyzer.main([str(ep_file), str(cached_jsonl), "--out", str(manifest_out)])
            self.assertEqual(rc, 0)
            manifest_doc = json.loads(manifest_out.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest_doc["models"]), 2)
            self.assertEqual(manifest_doc["holm_bonferroni_family_size"], 10)  # 2 models x 5 primary arms

    def test_assistant_prefill_and_max_new_tokens_defaults_and_subbatch_propagation(self) -> None:
        runner_mod = _load_module("run_arm3_colab_gpu", RUNNER_SCRIPT_PATH)
        self.assertEqual(runner_mod.DEFAULT_ASSISTANT_PREFILL, '{"notes": "')

        # 1. Verify CLI parser defaults
        parser = runner_mod.build_arg_parser()
        parsed_defaults = parser.parse_args(["--model-key", "phi4", "--repo-id", "microsoft/phi-4"])
        self.assertEqual(parsed_defaults.assistant_prefill, '{"notes": "')
        self.assertEqual(parsed_defaults.max_new_tokens, 1536)

        parsed_disabled = parser.parse_args(
            ["--model-key", "phi4", "--repo-id", "microsoft/phi-4", "--assistant-prefill", "", "--max-new-tokens", "768"]
        )
        self.assertEqual(parsed_disabled.assistant_prefill, "")
        self.assertEqual(parsed_disabled.max_new_tokens, 768)

        # 2. Verify HuggingFaceBatchBackend._run_subbatch appends prefill to prompt and prepends to decoded continuation
        captured_prompts: List[str] = []
        captured_gen_kwargs: Dict[str, Any] = {}

        class _FakeTensor:
            def __init__(self, rows: List[List[int]]) -> None:
                self.rows = rows
                self.shape = (len(rows), len(rows[0]) if rows else 0)

            def to(self, _device: Any) -> "_FakeTensor":
                return self

            def __getitem__(self, idx: int) -> "_FakeRow":
                return _FakeRow(self.rows[idx])

        class _FakeRow:
            def __init__(self, data: List[int]) -> None:
                self.data = data
                self.shape = (len(data),)

            def __getitem__(self, sl: slice) -> "_FakeRow":
                return _FakeRow(self.data[sl])

            def sum(self) -> "_FakeScalar":
                return _FakeScalar(sum(self.data))

            def __ne__(self, other: int) -> "_FakeRow":  # type: ignore[override]
                return _FakeRow([1 if x != other else 0 for x in self.data])

        class _FakeScalar:
            def __init__(self, val: int) -> None:
                self._val = val

            def item(self) -> int:
                return self._val

        class _FakeTorch:
            @staticmethod
            def inference_mode():
                class _Ctx:
                    def __enter__(self):
                        return None

                    def __exit__(self, exc_type, exc_val, exc_tb):
                        return False

                return _Ctx()

        class _FakeTokenizer:
            padding_side = "left"
            pad_token_id = 0

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
                return f"<|user|>{messages[-1]['content']}<|assistant|>"

            def __call__(self, prompts, return_tensors="pt", padding=True):
                captured_prompts.extend(prompts)
                n = len(prompts)
                return {
                    "input_ids": _FakeTensor([[1, 2, 3, 4] for _ in range(n)]),
                    "attention_mask": _FakeTensor([[1, 1, 1, 1] for _ in range(n)]),
                }

            def decode(self, token_ids, skip_special_tokens=True):
                return 'step trace", "step_id": "d4", "value": 42, "unit": "g", "retracted_ids": []}'

        class _FakeModel:
            device = "cpu"

            def generate(self, **kwargs):
                captured_gen_kwargs.update(kwargs)
                in_ids: _FakeTensor = kwargs["input_ids"]
                return _FakeTensor([row + [11, 12, 13] for row in in_ids.rows])

        backend = object.__new__(runner_mod.HuggingFaceBatchBackend)
        backend.tokenizer = _FakeTokenizer()
        backend.model = _FakeModel()
        backend.max_new_tokens = 1536
        backend.assistant_prefill = runner_mod.DEFAULT_ASSISTANT_PREFILL
        backend.enable_thinking = None

        sub_convs = [[{"role": "user", "content": "Compute d4."}]]
        with mock.patch.dict(sys.modules, {"torch": _FakeTorch}):
            out_with_prefill = backend._run_subbatch(sub_convs)
            self.assertEqual(len(captured_prompts), 1)
            self.assertTrue(captured_prompts[0].endswith('<|assistant|>{"notes": "'))
            self.assertEqual(captured_gen_kwargs.get("max_new_tokens"), 1536)
            self.assertEqual(
                out_with_prefill[0][0],
                '{"notes": "step trace", "step_id": "d4", "value": 42, "unit": "g", "retracted_ids": []}',
            )

            # Verify empty prefill disables prompt suffix and continuation prefix
            captured_prompts.clear()
            backend.assistant_prefill = ""
            out_no_prefill = backend._run_subbatch(sub_convs)
            self.assertTrue(captured_prompts[0].endswith("<|assistant|>"))
            self.assertEqual(
                out_no_prefill[0][0],
                'step trace", "step_id": "d4", "value": 42, "unit": "g", "retracted_ids": []}',
            )


if __name__ == "__main__":
    unittest.main()
