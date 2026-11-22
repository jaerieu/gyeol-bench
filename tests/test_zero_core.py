"""Unit tests for gyeol-bench clean-room Fraction oracle, PARSE_REJECT gate, and zero-core checker."""

from __future__ import annotations

from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gyeol_bench import (
    DimensionMismatchError,
    ExactRationalOracle,
    TurnSpec,
    evaluate_retraction_episode,
    generate_procedural_dataset,
    mixed_tolerance_match,
    parse_step_envelope,
)
from gyeol_bench._registry import retraction_dag

CHECKER_PATH = REPO_ROOT / "scripts" / "check_zero_core.py"


def _load_checker_module():
    spec = importlib.util.spec_from_file_location("check_zero_core", CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestZeroCoreAndOracle(unittest.TestCase):
    def test_repository_passes_zero_core_static_audit(self) -> None:
        checker = _load_checker_module()
        report = checker.audit_directory(REPO_ROOT)
        self.assertTrue(report["ok"], msg=f"Violations found: {report['violations']}")
        self.assertEqual(report["grade"], "[안전]")
        self.assertGreaterEqual(report["pythonFiles"], 4)

    def test_static_checker_blocks_private_symbols_and_spectrum_words(self) -> None:
        checker = _load_checker_module()
        forbidden_samples = [
            "import " + "gyeol" + "_lon\n",
            "x = " + "m" + "uk" + "_score\n",
            "# " + "ch" + "ae" + " leak\n",
            "def run_" + "calc" + "N():\n    return 1\n",
            "status = '" + "r" + "ed" + "'\n",
        ]
        for snippet in forbidden_samples:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                (tmp_path / "sample.py").write_text(snippet, encoding="utf-8")
                res = checker.audit_directory(tmp_path, allow_remotes=True)
                self.assertFalse(res["ok"], msg=f"Expected failure for snippet: {snippet!r}")
                self.assertEqual(res["grade"], "[금지]")

    def test_multi_turn_retraction_cascade_and_si_units(self) -> None:
        turns = [
            TurnSpec(node_id="p1", kind="premise", value=500, unit="mg"),
            TurnSpec(node_id="p2", kind="premise", value=2, unit="g"),
            TurnSpec(node_id="p3", kind="premise", value=3, unit="scalar"),
            TurnSpec(node_id="d1", kind="derive", arith_op="add", left_id="p1", right_id="p2"),
            TurnSpec(node_id="d2", kind="derive", arith_op="mul", left_id="d1", right_id="p3"),
            TurnSpec(
                node_id="p1_fix",
                kind="retract",
                retraction_op="fix->",
                target_id="p1",
                value=1500,
                unit="mg",
            ),
        ]
        # (1500 mg + 2 g) * 3 = 3.5 g * 3 = 10.5 g = 10500 mg
        good_json = json.dumps(
            {
                "step_id": "d2",
                "value": 10.5,
                "unit": "g",
                "retracted_ids": ["p1"],
            }
        )
        res = evaluate_retraction_episode(turns, good_json, target_node_id="d2", expected_unit="g")
        self.assertTrue(res.ok)
        self.assertEqual(res.verdict, "PASS")
        self.assertEqual(res.truth_target_fraction, Fraction(21, 2))

        # Equivalent answer reported in mg also aligns via exact base-unit conversion
        mg_json = json.dumps(
            {
                "step_id": "d2",
                "value": 10500,
                "unit": "mg",
                "retracted_ids": ["p1"],
            }
        )
        res_mg = evaluate_retraction_episode(turns, mg_json, target_node_id="d2", expected_unit="g")
        self.assertTrue(res_mg.ok)
        self.assertEqual(res_mg.verdict, "PASS")

        # Stale un-retracted value (2.5 g * 3 = 7.5 g) must fail
        stale_json = json.dumps(
            {
                "step_id": "d2",
                "value": 7.5,
                "unit": "g",
                "retracted_ids": ["p1"],
            }
        )
        res_stale = evaluate_retraction_episode(turns, stale_json, target_node_id="d2", expected_unit="g")
        self.assertFalse(res_stale.ok)
        self.assertEqual(res_stale.verdict, "VALUE_MISMATCH")

    def test_financial_scale_vs_meter_disambiguation(self) -> None:
        oracle = ExactRationalOracle()
        oracle.append_turn(TurnSpec(node_id="rev", kind="premise", value="5.2", unit="M_USD"))
        oracle.append_turn(TurnSpec(node_id="len", kind="premise", value=10, unit="m"))
        oracle.append_turn(TurnSpec(node_id="bad", kind="derive", arith_op="add", left_id="rev", right_id="len"))
        with self.assertRaises(DimensionMismatchError):
            oracle.evaluate()

    def test_parse_reject_gate_and_mixed_tolerance(self) -> None:
        for bad_raw in [
            "The answer is 10.5 g.",
            '{"step_id": "d2", "value": true, "unit": "g", "retracted_ids": []}',
            '{"step_id": "d2", "value": NaN, "unit": "g", "retracted_ids": []}',
            '{"step_id": "d2", "value": 10.5}',
        ]:
            parsed = parse_step_envelope(bad_raw)
            self.assertFalse(parsed.ok, msg=f"Expected PARSE_REJECT for {bad_raw!r}")
            self.assertEqual(parsed.status, "PARSE_REJECT")

        # Truth = 1000.0 -> allowed = 1e-4 + 1e-5 * 1000.0 = 0.0101
        ok_in, _, _ = mixed_tolerance_match("1000.0100", Fraction(1000, 1))
        ok_out, _, _ = mixed_tolerance_match("1000.0105", Fraction(1000, 1))
        self.assertTrue(ok_in)
        self.assertFalse(ok_out)

    def test_procedural_dataset_and_registry_entry(self) -> None:
        ds = generate_procedural_dataset(seed=20261122, count=40)
        self.assertEqual(len(ds), 40)
        self.assertEqual(sum(1 for e in ds if e.category == "retraction"), 10)
        self.assertEqual(sum(1 for e in ds if e.category == "unit_scale"), 10)
        self.assertEqual(sum(1 for e in ds if e.category == "normal"), 20)

        task_obj = retraction_dag(seed=20261122, sample_count=20)
        self.assertIsNotNone(task_obj)


if __name__ == "__main__":
    unittest.main()
