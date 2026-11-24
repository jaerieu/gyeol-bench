"""Unit tests for gyeol-bench v2.0 Fraction oracle, partial STALE_STATE_RETENTION, notes validation, and zero-core checker."""

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
    MultiTurnEpisode,
    TurnSpec,
    evaluate_multiturn_episode,
    evaluate_retraction_episode,
    generate_multiturn_dataset,
    generate_procedural_dataset,
    mixed_tolerance_match,
    parse_step_envelope,
    response_adapter,
)
from gyeol_bench._registry import retraction_dag, retraction_dag_multiturn
from gyeol_bench.oracle import _compute_stale_targets, convert_from_base_fraction

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
        good_json = json.dumps(
            {
                "notes": "1500 mg + 2 g = 3.5 g; * 3 = 10.5 g",
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

        # Optional notes omitted remains valid for backwards compatibility
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

        # Stale un-retracted value (2.5 g * 3 = 7.5 g) -> STALE_STATE_RETENTION
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
        self.assertEqual(res_stale.verdict, "STALE_STATE_RETENTION")

        # Arbitrary wrong numeric value -> CONDITIONAL_ARITH_ERROR
        bad_arith_json = json.dumps(
            {"step_id": "d2", "value": 9.25, "unit": "g", "retracted_ids": ["p1"]}
        )
        res_arith = evaluate_retraction_episode(turns, bad_arith_json, target_node_id="d2", expected_unit="g")
        self.assertFalse(res_arith.ok)
        self.assertEqual(res_arith.verdict, "CONDITIONAL_ARITH_ERROR")

    def test_downward_closed_partial_stale_state_retention(self) -> None:
        turns = [
            TurnSpec("p1", "premise", 100, "g"),
            TurnSpec("p2", "premise", 50, "g"),
            TurnSpec("p3", "premise", 40, "g"),
            TurnSpec("d1", "derive", arith_op="add", left_id="p1", right_id="p2"),
            TurnSpec("d2", "derive", arith_op="add", left_id="d1", right_id="p3"),
            TurnSpec("p1_r1", "retract", 200, "g", retraction_op="fix->", target_id="p1"),
            TurnSpec("p3_r1", "retract", 90, "g", retraction_op="err->", target_id="p3"),
            TurnSpec("p1_r2", "retract", 350, "g", retraction_op="sup->", target_id="p1_r1"),
        ]
        # Active truth: 350 + 50 + 90 = 490 g, retracted_ids = ["p1", "p1_r1", "p3"]
        # Downward-closed partial stale states:
        # 1) Drop {p1_r2} -> 200 + 50 + 90 = 340 g
        # 2) Drop {p1_r1, p1_r2} -> 100 + 50 + 90 = 240 g
        # 3) Drop {p3_r1} -> 350 + 50 + 40 = 440 g
        # 4) Drop {p1_r2, p3_r1} -> 200 + 50 + 40 = 290 g
        # 5) Drop {p1_r1, p1_r2, p3_r1} -> 100 + 50 + 40 = 190 g
        for partial_val in (340.0, 240.0, 440.0, 290.0, 190.0):
            payload = json.dumps(
                {
                    "notes": "partial trace",
                    "step_id": "d2",
                    "value": partial_val,
                    "unit": "g",
                    "retracted_ids": ["p1", "p1_r1", "p3", "extra_spurious"],
                }
            )
            res = evaluate_retraction_episode(turns, payload, target_node_id="d2", expected_unit="g")
            self.assertFalse(res.ok)
            self.assertEqual(
                res.verdict,
                "STALE_STATE_RETENTION",
                msg=f"Expected STALE_STATE_RETENTION before OVER_RETRACTION for {partial_val}",
            )

        ok_payload = json.dumps(
            {
                "notes": "350 + 50 + 90 = 490",
                "step_id": "d2",
                "value": 490.0,
                "unit": "g",
                "retracted_ids": ["p1", "p1_r1", "p3"],
            }
        )
        self.assertEqual(
            evaluate_retraction_episode(turns, ok_payload, target_node_id="d2", expected_unit="g").verdict,
            "PASS",
        )

    def test_financial_scale_vs_meter_disambiguation(self) -> None:
        oracle = ExactRationalOracle()
        oracle.append_turn(TurnSpec(node_id="rev", kind="premise", value="5.2", unit="M_USD"))
        oracle.append_turn(TurnSpec(node_id="len", kind="premise", value=10, unit="m"))
        oracle.append_turn(TurnSpec(node_id="bad", kind="derive", arith_op="add", left_id="rev", right_id="len"))
        with self.assertRaises(DimensionMismatchError):
            oracle.evaluate()

    def test_parse_reject_gate_notes_type_and_mixed_tolerance(self) -> None:
        valid_with_notes = '{"notes": "step trace", "step_id": "d2", "value": 10.5, "unit": "g", "retracted_ids": []}'
        self.assertTrue(response_adapter(valid_with_notes).ok)

        for bad_raw in [
            "The answer is 10.5 g.",
            '{"step_id": "d2", "value": true, "unit": "g", "retracted_ids": []}',
            '{"step_id": "d2", "value": NaN, "unit": "g", "retracted_ids": []}',
            '{"step_id": "d2", "value": 10.5}',
            '{"step_id": "d2", "value": 10.5, "unit": "g", "retracted_ids": [], "preserved": "bad"}',
            '{"notes": 123, "step_id": "d2", "value": 10.5, "unit": "g", "retracted_ids": []}',
            '{"notes": ["list"], "step_id": "d2", "value": 10.5, "unit": "g", "retracted_ids": []}',
            '{"notes": null, "step_id": "d2", "value": 10.5, "unit": "g", "retracted_ids": []}',
            '{"notes": true, "step_id": "d2", "value": 10.5, "unit": "g", "retracted_ids": []}',
        ]:
            for gate in (parse_step_envelope, response_adapter):
                parsed = gate(bad_raw)
                self.assertFalse(parsed.ok, msg=f"Expected PARSE_REJECT for {bad_raw!r}")
                self.assertEqual(parsed.status, "PARSE_REJECT")

        # Truth = 1000.0 -> allowed = 1e-4 + 1e-5 * 1000.0 = 0.0101
        ok_in, _, _ = mixed_tolerance_match("1000.0100", Fraction(1000, 1))
        ok_out, _, _ = mixed_tolerance_match("1000.0105", Fraction(1000, 1))
        self.assertTrue(ok_in)
        self.assertFalse(ok_out)

    def test_n800_v2_five_conditions_depths_and_four_way_error_split(self) -> None:
        ds = generate_procedural_dataset(seed=20261122, count=800)
        self.assertEqual(len(ds), 800)
        self.assertEqual(sum(1 for e in ds if e.category == "normal"), 400)
        self.assertEqual(sum(1 for e in ds if e.category == "perturbed"), 400)
        for cond, expected_n in [("A", 400), ("B", 100), ("C", 100), ("D", 100), ("E", 100)]:
            sub = [e for e in ds if e.condition == cond]
            self.assertEqual(len(sub), expected_n)
            for ep in sub:
                d_hops = sum(
                    1 for t in ep.turns if t.kind == "derive" and t.node_id.startswith("d") and t.node_id != "d_indep"
                )
                if cond == "A":
                    self.assertEqual(len(ep.turns), 17)
                    self.assertEqual(d_hops, 4)
                    self.assertEqual(ep.expected_retracted_ids, ())
                else:
                    self.assertGreaterEqual(d_hops, 8)
                    self.assertEqual(len(ep.expected_retracted_ids), 3)

        by_cond = {cond: next(e for e in ds if e.condition == cond) for cond in ("A", "B", "C", "D", "E")}

        def _score(ep, value, retracted_ids, preserved=None, unit=None):
            body = {
                "notes": "trace",
                "step_id": ep.target_node_id,
                "value": value,
                "unit": unit or ep.expected_unit,
                "retracted_ids": retracted_ids,
            }
            if preserved is not None:
                body["preserved"] = preserved
            return evaluate_retraction_episode(
                ep.turns, json.dumps(body), ep.target_node_id, ep.expected_unit,
                preserved_targets=dict(ep.preserved_targets) or None,
            )

        ep_a = by_cond["A"]
        self.assertEqual(_score(ep_a, ep_a.expected_value_float, []).verdict, "PASS")
        self.assertEqual(_score(ep_a, ep_a.expected_value_float + 5, []).verdict, "CONDITIONAL_ARITH_ERROR")
        self.assertEqual(_score(ep_a, ep_a.expected_value_float, ["p1"]).verdict, "OVER_RETRACTION")

        self.assertIn(by_cond["C"].target_node_id, {"d10", "d12"})
        for cond in ("B", "C", "D"):
            ep = by_cond[cond]
            self.assertEqual(_score(ep, ep.expected_value_float, list(ep.expected_retracted_ids)).verdict, "PASS")
            self.assertEqual(_score(ep, ep.pre_retraction_value_float, []).verdict, "STALE_STATE_RETENTION")

            oracle_c = ExactRationalOracle()
            for t in ep.turns:
                oracle_c.append_turn(t)
            snap_c = oracle_c.evaluate()
            bf, dim, _ = snap_c.active_values_base[ep.target_node_id]
            tf = convert_from_base_fraction(bf, dim, ep.expected_unit)
            stales = _compute_stale_targets(ep.turns, ep.target_node_id, ep.expected_unit, dim, tf)
            self.assertEqual(len(stales), 5)
            for st in stales:
                self.assertEqual(_score(ep, float(st), list(ep.expected_retracted_ids)).verdict, "STALE_STATE_RETENTION")

        ep_e = by_cond["E"]
        pres_map = dict(ep_e.preserved_targets)
        self.assertIn("d_indep", pres_map)
        oracle_e = ExactRationalOracle()
        for t in ep_e.turns:
            oracle_e.append_turn(t)
        indep_base, _, _ = oracle_e.evaluate().active_values_base["d_indep"]
        from gyeol_bench.oracle import UNIT_TABLE
        iu = pres_map["d_indep"]
        iv = float(indep_base / UNIT_TABLE[iu][1])
        ret_e = list(ep_e.expected_retracted_ids)

        self.assertEqual(_score(ep_e, ep_e.expected_value_float, ret_e, {"d_indep": {"value": iv, "unit": iu}}).verdict, "PASS")
        self.assertEqual(_score(ep_e, ep_e.expected_value_float, ret_e, {"d_indep": {"value": iv + 10.0, "unit": iu}}).verdict, "OVER_RETRACTION")
        self.assertEqual(_score(ep_e, ep_e.expected_value_float, ret_e + ["d_indep"], {"d_indep": {"value": iv, "unit": iu}}).verdict, "OVER_RETRACTION")
        self.assertEqual(_score(ep_e, ep_e.expected_value_float, ret_e).verdict, "PARSE_REJECT")

    def test_retraction_dag_registration_balance(self) -> None:
        task_obj = retraction_dag(seed=20261122, sample_count=16)
        self.assertIsNotNone(task_obj)
        if isinstance(task_obj, dict):
            self.assertEqual(task_obj["sample_count"], 16)
            self.assertEqual(task_obj["conditions"], {"A": 8, "B": 2, "C": 2, "D": 2, "E": 2})
        else:
            self.assertEqual(len(task_obj.dataset), 16)
            for cond, exp_n in [("A", 8), ("B", 2), ("C", 2), ("D", 2), ("E", 2)]:
                self.assertEqual(sum(1 for s in task_obj.dataset if (s.metadata or {}).get("condition") == cond), exp_n)

    def test_v3_multiturn_dataset_and_stale_turn1_node_substitution(self) -> None:
        ds = generate_multiturn_dataset(seed=20261122, count=800)
        self.assertEqual(len(ds), 800)
        for cond, expected_n in [("A", 400), ("B", 100), ("C", 100), ("D", 100), ("E", 100)]:
            sub = [e for e in ds if e.condition == cond]
            self.assertEqual(len(sub), expected_n)
            for ep in sub:
                self.assertIsInstance(ep, MultiTurnEpisode)
                self.assertEqual(ep.turn1_target_id, "d4")
                self.assertEqual(ep.target_node_id, "d6")
                self.assertNotIn("Turn 1.", ep.turn2_prompt)
                self.assertLessEqual(len(ep.turn1_prompt) / 3.6, 300.0)
                if cond == "A":
                    self.assertEqual(ep.expected_retracted_ids, ())
                    self.assertEqual(ep.stale_target_values_float, ())
                else:
                    self.assertGreaterEqual(len(ep.stale_target_values_float), 6)

        by_cond = {c: next(e for e in ds if e.condition == c) for c in ("A", "B", "C", "D", "E")}

        def _score_mt(ep: MultiTurnEpisode, val: float, ret_ids, preserved=None):
            payload = {
                "notes": "turn 2 trace",
                "step_id": "d6",
                "value": val,
                "unit": ep.expected_unit,
                "retracted_ids": ret_ids,
            }
            if preserved is not None:
                payload["preserved"] = preserved
            return evaluate_multiturn_episode(
                ep.turn1_turns, ep.turn2_turns, json.dumps(payload),
                target_node_id="d6", expected_unit=ep.expected_unit,
                preserved_targets=dict(ep.preserved_targets) or None,
            )

        # Verify frozen Turn-1 d4-only, d2-only, and {d4, d2} substitutions trigger STALE_STATE_RETENTION
        for cond in ("B", "C", "D"):
            ep = by_cond[cond]
            self.assertEqual(_score_mt(ep, ep.expected_value_float, list(ep.expected_retracted_ids)).verdict, "PASS")
            for sv in ep.stale_target_values_float:
                res_st = _score_mt(ep, sv, list(ep.expected_retracted_ids) + ["spurious_id"])
                self.assertEqual(res_st.verdict, "STALE_STATE_RETENTION")

        ep_e = by_cond["E"]
        o_e = ExactRationalOracle()
        for t in ep_e.all_turns:
            o_e.append_turn(t)
        snap_e = o_e.evaluate()
        iu = dict(ep_e.preserved_targets)["d_indep"]
        iv = float(convert_from_base_fraction(*snap_e.active_values_base["d_indep"][:2], iu))
        self.assertEqual(
            _score_mt(ep_e, ep_e.expected_value_float, list(ep_e.expected_retracted_ids), {"d_indep": {"value": iv, "unit": iu}}).verdict,
            "PASS",
        )
        for sv in ep_e.stale_target_values_float:
            self.assertEqual(
                _score_mt(ep_e, sv, list(ep_e.expected_retracted_ids), {"d_indep": {"value": iv + 50.0, "unit": iu}}).verdict,
                "STALE_STATE_RETENTION",
            )

        for arm in ("baseline", "lineage_k2"):
            mt_task = retraction_dag_multiturn(seed=20261151, sample_count=16, arm=arm)
            if isinstance(mt_task, dict):
                self.assertEqual(mt_task["sample_count"], 16)
                self.assertEqual(mt_task["conditions"], {"A": 8, "B": 2, "C": 2, "D": 2, "E": 2})
            else:
                self.assertEqual(len(mt_task.dataset), 16)


if __name__ == "__main__":
    unittest.main()
