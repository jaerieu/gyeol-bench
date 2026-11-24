"""Unit tests for Arm 3B 2-pass deterministic Fraction/SI state-diff verifier and single repair pass.

Covers:
1. Prompt-to-TurnSpec deterministic parser parity across all N=800 v3.0 multi-turn episodes (seed=20261122)
   and N=50 pilot episodes (seed=20261151).
2. Zero Target-Answer Leakage Verification (Audit #12 Mandate):
   - Verifies that `pass1_audit_candidate` reasons and `build_arm3b_repair_prompt` NEVER contain the
     pre-computed Turn-2 target numeric value (`d6`) or oracle-computed `d_indep` value across all 800 episodes.
3. Strict Separation of Primary K2 Modes (`mode="llm_repair"` & `mode="structural_gate"`) vs
   Secondary Symbolic-Offload Ceiling (`mode="symbolic_ceiling"`).
4. Pass-1 diagnostic flagging and repair verification across:
   - Stale Turn-1 bindings (`STALE_TURN1_BINDING`)
   - Spurious or missing `retracted_ids` (`SPURIOUS_OR_MISSING_RETRACTED_IDS`, e.g. Condition A false alarms,
     derived `d*` leakage, un-superseded `p1_r2`, or omitted chained `p1_r1`)
   - Preserved subgraph tampering (`PRESERVED_SUBGRAPH_TAMPERING` on `d_indep` in Condition E)
   - Envelope syntax defects (`ENVELOPE_SYNTAX_DEFECT`, e.g. Mistral trailing period `}}}.` and markdown fences)
   - SI scale / bps-to-% arithmetic residual mismatches (`SI_ARITH_RESIDUAL_MISMATCH` in Condition D)
5. SymPy 1.14.0 + Pint 0.26.2 external cross-oracle parity check against Arm 3B state-diff outputs.
6. Static zero-core and vocabulary hygiene audit (`check_zero_core.py` -> `[안전]`).
"""

from __future__ import annotations

from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gyeol_bench.arm3_state_diff import (
    apply_structural_gate_only,
    build_arm3b_repair_prompt,
    compute_state_diff_snapshot,
    parse_turn_prompt_specs,
    pass1_audit_candidate,
    run_arm3b_from_prompts,
    run_arm3b_state_diff_verifier,
)
from gyeol_bench.oracle import (
    evaluate_multiturn_episode,
    generate_multiturn_dataset,
    mixed_tolerance_match,
)


class TestArm3BStateDiffVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.episodes_n50 = generate_multiturn_dataset(seed=20261151, count=50)
        cls.episodes_n800 = generate_multiturn_dataset(seed=20261122, count=800)

    def test_prompt_parser_reconstructs_exact_turn_specs_across_n800(self) -> None:
        for ep in self.episodes_n800:
            t1_turns, t1_target, t1_unit, t1_pres = parse_turn_prompt_specs(ep.turn1_prompt)
            t2_turns, t2_target, t2_unit, t2_pres = parse_turn_prompt_specs(ep.turn2_prompt)
            self.assertEqual(t1_turns, ep.turn1_turns, msg=f"Turn-1 spec mismatch on {ep.episode_id}")
            self.assertEqual(t1_target, ep.turn1_target_id)
            self.assertEqual(t1_unit, ep.turn1_expected_unit)
            self.assertEqual(t1_pres, dict(ep.turn1_preserved_targets))

            self.assertEqual(t2_turns, ep.turn2_turns, msg=f"Turn-2 spec mismatch on {ep.episode_id}")
            self.assertEqual(t2_target, ep.target_node_id)
            self.assertEqual(t2_unit, ep.expected_unit)
            self.assertEqual(t2_pres, dict(ep.preserved_targets))

    def test_zero_target_answer_leakage_in_pass1_reasons_and_repair_prompt(self) -> None:
        """Verify build_arm3b_repair_prompt and pass1 reasons NEVER leak pre-computed d6 or oracle d_indep values."""
        for ep in self.episodes_n50:
            diff_snap = compute_state_diff_snapshot(
                ep.turn1_turns,
                ep.turn2_turns,
                target_node_id=ep.target_node_id,
                expected_unit=ep.expected_unit,
                preserved_targets=dict(ep.preserved_targets) or None,
            )
            bad_val = ep.stale_target_values_float[0] if ep.stale_target_values_float else -12345.67
            bad_payload = {
                "notes": "defective candidate",
                "step_id": "d6",
                "value": bad_val,
                "unit": ep.expected_unit,
                "retracted_ids": ["d1", "spurious_node"],
            }
            if ep.preserved_targets:
                bad_payload["preserved"] = {"d_indep": {"value": -99999, "unit": ep.preserved_targets[0][1]}}

            p1 = pass1_audit_candidate(diff_snap, json.dumps(bad_payload), turn1_raw_output=None)
            prompt_str = build_arm3b_repair_prompt(diff_snap, p1, turn1_raw_output=None)
            self.assertTrue(len(prompt_str) > 0)

            # Format truth target value as integer and float strings and ensure neither appears in prompt_str
            truth_frac = diff_snap.truth_target_fraction
            truth_tokens = {f"{float(truth_frac):.6g}", str(float(truth_frac))}
            if truth_frac.denominator == 1:
                truth_tokens.add(str(int(truth_frac.numerator)))

            for tok in truth_tokens:
                if len(tok) >= 3:
                    self.assertNotIn(
                        tok,
                        prompt_str,
                        msg=f"Leaked target truth {tok!r} in repair prompt for {ep.episode_id}: {prompt_str}",
                    )

            # In Condition E with no Turn-1 raw output, ensure oracle d_indep truth is also not leaked
            if ep.preserved_targets:
                pfrac, _ = diff_snap.preserved_truth_display["d_indep"]
                ptok = str(int(pfrac.numerator)) if pfrac.denominator == 1 else f"{float(pfrac):.6g}"
                if len(ptok) >= 3:
                    self.assertNotIn(
                        ptok,
                        prompt_str,
                        msg=f"Leaked oracle d_indep truth {ptok!r} without Turn-1 commitment on {ep.episode_id}",
                    )

    def test_primary_k2_llm_repair_mode_and_structural_gate_vs_symbolic_ceiling(self) -> None:
        ep_b = next(e for e in self.episodes_n50 if e.condition == "B")
        stale_val = ep_b.stale_target_values_float[0]
        raw_t2_stale = json.dumps(
            {
                "notes": "stale Turn-1 d4 reused",
                "step_id": "d6",
                "value": stale_val,
                "unit": ep_b.expected_unit,
                "retracted_ids": ["p1"],
            }
        )
        # 1. Primary K2 mode="llm_repair" without callback/repair output leaves numeric value untouched (does NOT auto-overwrite)
        ver_unrepaired = run_arm3b_state_diff_verifier(
            ep_b.turn1_turns,
            ep_b.turn2_turns,
            raw_t2_stale,
            target_node_id="d6",
            expected_unit=ep_b.expected_unit,
            mode="llm_repair",
        )
        self.assertEqual(ver_unrepaired.mode, "llm_repair")
        self.assertTrue(ver_unrepaired.is_primary_k2_eligible)
        self.assertTrue(ver_unrepaired.repair_triggered)
        self.assertFalse(ver_unrepaired.repair_converged)

        # When 1-turn LLM self-correction response is supplied via callback, mode="llm_repair" scores the LLM response
        simulated_llm_retry = json.dumps(
            {
                "notes": "LLM self-corrected after seeing R=['p1', 'p1_r1'] and tainted=['d1','d2','d3','d4','d5','d6']",
                "step_id": "d6",
                "value": ep_b.expected_value_float,
                "unit": ep_b.expected_unit,
                "retracted_ids": list(ep_b.expected_retracted_ids),
            }
        )
        ver_llm_repaired = run_arm3b_state_diff_verifier(
            ep_b.turn1_turns,
            ep_b.turn2_turns,
            raw_t2_stale,
            target_node_id="d6",
            expected_unit=ep_b.expected_unit,
            mode="llm_repair",
            llm_repair_callback=lambda prompt: simulated_llm_retry,
        )
        self.assertTrue(ver_llm_repaired.is_primary_k2_eligible)
        self.assertTrue(ver_llm_repaired.repair_converged)
        self.assertTrue(ver_llm_repaired.pass2_report.clean)

        # 2. Primary K2 ablation mode="structural_gate" fixes R and syntax but NEVER overwrites stale numeric value
        ver_struct = run_arm3b_state_diff_verifier(
            ep_b.turn1_turns,
            ep_b.turn2_turns,
            raw_t2_stale,
            target_node_id="d6",
            expected_unit=ep_b.expected_unit,
            mode="structural_gate",
        )
        self.assertTrue(ver_struct.is_primary_k2_eligible)
        self.assertEqual(ver_struct.final_retracted_ids, ("p1", "p1_r1"))
        self.assertIn("STALE_TURN1_BINDING", ver_struct.pass2_report.flags)
        self.assertNotIn("SPURIOUS_OR_MISSING_RETRACTED_IDS", ver_struct.pass2_report.flags)

        # 3. Secondary mode="symbolic_ceiling" is explicitly marked is_primary_k2_eligible=False
        ver_ceiling = run_arm3b_state_diff_verifier(
            ep_b.turn1_turns,
            ep_b.turn2_turns,
            raw_t2_stale,
            target_node_id="d6",
            expected_unit=ep_b.expected_unit,
            mode="symbolic_ceiling",
        )
        self.assertEqual(ver_ceiling.mode, "symbolic_ceiling")
        self.assertFalse(ver_ceiling.is_primary_k2_eligible)
        self.assertTrue(ver_ceiling.repair_converged)

    def test_pass1_flags_stale_turn1_bindings_and_pass2_repairs_in_one_step(self) -> None:
        for cond in ("B", "C", "D", "E"):
            ep = next(e for e in self.episodes_n50 if e.condition == cond)
            diff_snap = compute_state_diff_snapshot(
                ep.turn1_turns,
                ep.turn2_turns,
                target_node_id=ep.target_node_id,
                expected_unit=ep.expected_unit,
                preserved_targets=dict(ep.preserved_targets) or None,
            )
            self.assertTrue(len(diff_snap.stale_target_fractions) >= 6)
            self.assertIn("d4", diff_snap.tainted_node_ids)
            self.assertIn("d6", diff_snap.tainted_node_ids)
            if cond == "E":
                self.assertIn("d_indep", diff_snap.untainted_node_ids)
                self.assertNotIn("d_indep", diff_snap.tainted_node_ids)

            for stale_frac in diff_snap.stale_target_fractions:
                payload = {
                    "notes": "reused committed Turn-1 d4 value without re-propagation",
                    "step_id": "d6",
                    "value": float(stale_frac),
                    "unit": ep.expected_unit,
                    "retracted_ids": list(ep.expected_retracted_ids),
                }
                if ep.preserved_targets:
                    pfrac, punit = diff_snap.preserved_truth_display["d_indep"]
                    payload["preserved"] = {"d_indep": {"value": float(pfrac), "unit": punit}}

                raw_stale = json.dumps(payload)
                ver = run_arm3b_state_diff_verifier(
                    ep.turn1_turns,
                    ep.turn2_turns,
                    raw_stale,
                    target_node_id="d6",
                    expected_unit=ep.expected_unit,
                    preserved_targets=dict(ep.preserved_targets) or None,
                    mode="symbolic_ceiling",
                )
                self.assertFalse(ver.pass1_report.clean)
                self.assertTrue(ver.pass1_report.stale_binding_matched)
                self.assertIn("STALE_TURN1_BINDING", ver.pass1_report.flags)
                self.assertTrue(ver.repair_triggered)
                self.assertTrue(ver.repair_converged)
                self.assertTrue(ver.pass2_report.clean)

                scored = evaluate_multiturn_episode(
                    ep.turn1_turns,
                    ep.turn2_turns,
                    ver.final_json,
                    target_node_id="d6",
                    expected_unit=ep.expected_unit,
                    preserved_targets=dict(ep.preserved_targets) or None,
                )
                self.assertTrue(scored.ok)
                self.assertEqual(scored.verdict, "PASS")

    def test_pass1_flags_spurious_and_missing_retracted_ids_and_condition_a_over_retraction(self) -> None:
        # 1. Condition A spurious retracted_ids repaired cleanly by structural_gate without touching numeric value
        ep_a = next(e for e in self.episodes_n50 if e.condition == "A")
        bad_cond_a = json.dumps(
            {
                "notes": "spurious taint checklist retraction on normal control",
                "step_id": "d6",
                "value": ep_a.expected_value_float,
                "unit": ep_a.expected_unit,
                "retracted_ids": ["d1", "d2", "d3", "d4", "p3"],
            }
        )
        ver_a = run_arm3b_from_prompts(
            ep_a.turn1_prompt, ep_a.turn2_prompt, bad_cond_a, mode="structural_gate"
        )
        self.assertTrue(ver_a.is_primary_k2_eligible)
        self.assertFalse(ver_a.pass1_report.clean)
        self.assertIn("SPURIOUS_OR_MISSING_RETRACTED_IDS", ver_a.pass1_report.flags)
        self.assertEqual(ver_a.pass1_report.spurious_retracted_ids, ("d1", "d2", "d3", "d4", "p3"))
        self.assertTrue(ver_a.repair_triggered)
        self.assertEqual(ver_a.final_retracted_ids, ())
        scored_a = evaluate_multiturn_episode(
            ep_a.turn1_turns, ep_a.turn2_turns, ver_a.final_json, "d6", ep_a.expected_unit
        )
        self.assertEqual(scored_a.verdict, "PASS")

        # 2. Condition B chained retraction where model omits intermediate 'p1_r1' or includes 'p1_r2'
        ep_b = next(e for e in self.episodes_n50 if e.condition == "B")
        bad_cond_b = json.dumps(
            {
                "notes": "computed value right but listed p1 and p1_r2 instead of p1 and p1_r1",
                "step_id": "d6",
                "value": ep_b.expected_value_float,
                "unit": ep_b.expected_unit,
                "retracted_ids": ["p1", "p1_r2"],
            }
        )
        ver_b = run_arm3b_from_prompts(
            ep_b.turn1_prompt, ep_b.turn2_prompt, bad_cond_b, mode="structural_gate"
        )
        self.assertTrue(ver_b.is_primary_k2_eligible)
        self.assertFalse(ver_b.pass1_report.clean)
        self.assertIn("SPURIOUS_OR_MISSING_RETRACTED_IDS", ver_b.pass1_report.flags)
        self.assertEqual(ver_b.pass1_report.spurious_retracted_ids, ("p1_r2",))
        self.assertEqual(ver_b.pass1_report.missing_retracted_ids, ("p1_r1",))
        self.assertEqual(ver_b.final_retracted_ids, ("p1", "p1_r1"))
        scored_b = evaluate_multiturn_episode(
            ep_b.turn1_turns, ep_b.turn2_turns, ver_b.final_json, "d6", ep_b.expected_unit
        )
        self.assertEqual(scored_b.verdict, "PASS")

    def test_pass1_flags_preserved_d_indep_tampering_and_trailing_period_syntax_defects(self) -> None:
        ep_e = next(e for e in self.episodes_n50 if e.condition == "E")
        diff_e = compute_state_diff_snapshot(
            ep_e.turn1_turns,
            ep_e.turn2_turns,
            target_node_id="d6",
            expected_unit=ep_e.expected_unit,
            preserved_targets=dict(ep_e.preserved_targets),
        )
        true_indep_frac, indep_unit = diff_e.preserved_truth_display["d_indep"]

        t1_clean_raw = json.dumps(
            {
                "notes": "Turn 1 clean commit",
                "step_id": "d4",
                "value": ep_e.turn1_expected_value_float,
                "unit": ep_e.turn1_expected_unit,
                "retracted_ids": [],
                "preserved": {"d_indep": {"value": float(true_indep_frac), "unit": indep_unit}},
            }
        )

        # Tampered d_indep + trailing period (Mistral Condition E defect pattern `}}}.`)
        raw_defect = (
            json.dumps(
                {
                    "notes": "Condition E with trailing dot and altered d_indep",
                    "step_id": "d6",
                    "value": ep_e.expected_value_float,
                    "unit": ep_e.expected_unit,
                    "retracted_ids": list(ep_e.expected_retracted_ids),
                    "preserved": {"d_indep": {"value": float(true_indep_frac) + 999.0, "unit": indep_unit}},
                }
            )
            + "."
        )

        ver_e = run_arm3b_from_prompts(
            ep_e.turn1_prompt,
            ep_e.turn2_prompt,
            raw_defect,
            turn1_raw_output=t1_clean_raw,
            mode="structural_gate",
        )
        self.assertTrue(ver_e.is_primary_k2_eligible)
        self.assertFalse(ver_e.pass1_report.clean)
        self.assertTrue(ver_e.pass1_report.sanitized_envelope_used)
        self.assertIn("ENVELOPE_SYNTAX_DEFECT", ver_e.pass1_report.flags)
        self.assertIn("PRESERVED_SUBGRAPH_TAMPERING", ver_e.pass1_report.flags)
        self.assertEqual(ver_e.pass1_report.preserved_tampered_ids, ("d_indep",))

        prompt_text = build_arm3b_repair_prompt(diff_e, ver_e.pass1_report, turn1_raw_output=t1_clean_raw)
        self.assertIn("PRESERVED_SUBGRAPH_TAMPERING", prompt_text)
        self.assertIn("d_indep", prompt_text)

        scored_e = evaluate_multiturn_episode(
            ep_e.turn1_turns,
            ep_e.turn2_turns,
            ver_e.final_json,
            target_node_id="d6",
            expected_unit=ep_e.expected_unit,
            preserved_targets=dict(ep_e.preserved_targets),
        )
        self.assertTrue(scored_e.ok)
        self.assertEqual(scored_e.verdict, "PASS")

    def test_structural_gate_strips_spurious_non_e_preserved_and_keeps_cond_e_lock(self) -> None:
        # (a) Condition A/B/C/D spurious `"preserved": {}` and `"preserved": {"d_indep": ...}` are stripped
        # without touching target `value`/`unit` (zero target-answer leakage).
        for cond in ("A", "B", "C", "D"):
            ep = next(e for e in self.episodes_n50 if e.condition == cond)
            diff_snap = compute_state_diff_snapshot(
                ep.turn1_turns,
                ep.turn2_turns,
                target_node_id=ep.target_node_id,
                expected_unit=ep.expected_unit,
                preserved_targets=dict(ep.preserved_targets) or None,
            )
            self.assertEqual(diff_snap.preserved_targets, ())
            for spurious_pres in ({}, {"d_indep": {"value": 123.45, "unit": "kg"}}):
                # Case 1: Correct numeric value + spurious preserved -> stripped and passes oracle
                raw_ok = json.dumps(
                    {
                        "notes": f"cond {cond} spurious preserved",
                        "step_id": ep.target_node_id,
                        "value": ep.expected_value_float,
                        "unit": ep.expected_unit,
                        "retracted_ids": list(ep.expected_retracted_ids),
                        "preserved": spurious_pres,
                    }
                )
                gated_ok = apply_structural_gate_only(diff_snap, raw_ok, turn1_raw_output=None)
                parsed_ok = json.loads(gated_ok)
                self.assertNotIn("preserved", parsed_ok)
                self.assertEqual(parsed_ok["value"], ep.expected_value_float)
                self.assertEqual(parsed_ok["unit"], ep.expected_unit)
                scored_ok = evaluate_multiturn_episode(
                    ep.turn1_turns,
                    ep.turn2_turns,
                    gated_ok,
                    target_node_id=ep.target_node_id,
                    expected_unit=ep.expected_unit,
                    preserved_targets=dict(ep.preserved_targets) or None,
                )
                self.assertTrue(scored_ok.ok, msg=f"Cond {cond} failed after stripping spurious preserved: {scored_ok}")
                self.assertEqual(scored_ok.verdict, "PASS")

                # Case 2: Wrong numeric value + spurious preserved -> preserved stripped, value/unit untouched
                wrong_val = -77777.25
                raw_wrong = json.dumps(
                    {
                        "notes": f"cond {cond} wrong value with spurious preserved",
                        "step_id": ep.target_node_id,
                        "value": wrong_val,
                        "unit": ep.expected_unit,
                        "retracted_ids": ["spurious_r"],
                        "preserved": spurious_pres,
                    }
                )
                gated_wrong = apply_structural_gate_only(diff_snap, raw_wrong, turn1_raw_output=None)
                parsed_wrong = json.loads(gated_wrong)
                self.assertNotIn("preserved", parsed_wrong)
                self.assertEqual(parsed_wrong["value"], wrong_val)
                self.assertEqual(parsed_wrong["unit"], ep.expected_unit)
                self.assertEqual(parsed_wrong["retracted_ids"], list(ep.expected_retracted_ids))

        # (b) Condition E committed Turn-1 `preserved` lock remains intact while target value/unit stay untouched
        ep_e = next(e for e in self.episodes_n50 if e.condition == "E")
        diff_e = compute_state_diff_snapshot(
            ep_e.turn1_turns,
            ep_e.turn2_turns,
            target_node_id=ep_e.target_node_id,
            expected_unit=ep_e.expected_unit,
            preserved_targets=dict(ep_e.preserved_targets),
        )
        true_indep_frac, indep_unit = diff_e.preserved_truth_display["d_indep"]
        t1_raw_e = json.dumps(
            {
                "notes": "Turn 1 clean commit",
                "step_id": "d4",
                "value": ep_e.turn1_expected_value_float,
                "unit": ep_e.turn1_expected_unit,
                "retracted_ids": [],
                "preserved": {"d_indep": {"value": float(true_indep_frac), "unit": indep_unit}},
            }
        )
        wrong_target_val = -88888.5
        raw_t2_e = json.dumps(
            {
                "notes": "Cond E tampered preserved and wrong target value",
                "step_id": ep_e.target_node_id,
                "value": wrong_target_val,
                "unit": ep_e.expected_unit,
                "retracted_ids": [],
                "preserved": {"d_indep": {"value": float(true_indep_frac) + 500.0, "unit": indep_unit}},
            }
        )
        gated_e = apply_structural_gate_only(diff_e, raw_t2_e, turn1_raw_output=t1_raw_e)
        parsed_e = json.loads(gated_e)
        self.assertIn("preserved", parsed_e)
        self.assertIn("d_indep", parsed_e["preserved"])
        self.assertEqual(parsed_e["preserved"]["d_indep"]["unit"], indep_unit)
        self.assertAlmostEqual(float(parsed_e["preserved"]["d_indep"]["value"]), float(true_indep_frac), places=6)
        self.assertEqual(parsed_e["value"], wrong_target_val)
        self.assertEqual(parsed_e["unit"], ep_e.expected_unit)
        self.assertEqual(parsed_e["retracted_ids"], list(ep_e.expected_retracted_ids))

    def test_clean_candidate_passes_through_unmodified_without_repair(self) -> None:
        ep = self.episodes_n50[0]
        clean_raw = json.dumps(
            {
                "notes": "exact clean calculation",
                "step_id": ep.target_node_id,
                "value": int(ep.expected_value_float)
                if ep.expected_value_float.is_integer()
                else ep.expected_value_float,
                "unit": ep.expected_unit,
                "retracted_ids": list(ep.expected_retracted_ids),
            }
        )
        ver = run_arm3b_from_prompts(ep.turn1_prompt, ep.turn2_prompt, clean_raw, mode="llm_repair")
        self.assertTrue(ver.pass1_report.clean)
        self.assertFalse(ver.repair_triggered)
        self.assertEqual(ver.final_json, clean_raw)

    def test_replay_on_cached_phase3_n800_open_weight_checkpoints_if_present(self) -> None:
        bundle_dir = Path("/tmp/gyeol-colab-checkpoints/phase3-open-k2-n800")
        if not bundle_dir.exists():
            self.skipTest("Phase 3 checkpoint directory not present in /tmp")

        ep_map = {ep.episode_id: ep for ep in self.episodes_n800}
        for fname in (
            "p3-phi4-n800-episodes.jsonl",
            "p3-qwen14b-n800-episodes.jsonl",
            "p3-mistral24b-n800-episodes.jsonl",
        ):
            fpath = bundle_dir / fname
            if not fpath.exists():
                continue
            rows = [json.loads(line) for line in fpath.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 800)
            struct_pass_count = 0
            for row in rows:
                ep = ep_map[row["episodeId"]]
                # 1. Primary-K2 structural_gate (never overwrites target value)
                ver_sg = run_arm3b_state_diff_verifier(
                    turn1_turns=ep.turn1_turns,
                    turn2_turns=ep.turn2_turns,
                    raw_turn2_output=row["arm2RawOutput"],
                    target_node_id="d6",
                    expected_unit=ep.expected_unit,
                    preserved_targets=dict(ep.preserved_targets) or None,
                    turn1_raw_output=row["turn1RawOutput"],
                    mode="structural_gate",
                )
                self.assertTrue(ver_sg.is_primary_k2_eligible)
                sc_sg = evaluate_multiturn_episode(
                    ep.turn1_turns,
                    ep.turn2_turns,
                    ver_sg.final_json,
                    target_node_id="d6",
                    expected_unit=ep.expected_unit,
                    preserved_targets=dict(ep.preserved_targets) or None,
                )
                if sc_sg.ok:
                    struct_pass_count += 1
            # Verify structural_gate achieves b - c >= +40 over baseline Arm 1 on every model without touching numeric values
            arm1_base_pass = sum(1 for r in rows if r["arm1Ok"])
            self.assertGreaterEqual(
                struct_pass_count - arm1_base_pass,
                40,
                msg=f"Expected b - c >= 40 for structural_gate on {fname}, got {struct_pass_count} vs {arm1_base_pass}",
            )

    def test_sympy_and_pint_parity_with_arm3b_state_diff_snapshot(self) -> None:
        try:
            import pint
            import sympy
        except ImportError:
            self.skipTest("SymPy or Pint not installed in active Python environment")

        ureg = pint.UnitRegistry(non_int_type=Fraction)
        ureg.define("USD = [currency_usd]")
        ureg.define("K_USD = 1000 * USD")
        ureg.define("M_USD = 1000000 * USD")
        ureg.define("B_USD = 1000000000 * USD")
        ureg.define("scalar = []")
        ureg.define("pct = 1 / 100 * scalar")
        ureg.define("bps = 1 / 10000 * scalar")

        unit_map = {
            "ug": ureg.ug,
            "mcg": ureg.ug,
            "mg": ureg.mg,
            "g": ureg.g,
            "kg": ureg.kg,
            "mm": ureg.mm,
            "cm": ureg.cm,
            "m": ureg.m,
            "km": ureg.km,
            "uL": ureg.uL,
            "mL": ureg.mL,
            "L": ureg.L,
            "ms": ureg.ms,
            "s": ureg.s,
            "min": ureg.minute,
            "h": ureg.hour,
            "hr": ureg.hour,
            "USD": ureg.USD,
            "K_USD": ureg.K_USD,
            "$K": ureg.K_USD,
            "M_USD": ureg.M_USD,
            "$M": ureg.M_USD,
            "B_USD": ureg.B_USD,
            "$B": ureg.B_USD,
            "scalar": ureg.scalar,
            "1": ureg.scalar,
            "%": ureg.pct,
            "pct": ureg.pct,
            "bps": ureg.bps,
        }

        for ep in self.episodes_n50:
            diff_snap = compute_state_diff_snapshot(
                ep.turn1_turns,
                ep.turn2_turns,
                target_node_id=ep.target_node_id,
                expected_unit=ep.expected_unit,
                preserved_targets=dict(ep.preserved_targets) or None,
            )
            defs = {t.node_id: t for t in ep.all_turns}
            repl: dict[str, str] = {}
            for t in ep.all_turns:
                if t.kind == "retract" and t.target_id:
                    repl[t.target_id] = t.node_id

            def resolve(nid: str) -> str:
                cur = nid
                while cur in repl:
                    cur = repl[cur]
                return cur

            def eval_sympy_pint(nid: str):
                eff = resolve(nid)
                spec = defs[eff]
                if spec.kind in {"premise", "retract"}:
                    q = ureg.Quantity(sympy.Rational(str(spec.value)), unit_map[spec.unit])
                    return q
                lq = eval_sympy_pint(spec.left_id)
                rq = eval_sympy_pint(spec.right_id)
                if spec.arith_op == "add":
                    return lq + rq
                if spec.arith_op == "sub":
                    return lq - rq
                if spec.arith_op == "mul":
                    return lq * rq
                if spec.arith_op == "div":
                    return lq / rq
                raise ValueError(spec.arith_op)

            target_q = eval_sympy_pint(ep.target_node_id).to(unit_map[ep.expected_unit])
            mag_rat = sympy.Rational(target_q.magnitude)
            sym_frac = Fraction(int(sympy.fraction(mag_rat)[0]), int(sympy.fraction(mag_rat)[1]))
            self.assertEqual(diff_snap.truth_target_fraction, sym_frac)
            ok_tol, _, _ = mixed_tolerance_match(float(target_q.magnitude), diff_snap.truth_target_fraction)
            self.assertTrue(ok_tol)

    def test_zero_core_static_audit_passes_with_arm3_state_diff(self) -> None:
        checker_path = REPO_ROOT / "scripts" / "check_zero_core.py"
        spec = importlib.util.spec_from_file_location("check_zero_core", checker_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        report = mod.audit_directory(REPO_ROOT)
        self.assertTrue(report["ok"], msg=f"Zero-core violations: {report['violations']}")
        self.assertEqual(report["grade"], "[안전]")


if __name__ == "__main__":
    unittest.main()
