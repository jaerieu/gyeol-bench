"""Unit tests and SymPy+Pint cross-oracle parity checks for clean-room Arm 3A (`arm3_ledger_hook.py`).

Licensed under the Apache License, Version 2.0.
Contains zero lines and zero imports from any private engine.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gyeol_bench import (
    TurnSpec,
    evaluate_multiturn_episode,
    generate_multiturn_dataset,
    mixed_tolerance_match,
    response_adapter,
)
from gyeol_bench.arm3_ledger_hook import (
    apply_arm3a_invariant_guard,
    augment_turn2_prompt_with_ledger,
    build_arm3a_ledger,
    build_arm3a_ledger_from_prompts,
    extract_prompt_contract,
    extract_turns_from_prompt,
    format_exact_json_number,
    format_turn2_ledger_directive,
)

try:
    import pint
    import sympy as sp

    _SYMPY_PINT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional third-party cross-oracle
    _SYMPY_PINT_AVAILABLE = False


CHECKER_PATH = REPO_ROOT / "scripts" / "check_zero_core.py"


def _load_checker_module():
    spec = importlib.util.spec_from_file_location("check_zero_core", CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_sympy_pint_registry() -> "pint.UnitRegistry":
    ureg = pint.UnitRegistry(non_int_type=Fraction, autoconvert_offset_to_baseunit=True)
    ureg.define("USD = [currency_usd]")
    ureg.define("K_USD = 1000 * USD")
    ureg.define("M_USD = 1000000 * USD")
    ureg.define("B_USD = 1000000000 * USD")
    ureg.define("pct = 1 / 100 * dimensionless")
    ureg.define("bps = 1 / 10000 * dimensionless")
    ureg.define("scalar = 1 * dimensionless")
    ureg.define("mcg = ug")
    return ureg


def _sympy_pint_eval_episode(
    ureg: "pint.UnitRegistry",
    turns: tuple[TurnSpec, ...],
    target_id: str,
    target_unit: str,
) -> sp.Rational:
    """Independent SymPy Rational + Pint dimensional evaluator for parity verification."""
    unit_alias = {
        "%": "pct",
        "1": "scalar",
        "$K": "K_USD",
        "$M": "M_USD",
        "$B": "B_USD",
    }
    defs = {t.node_id: t for t in turns}
    replacements: dict[str, str] = {}
    for t in turns:
        if t.kind == "retract" and t.target_id and t.value is not None:
            replacements[t.target_id] = t.node_id

    def resolve(nid: str) -> str:
        cur = nid
        while cur in replacements:
            cur = replacements[cur]
        return cur

    base_for_family = {
        "ug": "g", "mcg": "g", "mg": "g", "g": "g", "kg": "g",
        "mm": "m", "cm": "m", "m": "m", "km": "m",
        "uL": "L", "mL": "L", "L": "L",
        "ms": "s", "s": "s", "min": "s", "h": "s", "hr": "s",
        "USD": "USD", "K_USD": "USD", "$K": "USD", "M_USD": "USD", "$M": "USD", "B_USD": "USD", "$B": "USD",
        "scalar": "scalar", "1": "scalar", "%": "scalar", "pct": "scalar", "bps": "scalar",
    }

    def unit_factor_rational(src_u: str, dst_u: str) -> sp.Rational:
        su = unit_alias.get(src_u, src_u)
        du = unit_alias.get(dst_u, dst_u)
        q = ureg.Quantity(Fraction(1, 1), su).to(du)
        mag = q.magnitude
        if isinstance(mag, Fraction):
            return sp.Rational(mag.numerator, mag.denominator)
        return sp.nsimplify(str(mag), rational=True)

    memo: dict[str, tuple[sp.Rational, str]] = {}

    def eval_node(nid: str) -> tuple[sp.Rational, str]:
        eff = resolve(nid)
        if eff in memo:
            return memo[eff]
        spec = defs[eff]
        if spec.kind in {"premise", "retract"}:
            bu = base_for_family[spec.unit]
            val_r = sp.Rational(str(spec.value)) * unit_factor_rational(spec.unit, bu)
            res = (val_r, bu)
        else:
            assert spec.left_id is not None and spec.right_id is not None
            lv, lu = eval_node(spec.left_id)
            rv, ru = eval_node(spec.right_id)
            if spec.arith_op == "add":
                assert lu == ru
                res = (lv + rv, lu)
            elif spec.arith_op == "sub":
                assert lu == ru
                res = (lv - rv, lu)
            elif spec.arith_op == "mul":
                if lu == "scalar":
                    res = (lv * rv, ru)
                else:
                    assert ru == "scalar"
                    res = (lv * rv, lu)
            elif spec.arith_op == "div":
                if lu == ru:
                    res = (lv / rv, "scalar")
                else:
                    assert ru == "scalar"
                    res = (lv / rv, lu)
            else:
                raise AssertionError(f"Unexpected op: {spec.arith_op}")
        memo[eff] = res
        return res

    b_val, b_unit = eval_node(target_id)
    factor_to_target = unit_factor_rational(b_unit, target_unit)
    return b_val * factor_to_target


class TestArm3ALedgerHook(unittest.TestCase):
    def test_zero_core_hygiene_passes_with_arm3a(self) -> None:
        checker = _load_checker_module()
        rep = checker.audit_directory(REPO_ROOT)
        self.assertTrue(rep["ok"], msg=f"Zero-core violations: {rep['violations']}")
        self.assertEqual(rep["grade"], "[안전]")

    def test_prompt_parser_reconstructs_exact_n800_turns_and_contracts(self) -> None:
        episodes = generate_multiturn_dataset(seed=20261122, count=800)
        for ep in episodes:
            parsed_t1 = extract_turns_from_prompt(ep.turn1_prompt)
            parsed_t2 = extract_turns_from_prompt(ep.turn2_prompt)
            self.assertEqual(parsed_t1, ep.turn1_turns, msg=f"Turn 1 mismatch on {ep.episode_id}")
            self.assertEqual(parsed_t2, ep.turn2_turns, msg=f"Turn 2 mismatch on {ep.episode_id}")

            t1_tid, t1_u, t1_pres = extract_prompt_contract(ep.turn1_prompt)
            t2_tid, t2_u, t2_pres = extract_prompt_contract(ep.turn2_prompt)
            self.assertEqual(t1_tid, ep.turn1_target_id)
            self.assertEqual(t1_u, ep.turn1_expected_unit)
            self.assertEqual(t1_pres, ep.turn1_preserved_targets)
            self.assertEqual(t2_tid, ep.target_node_id)
            self.assertEqual(t2_u, ep.expected_unit)
            self.assertEqual(t2_pres, ep.preserved_targets)

    def test_blocks_condition_a_spurious_over_retraction_and_extra_preserved(self) -> None:
        ds = generate_multiturn_dataset(seed=20261122, count=16)
        ep_a = next(e for e in ds if e.condition == "A")
        t1_json = json.dumps(
            {
                "notes": "Turn 1 clean checkpoint",
                "step_id": "d4",
                "value": ep_a.turn1_expected_value_float,
                "unit": ep_a.turn1_expected_unit,
                "retracted_ids": [],
            }
        )
        ledger = build_arm3a_ledger_from_prompts(ep_a.turn1_prompt, ep_a.turn2_prompt, t1_json)
        self.assertEqual(ledger.superseded_retracted_ids, ())
        self.assertEqual(ledger.tainted_turn1_ids, ())
        self.assertEqual(ledger.untainted_turn1_ids, ("d1", "d2", "d3", "d4"))
        self.assertTrue(ledger.committed_turn1.matches_turn1_truth)

        # Case 1: Model emits spurious d1..d4 retracted_ids and hallucinated preserved field on Condition A
        tampered_a = json.dumps(
            {
                "notes": "sup-> d1..d4 hallucinated taint",
                "step_id": "d6",
                "value": ep_a.expected_value_float,
                "unit": ep_a.expected_unit,
                "retracted_ids": ["d1", "d2", "d3", "d4"],
                "preserved": {"d4": {"value": 999}},
            }
        )
        raw_eval = evaluate_multiturn_episode(
            ep_a.turn1_turns, ep_a.turn2_turns, tampered_a, ep_a.target_node_id, ep_a.expected_unit
        )
        self.assertFalse(raw_eval.ok)

        guarded = apply_arm3a_invariant_guard(ledger, tampered_a)
        self.assertTrue(guarded.intervened)
        self.assertTrue(any("I2_COND_A_FALSE_RETRACTION" in v for v in guarded.violations_blocked))
        self.assertEqual(guarded.retracted_ids_verified, ())

        post_eval = evaluate_multiturn_episode(
            ep_a.turn1_turns, ep_a.turn2_turns, guarded.emitted_json, ep_a.target_node_id, ep_a.expected_unit
        )
        self.assertTrue(post_eval.ok)
        self.assertEqual(post_eval.verdict, "PASS")

        # Case 2: Clean Condition A output passes through untouched (intervened=False)
        clean_a = json.dumps(
            {
                "notes": "exact untainted calculation",
                "step_id": "d6",
                "value": format_exact_json_number(ledger.target_truth_fraction),
                "unit": ep_a.expected_unit,
                "retracted_ids": [],
            }
        )
        clean_decision = apply_arm3a_invariant_guard(ledger, clean_a)
        self.assertFalse(clean_decision.intervened)
        self.assertEqual(clean_decision.emitted_json, clean_a)

    def test_blocks_condition_e_d_indep_tampering_spurious_replacement_ids_and_trailing_period(self) -> None:
        ds = generate_multiturn_dataset(seed=20261122, count=16)
        ep_e = next(e for e in ds if e.condition == "E")

        # Build ledger first with valid Turn-1 commitment
        ledger_init = build_arm3a_ledger_from_prompts(ep_e.turn1_prompt, ep_e.turn2_prompt, "")
        indep_val, indep_unit = ledger_init.preserved_truth_bindings["d_indep"]
        t1_json = json.dumps(
            {
                "notes": "Turn 1 checkpoint with d_indep",
                "step_id": "d4",
                "value": ep_e.turn1_expected_value_float,
                "unit": ep_e.turn1_expected_unit,
                "retracted_ids": [],
                "preserved": {
                    "d_indep": {"value": format_exact_json_number(indep_val), "unit": indep_unit}
                },
            }
        )
        ledger = build_arm3a_ledger_from_prompts(ep_e.turn1_prompt, ep_e.turn2_prompt, t1_json)
        self.assertEqual(ledger.superseded_retracted_ids, ("p1", "p1_r1"))
        self.assertIn("d_indep", ledger.untainted_turn1_ids)
        self.assertIn("u_d1", ledger.untainted_turn1_ids)
        self.assertEqual(ledger.tainted_turn1_ids, ("d1", "d2", "d3", "d4"))
        self.assertTrue(ledger.committed_turn1.preserved_matches_turn1_truth["d_indep"])

        # Simulate combined failure modes observed in Phase 3:
        # 1) d_indep tampered by +2136 (Qwen14B OVER_RETRACTION)
        # 2) p1_r2 active replacement erroneously added to retracted_ids (Phi-4 OVER_RETRACTION)
        # 3) trailing period '.' after closing brace (Mistral-24B PARSE_REJECT)
        bad_turn2 = (
            json.dumps(
                {
                    "notes": "Retracted p1 -> p1_r1 -> p1_r2",
                    "step_id": "d6",
                    "value": ep_e.expected_value_float,
                    "unit": ep_e.expected_unit,
                    "retracted_ids": ["p1", "p1_r1", "p1_r2", "d_indep"],
                    "preserved": {
                        "d_indep": {
                            "value": float(indep_val) + 2136.0,
                            "unit": indep_unit,
                        }
                    },
                }
            )
            + "."
        )
        raw_res = evaluate_multiturn_episode(
            ep_e.turn1_turns,
            ep_e.turn2_turns,
            bad_turn2,
            target_node_id="d6",
            expected_unit=ep_e.expected_unit,
            preserved_targets=dict(ep_e.preserved_targets),
        )
        self.assertFalse(raw_res.ok)
        self.assertEqual(raw_res.verdict, "PARSE_REJECT")

        guarded = apply_arm3a_invariant_guard(ledger, bad_turn2)
        self.assertTrue(guarded.intervened)
        self.assertTrue(any("I1_ENVELOPE_FORMAT" in v for v in guarded.violations_blocked))
        self.assertTrue(any("I2_SPURIOUS_OVER_RETRACTION" in v for v in guarded.violations_blocked))
        self.assertTrue(any("I3_PRESERVED_VALUE_TAMPERED:d_indep" in v for v in guarded.violations_blocked))
        self.assertEqual(guarded.retracted_ids_verified, ("p1", "p1_r1"))

        fixed_res = evaluate_multiturn_episode(
            ep_e.turn1_turns,
            ep_e.turn2_turns,
            guarded.emitted_json,
            target_node_id="d6",
            expected_unit=ep_e.expected_unit,
            preserved_targets=dict(ep_e.preserved_targets),
        )
        self.assertTrue(fixed_res.ok, msg=f"Expected PASS after guard, got {fixed_res}")
        self.assertEqual(fixed_res.verdict, "PASS")

    def test_zero_answer_leakage_in_pre_inference_turn2_directive_and_ceiling_mode(self) -> None:
        ds = generate_multiturn_dataset(seed=20261151, count=50)
        for cond in ("A", "B", "C", "D", "E"):
            ep = next(e for e in ds if e.condition == cond)
            ledger = build_arm3a_ledger_from_prompts(ep.turn1_prompt, ep.turn2_prompt)
            directive = format_turn2_ledger_directive(ledger)
            aug_prompt = augment_turn2_prompt_with_ledger(ep.turn2_prompt, ledger)
            self.assertIn("[Arm3A Structural Dependency-Ledger Guard]", aug_prompt)
            self.assertIn(json.dumps(list(ep.expected_retracted_ids)), aug_prompt)

            # Strict Audit Blocker #1 check: ZERO pre-computed target or rebuilt numeric leakage in directive!
            target_str = str(format_exact_json_number(ledger.target_truth_fraction))
            self.assertNotIn("Verified target binding", directive)
            self.assertNotIn("Verified Turn-1 derived ledger", directive)
            self.assertNotIn("Active root overrides", directive)
            self.assertNotIn(f"= {target_str}", directive)
            self.assertNotIn(target_str, directive)
            for nid in ledger.tainted_turn1_ids:
                nb = ledger.node_bindings[nid]
                self.assertNotIn(f"{nid}[REBUILT]", directive)
                self.assertNotIn(f"= {format_exact_json_number(nb.base_value)}", directive)

            if cond in ("B", "C", "D", "E"):
                assert ep.pre_retraction_value_float is not None
                stale_candidate = json.dumps(
                    {
                        "notes": "reused stale Turn-1 d4 without recomputing",
                        "step_id": "d6",
                        "value": ep.pre_retraction_value_float,
                        "unit": ep.expected_unit,
                        "retracted_ids": [],
                    }
                )
                # Default mode ("structural_only", primary K2 co-logged guard) NEVER overwrites stale numeric value
                default_dec = apply_arm3a_invariant_guard(ledger, stale_candidate)
                self.assertEqual(default_dec.target_value_fraction, Fraction(str(ep.pre_retraction_value_float)))
                # Secondary ceiling mode ("full") explicitly repairs I4_STALE_STATE_BLOCKED
                ceiling_dec = apply_arm3a_invariant_guard(ledger, stale_candidate, mode="full")
                self.assertTrue(ceiling_dec.intervened)
                self.assertIn("I4_STALE_STATE_BLOCKED", ceiling_dec.violations_blocked)
                scored = evaluate_multiturn_episode(
                    ep.turn1_turns,
                    ep.turn2_turns,
                    ceiling_dec.emitted_json,
                    target_node_id=ep.target_node_id,
                    expected_unit=ep.expected_unit,
                    preserved_targets=dict(ep.preserved_targets) or None,
                )
                self.assertTrue(scored.ok)
                self.assertEqual(scored.verdict, "PASS")

    def test_sympy_pint_parity_across_n50_and_n800_datasets(self) -> None:
        if not _SYMPY_PINT_AVAILABLE:
            self.skipTest("SymPy / Pint not installed in active Python environment.")
        ureg = _build_sympy_pint_registry()

        for seed, count in ((20261151, 50), (20261122, 800)):
            episodes = generate_multiturn_dataset(seed=seed, count=count)
            self.assertEqual(len(episodes), count)
            for ep in episodes:
                ledger = build_arm3a_ledger_from_prompts(ep.turn1_prompt, ep.turn2_prompt)
                sp_target = _sympy_pint_eval_episode(
                    ureg, ep.all_turns, ep.target_node_id, ep.expected_unit
                )
                ledger_rational = sp.Rational(
                    ledger.target_truth_fraction.numerator,
                    ledger.target_truth_fraction.denominator,
                )
                self.assertEqual(
                    ledger_rational,
                    sp_target,
                    msg=f"SymPy+Pint target mismatch on {ep.episode_id} (seed={seed})",
                )
                self.assertEqual(ledger.superseded_retracted_ids, ep.expected_retracted_ids)

                for pid, punit in ep.preserved_targets:
                    sp_pres = _sympy_pint_eval_episode(ureg, ep.all_turns, pid, punit)
                    p_frac, p_u = ledger.preserved_truth_bindings[pid]
                    self.assertEqual(p_u, punit)
                    self.assertEqual(
                        sp.Rational(p_frac.numerator, p_frac.denominator),
                        sp_pres,
                        msg=f"SymPy+Pint preserved mismatch on {ep.episode_id}:{pid}",
                    )

                # Verify directive has zero target-answer leakage and secondary ceiling mode passes response_adapter
                directive = format_turn2_ledger_directive(ledger)
                self.assertTrue(directive.startswith("[Arm3A Structural Dependency-Ledger Guard]"))
                target_str = str(format_exact_json_number(ledger.target_truth_fraction))
                self.assertNotIn("Verified target binding", directive)
                self.assertNotIn(f"= {target_str}", directive)
                # Default structural_only mode on empty "{}" refuses to fabricate numeric answers
                struct_empty = apply_arm3a_invariant_guard(ledger, "{}")
                self.assertFalse(struct_empty.intervened)
                # Secondary symbolic-offload ceiling mode ("full") verifies exact rational emission
                decision = apply_arm3a_invariant_guard(ledger, "{}", mode="full")
                self.assertTrue(
                    response_adapter(
                        decision.emitted_json,
                        required_preserved=tuple(k for k, _ in ep.preserved_targets),
                    ).ok
                )
                scored_oracle = evaluate_multiturn_episode(
                    ep.turn1_turns,
                    ep.turn2_turns,
                    decision.emitted_json,
                    target_node_id=ep.target_node_id,
                    expected_unit=ep.expected_unit,
                    preserved_targets=dict(ep.preserved_targets) or None,
                )
                self.assertTrue(scored_oracle.ok, msg=f"Oracle failed on {ep.episode_id}: {scored_oracle}")
                self.assertEqual(scored_oracle.verdict, "PASS")
                matched, _, _ = mixed_tolerance_match(decision.target_value_fraction, ep.expected_value_float)
                self.assertTrue(matched)

    def test_structural_only_mode_preserves_model_arithmetic_while_blocking_over_retraction(self) -> None:
        ds = generate_multiturn_dataset(seed=20261151, count=16)
        ep_e = next(e for e in ds if e.condition == "E")
        ledger_init = build_arm3a_ledger_from_prompts(ep_e.turn1_prompt, ep_e.turn2_prompt)
        indep_val, indep_unit = ledger_init.preserved_truth_bindings["d_indep"]
        t1_json = json.dumps(
            {
                "notes": "Turn 1 valid checkpoint",
                "step_id": "d4",
                "value": ep_e.turn1_expected_value_float,
                "unit": ep_e.turn1_expected_unit,
                "retracted_ids": [],
                "preserved": {"d_indep": {"value": format_exact_json_number(indep_val), "unit": indep_unit}},
            }
        )
        ledger = build_arm3a_ledger_from_prompts(ep_e.turn1_prompt, ep_e.turn2_prompt, t1_json)
        wrong_arith_with_over_retraction = json.dumps(
            {
                "notes": "model arithmetic off by +99, plus over-retraction",
                "step_id": "d6",
                "value": ep_e.expected_value_float + 99.0,
                "unit": ep_e.expected_unit,
                "retracted_ids": ["p1", "p1_r1", "p1_r2", "d1"],
                "preserved": {"d_indep": {"value": 999999, "unit": indep_unit}},
            }
        )
        # Default mode is now "structural_only"
        struct_dec = apply_arm3a_invariant_guard(ledger, wrong_arith_with_over_retraction)
        self.assertTrue(struct_dec.intervened)
        self.assertEqual(struct_dec.retracted_ids_verified, ("p1", "p1_r1"))
        # Model's wrong arithmetic (+99.0) is preserved, so verdict becomes CONDITIONAL_ARITH_ERROR instead of OVER_RETRACTION
        ev_struct = evaluate_multiturn_episode(
            ep_e.turn1_turns,
            ep_e.turn2_turns,
            struct_dec.emitted_json,
            target_node_id="d6",
            expected_unit=ep_e.expected_unit,
            preserved_targets=dict(ep_e.preserved_targets),
        )
        self.assertFalse(ev_struct.ok)
        self.assertEqual(ev_struct.verdict, "CONDITIONAL_ARITH_ERROR")


if __name__ == "__main__":
    unittest.main()
