#!/usr/bin/env python3
"""Unit tests and SymPy+Pint parity checks for P1 Frontier Controls (P1.1 & P1.2)."""

from __future__ import annotations

from fractions import Fraction
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

from gyeol_bench import TurnSpec, generate_multiturn_dataset
import run_p1_frontier_controls as p1

try:
    import pint
    import sympy as sp

    _SYMPY_PINT_AVAILABLE = True
except ImportError:
    _SYMPY_PINT_AVAILABLE = False


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


def _sympy_pint_eval(
    ureg: "pint.UnitRegistry",
    turns: tuple[TurnSpec, ...],
    target_id: str,
    target_unit: str,
) -> "sp.Rational":
    unit_alias = {"%": "pct", "1": "scalar", "$K": "K_USD", "$M": "M_USD", "$B": "B_USD"}
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

    def unit_factor(src_u: str, dst_u: str) -> "sp.Rational":
        su = unit_alias.get(src_u, src_u)
        du = unit_alias.get(dst_u, dst_u)
        q = ureg.Quantity(Fraction(1, 1), su).to(du)
        mag = q.magnitude
        if isinstance(mag, Fraction):
            return sp.Rational(mag.numerator, mag.denominator)
        return sp.nsimplify(str(mag), rational=True)

    cache: dict[str, tuple["sp.Rational", str]] = {}

    def eval_node(nid: str) -> tuple["sp.Rational", str]:
        rnid = resolve(nid)
        if rnid in cache:
            return cache[rnid]
        t = defs[rnid]
        if t.kind in ("premise", "retract"):
            res = (sp.Rational(t.value), t.unit)
        else:
            lv, lu = eval_node(t.left_id)
            rv, ru = eval_node(t.right_id)
            if t.arith_op in ("add", "sub"):
                conv_rv = rv * unit_factor(ru, lu)
                res = (lv + conv_rv if t.arith_op == "add" else lv - conv_rv, lu)
            elif t.arith_op == "mul":
                if unit_alias.get(ru, ru) in ("scalar", "pct", "bps"):
                    res = (lv * rv * unit_factor(ru, "scalar"), lu)
                elif unit_alias.get(lu, lu) in ("scalar", "pct", "bps"):
                    res = (lv * unit_factor(lu, "scalar") * rv, ru)
                else:
                    raise ValueError(f"Unsupported mul units: {lu}, {ru}")
            else:
                raise ValueError(f"Unsupported arith_op: {t.arith_op}")
        cache[rnid] = res
        return res

    val, unit = eval_node(target_id)
    return val * unit_factor(unit, target_unit)


class TestP1FrontierControls(unittest.TestCase):
    def test_v03_aliasing_dataset_balance_and_invariants(self) -> None:
        items = p1.generate_value_aliasing_v03_dataset(seed=20261153, count=64)
        self.assertEqual(len(items), 64)
        counts: dict[str, int] = {}
        for item in items:
            ep = item.episode
            counts[ep.condition] = counts.get(ep.condition, 0) + 1
            p1_t = next(t for t in ep.turn1_turns if t.node_id == "p1")
            p4_t = next(t for t in ep.turn1_turns if t.node_id == "p4")
            self.assertEqual(p4_t.value, p1_t.value)
            self.assertEqual(p4_t.unit, p1_t.unit)
            if ep.condition == "E":
                u1_t = next(t for t in ep.turn1_turns if t.node_id == "u1")
                self.assertEqual(u1_t.value, p1_t.value)
            if ep.condition != "A":
                self.assertIsNotNone(item.aliased_both_retracted_float)
                self.assertIsNotNone(item.aliased_wrong_target_float)
                self.assertNotAlmostEqual(item.aliased_both_retracted_float, ep.expected_value_float, places=4)
                self.assertNotAlmostEqual(item.aliased_wrong_target_float, ep.expected_value_float, places=4)
        self.assertEqual(counts, {"A": 16, "B": 12, "C": 12, "D": 12, "E": 12})

    @unittest.skipUnless(_SYMPY_PINT_AVAILABLE, "SymPy+Pint not installed in this Python")
    def test_v03_aliasing_sympy_pint_parity_all_64(self) -> None:
        ureg = _build_sympy_pint_registry()
        items = p1.generate_value_aliasing_v03_dataset(seed=20261153, count=64)
        mismatches = 0
        for item in items:
            ep = item.episode
            sp_t1 = _sympy_pint_eval(ureg, ep.turn1_turns, ep.turn1_target_id, ep.turn1_expected_unit)
            if abs(float(sp_t1) - ep.turn1_expected_value_float) > 1e-9:
                mismatches += 1
            sp_t2 = _sympy_pint_eval(ureg, ep.all_turns, ep.target_node_id, ep.expected_unit)
            if abs(float(sp_t2) - ep.expected_value_float) > 1e-9:
                mismatches += 1
            for pid, punit in ep.preserved_targets:
                sp_p = _sympy_pint_eval(ureg, ep.all_turns, pid, punit)
                # Ensure SymPy+Pint evaluates preserved node cleanly
                self.assertTrue(sp_p > 0)
        self.assertEqual(mismatches, 0)

    def test_clopper_pearson_and_mcnemar_exactness(self) -> None:
        lo0, hi0 = p1.clopper_pearson_ci95(0, 50)
        self.assertEqual(lo0, 0.0)
        self.assertAlmostEqual(hi0, 1.0 - (0.025 ** (1.0 / 50.0)), places=5)
        lo_n, hi_n = p1.clopper_pearson_ci95(50, 50)
        self.assertEqual(hi_n, 1.0)
        self.assertAlmostEqual(lo_n, 0.025 ** (1.0 / 50.0), places=5)
        self.assertAlmostEqual(p1.exact_two_sided_mcnemar_p(10, 0), 2.0 / 1024.0, places=8)
        self.assertEqual(p1.exact_two_sided_mcnemar_p(0, 0), 1.0)

    def test_detect_value_aliasing_confusion(self) -> None:
        items = p1.generate_value_aliasing_v03_dataset(seed=20261153, count=16)
        b_item = next(it for it in items if it.episode.condition == "B")
        ep = b_item.episode
        raw_both = json.dumps({
            "notes": "aliased both",
            "step_id": "d6",
            "value": b_item.aliased_both_retracted_float,
            "unit": ep.expected_unit,
            "retracted_ids": list(ep.expected_retracted_ids),
        })
        flag, mode = p1.detect_value_aliasing_confusion(b_item, raw_both, verdict_ok=False)
        self.assertTrue(flag)
        self.assertEqual(mode, "aliased_both_retracted")

        raw_wrong = json.dumps({
            "notes": "aliased wrong",
            "step_id": "d6",
            "value": b_item.aliased_wrong_target_float,
            "unit": ep.expected_unit,
            "retracted_ids": ["p4"],
        })
        flag2, mode2 = p1.detect_value_aliasing_confusion(b_item, raw_wrong, verdict_ok=False)
        self.assertTrue(flag2)
        self.assertEqual(mode2, "aliased_wrong_target")

        flag_pass, mode_pass = p1.detect_value_aliasing_confusion(b_item, raw_both, verdict_ok=True)
        self.assertFalse(flag_pass)
        self.assertIsNone(mode_pass)

    def test_prelock_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p_a = Path(td) / "prelock_a.json"
            p_b = Path(td) / "prelock_b.json"
            m_a = p1.build_v03_aliasing_prelock(p_a)
            m_b = p1.build_v03_aliasing_prelock(p_b)
            self.assertEqual(m_a["dataset_sha256"], m_b["dataset_sha256"])
            self.assertEqual(m_a["condition_counts"], {"A": 16, "B": 12, "C": 12, "D": 12, "E": 12})

    def test_mocked_live_p1_1_and_p1_2_end_to_end(self) -> None:
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
                                "step_id": "d4" if "-t1-" in display_name else "d6",
                                "value": 1.0,
                                "unit": "M_USD",
                                "retracted_ids": [],
                            })}]},
                        }],
                        "usageMetadata": {
                            "promptTokenCount": 100,
                            "candidatesTokenCount": 20,
                            "thoughtsTokenCount": 10,
                        },
                    },
                })
            return f"batches/mock-{display_name}", items

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            spend_path = td_path / "spend.json"
            spend_path.write_text(json.dumps({"spentUsd": 1.635570}), encoding="utf-8")
            with patch.dict(os.environ, {"AI_STUDIO_PAID_TIER_CONFIRMED": "1", "GEMINI_API_KEY": "dummy"}):
                with patch.object(p1.fe, "_batch_generate_v1beta", side_effect=fake_batch):
                    r1 = p1.run_p1_1_for_model(
                        "models/gemini-4-carbon",
                        live=True,
                        checkpoint_dir=td_path / "ckpt",
                        spend_path=spend_path,
                    )
                    r2 = p1.run_p1_2_for_model(
                        "models/gemini-4-carbon",
                        live=True,
                        checkpoint_dir=td_path / "ckpt",
                        spend_path=spend_path,
                        prelock_path=td_path / "prelock.json",
                    )
            self.assertEqual(r1["count"], 50)
            self.assertEqual(r1["condition_counts"], {"A": 25, "B": 7, "C": 6, "D": 6, "E": 6})
            self.assertEqual(r1["finish_reason_max_tokens"], 0)
            self.assertEqual(r2["count"], 64)
            self.assertEqual(r2["finish_reason_max_tokens"], 0)
            self.assertIn("paired_mcnemar_c2_v2_vs_c1_itt_n64", r2)
            self.assertIn("paired_mcnemar_c2_v2_vs_c1_perturbed_n48", r2)

    def test_c2_v2_materialized_ledger_has_derivation_rules_and_n32_stratified(self) -> None:
        items = p1.generate_value_aliasing_v03_dataset(seed=20261153, count=64)
        ep = items[0].episode
        raw1 = json.dumps({
            "notes": "Turn 1 trace",
            "step_id": ep.turn1_target_id,
            "value": ep.turn1_expected_value_float,
            "unit": ep.turn1_expected_unit,
            "retracted_ids": [],
        })
        c2 = p1.build_c2_v2_materialized_ledger_contents(ep, raw1)
        payload = json.loads(c2[0]["parts"][0]["text"].split("\n", 1)[1])
        self.assertEqual(payload["ledger_version"], "materialized_ledger_v2")
        self.assertGreaterEqual(len(payload["turn1_derivation_rules"]), 4)

        n32_counts: dict[str, int] = {}
        for it in items[:32]:
            n32_counts[it.episode.condition] = n32_counts.get(it.episode.condition, 0) + 1
        self.assertEqual(n32_counts, {"A": 8, "B": 6, "C": 6, "D": 6, "E": 6})


if __name__ == "__main__":
    unittest.main()
