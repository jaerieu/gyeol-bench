#!/usr/bin/env python3
"""P1 Frontier Controls: Same-Harness Arm 1 Control (P1.1) & v0.3 Value-Aliasing Stress Slice (P1.2).

Zero external dependencies (pure Python stdlib, Apache-2.0, 0 private core lines).
Enforces:
- AI_STUDIO_PAID_TIER_CONFIRMED=1 for live Paid Batch API (50% discount)
- ONLY models/gemini-4-carbon and models/gemini-4-hydrogen (thinkingLevel="MEDIUM", temperature=0.0)
- 0 -latest aliases, 0 Google item-gen
- FINISH_REASON_MAX_TOKENS == 0
- Cumulative AI Studio spend < $5.00 USD hard cap
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gyeol_bench import (
    ExactRationalOracle,
    MultiTurnEpisode,
    TurnSpec,
    V3_MULTITURN_BASELINE_SYSTEM_PROMPT,
    evaluate_multiturn_episode,
    evaluate_retraction_episode,
    generate_multiturn_dataset,
    mixed_tolerance_match,
    normalize_to_base_fraction,
    parse_step_envelope,
)
from gyeol_bench.oracle import (
    DimensionMismatchError,
    _all_multiturn_primary_stales_nondegenerate,
    _compute_multiturn_stale_targets,
    _eval_turns,
    _format_turn_lines,
    _is_terminating_decimal,
    convert_from_base_fraction,
)
import run_frontier_eviction_v02 as fe

P1_1_SEED = 20261122
P1_1_COUNT = 50
P1_2_SEED = 20261153
P1_2_COUNT = 64

ALLOWED_MODELS = ("models/gemini-4-carbon", "models/gemini-4-hydrogen")
DEFAULT_SPEND_PATH = Path("/home/gyeol/.config/gyeol/aistudio-spend.json")
DEFAULT_CHECKPOINT_DIR = Path("/tmp/gyeol-colab-checkpoints/p1-frontier-controls")
DEFAULT_PRELOCK_PATH = REPO_ROOT / "manifests" / "p1-v03-aliasing-prelock.json"
DEFAULT_P1_1_OUT = REPO_ROOT / "manifests" / "p1-frontier-arm1-control-results.json"
DEFAULT_P1_2_OUT = REPO_ROOT / "manifests" / "p1-frontier-v03-aliasing-results.json"
HARD_CAP_CUMULATIVE_USD = 5.00

VERDICT_KEYS = (
    "PASS",
    "PARSE_REJECT",
    "CONDITIONAL_ARITH_ERROR",
    "STALE_STATE_RETENTION",
    "OVER_RETRACTION",
)


@dataclass(frozen=True)
class ValueAliasingEpisode:
    episode: MultiTurnEpisode
    aliased_both_retracted_frac: Optional[Fraction]
    aliased_wrong_target_frac: Optional[Fraction]

    @property
    def aliased_both_retracted_float(self) -> Optional[float]:
        return float(self.aliased_both_retracted_frac) if self.aliased_both_retracted_frac is not None else None

    @property
    def aliased_wrong_target_float(self) -> Optional[float]:
        return float(self.aliased_wrong_target_frac) if self.aliased_wrong_target_frac is not None else None


def clopper_pearson_ci95(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact two-sided 95% Clopper-Pearson confidence interval using pure stdlib math.comb."""
    if n <= 0:
        return (0.0, 1.0)
    if k < 0 or k > n:
        raise ValueError(f"Invalid k={k} for n={n}")

    half_alpha = alpha / 2.0

    def binom_cdf_upper_tail(prob: float, min_k: int) -> float:
        return sum(math.comb(n, i) * (prob ** i) * ((1.0 - prob) ** (n - i)) for i in range(min_k, n + 1))

    def binom_cdf_lower_tail(prob: float, max_k: int) -> float:
        return sum(math.comb(n, i) * (prob ** i) * ((1.0 - prob) ** (n - i)) for i in range(0, max_k + 1))

    if k == 0:
        lo = 0.0
    else:
        a, b = 0.0, 1.0
        for _ in range(80):
            mid = 0.5 * (a + b)
            if binom_cdf_upper_tail(mid, k) < half_alpha:
                a = mid
            else:
                b = mid
        lo = 0.5 * (a + b)

    if k == n:
        hi = 1.0
    else:
        a, b = 0.0, 1.0
        for _ in range(80):
            mid = 0.5 * (a + b)
            if binom_cdf_lower_tail(mid, k) < half_alpha:
                b = mid
            else:
                a = mid
        hi = 0.5 * (a + b)

    return (round(lo, 6), round(hi, 6))


def exact_two_sided_mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided binomial McNemar p-value via math.comb."""
    if b < 0 or c < 0:
        raise ValueError(f"Discordant counts must be non-negative: b={b}, c={c}")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _compute_aliasing_signatures(
    turn1_turns: Sequence[TurnSpec],
    turn2_turns: Sequence[TurnSpec],
    target_node_id: str,
    display_unit: str,
) -> Tuple[Optional[Fraction], Optional[Fraction]]:
    """Compute exact rational values for aliased_both_retracted and aliased_wrong_target."""
    retract_turns = [t for t in turn2_turns if t.kind == "retract"]
    if not retract_turns:
        return (None, None)

    p1_chain: List[TurnSpec] = []
    tracked = {"p1"}
    for t in retract_turns:
        if t.target_id in tracked:
            p1_chain.append(t)
            tracked.add(t.node_id)
    if not p1_chain:
        return (None, None)

    p1_final = p1_chain[-1]
    p1_chain_ids = {t.node_id for t in p1_chain}

    both_turns = tuple(turn1_turns) + tuple(turn2_turns) + (
        TurnSpec("p4_alias_both", "retract", p1_final.value, p1_final.unit, retraction_op="fix->", target_id="p4"),
    )
    wrong_turns = tuple(turn1_turns) + tuple(t for t in turn2_turns if t.node_id not in p1_chain_ids) + (
        TurnSpec("p4_alias_wrong", "retract", p1_final.value, p1_final.unit, retraction_op="fix->", target_id="p4"),
    )

    try:
        both_snap = _eval_turns(both_turns)
        wrong_snap = _eval_turns(wrong_turns)
    except (ValueError, ZeroDivisionError, DimensionMismatchError):
        return (None, None)

    if target_node_id not in both_snap.active_values_base or target_node_id not in wrong_snap.active_values_base:
        return (None, None)

    bb, bdim, _ = both_snap.active_values_base[target_node_id]
    wb, wdim, _ = wrong_snap.active_values_base[target_node_id]
    return (
        convert_from_base_fraction(bb, bdim, display_unit),
        convert_from_base_fraction(wb, wdim, display_unit),
    )


def generate_value_aliasing_v03_dataset(
    seed: int = P1_2_SEED,
    count: int = P1_2_COUNT,
) -> List[ValueAliasingEpisode]:
    """Clean-room v0.3 Targeted Value-Aliasing Stress Slice (16 A, 12 B, 12 C, 12 D, 12 E at count=64)."""
    if count <= 0:
        raise ValueError("count must be positive.")
    rng = random.Random(seed)
    # 16-item block: 4 A, 3 B, 3 C, 3 D, 3 E -> at count=64: 16 A, 12 B, 12 C, 12 D, 12 E.
    cond_block = (
        "A", "B", "C", "D",
        "E", "A", "B", "C",
        "D", "E", "A", "B",
        "C", "D", "E", "A",
    )
    retracted_rule = (
        'Rule for "retracted_ids": list ONLY unique node IDs directly on the left of ":=" in any '
        "fix->/err->/sup-> step, sorted alphabetically (exclude un-superseded replacement IDs and d* nodes; [] if none)."
    )
    items: List[ValueAliasingEpisode] = []

    for idx in range(count):
        ep_id = f"alias-{idx + 1:04d}"
        cond = cond_block[idx % len(cond_block)]
        grp = idx // 5
        t1_preserved: Tuple[Tuple[str, str], ...] = ()
        t2_preserved: Tuple[Tuple[str, str], ...] = ()
        t2_notes_extra = ""
        seq_ref = 100 + (idx % 90)
        lag_ms = 8 + (idx % 17)
        both_frac: Optional[Fraction] = None
        wrong_frac: Optional[Fraction] = None

        if cond == "A":
            unit = ("km", "g", "L", "s", "M_USD")[idx % 5]
            t1_unit, t2_unit = unit, unit
            p1 = rng.randint(20, 90)
            p2, p3, p4 = rng.randint(15, 75), rng.randint(2, 5), p1
            p5, s2 = rng.randint(12, 65), rng.randint(2, 5)
            turn1_turns: Tuple[TurnSpec, ...] = (
                TurnSpec("p1", "premise", p1, unit),
                TurnSpec("p2", "premise", p2, unit),
                TurnSpec("p3", "premise", p3, "scalar"),
                TurnSpec("p4", "premise", p4, unit),
                TurnSpec("d1", "derive", arith_op="add", left_id="p1", right_id="p2"),
                TurnSpec("d2", "derive", arith_op="mul", left_id="d1", right_id="p3"),
                TurnSpec("d3", "derive", arith_op="sub", left_id="d2", right_id="p4"),
                TurnSpec("d4", "derive", arith_op="add", left_id="d3", right_id="p1"),
            )
            turn2_turns: Tuple[TurnSpec, ...] = (
                TurnSpec("p5", "premise", p5, unit),
                TurnSpec("s2", "premise", s2, "scalar"),
                TurnSpec("d5", "derive", arith_op="add", left_id="d4", right_id="p5"),
                TurnSpec("d6", "derive", arith_op="mul", left_id="d5", right_id="s2"),
            )
            t2_notes_extra = (
                "[Status Note 1: Session heartbeat nominal; zero upstream premise amendments queued.]\n"
                f"[Status Note 2: Checkpoint buffer seq_ref = {seq_ref} verified ({lag_ms} ms).]\n"
            )
        elif cond == "B":
            unit = ("M_USD", "g", "km", "L")[grp % 4]
            t1_unit, t2_unit = unit, unit
            p1 = rng.randint(30, 120)
            p2, p3, p4 = rng.randint(20, 90), rng.randint(2, 5), p1
            base_r1, base_r2, s2 = rng.randint(160, 280), rng.randint(330, 490), rng.randint(2, 5)
            turn1_turns = (
                TurnSpec("p1", "premise", p1, unit),
                TurnSpec("p2", "premise", p2, unit),
                TurnSpec("p3", "premise", p3, "scalar"),
                TurnSpec("p4", "premise", p4, unit),
                TurnSpec("d1", "derive", arith_op="add", left_id="p1", right_id="p2"),
                TurnSpec("d2", "derive", arith_op="mul", left_id="d1", right_id="p3"),
                TurnSpec("d3", "derive", arith_op="sub", left_id="d2", right_id="p4"),
                TurnSpec("d4", "derive", arith_op="add", left_id="d3", right_id="p1"),
            )
            turn2_turns = ()
            for attempt in range(32):
                cand_t2 = (
                    TurnSpec("p1_r1", "retract", base_r1 + 17 * attempt, unit, retraction_op="fix->", target_id="p1"),
                    TurnSpec("p1_r2", "retract", base_r2 + 29 * attempt, unit, retraction_op="sup->", target_id="p1_r1"),
                    TurnSpec("s2", "premise", s2, "scalar"),
                    TurnSpec("d5", "derive", arith_op="add", left_id="d4", right_id="d2"),
                    TurnSpec("d6", "derive", arith_op="mul", left_id="d5", right_id="s2"),
                )
                if not _all_multiturn_primary_stales_nondegenerate(turn1_turns, cand_t2, "d6", t2_unit):
                    continue
                tsnap = _eval_turns(turn1_turns + cand_t2)
                tb, td, _ = tsnap.active_values_base["d6"]
                tf = convert_from_base_fraction(tb, td, t2_unit)
                bf, wf = _compute_aliasing_signatures(turn1_turns, cand_t2, "d6", t2_unit)
                if (
                    bf is not None
                    and wf is not None
                    and _is_terminating_decimal(bf)
                    and _is_terminating_decimal(wf)
                    and not mixed_tolerance_match(bf, tf)[0]
                    and not mixed_tolerance_match(wf, tf)[0]
                ):
                    turn2_turns = cand_t2
                    both_frac, wrong_frac = bf, wf
                    break
            assert turn2_turns, f"Failed v0.3 aliasing non-degeneracy on {ep_id}"
        elif cond == "C":
            unit = ("g", "km", "M_USD", "L")[grp % 4]
            dist_u = ("km", "M_USD", "L", "g")[grp % 4]
            t1_unit, t2_unit = unit, unit
            p1 = rng.randint(30, 110)
            p2, p3, p4 = rng.randint(2, 5), rng.randint(25, 100), p1
            x1, x2 = rng.randint(40, 150), rng.randint(20, 95)
            base_r1, base_r3 = rng.randint(160, 280), rng.randint(150, 260)
            p5, s2 = rng.randint(15, 80), rng.randint(2, 5)
            turn1_turns = (
                TurnSpec("p1", "premise", p1, unit),
                TurnSpec("p2", "premise", p2, "scalar"),
                TurnSpec("p3", "premise", p3, unit),
                TurnSpec("p4", "premise", p4, unit),
                TurnSpec("d1", "derive", arith_op="mul", left_id="p1", right_id="p2"),
                TurnSpec("d2", "derive", arith_op="add", left_id="d1", right_id="p4"),
                TurnSpec("d3", "derive", arith_op="add", left_id="d2", right_id="p3"),
                TurnSpec("d4", "derive", arith_op="add", left_id="d3", right_id="p1"),
            )
            turn2_turns = ()
            for attempt in range(32):
                cand_t2 = (
                    TurnSpec("x1", "premise", x1, dist_u),
                    TurnSpec("p1_r1", "retract", base_r1 + 19 * attempt, unit, retraction_op="err->", target_id="p1"),
                    TurnSpec("x2", "premise", x2, dist_u),
                    TurnSpec("p3_r1", "retract", base_r3 + 23 * attempt, unit, retraction_op="fix->", target_id="p3"),
                    TurnSpec("p5", "premise", p5, unit),
                    TurnSpec("s2", "premise", s2, "scalar"),
                    TurnSpec("d5a", "derive", arith_op="add", left_id="d2", right_id="p5"),
                    TurnSpec("d5", "derive", arith_op="add", left_id="d4", right_id="d5a"),
                    TurnSpec("d6", "derive", arith_op="mul", left_id="d5", right_id="s2"),
                )
                if not _all_multiturn_primary_stales_nondegenerate(turn1_turns, cand_t2, "d6", t2_unit):
                    continue
                tsnap = _eval_turns(turn1_turns + cand_t2)
                tb, td, _ = tsnap.active_values_base["d6"]
                tf = convert_from_base_fraction(tb, td, t2_unit)
                bf, wf = _compute_aliasing_signatures(turn1_turns, cand_t2, "d6", t2_unit)
                if (
                    bf is not None
                    and wf is not None
                    and _is_terminating_decimal(bf)
                    and _is_terminating_decimal(wf)
                    and not mixed_tolerance_match(bf, tf)[0]
                    and not mixed_tolerance_match(wf, tf)[0]
                ):
                    turn2_turns = cand_t2
                    both_frac, wrong_frac = bf, wf
                    break
            assert turn2_turns, f"Failed v0.3 aliasing non-degeneracy on {ep_id}"
        elif cond == "D":
            fam = grp % 5
            alt = (grp // 5) % 2
            if fam == 0:
                t1_unit, r_unit, t2_unit = "mg", "kg", "g"
                v1 = 100 * rng.randint(2, 9)
                v2, v4 = 100 * rng.randint(2, 8), v1
                rv1, step_r = rng.randint(1, 4), 1
            elif fam == 1:
                t1_unit, r_unit, t2_unit = "cm", "km", "m"
                v1 = 20 * rng.randint(2, 9)
                v2, v4 = 20 * rng.randint(2, 7), v1
                rv1, step_r = rng.randint(1, 4), 1
            elif fam == 2:
                if alt == 0:
                    t1_unit, r_unit, t2_unit = "mL", "L", "uL"
                else:
                    t1_unit, r_unit, t2_unit = "uL", "mL", "L"
                v1 = 100 * rng.randint(2, 8)
                v2, v4 = 100 * rng.randint(1, 6), v1
                rv1, step_r = rng.randint(2, 6), 1
            elif fam == 3:
                t1_unit, r_unit, t2_unit = "s", ("h" if alt == 0 else "min"), ("min" if alt == 0 else "ms")
                v1 = 120 * rng.randint(1, 4)
                v2, v4 = 120 * rng.randint(1, 3), v1
                rv1, step_r = (rng.randint(1, 3) if alt == 0 else rng.randint(12, 30)), 1
            else:
                t1_unit, r_unit, t2_unit = "K_USD", "M_USD", ("USD" if alt == 0 else "B_USD")
                v1 = 100 * rng.randint(2, 8)
                v2, v4 = 100 * rng.randint(1, 6), v1
                rv1, step_r = rng.randint(2, 7), 1

            bps_v = rng.choice((500, 1000, 1500))
            pct_choices = (20, 25, 30, 40)
            pct_r = rng.choice(pct_choices)
            s2 = rng.choice((2, 4, 5))
            turn1_turns = (
                TurnSpec("p1", "premise", v1, t1_unit),
                TurnSpec("p2", "premise", v2, t1_unit),
                TurnSpec("r1", "premise", bps_v, "bps"),
                TurnSpec("p4", "premise", v4, t1_unit),
                TurnSpec("d1", "derive", arith_op="add", left_id="p1", right_id="p2"),
                TurnSpec("d2", "derive", arith_op="mul", left_id="d1", right_id="r1"),
                TurnSpec("d3", "derive", arith_op="add", left_id="d1", right_id="d2"),
                TurnSpec("d4", "derive", arith_op="sub", left_id="d3", right_id="p4"),
            )
            turn2_turns = ()
            for attempt in range(32):
                cand_t2 = (
                    TurnSpec("p1_r1", "retract", rv1 + step_r * attempt, r_unit, retraction_op="fix->", target_id="p1"),
                    TurnSpec(
                        "r1_r1",
                        "retract",
                        pct_choices[(pct_choices.index(pct_r) + attempt) % len(pct_choices)],
                        "%",
                        retraction_op="err->",
                        target_id="r1",
                    ),
                    TurnSpec("s2", "premise", s2, "scalar"),
                    TurnSpec("d5", "derive", arith_op="add", left_id="d4", right_id="d2"),
                    TurnSpec("d6", "derive", arith_op="mul", left_id="d5", right_id="s2"),
                )
                if not _all_multiturn_primary_stales_nondegenerate(turn1_turns, cand_t2, "d6", t2_unit):
                    continue
                tsnap = _eval_turns(turn1_turns + cand_t2)
                tb, td, _ = tsnap.active_values_base["d6"]
                tf = convert_from_base_fraction(tb, td, t2_unit)
                bf, wf = _compute_aliasing_signatures(turn1_turns, cand_t2, "d6", t2_unit)
                if (
                    bf is not None
                    and wf is not None
                    and _is_terminating_decimal(bf)
                    and _is_terminating_decimal(wf)
                    and not mixed_tolerance_match(bf, tf)[0]
                    and not mixed_tolerance_match(wf, tf)[0]
                ):
                    turn2_turns = cand_t2
                    both_frac, wrong_frac = bf, wf
                    break
            assert turn2_turns, f"Failed v0.3 aliasing non-degeneracy on {ep_id}"
        else:
            unit, indep_u = (("M_USD", "g"), ("g", "km"), ("km", "L"), ("L", "M_USD"))[grp % 4]
            t1_unit, t2_unit = unit, unit
            t1_preserved = (("d_indep", indep_u),)
            t2_preserved = (("d_indep", indep_u),)
            p1 = rng.randint(30, 120)
            p2, p3, p4 = rng.randint(20, 90), rng.randint(2, 5), p1
            # Enforce u1.value == p1.value in Condition E
            u1, u2, u3 = p1, rng.randint(15, 85), rng.randint(2, 5)
            u_near_v = rng.randint(35, 145)
            base_r1, base_r2, s2 = rng.randint(160, 280), rng.randint(330, 490), rng.randint(2, 5)
            turn1_turns = (
                TurnSpec("p1", "premise", p1, unit),
                TurnSpec("u1", "premise", u1, indep_u),
                TurnSpec("p2", "premise", p2, unit),
                TurnSpec("u2", "premise", u2, indep_u),
                TurnSpec("p3", "premise", p3, "scalar"),
                TurnSpec("u3", "premise", u3, "scalar"),
                TurnSpec("p4", "premise", p4, unit),
                TurnSpec("u_d1", "derive", arith_op="add", left_id="u1", right_id="u2"),
                TurnSpec("d1", "derive", arith_op="add", left_id="p1", right_id="p2"),
                TurnSpec("d_indep", "derive", arith_op="mul", left_id="u_d1", right_id="u3"),
                TurnSpec("d2", "derive", arith_op="mul", left_id="d1", right_id="p3"),
                TurnSpec("d3", "derive", arith_op="sub", left_id="d2", right_id="p4"),
                TurnSpec("d4", "derive", arith_op="add", left_id="d3", right_id="p1"),
            )
            turn2_turns = ()
            for attempt in range(32):
                cand_t2 = (
                    TurnSpec("u_near", "premise", u_near_v, unit),
                    TurnSpec("p1_r1", "retract", base_r1 + 17 * attempt, unit, retraction_op="fix->", target_id="p1"),
                    TurnSpec("p1_r2", "retract", base_r2 + 29 * attempt, unit, retraction_op="sup->", target_id="p1_r1"),
                    TurnSpec("s2", "premise", s2, "scalar"),
                    TurnSpec("d5", "derive", arith_op="add", left_id="d4", right_id="d2"),
                    TurnSpec("d6", "derive", arith_op="mul", left_id="d5", right_id="s2"),
                )
                if not _all_multiturn_primary_stales_nondegenerate(turn1_turns, cand_t2, "d6", t2_unit):
                    continue
                tsnap = _eval_turns(turn1_turns + cand_t2)
                tb, td, _ = tsnap.active_values_base["d6"]
                tf = convert_from_base_fraction(tb, td, t2_unit)
                bf, wf = _compute_aliasing_signatures(turn1_turns, cand_t2, "d6", t2_unit)
                if (
                    bf is not None
                    and wf is not None
                    and _is_terminating_decimal(bf)
                    and _is_terminating_decimal(wf)
                    and not mixed_tolerance_match(bf, tf)[0]
                    and not mixed_tolerance_match(wf, tf)[0]
                ):
                    turn2_turns = cand_t2
                    both_frac, wrong_frac = bf, wf
                    break
            assert turn2_turns, f"Failed v0.3 aliasing non-degeneracy on {ep_id}"

        all_turns = turn1_turns + turn2_turns
        t1_snap = _eval_turns(turn1_turns)
        t1_base, t1_dim, _ = t1_snap.active_values_base["d4"]
        t1_frac = convert_from_base_fraction(t1_base, t1_dim, t1_unit)
        assert _is_terminating_decimal(t1_frac), f"Non-terminating Turn-1 decimal on {ep_id}: {t1_frac}"

        t2_snap = _eval_turns(all_turns)
        t2_base, t2_dim, _ = t2_snap.active_values_base["d6"]
        t2_frac = convert_from_base_fraction(t2_base, t2_dim, t2_unit)
        assert _is_terminating_decimal(t2_frac), f"Non-terminating Turn-2 decimal on {ep_id}: {t2_frac}"

        retracted_sorted = tuple(sorted(t2_snap.retracted_ids))
        pre_float: Optional[float] = None
        if retracted_sorted:
            pb, pdim, _ = _eval_turns([t for t in all_turns if t.kind != "retract"]).active_values_base["d6"]
            pre_float = float(convert_from_base_fraction(pb, pdim, t2_unit))

        stale_fracs = _compute_multiturn_stale_targets(
            turn1_turns, turn2_turns, "d6", t2_unit, t2_dim, t2_frac
        )
        stale_floats = tuple(float(sf) for sf in stale_fracs)

        t1_pres_schema = (
            f', "preserved": {{"{t1_preserved[0][0]}": {{"value": <number>, "unit": "{t1_preserved[0][1]}"}}}}'
            if t1_preserved else ""
        )
        t2_pres_schema = (
            f', "preserved": {{"{t2_preserved[0][0]}": {{"value": <number>, "unit": "{t2_preserved[0][1]}"}}}}'
            if t2_preserved else ""
        )

        turn1_prompt = (
            "[Context Note 1: Session ledger active; retain all premises and derived nodes in working memory.]\n"
            f"[Context Note 2: Telemetry nominal (seq={seq_ref}, lag={lag_ms} ms).]\n"
            f"Compute checkpoint 'd4' in '{t1_unit}'. Output ONLY standalone JSON: "
            f'{{"notes": "<step trace>", "step_id": "d4", "value": <number>, "unit": "{t1_unit}", '
            f'"retracted_ids": []{t1_pres_schema}}}. '
            f"{retracted_rule}\n" + "\n".join(_format_turn_lines("Turn 1", turn1_turns))
        )
        turn2_prompt = (
            "Follow-up update on the active session above (use the prior Turn-1 definitions from conversation "
            "history; recompute any invalidated Turn-1 dependencies and preserve unrelated nodes).\n"
            f"{t2_notes_extra}"
            f"Compute the final value of node 'd6' in unit '{t2_unit}'. "
            "Respond ONLY with a standalone JSON object containing keys "
            f'{{"notes": "<concise step-by-step calculation and retraction trace>", '
            f'"step_id": "d6", "value": <number>, "unit": "{t2_unit}", '
            f'"retracted_ids": [<sorted superseded target node IDs>]{t2_pres_schema}}}. '
            f"{retracted_rule}\n" + "\n".join(_format_turn_lines("Turn 2", turn2_turns))
        )

        ep = MultiTurnEpisode(
            episode_id=ep_id,
            condition=cond,
            category="normal" if cond == "A" else "perturbed",
            turn1_turns=turn1_turns,
            turn1_prompt=turn1_prompt,
            turn1_target_id="d4",
            turn1_expected_unit=t1_unit,
            turn1_expected_value_float=float(t1_frac),
            turn1_preserved_targets=t1_preserved,
            turn2_turns=turn2_turns,
            all_turns=all_turns,
            turn2_prompt=turn2_prompt,
            target_node_id="d6",
            expected_unit=t2_unit,
            expected_value_float=float(t2_frac),
            expected_retracted_ids=retracted_sorted,
            preserved_targets=t2_preserved,
            pre_retraction_value_float=pre_float,
            stale_target_values_float=stale_floats,
        )
        items.append(ValueAliasingEpisode(ep, both_frac, wrong_frac))

    return items


def build_v03_aliasing_prelock(
    prelock_path: Path = DEFAULT_PRELOCK_PATH,
    seed: int = P1_2_SEED,
    count: int = P1_2_COUNT,
) -> Dict[str, Any]:
    """Generate and write deterministic SHA-256 pre-registration lock for v0.3 aliasing slice."""
    items = generate_value_aliasing_v03_dataset(seed=seed, count=count)
    cond_counts: Dict[str, int] = {}
    records: List[Dict[str, Any]] = []
    for item in items:
        ep = item.episode
        cond_counts[ep.condition] = cond_counts.get(ep.condition, 0) + 1
        p1_turn = next(t for t in ep.turn1_turns if t.node_id == "p1")
        p4_turn = next(t for t in ep.turn1_turns if t.node_id == "p4")
        assert p4_turn.value == p1_turn.value and p4_turn.unit == p1_turn.unit
        if ep.condition == "E":
            u1_turn = next(t for t in ep.turn1_turns if t.node_id == "u1")
            assert u1_turn.value == p1_turn.value
        records.append({
            "episode_id": ep.episode_id,
            "condition": ep.condition,
            "p1": {"value": p1_turn.value, "unit": p1_turn.unit},
            "p4": {"value": p4_turn.value, "unit": p4_turn.unit},
            "turn1_expected_value_float": ep.turn1_expected_value_float,
            "turn1_expected_unit": ep.turn1_expected_unit,
            "expected_value_float": ep.expected_value_float,
            "expected_unit": ep.expected_unit,
            "expected_retracted_ids": list(ep.expected_retracted_ids),
            "aliased_both_retracted_float": item.aliased_both_retracted_float,
            "aliased_wrong_target_float": item.aliased_wrong_target_float,
        })

    canonical_bytes = "\n".join(json.dumps(r, sort_keys=True) for r in records).encode("utf-8")
    manifest = {
        "schema_version": "p1-v03-aliasing-prelock-v1",
        "track": "Track 1C-Alias",
        "seed": seed,
        "count": count,
        "condition_counts": cond_counts,
        "dataset_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(V3_MULTITURN_BASELINE_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "episodes": records,
    }
    prelock_path.parent.mkdir(parents=True, exist_ok=True)
    prelock_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def detect_value_aliasing_confusion(
    item: ValueAliasingEpisode,
    raw2: str,
    verdict_ok: bool,
) -> Tuple[bool, Optional[str]]:
    """Deterministically tag VALUE_ALIASING_CONFUSION alongside evaluate_multiturn_episode."""
    ep = item.episode
    if verdict_ok or ep.condition == "A":
        return (False, None)
    parsed = parse_step_envelope(raw2)
    if not parsed.ok or parsed.value_fraction is None or not parsed.unit:
        return (False, None)

    try:
        cand_base, cand_dim, _ = normalize_to_base_fraction(parsed.value_fraction, parsed.unit)
        _, truth_dim, _ = normalize_to_base_fraction(Fraction(1, 1), ep.expected_unit)
    except DimensionMismatchError:
        return (False, None)

    if cand_dim == truth_dim:
        cand_in_display = convert_from_base_fraction(cand_base, cand_dim, ep.expected_unit)
        if item.aliased_both_retracted_frac is not None and mixed_tolerance_match(
            cand_in_display, item.aliased_both_retracted_frac
        )[0]:
            return (True, "aliased_both_retracted")
        if item.aliased_wrong_target_frac is not None and mixed_tolerance_match(
            cand_in_display, item.aliased_wrong_target_frac
        )[0]:
            return (True, "aliased_wrong_target")

    if "p4" in parsed.retracted_ids:
        return (True, "aliased_p4_retracted_id")
    return (False, None)


def _summarize_verdicts(per_ep: List[Dict[str, Any]], clean_only: bool = False) -> Dict[str, Any]:
    subset = [e for e in per_ep if (not clean_only or e["t1_ok"])]
    n = len(subset)
    counts = {k: 0 for k in VERDICT_KEYS}
    for e in subset:
        v = e["verdict"]
        counts[v] = counts.get(v, 0) + 1
    p_cnt = counts["PASS"]
    s_cnt = counts["STALE_STATE_RETENTION"]
    return {
        "n": n,
        "pass": p_cnt,
        "fail": n - p_cnt,
        "pass_rate": round(p_cnt / n, 6) if n else 0.0,
        "pass_ci95": list(clopper_pearson_ci95(p_cnt, n)) if n else [0.0, 1.0],
        "stale_count": s_cnt,
        "stale_rate": round(s_cnt / n, 6) if n else 0.0,
        "stale_ci95": list(clopper_pearson_ci95(s_cnt, n)) if n else [0.0, 1.0],
        "verdicts": counts,
    }


def _tally_p1_1(episodes: List[MultiTurnEpisode], raw2_list: List[str], t1_ok_list: List[bool]) -> Dict[str, Any]:
    per_ep: List[Dict[str, Any]] = []
    cond_a_fail = 0
    cond_be_fail = 0
    for ep, r2, t1_ok in zip(episodes, raw2_list, t1_ok_list):
        ev = evaluate_multiturn_episode(
            ep.turn1_turns,
            ep.turn2_turns,
            r2,
            target_node_id=ep.target_node_id,
            expected_unit=ep.expected_unit,
            preserved_targets=dict(ep.preserved_targets) if ep.preserved_targets else None,
        )
        if not ev.ok:
            if ep.condition == "A":
                cond_a_fail += 1
            else:
                cond_be_fail += 1
        per_ep.append({
            "episode_id": ep.episode_id,
            "condition": ep.condition,
            "t1_ok": t1_ok,
            "t2_ok": ev.ok,
            "verdict": ev.verdict,
            "reason": ev.reason,
        })
    return {
        "itt": _summarize_verdicts(per_ep, clean_only=False),
        "clean_t1": _summarize_verdicts(per_ep, clean_only=True),
        "condition_a_fail": cond_a_fail,
        "condition_be_fail": cond_be_fail,
        "episodes": per_ep,
    }


def _tally_p1_2_branch(
    items: List[ValueAliasingEpisode],
    raw2_list: List[str],
    t1_ok_list: List[bool],
) -> Dict[str, Any]:
    per_ep: List[Dict[str, Any]] = []
    cond_a_fail = 0
    cond_be_fail = 0
    alias_conf_count = 0
    alias_mode_counts: Dict[str, int] = {}
    alias_by_verdict: Dict[str, int] = {}

    for item, r2, t1_ok in zip(items, raw2_list, t1_ok_list):
        ep = item.episode
        ev = evaluate_multiturn_episode(
            ep.turn1_turns,
            ep.turn2_turns,
            r2,
            target_node_id=ep.target_node_id,
            expected_unit=ep.expected_unit,
            preserved_targets=dict(ep.preserved_targets) if ep.preserved_targets else None,
        )
        alias_conf, alias_mode = detect_value_aliasing_confusion(item, r2, ev.ok)
        if alias_conf:
            alias_conf_count += 1
            if alias_mode:
                alias_mode_counts[alias_mode] = alias_mode_counts.get(alias_mode, 0) + 1
            alias_by_verdict[ev.verdict] = alias_by_verdict.get(ev.verdict, 0) + 1

        if not ev.ok:
            if ep.condition == "A":
                cond_a_fail += 1
            else:
                cond_be_fail += 1

        per_ep.append({
            "episode_id": ep.episode_id,
            "condition": ep.condition,
            "t1_ok": t1_ok,
            "t2_ok": ev.ok,
            "verdict": ev.verdict,
            "value_aliasing_confusion": alias_conf,
            "value_aliasing_mode": alias_mode,
            "reason": ev.reason,
        })

    return {
        "itt": _summarize_verdicts(per_ep, clean_only=False),
        "clean_t1": _summarize_verdicts(per_ep, clean_only=True),
        "condition_a_fail": cond_a_fail,
        "condition_be_fail": cond_be_fail,
        "value_aliasing_confusion_count": alias_conf_count,
        "value_aliasing_confusion_by_mode": alias_mode_counts,
        "value_aliasing_confusion_by_verdict": alias_by_verdict,
        "episodes": per_ep,
    }


def _paired_mcnemar_summary(
    c1_eps: List[Dict[str, Any]],
    c2_eps: List[Dict[str, Any]],
    perturbed_only: bool = False,
) -> Dict[str, Any]:
    b_c1_fail_c2_pass = 0
    c_c1_pass_c2_fail = 0
    both_pass = 0
    both_fail = 0
    for e1, e2 in zip(c1_eps, c2_eps):
        if perturbed_only and e1["condition"] == "A":
            continue
        ok1, ok2 = e1["t2_ok"], e2["t2_ok"]
        if not ok1 and ok2:
            b_c1_fail_c2_pass += 1
        elif ok1 and not ok2:
            c_c1_pass_c2_fail += 1
        elif ok1 and ok2:
            both_pass += 1
        else:
            both_fail += 1
    return {
        "n": both_pass + both_fail + b_c1_fail_c2_pass + c_c1_pass_c2_fail,
        "both_pass": both_pass,
        "both_fail": both_fail,
        "b_c1_fail_c2_pass": b_c1_fail_c2_pass,
        "c_c1_pass_c2_fail": c_c1_pass_c2_fail,
        "two_sided_exact_p": round(exact_two_sided_mcnemar_p(b_c1_fail_c2_pass, c_c1_pass_c2_fail), 8),
    }


def run_p1_1_for_model(
    model: str,
    live: bool,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    spend_path: Path = DEFAULT_SPEND_PATH,
) -> Dict[str, Any]:
    prior_spend = fe._assert_safety_guards(model, live=live, spend_path=spend_path)
    episodes = generate_multiturn_dataset(seed=P1_1_SEED, count=P1_1_COUNT)
    cond_counts: Dict[str, int] = {}
    for ep in episodes:
        cond_counts[ep.condition] = cond_counts.get(ep.condition, 0) + 1

    if not live:
        return {
            "mode": "dry_run",
            "model": model,
            "seed": P1_1_SEED,
            "count": len(episodes),
            "condition_counts": cond_counts,
            "prior_spend_usd": prior_spend,
        }

    api_key = os.environ["GEMINI_API_KEY"]
    tag = model.split("/")[-1]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    t1_reqs = [
        (ep.episode_id, {
            "systemInstruction": {"parts": [{"text": V3_MULTITURN_BASELINE_SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": ep.turn1_prompt}]}],
            "generationConfig": {"temperature": 0.0, "thinkingConfig": {"thinkingLevel": "MEDIUM"}},
        })
        for ep in episodes
    ]
    b_t1, raw1_list, mtok_t1, u_t1, cached_t1 = fe._run_or_load_stage(
        checkpoint_dir / f"p1_1-{tag}-t1.json",
        episodes,
        model,
        f"p1-1-arm1-t1-{tag}",
        t1_reqs,
        api_key,
    )
    t1_ok_list = [
        evaluate_retraction_episode(
            ep.turn1_turns,
            txt,
            target_node_id=ep.turn1_target_id,
            expected_unit=ep.turn1_expected_unit,
            preserved_targets=dict(ep.turn1_preserved_targets) if ep.turn1_preserved_targets else None,
        ).ok
        for ep, txt in zip(episodes, raw1_list)
    ]

    t2_reqs = [
        (ep.episode_id, {
            "systemInstruction": {"parts": [{"text": V3_MULTITURN_BASELINE_SYSTEM_PROMPT}]},
            "contents": [
                {"role": "user", "parts": [{"text": ep.turn1_prompt}]},
                {"role": "model", "parts": [{"text": r1}]},
                {"role": "user", "parts": [{"text": ep.turn2_prompt}]},
            ],
            "generationConfig": {"temperature": 0.0, "thinkingConfig": {"thinkingLevel": "MEDIUM"}},
        })
        for ep, r1 in zip(episodes, raw1_list)
    ]
    b_t2, raw2_list, mtok_t2, u_t2, cached_t2 = fe._run_or_load_stage(
        checkpoint_dir / f"p1_1-{tag}-t2.json",
        episodes,
        model,
        f"p1-1-arm1-t2-{tag}",
        t2_reqs,
        api_key,
    )

    usage_total = {
        k: u_t1[k] + u_t2[k]
        for k in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount")
    }
    max_tokens_hits = mtok_t1 + mtok_t2
    run_cost = fe._estimate_cost_usd(
        model, usage_total["promptTokenCount"], usage_total["candidatesTokenCount"], usage_total["thoughtsTokenCount"]
    )
    uncached_cost = sum(
        fe._estimate_cost_usd(model, u["promptTokenCount"], u["candidatesTokenCount"], u["thoughtsTokenCount"])
        for u, is_c in ((u_t1, cached_t1), (u_t2, cached_t2))
        if not is_c
    )
    new_batches = [b for b, is_c in ((b_t1, cached_t1), (b_t2, cached_t2)) if not is_c]
    cumulative_spend = fe._update_spend_ledger(spend_path, uncached_cost, new_batches)

    tally = _tally_p1_1(episodes, raw2_list, t1_ok_list)
    (checkpoint_dir / f"p1_1-{tag}-episodes.jsonl").write_text(
        "\n".join(
            json.dumps({
                "episode_id": ep.episode_id,
                "condition": ep.condition,
                "t1_ok": t1_ok,
                "raw1": r1,
                "raw2": r2,
                "verdict": per["verdict"],
            }, ensure_ascii=False)
            for ep, t1_ok, r1, r2, per in zip(episodes, t1_ok_list, raw1_list, raw2_list, tally["episodes"])
        ) + "\n",
        encoding="utf-8",
    )

    return {
        "mode": "live",
        "model": model,
        "seed": P1_1_SEED,
        "count": len(episodes),
        "condition_counts": cond_counts,
        "batches": {"t1": b_t1, "t2": b_t2},
        "finish_reason_max_tokens": max_tokens_hits,
        "usageMetadata": usage_total,
        "run_spend_usd": round(run_cost, 6),
        "uncached_spend_delta_usd": round(uncached_cost, 6),
        "cumulative_spend_usd": cumulative_spend,
        "t1_clean_count": sum(1 for x in t1_ok_list if x),
        "itt": tally["itt"],
        "clean_t1": tally["clean_t1"],
        "condition_a_fail": tally["condition_a_fail"],
        "condition_be_fail": tally["condition_be_fail"],
    }


def build_c2_v2_materialized_ledger_contents(
    ep: MultiTurnEpisode,
    raw_turn1_json: str,
) -> List[Dict[str, Any]]:
    """C2_v2: Materialized Turn-1 State Ledger with fixed t.kind in ('derive', 'derived') rules."""
    parsed = parse_step_envelope(raw_turn1_json)
    raw_obj = fe._parse_json_fenced(raw_turn1_json)
    val = raw_obj.get("value")
    if val is None and parsed.ok and parsed.value_fraction is not None:
        val = float(parsed.value_fraction)

    oracle = ExactRationalOracle()
    primitives: Dict[str, str] = {}
    derivations: List[str] = []
    for t in ep.turn1_turns:
        oracle.append_turn(t)
        if t.kind == "premise":
            primitives[t.node_id] = f"{t.value} {t.unit}"
        elif t.kind in ("derive", "derived"):
            derivations.append(f"{t.node_id} := {t.left_id} {t.arith_op} {t.right_id} (unit: {t.unit})")

    assert len(derivations) >= 4, f"Expected >= 4 Turn-1 derivation rules, got {len(derivations)} on {ep.episode_id}"
    snap = oracle.evaluate()
    stale_intermediates: Dict[str, str] = {}
    for nid, (base_val, dim, orig_unit) in snap.active_values_base.items():
        if nid.startswith("d") or nid.startswith("u_d"):
            stale_intermediates[nid] = f"base={float(base_val):.6g} (dim={dim}, unit={orig_unit})"

    ledger = {
        "ledger_version": "materialized_ledger_v2",
        "episode_id": ep.episode_id,
        "committed_turn1_checkpoint": {
            "step_id": ep.turn1_target_id,
            "value": val,
            "unit": raw_obj.get("unit") or (parsed.unit if parsed.ok else ep.turn1_expected_unit),
            "preserved": raw_obj.get("preserved") if "preserved" in raw_obj else (parsed.preserved if parsed.ok else None),
        },
        "turn1_primitive_bindings": primitives,
        "turn1_derivation_rules": derivations,
        "stale_cached_turn1_intermediates": stale_intermediates,
    }
    ledger_text = (
        "Materialized Turn-1 State Ledger (raw conversational Turn-1 prompt evicted):\n"
        + json.dumps(ledger, ensure_ascii=False)
    )
    return [
        {"role": "user", "parts": [{"text": ledger_text}]},
        {"role": "model", "parts": [{"text": "Acknowledged materialized state ledger."}]},
        {"role": "user", "parts": [{"text": ep.turn2_prompt}]},
    ]


def run_p1_2_for_model(
    model: str,
    live: bool,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    spend_path: Path = DEFAULT_SPEND_PATH,
    prelock_path: Path = DEFAULT_PRELOCK_PATH,
    recover_t1_batch: Optional[str] = None,
    v03_limit: int = P1_2_COUNT,
    c1_spend_gate_usd: float = 3.50,
) -> Dict[str, Any]:
    prior_spend = fe._assert_safety_guards(model, live=live, spend_path=spend_path)
    items = generate_value_aliasing_v03_dataset(seed=P1_2_SEED, count=P1_2_COUNT)[:v03_limit]
    episodes = [it.episode for it in items]
    cond_counts: Dict[str, int] = {}
    for ep in episodes:
        cond_counts[ep.condition] = cond_counts.get(ep.condition, 0) + 1

    if not live:
        prelock = build_v03_aliasing_prelock(prelock_path=prelock_path, seed=P1_2_SEED, count=P1_2_COUNT)
        dummy_r1 = json.dumps({
            "notes": "Full untruncated Turn-1 notes trace.",
            "step_id": episodes[0].turn1_target_id,
            "value": episodes[0].turn1_expected_value_float,
            "unit": episodes[0].turn1_expected_unit,
            "retracted_ids": [],
        })
        c1_sample = fe.build_c1_self_summary_contents(episodes[0], dummy_r1)
        c2_sample = build_c2_v2_materialized_ledger_contents(episodes[0], dummy_r1)
        return {
            "mode": "dry_run",
            "model": model,
            "seed": P1_2_SEED,
            "count": len(episodes),
            "condition_counts": cond_counts,
            "prelock_dataset_sha256": prelock["dataset_sha256"],
            "prior_spend_usd": prior_spend,
            "c1_roles": [m["role"] for m in c1_sample],
            "c2_roles": [m["role"] for m in c2_sample],
        }

    if not prelock_path.exists():
        build_v03_aliasing_prelock(prelock_path=prelock_path, seed=P1_2_SEED, count=P1_2_COUNT)

    api_key = os.environ["GEMINI_API_KEY"]
    tag = model.split("/")[-1]
    ckpt_prefix = f"p1_2-{tag}" if v03_limit == P1_2_COUNT else f"p1_2-{tag}-n{v03_limit}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    t1_reqs = [
        (ep.episode_id, {
            "systemInstruction": {"parts": [{"text": V3_MULTITURN_BASELINE_SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": ep.turn1_prompt}]}],
            "generationConfig": {"temperature": 0.0, "thinkingConfig": {"thinkingLevel": "MEDIUM"}},
        })
        for ep in episodes
    ]
    b_t1, raw1_list, mtok_t1, u_t1, cached_t1 = fe._run_or_load_stage(
        checkpoint_dir / f"{ckpt_prefix}-t1.json",
        episodes,
        model,
        f"{ckpt_prefix}-t1",
        t1_reqs,
        api_key,
        recover_batch_name=recover_t1_batch,
    )
    t1_ok_list = [
        evaluate_retraction_episode(
            ep.turn1_turns,
            txt,
            target_node_id=ep.turn1_target_id,
            expected_unit=ep.turn1_expected_unit,
            preserved_targets=dict(ep.turn1_preserved_targets) if ep.turn1_preserved_targets else None,
        ).ok
        for ep, txt in zip(episodes, raw1_list)
    ]

    # Run C2_v2 first (cost-safe order for gated N=32 runs; reuses cached t1)
    c2_reqs = [
        (ep.episode_id, {
            "systemInstruction": {"parts": [{"text": V3_MULTITURN_BASELINE_SYSTEM_PROMPT}]},
            "contents": build_c2_v2_materialized_ledger_contents(ep, r1),
            "generationConfig": {"temperature": 0.0, "thinkingConfig": {"thinkingLevel": "MEDIUM"}},
        })
        for ep, r1 in zip(episodes, raw1_list)
    ]
    b_c2, c2_raw2, mtok_c2, u_c2, cached_c2 = fe._run_or_load_stage(
        checkpoint_dir / f"{ckpt_prefix}-c2_v2.json",
        episodes,
        model,
        f"{ckpt_prefix}-c2-v2",
        c2_reqs,
        api_key,
    )

    pre_c1_uncached = sum(
        fe._estimate_cost_usd(model, u["promptTokenCount"], u["candidatesTokenCount"], u["thoughtsTokenCount"])
        for u, is_c in ((u_t1, cached_t1), (u_c2, cached_c2))
        if not is_c
    )
    pre_c1_batches = [b for b, is_c in ((b_t1, cached_t1), (b_c2, cached_c2)) if not is_c]
    cumulative_spend = fe._update_spend_ledger(spend_path, pre_c1_uncached, pre_c1_batches)

    c1_ckpt = checkpoint_dir / f"{ckpt_prefix}-c1.json"
    c1_skipped_budget_gate = False
    if not c1_ckpt.exists() and cumulative_spend > c1_spend_gate_usd:
        c1_skipped_budget_gate = True
        b_c1, c1_raw2, mtok_c1, u_c1, cached_c1 = (
            "",
            [],
            0,
            {"promptTokenCount": 0, "candidatesTokenCount": 0, "thoughtsTokenCount": 0},
            True,
        )
    else:
        c1_reqs = [
            (ep.episode_id, {
                "systemInstruction": {"parts": [{"text": V3_MULTITURN_BASELINE_SYSTEM_PROMPT}]},
                "contents": fe.build_c1_self_summary_contents(ep, r1),
                "generationConfig": {"temperature": 0.0, "thinkingConfig": {"thinkingLevel": "MEDIUM"}},
            })
            for ep, r1 in zip(episodes, raw1_list)
        ]
        b_c1, c1_raw2, mtok_c1, u_c1, cached_c1 = fe._run_or_load_stage(
            c1_ckpt,
            episodes,
            model,
            f"{ckpt_prefix}-c1",
            c1_reqs,
            api_key,
        )
        if not cached_c1:
            c1_cost = fe._estimate_cost_usd(
                model, u_c1["promptTokenCount"], u_c1["candidatesTokenCount"], u_c1["thoughtsTokenCount"]
            )
            cumulative_spend = fe._update_spend_ledger(spend_path, c1_cost, [b_c1])

    usage_total = {
        k: u_t1[k] + u_c1[k] + u_c2[k]
        for k in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount")
    }
    max_tokens_hits = mtok_t1 + mtok_c1 + mtok_c2
    run_cost = fe._estimate_cost_usd(
        model, usage_total["promptTokenCount"], usage_total["candidatesTokenCount"], usage_total["thoughtsTokenCount"]
    )
    uncached_cost = sum(
        fe._estimate_cost_usd(model, u["promptTokenCount"], u["candidatesTokenCount"], u["thoughtsTokenCount"])
        for u, is_c in ((u_t1, cached_t1), (u_c1, cached_c1), (u_c2, cached_c2))
        if not is_c
    )

    c2_tally = _tally_p1_2_branch(items, c2_raw2, t1_ok_list)
    out: Dict[str, Any] = {
        "mode": "live",
        "model": model,
        "seed": P1_2_SEED,
        "count": len(episodes),
        "condition_counts": cond_counts,
        "batches": {"t1": b_t1, "c1": b_c1 or None, "c2_v2": b_c2},
        "finish_reason_max_tokens": max_tokens_hits,
        "finish_reason_max_tokens_by_stage": {"t1": mtok_t1, "c1": mtok_c1, "c2_v2": mtok_c2},
        "usageMetadata": usage_total,
        "run_spend_usd": round(run_cost, 6),
        "uncached_spend_delta_usd": round(uncached_cost, 6),
        "cumulative_spend_usd": cumulative_spend,
        "t1_clean_count": sum(1 for x in t1_ok_list if x),
        "c1_skipped_budget_gate": c1_skipped_budget_gate,
        "C2_v2_materialized_ledger_eviction": {k: v for k, v in c2_tally.items() if k != "episodes"},
    }

    if not c1_skipped_budget_gate:
        c1_tally = _tally_p1_2_branch(items, c1_raw2, t1_ok_list)
        mcnemar_itt = _paired_mcnemar_summary(c1_tally["episodes"], c2_tally["episodes"], perturbed_only=False)
        mcnemar_be = _paired_mcnemar_summary(c1_tally["episodes"], c2_tally["episodes"], perturbed_only=True)
        out["C1_self_summary_eviction"] = {k: v for k, v in c1_tally.items() if k != "episodes"}
        out[f"paired_mcnemar_c2_v2_vs_c1_itt_n{len(episodes)}"] = mcnemar_itt
        out[f"paired_mcnemar_c2_v2_vs_c1_perturbed_n{len(episodes) - cond_counts.get('A', 0)}"] = mcnemar_be

        (checkpoint_dir / f"{ckpt_prefix}-t1-c1-c2_v2.jsonl").write_text(
            "\n".join(
                json.dumps({
                    "episode_id": ep.episode_id,
                    "condition": ep.condition,
                    "t1_ok": t1_ok,
                    "raw1": r1,
                    "c1_raw2": c1r2,
                    "c2_v2_raw2": c2r2,
                    "c1_verdict": e1["verdict"],
                    "c1_value_aliasing_confusion": e1["value_aliasing_confusion"],
                    "c1_value_aliasing_mode": e1["value_aliasing_mode"],
                    "c2_v2_verdict": e2["verdict"],
                    "c2_v2_value_aliasing_confusion": e2["value_aliasing_confusion"],
                    "c2_v2_value_aliasing_mode": e2["value_aliasing_mode"],
                }, ensure_ascii=False)
                for ep, t1_ok, r1, c1r2, c2r2, e1, e2 in zip(
                    episodes, t1_ok_list, raw1_list, c1_raw2, c2_raw2, c1_tally["episodes"], c2_tally["episodes"]
                )
            ) + "\n",
            encoding="utf-8",
        )

    c2_v1_path = checkpoint_dir / f"{ckpt_prefix}-c2-empty-rules-v1.json"
    if v03_limit == P1_2_COUNT and c2_v1_path.exists():
        c2_v1_cached = json.loads(c2_v1_path.read_text(encoding="utf-8"))
        c2_v1_tally = _tally_p1_2_branch(items, c2_v1_cached["raw_texts"], t1_ok_list)
        out["C2_v1_values_only_empty_rules"] = {
            "batch_name": c2_v1_cached["batch_name"],
            "finish_reason_max_tokens": c2_v1_cached["max_tokens_hits"],
            "usageMetadata": c2_v1_cached["usage"],
            **{k: v for k, v in c2_v1_tally.items() if k != "episodes"},
        }

    return out


def _merge_manifest(out_path: Path, model: str, result: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    existing: Dict[str, Any] = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    models_map = dict(existing.get("models", {}))
    models_map[model] = result
    payload = {
        **meta,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "models": models_map,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _poll_batch_v1beta_long(op_name: str, api_key: str) -> List[Dict[str, Any]]:
    import time as _time
    import urllib.request as _urllib_request

    poll_url = f"https://generativelanguage.googleapis.com/v1beta/{op_name}"
    for _ in range(540):
        poll_req = _urllib_request.Request(poll_url, headers={"x-goog-api-key": api_key}, method="GET")
        with _urllib_request.urlopen(poll_req, timeout=60) as presp:
            pdata = json.loads(presp.read().decode("utf-8"))
        if pdata.get("done"):
            return pdata.get("response", {}).get("inlinedResponses", {}).get("inlinedResponses", [])
        _time.sleep(10)
    raise TimeoutError(f"Batch operation timed out after 90m: {op_name}")


fe._poll_batch_v1beta = _poll_batch_v1beta_long


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="P1 Frontier Controls Runner (P1.1 Arm 1 + P1.2 v0.3 Aliasing)")
    parser.add_argument("--models", nargs="+", default=["models/gemini-4-carbon"], choices=ALLOWED_MODELS)
    parser.add_argument("--part", choices=("all", "p1_1", "p1_2"), default="all")
    parser.add_argument("--live", action="store_true", help="Run live Paid Batch API")
    parser.add_argument("--dry-run", action="store_true", help="Run offline dry-run + prelock generation")
    parser.add_argument("--recover-p1-2-t1-batch", default=None, help="Existing P1.2 Turn-1 batch name to poll")
    parser.add_argument("--v03-limit", type=int, default=P1_2_COUNT, help="Stratified v0.3 episode limit (e.g. 64 or 32)")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--spend-path", type=Path, default=DEFAULT_SPEND_PATH)
    parser.add_argument("--prelock-path", type=Path, default=DEFAULT_PRELOCK_PATH)
    parser.add_argument("--p1-1-out", type=Path, default=DEFAULT_P1_1_OUT)
    parser.add_argument("--p1-2-out", type=Path, default=DEFAULT_P1_2_OUT)
    args = parser.parse_args(argv)

    live = bool(args.live and not args.dry_run)
    report: Dict[str, Any] = {"live": live, "runs": {}}

    for model in args.models:
        m_rep: Dict[str, Any] = {}
        if args.part in ("all", "p1_1"):
            r1 = run_p1_1_for_model(model, live=live, checkpoint_dir=args.checkpoint_dir, spend_path=args.spend_path)
            m_rep["p1_1"] = r1
            if live:
                _merge_manifest(
                    args.p1_1_out,
                    model,
                    r1,
                    {
                        "schema_version": "p1-frontier-arm1-control-v1",
                        "harness": "standard_2_turn_arm1",
                        "seed": P1_1_SEED,
                        "count": P1_1_COUNT,
                    },
                )
        if args.part in ("all", "p1_2"):
            r2 = run_p1_2_for_model(
                model,
                live=live,
                checkpoint_dir=args.checkpoint_dir,
                spend_path=args.spend_path,
                prelock_path=args.prelock_path,
                recover_t1_batch=args.recover_p1_2_t1_batch,
                v03_limit=args.v03_limit,
            )
            m_rep["p1_2"] = r2
            if live:
                _merge_manifest(
                    args.p1_2_out,
                    model,
                    r2,
                    {
                        "schema_version": "p1-frontier-v03-aliasing-v1",
                        "track": "Track 1C-Alias",
                        "seed": P1_2_SEED,
                        "count": P1_2_COUNT,
                        "prelock_path": str(args.prelock_path.relative_to(REPO_ROOT)),
                        "v0_2_derived_typo_disclosure": (
                            "scripts/run_frontier_eviction_v02.py originally checked t.kind == 'derived' "
                            "instead of 'derive', leaving turn1_derivation_rules empty in v0.2 C2 (and initial "
                            "carbon P1.2 C2_v1). Fixed to t.kind in ('derive', 'derived') in C2_v2; "
                            "manifests/frontier-eviction-v02-*.json remain untouched."
                        ),
                    },
                )
        report["runs"][model] = m_rep

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
