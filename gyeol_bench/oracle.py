"""Clean-room pure Python rational oracle for quantitative premise retraction and SI unit cascades.

Licensed under the Apache License, Version 2.0.
Contains zero lines and zero imports from any private engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
import math
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

RETRACTION_OPERATORS: frozenset[str] = frozenset({"fix->", "err->", "sup->"})

# Exact rational unit table: unit_token -> (dimension_family, factor_to_base_unit, base_unit_token)
UNIT_TABLE: Dict[str, Tuple[str, Fraction, str]] = {
    # Mass (base: g)
    "ug": ("mass", Fraction(1, 1_000_000), "g"),
    "mcg": ("mass", Fraction(1, 1_000_000), "g"),
    "mg": ("mass", Fraction(1, 1_000), "g"),
    "g": ("mass", Fraction(1, 1), "g"),
    "kg": ("mass", Fraction(1_000, 1), "g"),
    # Length (base: m)
    "mm": ("length", Fraction(1, 1_000), "m"),
    "cm": ("length", Fraction(1, 100), "m"),
    "m": ("length", Fraction(1, 1), "m"),
    "km": ("length", Fraction(1_000, 1), "m"),
    # Volume (base: L)
    "uL": ("volume", Fraction(1, 1_000_000), "L"),
    "mL": ("volume", Fraction(1, 1_000), "L"),
    "L": ("volume", Fraction(1, 1), "L"),
    # Time (base: s)
    "ms": ("time", Fraction(1, 1_000), "s"),
    "s": ("time", Fraction(1, 1), "s"),
    "min": ("time", Fraction(60, 1), "s"),
    "h": ("time", Fraction(3_600, 1), "s"),
    "hr": ("time", Fraction(3_600, 1), "s"),
    # Financial USD scale (base: USD) — distinct from length 'm'
    "USD": ("currency_usd", Fraction(1, 1), "USD"),
    "K_USD": ("currency_usd", Fraction(1_000, 1), "USD"),
    "$K": ("currency_usd", Fraction(1_000, 1), "USD"),
    "M_USD": ("currency_usd", Fraction(1_000_000, 1), "USD"),
    "$M": ("currency_usd", Fraction(1_000_000, 1), "USD"),
    "B_USD": ("currency_usd", Fraction(1_000_000_000, 1), "USD"),
    "$B": ("currency_usd", Fraction(1_000_000_000, 1), "USD"),
    # Dimensionless ratios (base: scalar)
    "scalar": ("dimensionless", Fraction(1, 1), "scalar"),
    "1": ("dimensionless", Fraction(1, 1), "scalar"),
    "%": ("dimensionless", Fraction(1, 100), "scalar"),
    "pct": ("dimensionless", Fraction(1, 100), "scalar"),
    "bps": ("dimensionless", Fraction(1, 10_000), "scalar"),
}

NUMERIC_STR_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:/[+-]?[1-9]\d*)?$")


class DimensionMismatchError(ValueError):
    """Raised when incompatible SI or financial dimensions are combined."""


def to_exact_fraction(raw: Any) -> Fraction:
    """Convert int, finite float, Decimal, Fraction, or decimal/rational string to exact Fraction."""
    if isinstance(raw, bool) or raw is None:
        raise ValueError("Boolean or None is not a valid numeric value.")
    if isinstance(raw, Fraction):
        return raw
    if isinstance(raw, int):
        return Fraction(raw, 1)
    if isinstance(raw, Decimal):
        if not raw.is_finite():
            raise ValueError("Non-finite Decimal value rejected.")
        return Fraction(raw)
    if isinstance(raw, float):
        if not math.isfinite(raw):
            raise ValueError("Non-finite float (NaN/Inf) rejected.")
        return Fraction(str(raw))
    if isinstance(raw, str):
        text = raw.strip()
        if not text or not NUMERIC_STR_RE.match(text):
            raise ValueError(f"Invalid numeric string: {raw!r}")
        if "/" in text:
            num_s, den_s = text.split("/", 1)
            return Fraction(int(num_s), int(den_s))
        try:
            dec = Decimal(text)
        except InvalidOperation as exc:
            raise ValueError(f"Malformed decimal string: {raw!r}") from exc
        if not dec.is_finite():
            raise ValueError("Non-finite string numeric rejected.")
        return Fraction(dec)
    raise ValueError(f"Unsupported numeric type: {type(raw).__name__}")


def normalize_to_base_fraction(value: Any, unit: str) -> Tuple[Fraction, str, str]:
    """Return (value_in_base_unit, dimension_family, base_unit_token)."""
    if not isinstance(unit, str) or unit not in UNIT_TABLE:
        raise DimensionMismatchError(f"Unknown or unsupported unit token: {unit!r}")
    dim, factor, base_unit = UNIT_TABLE[unit]
    exact_val = to_exact_fraction(value)
    return exact_val * factor, dim, base_unit


def convert_from_base_fraction(base_value: Fraction, dimension: str, target_unit: str) -> Fraction:
    """Convert a base-unit Fraction into target_unit after checking dimensional compatibility."""
    if target_unit not in UNIT_TABLE:
        raise DimensionMismatchError(f"Unknown target unit token: {target_unit!r}")
    target_dim, factor, _ = UNIT_TABLE[target_unit]
    if target_dim != dimension:
        raise DimensionMismatchError(
            f"Dimension mismatch: cannot convert {dimension} into {target_unit} ({target_dim})."
        )
    return base_value / factor


def mixed_tolerance_match(
    candidate: Any,
    truth: Any,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-5,
) -> Tuple[bool, Fraction, Fraction]:
    """Evaluate |a - b| <= abs_tol + rel_tol * max(|a|, |b|) in exact Fraction arithmetic."""
    try:
        cand_f = to_exact_fraction(candidate)
        truth_f = to_exact_fraction(truth)
        abs_tol_f = to_exact_fraction(abs_tol)
        rel_tol_f = to_exact_fraction(rel_tol)
    except ValueError:
        return False, Fraction(0, 1), Fraction(0, 1)

    if abs_tol_f < 0 or rel_tol_f < 0:
        raise ValueError("Tolerance parameters must be non-negative.")

    diff = abs(cand_f - truth_f)
    scale = max(abs(cand_f), abs(truth_f))
    allowed = abs_tol_f + rel_tol_f * scale
    return diff <= allowed, diff, allowed


@dataclass(frozen=True)
class EnvelopeParseResult:
    ok: bool
    status: str
    step_id: str = ""
    value_fraction: Optional[Fraction] = None
    unit: str = ""
    retracted_ids: Tuple[str, ...] = ()
    reason: str = ""


def parse_step_envelope(raw_text: str) -> EnvelopeParseResult:
    """Strict JSON step envelope gate; rejects prose, missing keys, booleans, and NaN/Inf."""
    if not isinstance(raw_text, str):
        return EnvelopeParseResult(ok=False, status="PARSE_REJECT", reason="Response is not a string.")
    stripped = raw_text.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return EnvelopeParseResult(
            ok=False,
            status="PARSE_REJECT",
            reason="Response must be a standalone JSON object without free-form prose.",
        )

    def _reject_constant(token: str) -> Any:
        raise ValueError(f"Forbidden non-finite JSON constant: {token}")

    try:
        payload = json.loads(stripped, parse_constant=_reject_constant)
    except Exception as exc:
        return EnvelopeParseResult(ok=False, status="PARSE_REJECT", reason=f"Invalid JSON: {exc}")

    if not isinstance(payload, dict):
        return EnvelopeParseResult(ok=False, status="PARSE_REJECT", reason="Envelope root must be an object.")

    required = {"step_id", "value", "unit", "retracted_ids"}
    missing = required - set(payload.keys())
    if missing:
        return EnvelopeParseResult(
            ok=False,
            status="PARSE_REJECT",
            reason=f"Missing required keys: {sorted(missing)}",
        )
    extra = set(payload.keys()) - (required | {"notes"})
    if extra:
        return EnvelopeParseResult(
            ok=False,
            status="PARSE_REJECT",
            reason=f"Unexpected envelope keys: {sorted(extra)}",
        )

    step_id = payload["step_id"]
    if not isinstance(step_id, str) or not step_id.strip():
        return EnvelopeParseResult(ok=False, status="PARSE_REJECT", reason="Invalid step_id.")

    unit = payload["unit"]
    if not isinstance(unit, str) or not unit.strip():
        return EnvelopeParseResult(ok=False, status="PARSE_REJECT", reason="Invalid unit field.")

    retracted_raw = payload["retracted_ids"]
    if not isinstance(retracted_raw, list) or any(
        not isinstance(item, str) or not item.strip() for item in retracted_raw
    ):
        return EnvelopeParseResult(ok=False, status="PARSE_REJECT", reason="retracted_ids must be list[str].")

    try:
        val_frac = to_exact_fraction(payload["value"])
    except ValueError as exc:
        return EnvelopeParseResult(ok=False, status="PARSE_REJECT", reason=str(exc))

    return EnvelopeParseResult(
        ok=True,
        status="OK",
        step_id=step_id.strip(),
        value_fraction=val_frac,
        unit=unit.strip(),
        retracted_ids=tuple(sorted(set(item.strip() for item in retracted_raw))),
    )


@dataclass(frozen=True)
class TurnSpec:
    node_id: str
    kind: str  # "premise" | "retract" | "derive"
    value: Optional[Any] = None
    unit: str = "scalar"
    retraction_op: Optional[str] = None  # "fix->" | "err->" | "sup->"
    target_id: Optional[str] = None
    arith_op: Optional[str] = None  # "add" | "sub" | "mul" | "div"
    left_id: Optional[str] = None
    right_id: Optional[str] = None


@dataclass
class OracleStateSnapshot:
    active_values_base: Dict[str, Tuple[Fraction, str, str]]
    retracted_ids: Set[str]
    replacement_map: Dict[str, str]


class ExactRationalOracle:
    """Evaluates multi-turn premise graphs with non-destructive retraction and cascading re-propagation."""

    def __init__(self) -> None:
        self._turns: List[TurnSpec] = []

    def append_turn(self, turn: TurnSpec) -> None:
        if turn.kind not in {"premise", "retract", "derive"}:
            raise ValueError(f"Unsupported turn kind: {turn.kind!r}")
        if turn.kind == "retract" and turn.retraction_op not in RETRACTION_OPERATORS:
            raise ValueError(f"Invalid retraction operator: {turn.retraction_op!r}")
        self._turns.append(turn)

    def evaluate(self) -> OracleStateSnapshot:
        definitions: Dict[str, TurnSpec] = {}
        order: List[str] = []
        retracted: Set[str] = set()
        replacements: Dict[str, str] = {}

        for turn in self._turns:
            definitions[turn.node_id] = turn
            if turn.node_id not in order:
                order.append(turn.node_id)
            if turn.kind == "retract":
                if not turn.target_id or turn.target_id not in definitions:
                    raise ValueError(f"Retraction target {turn.target_id!r} does not exist.")
                retracted.add(turn.target_id)
                if turn.value is not None:
                    replacements[turn.target_id] = turn.node_id

        def resolve_latest(node_id: str) -> str:
            seen: Set[str] = set()
            cur = node_id
            while cur in replacements:
                if cur in seen:
                    raise ValueError(f"Cycle detected in retraction chain at {cur!r}")
                seen.add(cur)
                cur = replacements[cur]
            return cur

        memo: Dict[str, Tuple[Fraction, str, str]] = {}
        visiting: Set[str] = set()

        def compute_node(node_id: str) -> Tuple[Fraction, str, str]:
            effective_id = resolve_latest(node_id)
            if effective_id in memo:
                return memo[effective_id]
            if effective_id in retracted and effective_id not in replacements:
                raise ValueError(f"Node {effective_id!r} was retracted without replacement.")
            if effective_id in visiting:
                raise ValueError(f"Dependency cycle detected at {effective_id!r}")

            visiting.add(effective_id)
            spec = definitions[effective_id]
            if spec.kind in {"premise", "retract"}:
                res = normalize_to_base_fraction(spec.value, spec.unit)
            elif spec.kind == "derive":
                if not spec.left_id or not spec.right_id:
                    raise ValueError(f"Derived node {effective_id!r} missing operands.")
                lv, ldim, lbase = compute_node(spec.left_id)
                rv, rdim, rbase = compute_node(spec.right_id)
                op = spec.arith_op
                if op in {"add", "sub"}:
                    if ldim != rdim:
                        raise DimensionMismatchError(
                            f"Cannot {op} incompatible dimensions: {ldim} vs {rdim}"
                        )
                    val = lv + rv if op == "add" else lv - rv
                    res = (val, ldim, lbase)
                elif op == "mul":
                    if ldim == "dimensionless":
                        res = (lv * rv, rdim, rbase)
                    elif rdim == "dimensionless":
                        res = (lv * rv, ldim, lbase)
                    else:
                        raise DimensionMismatchError(
                            f"Unsupported compound product dimension: {ldim} * {rdim}"
                        )
                elif op == "div":
                    if rv == 0:
                        raise ZeroDivisionError(f"Division by zero at {effective_id!r}")
                    if ldim == rdim:
                        res = (lv / rv, "dimensionless", "scalar")
                    elif rdim == "dimensionless":
                        res = (lv / rv, ldim, lbase)
                    else:
                        raise DimensionMismatchError(
                            f"Unsupported compound quotient dimension: {ldim} / {rdim}"
                        )
                else:
                    raise ValueError(f"Unsupported arithmetic operator: {op!r}")
            else:
                raise ValueError(f"Unexpected turn kind: {spec.kind!r}")

            visiting.remove(effective_id)
            memo[effective_id] = res
            return res

        active_values: Dict[str, Tuple[Fraction, str, str]] = {}
        for nid in order:
            if nid in retracted and nid not in replacements:
                continue
            latest_id = resolve_latest(nid)
            if latest_id in retracted and latest_id not in replacements:
                continue
            active_values[nid] = compute_node(nid)

        return OracleStateSnapshot(
            active_values_base=active_values,
            retracted_ids=retracted,
            replacement_map=replacements,
        )


@dataclass(frozen=True)
class OracleEvaluationResult:
    ok: bool
    verdict: str
    truth_base_fraction: Optional[Fraction]
    truth_target_fraction: Optional[Fraction]
    truth_dimension: str
    candidate_base_fraction: Optional[Fraction]
    abs_diff: Optional[Fraction]
    allowed_tol: Optional[Fraction]
    expected_retracted_ids: Tuple[str, ...]
    observed_retracted_ids: Tuple[str, ...]
    reason: str = ""


def evaluate_retraction_episode(
    turns: Sequence[TurnSpec],
    raw_model_output: str,
    target_node_id: str,
    expected_unit: Optional[str] = None,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-5,
) -> OracleEvaluationResult:
    """End-to-end deterministic scoring of a multi-turn episode against a JSON step response."""
    oracle = ExactRationalOracle()
    for t in turns:
        oracle.append_turn(t)
    snap = oracle.evaluate()
    if target_node_id not in snap.active_values_base:
        raise ValueError(f"Target node {target_node_id!r} not active in oracle state.")

    truth_base, truth_dim, truth_base_unit = snap.active_values_base[target_node_id]
    display_unit = expected_unit or truth_base_unit
    truth_target = convert_from_base_fraction(truth_base, truth_dim, display_unit)
    expected_retracted = tuple(sorted(snap.retracted_ids))

    env = parse_step_envelope(raw_model_output)
    if not env.ok or env.value_fraction is None:
        return OracleEvaluationResult(
            ok=False,
            verdict="PARSE_REJECT",
            truth_base_fraction=truth_base,
            truth_target_fraction=truth_target,
            truth_dimension=truth_dim,
            candidate_base_fraction=None,
            abs_diff=None,
            allowed_tol=None,
            expected_retracted_ids=expected_retracted,
            observed_retracted_ids=(),
            reason=env.reason,
        )

    try:
        cand_base, cand_dim, _ = normalize_to_base_fraction(env.value_fraction, env.unit)
    except DimensionMismatchError as exc:
        return OracleEvaluationResult(
            ok=False,
            verdict="UNIT_MISMATCH",
            truth_base_fraction=truth_base,
            truth_target_fraction=truth_target,
            truth_dimension=truth_dim,
            candidate_base_fraction=None,
            abs_diff=None,
            allowed_tol=None,
            expected_retracted_ids=expected_retracted,
            observed_retracted_ids=env.retracted_ids,
            reason=str(exc),
        )

    if cand_dim != truth_dim:
        return OracleEvaluationResult(
            ok=False,
            verdict="UNIT_MISMATCH",
            truth_base_fraction=truth_base,
            truth_target_fraction=truth_target,
            truth_dimension=truth_dim,
            candidate_base_fraction=cand_base,
            abs_diff=None,
            allowed_tol=None,
            expected_retracted_ids=expected_retracted,
            observed_retracted_ids=env.retracted_ids,
            reason=f"Dimension mismatch: candidate {cand_dim} vs truth {truth_dim}.",
        )

    if env.retracted_ids != expected_retracted:
        return OracleEvaluationResult(
            ok=False,
            verdict="RETRACTION_SET_MISMATCH",
            truth_base_fraction=truth_base,
            truth_target_fraction=truth_target,
            truth_dimension=truth_dim,
            candidate_base_fraction=cand_base,
            abs_diff=None,
            allowed_tol=None,
            expected_retracted_ids=expected_retracted,
            observed_retracted_ids=env.retracted_ids,
            reason=f"Expected retracted_ids={list(expected_retracted)}, got {list(env.retracted_ids)}.",
        )

    cand_in_display = convert_from_base_fraction(cand_base, cand_dim, display_unit)
    matched, diff, allowed = mixed_tolerance_match(
        cand_in_display,
        truth_target,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    return OracleEvaluationResult(
        ok=matched,
        verdict="PASS" if matched else "VALUE_MISMATCH",
        truth_base_fraction=truth_base,
        truth_target_fraction=truth_target,
        truth_dimension=truth_dim,
        candidate_base_fraction=cand_base,
        abs_diff=diff,
        allowed_tol=allowed,
        expected_retracted_ids=expected_retracted,
        observed_retracted_ids=env.retracted_ids,
        reason="" if matched else f"Difference {float(diff):.6g} exceeds tolerance {float(allowed):.6g}.",
    )


@dataclass(frozen=True)
class RetractionEpisode:
    episode_id: str
    category: str  # "retraction" | "unit_scale" | "normal"
    prompt: str
    turns: Tuple[TurnSpec, ...]
    target_node_id: str
    expected_unit: str
    expected_value_float: float
    expected_retracted_ids: Tuple[str, ...] = field(default_factory=tuple)


def generate_procedural_dataset(seed: int = 20261122, count: int = 800) -> List[RetractionEpisode]:
    """Deterministically generate balanced retraction, unit-scale, and normal control episodes."""
    if count <= 0:
        raise ValueError("count must be positive.")
    rng = random.Random(seed)
    n_retract = count // 4
    n_unit = count // 4
    n_normal = count - n_retract - n_unit

    categories = (
        ["retraction"] * n_retract
        + ["unit_scale"] * n_unit
        + ["normal"] * n_normal
    )

    episodes: List[RetractionEpisode] = []
    retraction_ops = ["fix->", "err->", "sup->"]

    for idx, cat in enumerate(categories):
        ep_id = f"gb-{idx + 1:04d}"
        if cat == "retraction":
            a_init = rng.randint(10, 90)
            a_fix = rng.randint(100, 250)
            b_val = rng.randint(5, 60)
            rate_pct = rng.choice([10, 15, 20, 25])
            op_tok = retraction_ops[idx % len(retraction_ops)]
            turns = (
                TurnSpec(node_id="p1", kind="premise", value=a_init, unit="M_USD"),
                TurnSpec(node_id="p2", kind="premise", value=b_val, unit="M_USD"),
                TurnSpec(node_id="p3", kind="premise", value=rate_pct, unit="%"),
                TurnSpec(node_id="d1", kind="derive", arith_op="add", left_id="p1", right_id="p2"),
                TurnSpec(node_id="d2", kind="derive", arith_op="mul", left_id="d1", right_id="p3"),
                TurnSpec(
                    node_id="p1_r1",
                    kind="retract",
                    retraction_op=op_tok,
                    target_id="p1",
                    value=a_fix,
                    unit="M_USD",
                ),
            )
            expected_unit = "M_USD"
        elif cat == "unit_scale":
            mg_val = rng.randint(250, 1750)
            g_val = rng.randint(1, 6)
            mult = rng.randint(2, 5)
            turns = (
                TurnSpec(node_id="p1", kind="premise", value=mg_val, unit="mg"),
                TurnSpec(node_id="p2", kind="premise", value=g_val, unit="g"),
                TurnSpec(node_id="p3", kind="premise", value=mult, unit="scalar"),
                TurnSpec(node_id="d1", kind="derive", arith_op="add", left_id="p1", right_id="p2"),
                TurnSpec(node_id="d2", kind="derive", arith_op="mul", left_id="d1", right_id="p3"),
            )
            expected_unit = "g"
        else:
            v1 = rng.randint(12, 80)
            v2 = rng.randint(8, 40)
            v3 = rng.randint(2, 6)
            turns = (
                TurnSpec(node_id="p1", kind="premise", value=v1, unit="km"),
                TurnSpec(node_id="p2", kind="premise", value=v2, unit="km"),
                TurnSpec(node_id="p3", kind="premise", value=v3, unit="scalar"),
                TurnSpec(node_id="d1", kind="derive", arith_op="add", left_id="p1", right_id="p2"),
                TurnSpec(node_id="d2", kind="derive", arith_op="mul", left_id="d1", right_id="p3"),
            )
            expected_unit = "km"

        oracle = ExactRationalOracle()
        for t in turns:
            oracle.append_turn(t)
        snap = oracle.evaluate()
        base_val, dim, _ = snap.active_values_base["d2"]
        target_frac = convert_from_base_fraction(base_val, dim, expected_unit)
        retracted_sorted = tuple(sorted(snap.retracted_ids))

        turn_lines = []
        for i, t in enumerate(turns, start=1):
            if t.kind == "premise":
                turn_lines.append(f"Turn {i} [{t.node_id}]: value = {t.value} {t.unit}")
            elif t.kind == "derive":
                turn_lines.append(f"Turn {i} [{t.node_id}]: {t.arith_op}({t.left_id}, {t.right_id})")
            elif t.kind == "retract":
                turn_lines.append(
                    f"Turn {i} [{t.node_id}]: {t.retraction_op} {t.target_id} := {t.value} {t.unit}"
                )

        prompt = (
            "Compute the final value of node 'd2' in unit "
            f"'{expected_unit}'. Respond ONLY with a JSON object containing keys "
            '{"step_id": "d2", "value": <number>, "unit": "'
            + expected_unit
            + '", "retracted_ids": [<retracted node ids>]}.\n'
            + "\n".join(turn_lines)
        )

        episodes.append(
            RetractionEpisode(
                episode_id=ep_id,
                category=cat,
                prompt=prompt,
                turns=turns,
                target_node_id="d2",
                expected_unit=expected_unit,
                expected_value_float=float(target_frac),
                expected_retracted_ids=retracted_sorted,
            )
        )

    return episodes
