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
    preserved: Tuple[Tuple[str, Fraction, str], ...] = ()


def response_adapter(raw_text: str, required_preserved: Sequence[str] = ()) -> EnvelopeParseResult:
    """Strict JSON step-envelope gate; rejects prose, missing keys, booleans, and NaN/Inf."""
    if not isinstance(raw_text, str):
        return EnvelopeParseResult(ok=False, status="PARSE_REJECT", reason="Response is not a string.")
    stripped = raw_text.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return EnvelopeParseResult(
            ok=False, status="PARSE_REJECT",
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
        return EnvelopeParseResult(ok=False, status="PARSE_REJECT", reason=f"Missing required keys: {sorted(missing)}")
    extra = set(payload.keys()) - (required | {"notes", "preserved"})
    if extra:
        return EnvelopeParseResult(ok=False, status="PARSE_REJECT", reason=f"Unexpected envelope keys: {sorted(extra)}")

    step_id, unit, retracted_raw = payload["step_id"], payload["unit"], payload["retracted_ids"]
    if not isinstance(step_id, str) or not step_id.strip():
        return EnvelopeParseResult(ok=False, status="PARSE_REJECT", reason="Invalid step_id.")
    if not isinstance(unit, str) or not unit.strip():
        return EnvelopeParseResult(ok=False, status="PARSE_REJECT", reason="Invalid unit field.")
    if not isinstance(retracted_raw, list) or any(not isinstance(item, str) or not item.strip() for item in retracted_raw):
        return EnvelopeParseResult(ok=False, status="PARSE_REJECT", reason="retracted_ids must be list[str].")

    try:
        val_frac = to_exact_fraction(payload["value"])
    except ValueError as exc:
        return EnvelopeParseResult(ok=False, status="PARSE_REJECT", reason=str(exc))

    preserved_items: List[Tuple[str, Fraction, str]] = []
    if required_preserved and "preserved" not in payload:
        return EnvelopeParseResult(ok=False, status="PARSE_REJECT", reason="Missing required 'preserved' mapping.")
    if "preserved" in payload:
        raw_pres = payload["preserved"]
        if not isinstance(raw_pres, dict):
            return EnvelopeParseResult(ok=False, status="PARSE_REJECT", reason="preserved must be an object.")
        for k, v in sorted(raw_pres.items()):
            if not isinstance(k, str) or not k.strip() or not isinstance(v, dict) or set(v.keys()) != {"value", "unit"}:
                return EnvelopeParseResult(ok=False, status="PARSE_REJECT", reason=f"Invalid preserved entry for {k!r}.")
            if not isinstance(v["unit"], str) or not v["unit"].strip():
                return EnvelopeParseResult(ok=False, status="PARSE_REJECT", reason=f"Invalid preserved unit for {k!r}.")
            try:
                pv = to_exact_fraction(v["value"])
            except ValueError as exc:
                return EnvelopeParseResult(ok=False, status="PARSE_REJECT", reason=f"Invalid preserved value for {k!r}: {exc}")
            preserved_items.append((k.strip(), pv, v["unit"].strip()))
        missing_p = set(required_preserved) - {k for k, _, _ in preserved_items}
        if missing_p:
            return EnvelopeParseResult(ok=False, status="PARSE_REJECT", reason=f"Missing preserved keys: {sorted(missing_p)}")

    return EnvelopeParseResult(
        ok=True,
        status="OK",
        step_id=step_id.strip(),
        value_fraction=val_frac,
        unit=unit.strip(),
        retracted_ids=tuple(sorted(set(item.strip() for item in retracted_raw))),
        preserved=tuple(preserved_items),
    )


def parse_step_envelope(raw_text: str) -> EnvelopeParseResult:
    """Backwards-compatible alias for `response_adapter`."""
    return response_adapter(raw_text)


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


def _eval_turns(turns: Sequence[TurnSpec]) -> OracleStateSnapshot:
    oracle = ExactRationalOracle()
    for t in turns:
        oracle.append_turn(t)
    return oracle.evaluate()


def evaluate_retraction_episode(
    turns: Sequence[TurnSpec],
    raw_model_output: str,
    target_node_id: str,
    expected_unit: Optional[str] = None,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-5,
    preserved_targets: Optional[Dict[str, str]] = None,
) -> OracleEvaluationResult:
    """Deterministic 4-way split scoring of a multi-turn episode against a JSON step envelope."""
    snap = _eval_turns(turns)
    if target_node_id not in snap.active_values_base:
        raise ValueError(f"Target node {target_node_id!r} not active in oracle state.")

    truth_base, truth_dim, truth_base_unit = snap.active_values_base[target_node_id]
    display_unit = expected_unit or truth_base_unit
    truth_target = convert_from_base_fraction(truth_base, truth_dim, display_unit)
    expected_retracted = tuple(sorted(snap.retracted_ids))

    pre_truth_target: Optional[Fraction] = None
    if expected_retracted:
        pre_snap = _eval_turns([t for t in turns if t.kind != "retract"])
        if target_node_id in pre_snap.active_values_base:
            pb, pdim, _ = pre_snap.active_values_base[target_node_id]
            if pdim == truth_dim:
                pre_truth_target = convert_from_base_fraction(pb, pdim, display_unit)

    req_pres = tuple(preserved_targets.keys()) if preserved_targets else ()
    env = response_adapter(raw_model_output, required_preserved=req_pres)
    if not env.ok or env.value_fraction is None or env.step_id != target_node_id:
        reason = env.reason if not env.ok else f"Expected step_id={target_node_id!r}, got {env.step_id!r}."
        return OracleEvaluationResult(
            ok=False, verdict="PARSE_REJECT", truth_base_fraction=truth_base,
            truth_target_fraction=truth_target, truth_dimension=truth_dim,
            candidate_base_fraction=None, abs_diff=None, allowed_tol=None,
            expected_retracted_ids=expected_retracted, observed_retracted_ids=(), reason=reason,
        )

    try:
        cand_base, cand_dim, _ = normalize_to_base_fraction(env.value_fraction, env.unit)
    except DimensionMismatchError as exc:
        return OracleEvaluationResult(
            ok=False, verdict="CONDITIONAL_ARITH_ERROR", truth_base_fraction=truth_base,
            truth_target_fraction=truth_target, truth_dimension=truth_dim,
            candidate_base_fraction=None, abs_diff=None, allowed_tol=None,
            expected_retracted_ids=expected_retracted, observed_retracted_ids=env.retracted_ids, reason=str(exc),
        )

    if cand_dim != truth_dim:
        return OracleEvaluationResult(
            ok=False, verdict="CONDITIONAL_ARITH_ERROR", truth_base_fraction=truth_base,
            truth_target_fraction=truth_target, truth_dimension=truth_dim,
            candidate_base_fraction=cand_base, abs_diff=None, allowed_tol=None,
            expected_retracted_ids=expected_retracted, observed_retracted_ids=env.retracted_ids,
            reason=f"Dimension mismatch: candidate {cand_dim} vs truth {truth_dim}.",
        )

    cand_in_display = convert_from_base_fraction(cand_base, cand_dim, display_unit)
    matched, diff, allowed = mixed_tolerance_match(cand_in_display, truth_target, abs_tol=abs_tol, rel_tol=rel_tol)

    if (
        pre_truth_target is not None
        and not matched
        and mixed_tolerance_match(cand_in_display, pre_truth_target, abs_tol=abs_tol, rel_tol=rel_tol)[0]
    ):
        return OracleEvaluationResult(
            ok=False, verdict="STALE_STATE_RETENTION", truth_base_fraction=truth_base,
            truth_target_fraction=truth_target, truth_dimension=truth_dim,
            candidate_base_fraction=cand_base, abs_diff=diff, allowed_tol=allowed,
            expected_retracted_ids=expected_retracted, observed_retracted_ids=env.retracted_ids,
            reason="Candidate retained stale pre-retraction value.",
        )

    extra_retracted = sorted(set(env.retracted_ids) - set(expected_retracted))
    pres_err = ""
    pres_map = {k: (v, u) for k, v, u in env.preserved}
    for pid, punit in (preserved_targets or {}).items():
        pbase, pdim, _ = snap.active_values_base[pid]
        ptruth = convert_from_base_fraction(pbase, pdim, punit)
        pv, pu = pres_map[pid]
        try:
            pc_base, pc_dim, _ = normalize_to_base_fraction(pv, pu)
            if pc_dim != pdim or not mixed_tolerance_match(
                convert_from_base_fraction(pc_base, pc_dim, punit), ptruth, abs_tol=abs_tol, rel_tol=rel_tol
            )[0]:
                pres_err = f"Preserved node {pid!r} value or scale altered."
                break
        except DimensionMismatchError as exc:
            pres_err = f"Preserved node {pid!r} unit error: {exc}"
            break

    if extra_retracted or pres_err:
        return OracleEvaluationResult(
            ok=False, verdict="OVER_RETRACTION", truth_base_fraction=truth_base,
            truth_target_fraction=truth_target, truth_dimension=truth_dim,
            candidate_base_fraction=cand_base, abs_diff=diff, allowed_tol=allowed,
            expected_retracted_ids=expected_retracted, observed_retracted_ids=env.retracted_ids,
            reason=pres_err or f"Spurious retracted_ids: {extra_retracted}.",
        )

    if env.retracted_ids != expected_retracted or not matched:
        reason = (
            f"Expected retracted_ids={list(expected_retracted)}, got {list(env.retracted_ids)}."
            if env.retracted_ids != expected_retracted
            else f"Difference {float(diff):.6g} exceeds tolerance {float(allowed):.6g}."
        )
        return OracleEvaluationResult(
            ok=False, verdict="CONDITIONAL_ARITH_ERROR", truth_base_fraction=truth_base,
            truth_target_fraction=truth_target, truth_dimension=truth_dim,
            candidate_base_fraction=cand_base, abs_diff=diff, allowed_tol=allowed,
            expected_retracted_ids=expected_retracted, observed_retracted_ids=env.retracted_ids, reason=reason,
        )

    return OracleEvaluationResult(
        ok=True, verdict="PASS", truth_base_fraction=truth_base,
        truth_target_fraction=truth_target, truth_dimension=truth_dim,
        candidate_base_fraction=cand_base, abs_diff=diff, allowed_tol=allowed,
        expected_retracted_ids=expected_retracted, observed_retracted_ids=env.retracted_ids,
    )


@dataclass(frozen=True)
class RetractionEpisode:
    episode_id: str
    category: str  # "normal" | "perturbed"
    prompt: str
    turns: Tuple[TurnSpec, ...]
    target_node_id: str
    expected_unit: str
    expected_value_float: float
    expected_retracted_ids: Tuple[str, ...] = field(default_factory=tuple)
    condition: str = "A"  # "A" | "B" | "C" | "D" | "E"
    preserved_targets: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    pre_retraction_value_float: Optional[float] = None


def generate_procedural_dataset(seed: int = 20261122, count: int = 800) -> List[RetractionEpisode]:
    """Deterministically generate N=800 balanced 5-condition factorial episodes (A=400, B..E=100 each)."""
    if count <= 0:
        raise ValueError("count must be positive.")
    rng = random.Random(seed)
    cond_cycle = ("A", "B", "A", "C", "A", "D", "A", "E")
    retraction_ops = ("fix->", "err->", "sup->")
    scale_specs = (
        ("mg", "g", "%", 1, 6, 250, 1750, 2, 8),
        ("K_USD", "M_USD", "bps", 1, 5, 200, 900, 2, 8),
        ("cm", "m", "%", 1, 6, 20, 90, 2, 7),
        ("mL", "L", "%", 1, 5, 200, 800, 2, 7),
    )
    episodes: List[RetractionEpisode] = []

    for idx in range(count):
        ep_id, cond = f"gb-{idx + 1:04d}", cond_cycle[idx % 8]
        op_tok = retraction_ops[idx % 3]
        target_id, preserved = "d2", ()
        if cond == "A":
            unit = ("km", "g", "L", "s", "M_USD")[idx % 5]
            turns = (
                TurnSpec("p1", "premise", rng.randint(12, 80), unit),
                TurnSpec("p2", "premise", rng.randint(8, 40), unit),
                TurnSpec("p3", "premise", rng.randint(2, 6), "scalar"),
                TurnSpec("d1", "derive", arith_op="add", left_id="p1", right_id="p2"),
                TurnSpec("d2", "derive", arith_op="mul", left_id="d1", right_id="p3"),
            )
            expected_unit = unit
        elif cond == "B":
            unit = ("M_USD", "g", "km", "L")[idx % 4]
            turns = (
                TurnSpec("p1", "premise", rng.randint(10, 90), unit),
                TurnSpec("p2", "premise", rng.randint(5, 60), unit),
                TurnSpec("p3", "premise", rng.randint(2, 6), "scalar"),
                TurnSpec("d1", "derive", arith_op="add", left_id="p1", right_id="p2"),
                TurnSpec("d2", "derive", arith_op="mul", left_id="d1", right_id="p3"),
                TurnSpec("p1_r1", "retract", rng.randint(100, 250), unit, retraction_op=op_tok, target_id="p1"),
            )
            expected_unit = unit
        elif cond == "C":
            unit, target_id = ("g", "km", "M_USD", "L")[idx % 4], "d3"
            turns = (
                TurnSpec("p1", "premise", rng.randint(15, 75), unit),
                TurnSpec("p2", "premise", rng.randint(10, 50), unit),
                TurnSpec("p3", "premise", rng.randint(2, 5), "scalar"),
                TurnSpec("p4", "premise", rng.randint(5, 30), unit),
                TurnSpec("d1", "derive", arith_op="add", left_id="p1", right_id="p2"),
                TurnSpec("d2", "derive", arith_op="mul", left_id="d1", right_id="p3"),
                TurnSpec("d3", "derive", arith_op="sub", left_id="d2", right_id="p4"),
                TurnSpec("p1_r1", "retract", rng.randint(90, 220), unit, retraction_op=op_tok, target_id="p1"),
            )
            expected_unit = unit
        elif cond == "D":
            u_sm, u_lg, u_rt, lo2, hi2, lo1, hi1, rlo, rhi = scale_specs[idx % 4]
            rt_v = rng.choice((500, 1000, 1500, 2000) if u_rt == "bps" else (10, 15, 20, 25))
            turns = (
                TurnSpec("p1", "premise", rng.randint(lo1, hi1), u_sm),
                TurnSpec("p2", "premise", rng.randint(lo2, hi2), u_lg),
                TurnSpec("p3", "premise", rt_v, u_rt),
                TurnSpec("d1", "derive", arith_op="add", left_id="p1", right_id="p2"),
                TurnSpec("d2", "derive", arith_op="mul", left_id="d1", right_id="p3"),
                TurnSpec("p1_r1", "retract", rng.randint(rlo, rhi), u_lg, retraction_op=op_tok, target_id="p1"),
            )
            expected_unit = u_sm if (idx // 8) % 2 else u_lg
        else:
            unit, indep_u = (("M_USD", "g"), ("g", "km"), ("km", "L"), ("L", "M_USD"))[idx % 4]
            turns = (
                TurnSpec("p1", "premise", rng.randint(10, 90), unit),
                TurnSpec("p2", "premise", rng.randint(5, 60), unit),
                TurnSpec("p3", "premise", rng.randint(2, 5), "scalar"),
                TurnSpec("u1", "premise", rng.randint(15, 85), indep_u),
                TurnSpec("u2", "premise", rng.randint(10, 40), indep_u),
                TurnSpec("d1", "derive", arith_op="add", left_id="p1", right_id="p2"),
                TurnSpec("d2", "derive", arith_op="mul", left_id="d1", right_id="p3"),
                TurnSpec("d_indep", "derive", arith_op="add", left_id="u1", right_id="u2"),
                TurnSpec("p1_r1", "retract", rng.randint(100, 250), unit, retraction_op=op_tok, target_id="p1"),
            )
            expected_unit, preserved = unit, (("d_indep", indep_u),)

        snap = _eval_turns(turns)
        base_val, dim, _ = snap.active_values_base[target_id]
        target_frac = convert_from_base_fraction(base_val, dim, expected_unit)
        retracted_sorted = tuple(sorted(snap.retracted_ids))
        pre_float: Optional[float] = None
        if retracted_sorted:
            pb, pdim, _ = _eval_turns([t for t in turns if t.kind != "retract"]).active_values_base[target_id]
            pre_float = float(convert_from_base_fraction(pb, pdim, expected_unit))

        turn_lines = [
            f"Turn {i} [{t.node_id}]: value = {t.value} {t.unit}" if t.kind == "premise"
            else f"Turn {i} [{t.node_id}]: {t.arith_op}({t.left_id}, {t.right_id})" if t.kind == "derive"
            else f"Turn {i} [{t.node_id}]: {t.retraction_op} {t.target_id} := {t.value} {t.unit}"
            for i, t in enumerate(turns, start=1)
        ]
        pres_schema = f', "preserved": {{"{preserved[0][0]}": {{"value": <number>, "unit": "{preserved[0][1]}"}}}}' if preserved else ""
        prompt = (
            f"Compute the final value of node '{target_id}' in unit '{expected_unit}'. Respond ONLY with a JSON object "
            f'containing keys {{"step_id": "{target_id}", "value": <number>, "unit": "{expected_unit}", '
            f'"retracted_ids": [<retracted node ids>]{pres_schema}}}.\n' + "\n".join(turn_lines)
        )
        episodes.append(
            RetractionEpisode(
                episode_id=ep_id, category="normal" if cond == "A" else "perturbed", prompt=prompt,
                turns=turns, target_node_id=target_id, expected_unit=expected_unit,
                expected_value_float=float(target_frac), expected_retracted_ids=retracted_sorted,
                condition=cond, preserved_targets=preserved, pre_retraction_value_float=pre_float,
            )
        )
    return episodes
