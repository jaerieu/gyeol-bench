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
    """Strict JSON step-envelope gate; rejects prose, missing keys, non-string notes, booleans, and NaN/Inf."""
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
    if "notes" in payload and not isinstance(payload["notes"], str):
        return EnvelopeParseResult(ok=False, status="PARSE_REJECT", reason="Invalid notes field: must be str.")

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


def _downward_closed_dropped_sets(retract_turns: Sequence[TurnSpec]) -> List[frozenset[str]]:
    """Enumerate unique non-empty downward-closed dropped retraction node_id subsets (1 <= k <= 5)."""
    k = len(retract_turns)
    if k < 1 or k > 5:
        return []
    seen: Set[frozenset[str]] = set()
    ordered: List[frozenset[str]] = []
    for mask in range(1, 1 << k):
        dropped: Set[str] = {
            retract_turns[i].node_id for i in range(k) if (mask & (1 << i)) != 0
        }
        changed = True
        while changed:
            changed = False
            for rt in retract_turns:
                if rt.node_id not in dropped and rt.target_id in dropped:
                    dropped.add(rt.node_id)
                    changed = True
        frozen = frozenset(dropped)
        if frozen not in seen:
            seen.add(frozen)
            ordered.append(frozen)
    return ordered


def _compute_stale_targets(
    turns: Sequence[TurnSpec],
    target_node_id: str,
    display_unit: str,
    truth_dim: str,
    truth_target: Fraction,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-5,
) -> Tuple[Fraction, ...]:
    retract_turns = [t for t in turns if t.kind == "retract"]
    stale: List[Fraction] = []
    for dropped in _downward_closed_dropped_sets(retract_turns):
        try:
            sub_snap = _eval_turns([t for t in turns if t.node_id not in dropped])
        except (ValueError, ZeroDivisionError, DimensionMismatchError):
            continue
        if target_node_id not in sub_snap.active_values_base:
            continue
        sb, sdim, _ = sub_snap.active_values_base[target_node_id]
        if sdim != truth_dim:
            continue
        starget = convert_from_base_fraction(sb, sdim, display_unit)
        if not mixed_tolerance_match(starget, truth_target, abs_tol=abs_tol, rel_tol=rel_tol)[0]:
            if starget not in stale:
                stale.append(starget)
    return tuple(stale)


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
    stale_targets = _compute_stale_targets(
        turns, target_node_id, display_unit, truth_dim, truth_target, abs_tol=abs_tol, rel_tol=rel_tol
    )

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
        stale_targets
        and not matched
        and any(mixed_tolerance_match(cand_in_display, st, abs_tol=abs_tol, rel_tol=rel_tol)[0] for st in stale_targets)
    ):
        return OracleEvaluationResult(
            ok=False, verdict="STALE_STATE_RETENTION", truth_base_fraction=truth_base,
            truth_target_fraction=truth_target, truth_dimension=truth_dim,
            candidate_base_fraction=cand_base, abs_diff=diff, allowed_tol=allowed,
            expected_retracted_ids=expected_retracted, observed_retracted_ids=env.retracted_ids,
            reason="Candidate retained stale full or partial pre-retraction value.",
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


def _all_subsets_nondegenerate(turns: Sequence[TurnSpec], target_id: str, expected_unit: str) -> bool:
    try:
        snap = _eval_turns(turns)
    except (ValueError, ZeroDivisionError, DimensionMismatchError):
        return False
    truth_base, truth_dim, _ = snap.active_values_base[target_id]
    truth_target = convert_from_base_fraction(truth_base, truth_dim, expected_unit)
    retract_turns = [t for t in turns if t.kind == "retract"]
    subsets = _downward_closed_dropped_sets(retract_turns)
    if not subsets:
        return False
    for dropped in subsets:
        try:
            sub_snap = _eval_turns([t for t in turns if t.node_id not in dropped])
        except (ValueError, ZeroDivisionError, DimensionMismatchError):
            return False
        sb, sdim, _ = sub_snap.active_values_base[target_id]
        if sdim != truth_dim:
            return False
        starget = convert_from_base_fraction(sb, sdim, expected_unit)
        if mixed_tolerance_match(starget, truth_target)[0]:
            return False
    return True


def _build_diamond_core(
    rng: random.Random,
    unit: str,
    r_vals: Tuple[int, int, int],
) -> Tuple[List[TurnSpec], List[TurnSpec], List[TurnSpec]]:
    p1 = TurnSpec("p1", "premise", rng.randint(110, 290), unit)
    p2 = TurnSpec("p2", "premise", rng.randint(105, 240), unit)
    p3 = TurnSpec("p3", "premise", rng.randint(115, 280), unit)
    p4 = TurnSpec("p4", "premise", rng.choice((20, 25, 40, 50, 75, 80)), "%")
    p5 = TurnSpec("p5", "premise", rng.randint(2, 5), "scalar")
    p6 = TurnSpec("p6", "premise", rng.randint(100, 210), unit)
    p7 = TurnSpec("p7", "premise", rng.choice((2, 3, 4, 5)), "scalar")
    premises = [p1, p2, p3, p4, p5, p6, p7]
    derives = [
        TurnSpec("d1", "derive", arith_op="add", left_id="p1", right_id="p2"),
        TurnSpec("d2", "derive", arith_op="sub", left_id="p1", right_id="p6"),
        TurnSpec("d3", "derive", arith_op="mul", left_id="d1", right_id="p4"),
        TurnSpec("d4", "derive", arith_op="mul", left_id="d2", right_id="p5"),
        TurnSpec("d5", "derive", arith_op="add", left_id="d3", right_id="p3"),
        TurnSpec("d6", "derive", arith_op="add", left_id="d5", right_id="d4"),
        TurnSpec("d7", "derive", arith_op="sub", left_id="d6", right_id="p2"),
        TurnSpec("d8", "derive", arith_op="mul", left_id="d7", right_id="p7"),
    ]
    retracts = [
        TurnSpec("p1_r1", "retract", r_vals[0], unit, retraction_op="fix->", target_id="p1"),
        TurnSpec("p3_r1", "retract", r_vals[1], unit, retraction_op="err->", target_id="p3"),
        TurnSpec("p1_r2", "retract", r_vals[2], unit, retraction_op="sup->", target_id="p1_r1"),
    ]
    return premises, derives, retracts


def generate_procedural_dataset(seed: int = 20261122, count: int = 800) -> List[RetractionEpisode]:
    """Deterministically generate v2.0 N=800 balanced 5-condition episodes (A=400, B..E=100 each)."""
    if count <= 0:
        raise ValueError("count must be positive.")
    rng = random.Random(seed)
    cond_cycle = ("A", "B", "A", "C", "A", "D", "A", "E")
    episodes: List[RetractionEpisode] = []

    for idx in range(count):
        ep_id, cond = f"gb-{idx + 1:04d}", cond_cycle[idx % 8]
        grp = idx // 8
        preserved: Tuple[Tuple[str, str], ...] = ()

        if cond == "A":
            unit = ("km", "g", "L", "s", "M_USD")[(idx // 2) % 5]
            dist_u = ("g", "L", "s", "M_USD", "km")[(idx // 2) % 5]
            target_id, expected_unit = "d4", unit
            p1, p2 = rng.randint(15, 85), rng.randint(12, 68)
            p3, p4, p5 = rng.randint(11, 24), rng.randint(10, 55), rng.randint(11, 19)
            x1, x2 = rng.randint(14, 79), rng.randint(11, 58)
            x3, x4 = rng.randint(11, 18), rng.randint(10, 49)
            turns: Tuple[TurnSpec, ...] = (
                TurnSpec("p1", "premise", p1, unit),
                TurnSpec("x1", "premise", x1, dist_u),
                TurnSpec("p2", "premise", p2, unit),
                TurnSpec("x2", "premise", x2, dist_u),
                TurnSpec("d1", "derive", arith_op="add", left_id="p1", right_id="p2"),
                TurnSpec("z1", "derive", arith_op="add", left_id="x1", right_id="x2"),
                TurnSpec("p3", "premise", p3, "scalar"),
                TurnSpec("x3", "premise", x3, "scalar"),
                TurnSpec("d2", "derive", arith_op="mul", left_id="d1", right_id="p3"),
                TurnSpec("z2", "derive", arith_op="mul", left_id="z1", right_id="x3"),
                TurnSpec("p4", "premise", p4, unit),
                TurnSpec("x4", "premise", x4, dist_u),
                TurnSpec("d3", "derive", arith_op="sub", left_id="d2", right_id="p4"),
                TurnSpec("z3", "derive", arith_op="sub", left_id="z2", right_id="x4"),
                TurnSpec("p5", "premise", p5, "scalar"),
                TurnSpec("z4", "derive", arith_op="add", left_id="z3", right_id="x2"),
                TurnSpec("d4", "derive", arith_op="mul", left_id="d3", right_id="p5"),
            )
        elif cond == "B":
            unit = ("M_USD", "g", "km", "L")[grp % 4]
            dist_u = ("g", "km", "L", "M_USD")[grp % 4]
            target_id, expected_unit = "d8", unit
            base_r = (rng.randint(340, 520), rng.randint(330, 510), rng.randint(580, 820))
            sub_seed = rng.randint(1, 10**9)
            x_vals = (rng.randint(120, 310), rng.randint(105, 230), rng.randint(110, 260), rng.randint(100, 195))
            turns = ()
            for attempt in range(32):
                pr, dr, rr = _build_diamond_core(
                    random.Random(sub_seed),
                    unit,
                    (base_r[0] + 19 * attempt, base_r[1] + 23 * attempt, base_r[2] + 31 * attempt),
                )
                cand_turns = (
                    pr[0], TurnSpec("x1", "premise", x_vals[0], dist_u),
                    pr[1], TurnSpec("x2", "premise", x_vals[1], dist_u),
                    pr[2], pr[3], pr[4], pr[5], pr[6],
                    TurnSpec("z1", "derive", arith_op="add", left_id="x1", right_id="x2"),
                    dr[0], dr[1], dr[2], dr[3],
                    rr[0],
                    TurnSpec("x3", "premise", x_vals[2], dist_u),
                    TurnSpec("x4", "premise", x_vals[3], dist_u),
                    dr[4], dr[5],
                    rr[1],
                    TurnSpec("z2", "derive", arith_op="sub", left_id="x3", right_id="x4"),
                    dr[6], dr[7],
                    rr[2],
                )
                if _all_subsets_nondegenerate(cand_turns, target_id, expected_unit):
                    turns = cand_turns
                    break
            assert turns, f"Failed non-degeneracy guard on {ep_id}"
        elif cond == "C":
            unit = ("g", "km", "M_USD", "L")[grp % 4]
            dist_u = ("km", "M_USD", "L", "g")[grp % 4]
            use_12 = (grp % 2) == 1
            target_id, expected_unit = ("d12" if use_12 else "d10"), unit
            p_vals = (
                rng.randint(120, 280), rng.randint(60, 170), rng.randint(2, 5),
                rng.randint(75, 195), rng.choice((2, 4, 5, 8, 10)),
                rng.randint(45, 145), rng.choice((2, 4, 5, 8, 10)),
            )
            x_vals = (rng.randint(110, 270), rng.randint(80, 190), rng.randint(120, 250), rng.randint(60, 140))
            base_r = (rng.randint(330, 520), rng.randint(260, 440), rng.randint(560, 790))
            turns = ()
            for attempt in range(32):
                r1v, r4v, r2v = base_r[0] + 17 * attempt, base_r[1] + 23 * attempt, base_r[2] + 29 * attempt
                tail = (
                    (
                        TurnSpec("d10", "derive", arith_op="mul", left_id="d9", right_id="p3"),
                        TurnSpec("d11", "derive", arith_op="sub", left_id="d10", right_id="p6"),
                        TurnSpec("d12", "derive", arith_op="add", left_id="d11", right_id="d5"),
                    )
                    if use_12
                    else (TurnSpec("d10", "derive", arith_op="add", left_id="d9", right_id="d5"),)
                )
                cand_turns = (
                    TurnSpec("p1", "premise", p_vals[0], unit),
                    TurnSpec("x1", "premise", x_vals[0], dist_u),
                    TurnSpec("p2", "premise", p_vals[1], unit),
                    TurnSpec("x2", "premise", x_vals[1], dist_u),
                    TurnSpec("p3", "premise", p_vals[2], "scalar"),
                    TurnSpec("p4", "premise", p_vals[3], unit),
                    TurnSpec("p5", "premise", p_vals[4], "scalar"),
                    TurnSpec("p6", "premise", p_vals[5], unit),
                    TurnSpec("p7", "premise", p_vals[6], "scalar"),
                    TurnSpec("z1", "derive", arith_op="add", left_id="x1", right_id="x2"),
                    TurnSpec("d1", "derive", arith_op="add", left_id="p1", right_id="p2"),
                    TurnSpec("d2", "derive", arith_op="mul", left_id="d1", right_id="p3"),
                    TurnSpec("d3", "derive", arith_op="sub", left_id="d2", right_id="p4"),
                    TurnSpec("d4", "derive", arith_op="div", left_id="d3", right_id="p5"),
                    TurnSpec("p1_r1", "retract", r1v, unit, retraction_op="fix->", target_id="p1"),
                    TurnSpec("x3", "premise", x_vals[2], dist_u),
                    TurnSpec("x4", "premise", x_vals[3], dist_u),
                    TurnSpec("d5", "derive", arith_op="add", left_id="d4", right_id="p6"),
                    TurnSpec("d6", "derive", arith_op="add", left_id="d5", right_id="d2"),
                    TurnSpec("d7", "derive", arith_op="sub", left_id="d6", right_id="p2"),
                    TurnSpec("p4_r1", "retract", r4v, unit, retraction_op="err->", target_id="p4"),
                    TurnSpec("z2", "derive", arith_op="sub", left_id="x3", right_id="x4"),
                    TurnSpec("d8", "derive", arith_op="div", left_id="d7", right_id="p7"),
                    TurnSpec("d9", "derive", arith_op="add", left_id="d8", right_id="d3"),
                    *tail,
                    TurnSpec("p1_r2", "retract", r2v, unit, retraction_op="sup->", target_id="p1_r1"),
                )
                if _all_subsets_nondegenerate(cand_turns, target_id, expected_unit):
                    turns = cand_turns
                    break
            assert turns, f"Failed non-degeneracy guard on {ep_id}"
        elif cond == "D":
            fam_idx, sub_var = grp % 5, (grp // 5) % 3
            if fam_idx == 0:
                u1, u2, u3, dist_u = "mg", "g", "kg", "m"
                v1, v2, v3 = rng.randint(250, 1750), rng.randint(12, 85), rng.randint(1, 4)
                rv1, rv2, step1, step2 = rng.randint(18, 95), rng.randint(2, 7), 7, 1
            elif fam_idx == 1:
                u1, u2, u3 = ("mm", "cm", "m") if (sub_var % 2 == 0) else ("cm", "m", "km")
                dist_u = "g"
                v1, v2, v3 = rng.randint(120, 850), rng.randint(15, 90), rng.randint(2, 6)
                rv1, rv2, step1, step2 = rng.randint(25, 110), rng.randint(3, 9), 9, 1
            elif fam_idx == 2:
                u1, u2, u3, dist_u = "uL", "mL", "L", "km"
                v1, v2, v3 = rng.randint(500, 4500), rng.randint(20, 140), rng.randint(1, 5)
                rv1, rv2, step1, step2 = rng.randint(45, 260), rng.randint(2, 8), 13, 1
            elif fam_idx == 3:
                if (sub_var % 2) == 0:
                    u1, u2, u3 = "ms", "s", "min"
                    v1, v2, v3 = 300 * rng.randint(2, 9), 6 * rng.randint(2, 9), rng.randint(1, 5)
                    rv1, rv2, step1, step2 = 6 * rng.randint(4, 14), rng.randint(2, 8), 6, 1
                else:
                    u1, u2, u3 = "s", "min", "h"
                    v1, v2, v3 = 36 * rng.randint(2, 9), 6 * rng.randint(2, 9), rng.randint(1, 4)
                    rv1, rv2, step1, step2 = 6 * rng.randint(4, 14), rng.randint(2, 7), 6, 1
                dist_u = "L"
            else:
                if sub_var == 0:
                    u1, u2, u3 = "K_USD", "M_USD", "USD"
                    v1, v2, v3 = rng.randint(150, 850), rng.randint(1, 5), 250_000 * rng.randint(2, 8)
                    rv1, rv2, step1, step2 = rng.randint(2, 8), 250_000 * rng.randint(6, 18), 1, 250_000
                elif sub_var == 1:
                    u1, u2, u3 = "USD", "K_USD", "M_USD"
                    v1, v2, v3 = 500 * rng.randint(2, 15), rng.randint(25, 180), rng.randint(1, 4)
                    rv1, rv2, step1, step2 = rng.randint(60, 320), rng.randint(2, 7), 15, 1
                else:
                    u1, u2, u3 = "K_USD", "M_USD", "B_USD"
                    v1, v2, v3 = rng.randint(200, 900), rng.randint(15, 120), rng.randint(1, 3)
                    rv1, rv2, step1, step2 = rng.randint(40, 260), rng.randint(2, 5), 15, 1
                dist_u = "g"

            use_10 = (grp % 2) == 1
            target_id = "d10" if use_10 else "d8"
            expected_unit = u2 if (grp // 2) % 2 == 0 else u1
            bps_v = rng.choice((250, 500, 750, 1250, 1500, 2500))
            pct_choices = [p for p in (10, 15, 20, 30, 35, 40) if p * 100 != bps_v]
            pct_r = rng.choice(pct_choices)
            r2_pct = rng.choice((10, 20, 25, 50))
            s1_v = rng.choice((2, 4, 5))
            x_vals = (rng.randint(20, 95), rng.randint(15, 75), rng.randint(30, 110), rng.randint(10, 50))
            turns = ()
            for attempt in range(32):
                adj_rv1 = rv1 + step1 * attempt
                adj_rv2 = rv2 + step2 * attempt
                adj_pct = pct_choices[(pct_choices.index(pct_r) + attempt) % len(pct_choices)]
                tail_d = (
                    (
                        TurnSpec("d9", "derive", arith_op="div", left_id="d8", right_id="s1"),
                        TurnSpec("d10", "derive", arith_op="add", left_id="d9", right_id="d3"),
                    )
                    if use_10
                    else ()
                )
                cand_turns = (
                    TurnSpec("p1", "premise", v1, u1),
                    TurnSpec("x1", "premise", x_vals[0], dist_u),
                    TurnSpec("p2", "premise", v2, u2),
                    TurnSpec("x2", "premise", x_vals[1], dist_u),
                    TurnSpec("p3", "premise", v3, u3),
                    TurnSpec("r1", "premise", bps_v, "bps"),
                    TurnSpec("r2", "premise", r2_pct, "%"),
                    TurnSpec("s1", "premise", s1_v, "scalar"),
                    TurnSpec("z1", "derive", arith_op="add", left_id="x1", right_id="x2"),
                    TurnSpec("d1", "derive", arith_op="add", left_id="p1", right_id="p2"),
                    TurnSpec("d2", "derive", arith_op="add", left_id="d1", right_id="p3"),
                    TurnSpec("d3", "derive", arith_op="mul", left_id="d2", right_id="r1"),
                    TurnSpec("p1_r1", "retract", adj_rv1, u2, retraction_op="fix->", target_id="p1"),
                    TurnSpec("x3", "premise", x_vals[2], dist_u),
                    TurnSpec("x4", "premise", x_vals[3], dist_u),
                    TurnSpec("d4", "derive", arith_op="add", left_id="d2", right_id="d3"),
                    TurnSpec("d5", "derive", arith_op="mul", left_id="p3", right_id="r2"),
                    TurnSpec("r1_r1", "retract", adj_pct, "%", retraction_op="err->", target_id="r1"),
                    TurnSpec("z2", "derive", arith_op="sub", left_id="x3", right_id="x4"),
                    TurnSpec("d6", "derive", arith_op="sub", left_id="d4", right_id="d5"),
                    TurnSpec("d7", "derive", arith_op="mul", left_id="d6", right_id="s1"),
                    TurnSpec("d8", "derive", arith_op="add", left_id="d7", right_id="d1"),
                    *tail_d,
                    TurnSpec("p1_r2", "retract", adj_rv2, u3, retraction_op="sup->", target_id="p1_r1"),
                )
                if _all_subsets_nondegenerate(cand_turns, target_id, expected_unit):
                    turns = cand_turns
                    break
            assert turns, f"Failed non-degeneracy guard on {ep_id}"
        else:
            unit, indep_u = (("M_USD", "g"), ("g", "km"), ("km", "L"), ("L", "M_USD"))[grp % 4]
            target_id, expected_unit = "d8", unit
            preserved = (("d_indep", indep_u),)
            base_r = (rng.randint(340, 520), rng.randint(330, 510), rng.randint(580, 820))
            sub_seed = rng.randint(1, 10**9)
            u_vals = (rng.randint(110, 290), rng.randint(60, 180), rng.randint(2, 5), rng.randint(40, 130))
            u_near_v = rng.randint(115, 275)
            turns = ()
            for attempt in range(32):
                pr, dr, rr = _build_diamond_core(
                    random.Random(sub_seed),
                    unit,
                    (base_r[0] + 19 * attempt, base_r[1] + 23 * attempt, base_r[2] + 31 * attempt),
                )
                cand_turns = (
                    pr[0], TurnSpec("u1", "premise", u_vals[0], indep_u),
                    pr[1], TurnSpec("u2", "premise", u_vals[1], indep_u),
                    pr[2], pr[3], pr[4], pr[5], pr[6],
                    TurnSpec("u3", "premise", u_vals[2], "scalar"),
                    TurnSpec("u4", "premise", u_vals[3], indep_u),
                    TurnSpec("u_near", "premise", u_near_v, unit),
                    TurnSpec("u_d1", "derive", arith_op="add", left_id="u1", right_id="u2"),
                    dr[0], dr[1], dr[2], dr[3],
                    rr[0],
                    TurnSpec("u_d2", "derive", arith_op="mul", left_id="u_d1", right_id="u3"),
                    dr[4], dr[5],
                    rr[1],
                    TurnSpec("d_indep", "derive", arith_op="sub", left_id="u_d2", right_id="u4"),
                    dr[6], dr[7],
                    rr[2],
                )
                if _all_subsets_nondegenerate(cand_turns, target_id, expected_unit):
                    turns = cand_turns
                    break
            assert turns, f"Failed non-degeneracy guard on {ep_id}"

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
        pres_schema = (
            f', "preserved": {{"{preserved[0][0]}": {{"value": <number>, "unit": "{preserved[0][1]}"}}}}'
            if preserved else ""
        )
        prompt = (
            f"Compute the final value of node '{target_id}' in unit '{expected_unit}'. "
            "Respond ONLY with a standalone JSON object containing keys "
            f'{{"notes": "<concise step-by-step calculation and retraction trace>", '
            f'"step_id": "{target_id}", "value": <number>, "unit": "{expected_unit}", '
            f'"retracted_ids": [<sorted superseded target node IDs>]{pres_schema}}}. '
            'Rule for "retracted_ids": include ONLY unique node IDs appearing directly as the superseded '
            "target on the left of ':=' in any retraction turn (fix->, err->, sup->), sorted alphabetically; "
            "do NOT include replacement turn IDs (unless later superseded themselves) or derived nodes (d*); "
            "output [] if no retraction turn occurs.\n" + "\n".join(turn_lines)
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


V3_MULTITURN_BASELINE_SYSTEM_PROMPT = (
    "You are a precise multi-turn quantitative reasoning assistant. Across sequential user turns:\n"
    "1. Track all base premises and their exact units (SI mass ug/mg/g/kg, length mm/cm/m/km, volume uL/mL/L, "
    "time ms/s/min/h, USD financial scales USD/K_USD/M_USD/B_USD, percentages %, and basis points bps), "
    "ignoring inert context or distractor notes unrelated to the target node.\n"
    "2. Whenever a retraction step (fix->, err->, or sup->) supersedes an earlier node (including chained "
    "retractions such as sup-> p1_r1), record ONLY unique node IDs appearing directly on the left of ':=' in "
    "\"retracted_ids\" sorted alphabetically (never include un-superseded replacement turn IDs or derived d* "
    "nodes; output [] if zero retractions occur), and recompute every downstream derived node from earlier "
    "turns that depends on it using exact dimensional conversion.\n"
    "3. Keep any independent derived subgraph (such as d_indep) that has no directed dependency on a retracted "
    "node strictly unchanged in \"preserved\" when requested.\n"
    "4. On every turn, output ONLY a single standalone valid JSON object starting with \"notes\" (containing a "
    "concise step-by-step calculation and retraction trace) followed by \"step_id\", \"value\", \"unit\", "
    "\"retracted_ids\" (and \"preserved\" when requested), with zero surrounding prose or markdown fences."
)

V3_MULTITURN_LINEAGE_K2_SYSTEM_PROMPT = (
    "You are a lineage-verified multi-turn quantitative reasoning engine. Enforce strict dependency-graph "
    "taint invalidation across sequential user turns:\n"
    "1. In Turn 1 \"notes\", record an explicit ancestor ledger for every derived node (e.g., d1<-{p1,p2}; "
    "d2<-{p1,p2,p3}; d3<-{...}; d4<-{...}; d_indep<-{u1,u2,u3}) alongside canonical base-unit values "
    "(g, m, L, s, USD, scalar).\n"
    "2. In Turn 2 \"notes\", BEFORE evaluating d5 or d6: (a) extract the exact sorted set R of superseded "
    "target IDs appearing directly on the left of ':=' in any fix->, err->, or sup-> step (output R=[] if none, "
    "never include un-superseded replacement IDs or derived d* IDs); (b) mark as TAINTED every Turn-1 derived "
    "node whose ancestor set intersects R, and NEVER reuse a committed Turn-1 numeric value for any TAINTED "
    "node; (c) rebuild every TAINTED Turn-1 node bottom-up from the latest active root premises in canonical "
    "base units before computing d5 and d6 and converting to the requested output unit; (d) copy any UNTAINTED "
    "disjoint node (d_indep) verbatim from Turn 1 into \"preserved\".\n"
    "3. On every turn, output ONLY a single standalone valid JSON object starting with \"notes\" followed by "
    "\"step_id\", \"value\", \"unit\", \"retracted_ids\" (and \"preserved\" when requested), with zero "
    "surrounding prose or markdown fences."
)


@dataclass(frozen=True)
class MultiTurnEpisode:
    episode_id: str
    condition: str  # "A" | "B" | "C" | "D" | "E"
    category: str  # "normal" | "perturbed"
    turn1_turns: Tuple[TurnSpec, ...]
    turn1_prompt: str
    turn1_target_id: str
    turn1_expected_unit: str
    turn1_expected_value_float: float
    turn1_preserved_targets: Tuple[Tuple[str, str], ...]
    turn2_turns: Tuple[TurnSpec, ...]
    all_turns: Tuple[TurnSpec, ...]
    turn2_prompt: str
    target_node_id: str
    expected_unit: str
    expected_value_float: float
    expected_retracted_ids: Tuple[str, ...] = field(default_factory=tuple)
    preserved_targets: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    pre_retraction_value_float: Optional[float] = None
    stale_target_values_float: Tuple[float, ...] = field(default_factory=tuple)


def _compute_multiturn_stale_targets(
    turn1_turns: Sequence[TurnSpec],
    turn2_turns: Sequence[TurnSpec],
    target_node_id: str,
    display_unit: str,
    truth_dim: str,
    truth_target: Fraction,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-5,
) -> Tuple[Fraction, ...]:
    all_turns = tuple(turn1_turns) + tuple(turn2_turns)
    retract_turns = [t for t in all_turns if t.kind == "retract"]
    if not retract_turns:
        return ()

    stale: List[Fraction] = list(
        _compute_stale_targets(
            all_turns, target_node_id, display_unit, truth_dim, truth_target, abs_tol=abs_tol, rel_tol=rel_tol
        )
    )

    t1_snap = _eval_turns(turn1_turns)
    candidate_frozen_sets: Tuple[Tuple[str, ...], ...] = (
        ("d4",),
        ("d2",),
        ("d4", "d2"),
        ("d3",),
        ("d1",),
    )
    dropped_options: List[frozenset[str]] = [frozenset()] + _downward_closed_dropped_sets(retract_turns)

    for fset in candidate_frozen_sets:
        if any(nid not in t1_snap.active_values_base for nid in fset):
            continue
        f_lookup = set(fset)
        mod_t1: List[TurnSpec] = []
        for t in turn1_turns:
            if t.node_id in f_lookup:
                bval, _, bunit = t1_snap.active_values_base[t.node_id]
                mod_t1.append(TurnSpec(node_id=t.node_id, kind="premise", value=bval, unit=bunit))
            else:
                mod_t1.append(t)
        for dropped in dropped_options:
            mod_all = mod_t1 + [t for t in turn2_turns if t.node_id not in dropped]
            try:
                sub_snap = _eval_turns(mod_all)
            except (ValueError, ZeroDivisionError, DimensionMismatchError):
                continue
            if target_node_id not in sub_snap.active_values_base:
                continue
            sb, sdim, _ = sub_snap.active_values_base[target_node_id]
            if sdim != truth_dim:
                continue
            starget = convert_from_base_fraction(sb, sdim, display_unit)
            if not mixed_tolerance_match(starget, truth_target, abs_tol=abs_tol, rel_tol=rel_tol)[0]:
                if starget not in stale:
                    stale.append(starget)
    return tuple(stale)


def _all_multiturn_primary_stales_nondegenerate(
    turn1_turns: Sequence[TurnSpec],
    turn2_turns: Sequence[TurnSpec],
    target_node_id: str,
    display_unit: str,
) -> bool:
    all_turns = tuple(turn1_turns) + tuple(turn2_turns)
    try:
        t1_snap = _eval_turns(turn1_turns)
        snap = _eval_turns(all_turns)
    except (ValueError, ZeroDivisionError, DimensionMismatchError):
        return False
    truth_base, truth_dim, _ = snap.active_values_base[target_node_id]
    truth_target = convert_from_base_fraction(truth_base, truth_dim, display_unit)

    retract_turns = [t for t in all_turns if t.kind == "retract"]
    subsets = _downward_closed_dropped_sets(retract_turns)
    if not subsets:
        return False

    for dropped in subsets:
        try:
            sub_snap = _eval_turns([t for t in all_turns if t.node_id not in dropped])
        except (ValueError, ZeroDivisionError, DimensionMismatchError):
            return False
        sb, sdim, _ = sub_snap.active_values_base[target_node_id]
        if sdim != truth_dim:
            return False
        starget = convert_from_base_fraction(sb, sdim, display_unit)
        if mixed_tolerance_match(starget, truth_target)[0]:
            return False

    for fset in (("d4",), ("d2",), ("d4", "d2")):
        if any(nid not in t1_snap.active_values_base for nid in fset):
            return False
        mod_t1 = [
            TurnSpec(t.node_id, "premise", t1_snap.active_values_base[t.node_id][0], t1_snap.active_values_base[t.node_id][2])
            if t.node_id in fset else t
            for t in turn1_turns
        ]
        try:
            sub_snap = _eval_turns(mod_t1 + list(turn2_turns))
        except (ValueError, ZeroDivisionError, DimensionMismatchError):
            return False
        sb, sdim, _ = sub_snap.active_values_base[target_node_id]
        if sdim != truth_dim:
            return False
        starget = convert_from_base_fraction(sb, sdim, display_unit)
        if mixed_tolerance_match(starget, truth_target)[0]:
            return False
    return True


def evaluate_multiturn_episode(
    turn1_turns: Sequence[TurnSpec],
    turn2_turns: Sequence[TurnSpec],
    raw_model_output: str,
    target_node_id: str = "d6",
    expected_unit: Optional[str] = None,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-5,
    preserved_targets: Optional[Dict[str, str]] = None,
) -> OracleEvaluationResult:
    """Deterministic 4-way split scorer for v3.0 2-turn episodes (includes stale Turn-1 node substitution)."""
    all_turns = tuple(turn1_turns) + tuple(turn2_turns)
    snap = _eval_turns(all_turns)
    if target_node_id not in snap.active_values_base:
        raise ValueError(f"Target node {target_node_id!r} not active in oracle state.")

    truth_base, truth_dim, truth_base_unit = snap.active_values_base[target_node_id]
    display_unit = expected_unit or truth_base_unit
    truth_target = convert_from_base_fraction(truth_base, truth_dim, display_unit)
    expected_retracted = tuple(sorted(snap.retracted_ids))
    stale_targets = _compute_multiturn_stale_targets(
        turn1_turns, turn2_turns, target_node_id, display_unit, truth_dim, truth_target,
        abs_tol=abs_tol, rel_tol=rel_tol,
    )

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
        stale_targets
        and not matched
        and any(mixed_tolerance_match(cand_in_display, st, abs_tol=abs_tol, rel_tol=rel_tol)[0] for st in stale_targets)
    ):
        return OracleEvaluationResult(
            ok=False, verdict="STALE_STATE_RETENTION", truth_base_fraction=truth_base,
            truth_target_fraction=truth_target, truth_dimension=truth_dim,
            candidate_base_fraction=cand_base, abs_diff=diff, allowed_tol=allowed,
            expected_retracted_ids=expected_retracted, observed_retracted_ids=env.retracted_ids,
            reason="Candidate retained stale Turn-1 committed value or dropped retraction.",
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


def _format_turn_lines(prefix: str, turns: Sequence[TurnSpec]) -> List[str]:
    return [
        f"{prefix}.{i} [{t.node_id}]: value = {t.value} {t.unit}" if t.kind == "premise"
        else f"{prefix}.{i} [{t.node_id}]: {t.arith_op}({t.left_id}, {t.right_id})" if t.kind == "derive"
        else f"{prefix}.{i} [{t.node_id}]: {t.retraction_op} {t.target_id} := {t.value} {t.unit}"
        for i, t in enumerate(turns, start=1)
    ]


def _is_terminating_decimal(frac: Fraction) -> bool:
    den = frac.denominator
    while den % 2 == 0:
        den //= 2
    while den % 5 == 0:
        den //= 5
    return den == 1


def generate_multiturn_dataset(seed: int = 20261122, count: int = 800) -> List[MultiTurnEpisode]:
    """Deterministically generate v3.0 2-live-turn Shared-Turn-1 episodes (A=400, B..E=100 each)."""
    if count <= 0:
        raise ValueError("count must be positive.")
    rng = random.Random(seed)
    cond_cycle = ("A", "B", "A", "C", "A", "D", "A", "E")
    retracted_rule = (
        'Rule for "retracted_ids": list ONLY unique node IDs directly on the left of ":=" in any '
        "fix->/err->/sup-> step, sorted alphabetically (exclude un-superseded replacement IDs and d* nodes; [] if none)."
    )
    episodes: List[MultiTurnEpisode] = []

    for idx in range(count):
        ep_id, cond = f"gb-{idx + 1:04d}", cond_cycle[idx % 8]
        grp = idx // 8
        t1_preserved: Tuple[Tuple[str, str], ...] = ()
        t2_preserved: Tuple[Tuple[str, str], ...] = ()
        t2_notes_extra = ""
        seq_ref = 100 + (idx % 90)
        lag_ms = 8 + (idx % 17)

        if cond == "A":
            unit = ("km", "g", "L", "s", "M_USD")[(idx // 2) % 5]
            t1_unit, t2_unit = unit, unit
            p1, p2, p3, p4 = rng.randint(20, 90), rng.randint(15, 75), rng.randint(2, 5), rng.randint(10, 60)
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
                f"[Status Note 1: Session heartbeat nominal; zero upstream premise amendments queued.]\n"
                f"[Status Note 2: Checkpoint buffer seq_ref = {seq_ref} verified ({lag_ms} ms).]\n"
            )
        elif cond == "B":
            unit = ("M_USD", "g", "km", "L")[grp % 4]
            t1_unit, t2_unit = unit, unit
            p1, p2, p3, p4 = rng.randint(30, 120), rng.randint(20, 90), rng.randint(2, 5), rng.randint(15, 75)
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
            for attempt in range(16):
                cand_t2 = (
                    TurnSpec("p1_r1", "retract", base_r1 + 17 * attempt, unit, retraction_op="fix->", target_id="p1"),
                    TurnSpec("p1_r2", "retract", base_r2 + 29 * attempt, unit, retraction_op="sup->", target_id="p1_r1"),
                    TurnSpec("s2", "premise", s2, "scalar"),
                    TurnSpec("d5", "derive", arith_op="add", left_id="d4", right_id="d2"),
                    TurnSpec("d6", "derive", arith_op="mul", left_id="d5", right_id="s2"),
                )
                if _all_multiturn_primary_stales_nondegenerate(turn1_turns, cand_t2, "d6", t2_unit):
                    turn2_turns = cand_t2
                    break
            assert turn2_turns, f"Failed v3.0 non-degeneracy on {ep_id}"
        elif cond == "C":
            unit = ("g", "km", "M_USD", "L")[grp % 4]
            dist_u = ("km", "M_USD", "L", "g")[grp % 4]
            t1_unit, t2_unit = unit, unit
            p1, p2, p3, p4 = rng.randint(30, 110), rng.randint(2, 5), rng.randint(25, 100), rng.randint(15, 70)
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
            for attempt in range(16):
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
                if _all_multiturn_primary_stales_nondegenerate(turn1_turns, cand_t2, "d6", t2_unit):
                    turn2_turns = cand_t2
                    break
            assert turn2_turns, f"Failed v3.0 non-degeneracy on {ep_id}"
        elif cond == "D":
            fam = grp % 5
            alt = (grp // 5) % 2
            if fam == 0:
                t1_unit, r_unit, t2_unit = "mg", "kg", "g"
                v1, v2, v4 = 100 * rng.randint(2, 9), 100 * rng.randint(2, 8), 100 * rng.randint(1, 3)
                rv1, step_r = rng.randint(1, 4), 1
            elif fam == 1:
                t1_unit, r_unit, t2_unit = "cm", "km", "m"
                v1, v2, v4 = 20 * rng.randint(2, 9), 20 * rng.randint(2, 7), 20 * rng.randint(1, 3)
                rv1, step_r = rng.randint(1, 4), 1
            elif fam == 2:
                if alt == 0:
                    t1_unit, r_unit, t2_unit = "mL", "L", "uL"
                    v1, v2, v4 = 100 * rng.randint(2, 8), 100 * rng.randint(1, 6), 50 * rng.randint(1, 4)
                    rv1, step_r = rng.randint(2, 6), 1
                else:
                    t1_unit, r_unit, t2_unit = "uL", "mL", "L"
                    v1, v2, v4 = 100 * rng.randint(2, 8), 100 * rng.randint(1, 6), 50 * rng.randint(1, 4)
                    rv1, step_r = rng.randint(2, 6), 1
            elif fam == 3:
                t1_unit, r_unit, t2_unit = "s", ("h" if alt == 0 else "min"), ("min" if alt == 0 else "ms")
                v1, v2, v4 = 120 * rng.randint(1, 4), 120 * rng.randint(1, 3), 60 * rng.randint(1, 3)
                rv1, step_r = (rng.randint(1, 3) if alt == 0 else rng.randint(12, 30)), 1
            else:
                t1_unit, r_unit, t2_unit = "K_USD", "M_USD", ("USD" if alt == 0 else "B_USD")
                v1, v2, v4 = 100 * rng.randint(2, 8), 100 * rng.randint(1, 6), 50 * rng.randint(1, 4)
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
            for attempt in range(16):
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
                if _all_multiturn_primary_stales_nondegenerate(turn1_turns, cand_t2, "d6", t2_unit):
                    turn2_turns = cand_t2
                    break
            assert turn2_turns, f"Failed v3.0 non-degeneracy on {ep_id}"
        else:
            unit, indep_u = (("M_USD", "g"), ("g", "km"), ("km", "L"), ("L", "M_USD"))[grp % 4]
            t1_unit, t2_unit = unit, unit
            t1_preserved = (("d_indep", indep_u),)
            t2_preserved = (("d_indep", indep_u),)
            p1, p2, p3, p4 = rng.randint(30, 120), rng.randint(20, 90), rng.randint(2, 5), rng.randint(15, 75)
            u1, u2, u3 = rng.randint(25, 130), rng.randint(15, 85), rng.randint(2, 5)
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
            for attempt in range(16):
                cand_t2 = (
                    TurnSpec("u_near", "premise", u_near_v, unit),
                    TurnSpec("p1_r1", "retract", base_r1 + 17 * attempt, unit, retraction_op="fix->", target_id="p1"),
                    TurnSpec("p1_r2", "retract", base_r2 + 29 * attempt, unit, retraction_op="sup->", target_id="p1_r1"),
                    TurnSpec("s2", "premise", s2, "scalar"),
                    TurnSpec("d5", "derive", arith_op="add", left_id="d4", right_id="d2"),
                    TurnSpec("d6", "derive", arith_op="mul", left_id="d5", right_id="s2"),
                )
                if _all_multiturn_primary_stales_nondegenerate(turn1_turns, cand_t2, "d6", t2_unit):
                    turn2_turns = cand_t2
                    break
            assert turn2_turns, f"Failed v3.0 non-degeneracy on {ep_id}"

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
            f"Compute checkpoint '{'d4'}' in '{t1_unit}'. Output ONLY standalone JSON: "
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

        episodes.append(
            MultiTurnEpisode(
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
        )
    return episodes
