"""Clean-room Arm 3A: Turn-2 deterministic dependency-ledger hook and invariant guard.

Licensed under the Apache License, Version 2.0.
Contains zero lines and zero imports from any private engine.

Architecture & Evaluation Hierarchy (Classical JTMS / AGM Inclusion & Preservation Guard, [안전]):
1. Extracts structured Turn-1 node definitions (`premise` / `derive`) and Turn-2 incremental
   amendments (`fix->`, `err->`, `sup->`, `premise`, `derive`) from user turn transcripts
   (or explicit `TurnSpec` tuples) plus the model's committed Turn-1 JSON envelope (`state_v1`).
2. Builds the exact directed dependency ledger (`ancestors(v)`) and superseded target set `R`
   (unique node IDs appearing directly on the left of `:=` in any `fix-> / err-> / sup->` step).
3. Primary `K2` Path (`Arm 3A-Structural-Prompt` + co-logged `mode="structural_only"`):
   - Pre-inference `format_turn2_ledger_directive` / `augment_turn2_prompt_with_ledger` injects ONLY
     zero-numeric-leakage structural lineage: `R`, `TAINTED` vs `UNTAINTED` Turn-1 node ID sets,
     structural replacement chain IDs (`p1 -> p1_r1 -> p1_r2`, with ZERO pre-computed numeric values),
     and the model's own committed Turn-1 `state_v1` `preserved` lock when untainted.
   - Raw Turn-2 LLM output is scored directly for primary `K2`, alongside `apply_arm3a_invariant_guard`
     with default `mode="structural_only"` (`I1..I3` envelope/retracted_ids/`state_v1` preserved hygiene,
     never overwriting the model's Turn-2 numeric `value` or `unit`).
4. Secondary Symbolic-Offload Ceiling Only (`mode="full"`):
   - `mode="full"` (`I1..I4`) additionally substitutes verified rational target values and is strictly
     relegated to a secondary neuro-symbolic offload upper-bound reference table (never used to claim
     primary `K2` LLM belief-revision superiority).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import json
import re
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

from .oracle import (
    RETRACTION_OPERATORS,
    UNIT_TABLE,
    DimensionMismatchError,
    EnvelopeParseResult,
    ExactRationalOracle,
    TurnSpec,
    convert_from_base_fraction,
    mixed_tolerance_match,
    normalize_to_base_fraction,
    response_adapter,
    to_exact_fraction,
)

# Structured turn-line parsers for Turn 1 / Turn 2 user prompt transcripts (`Turn k.i [id]: ...` or `Turn i [id]: ...`)
_TURN_HEADER_RE = re.compile(r"^\s*Turn\s+\d+(?:\.\d+)?\s*\[([A-Za-z_][A-Za-z0-9_]*)\]\s*:\s*(.+?)\s*$")
_PREMISE_BODY_RE = re.compile(r"^value\s*=\s*([^\s]+)\s+([^\s]+)$")
_DERIVE_BODY_RE = re.compile(
    r"^(add|sub|mul|div)\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)$"
)
_RETRACT_BODY_RE = re.compile(
    r"^(fix->|err->|sup->)\s+([A-Za-z_][A-Za-z0-9_]*)\s*:=\s*([^\s]+)\s+([^\s]+)$"
)
_TARGET_UNIT_PROMPT_RE = re.compile(
    r"Compute\s+(?:the\s+final\s+value\s+of\s+node|checkpoint)\s+'([A-Za-z_][A-Za-z0-9_]*)'\s+in\s+(?:unit\s+)?'([^']+)'"
)
_PRESERVED_SCHEMA_RE = re.compile(
    r'"preserved"\s*:\s*\{\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*\{\s*"value"\s*:\s*<number>\s*,\s*"unit"\s*:\s*"([^"]+)"\s*\}\s*\}'
)
_JSON_OBJECT_SPAN_RE = re.compile(r"\{[\s\S]*\}")


@dataclass(frozen=True)
class DependencyNodeBinding:
    """Verified dependency-ledger entry for a premise, retraction, or derived node."""

    node_id: str
    turn_origin: int  # 1 or 2
    kind: str  # "premise" | "retract" | "derive"
    effective_node_id: str
    ancestors: Tuple[str, ...]
    tainted: bool
    base_value: Fraction
    dimension: str
    base_unit: str
    declared_unit: str


@dataclass(frozen=True)
class CommittedTurn1State:
    """Extracted commitment from the model's Turn-1 assistant response (`state_v1`)."""

    parse_ok: bool
    step_id: str
    committed_value: Optional[Fraction]
    committed_unit: str
    committed_retracted_ids: Tuple[str, ...]
    committed_preserved: Dict[str, Tuple[Fraction, str]]
    matches_turn1_truth: bool
    preserved_matches_turn1_truth: Dict[str, bool]


@dataclass(frozen=True)
class Arm3ALedgerState:
    """Deterministic dependency-ledger and invariant snapshot across Turn 1 and Turn 2."""

    turn1_turns: Tuple[TurnSpec, ...]
    turn2_turns: Tuple[TurnSpec, ...]
    target_node_id: str
    expected_unit: str
    preserved_targets: Tuple[Tuple[str, str], ...]
    superseded_retracted_ids: Tuple[str, ...]
    replacement_map: Dict[str, str]
    tainted_turn1_ids: Tuple[str, ...]
    untainted_turn1_ids: Tuple[str, ...]
    node_bindings: Dict[str, DependencyNodeBinding]
    target_truth_fraction: Fraction
    target_dimension: str
    pre_retraction_target_fraction: Optional[Fraction]
    preserved_truth_bindings: Dict[str, Tuple[Fraction, str]]
    committed_turn1: CommittedTurn1State


@dataclass(frozen=True)
class Arm3AGuardDecision:
    """Result of applying the Arm 3A invariant guard to a Turn-2 candidate response."""

    emitted_json: str
    raw_parse_ok: bool
    intervened: bool
    violations_blocked: Tuple[str, ...]
    retracted_ids_verified: Tuple[str, ...]
    target_value_fraction: Fraction
    target_unit: str
    preserved_verified: Dict[str, Tuple[Fraction, str]]


def format_exact_json_number(frac: Fraction) -> int | float:
    """Convert an exact Fraction into an int or clean decimal float for standalone JSON emission."""
    if frac.denominator == 1:
        return int(frac.numerator)
    # Check if denominator is 2^a * 5^b for exact terminating decimal representation
    den = frac.denominator
    pow2 = 0
    pow5 = 0
    while den % 2 == 0:
        den //= 2
        pow2 += 1
    while den % 5 == 0:
        den //= 5
        pow5 += 1
    if den == 1:
        scale = max(pow2, pow5)
        scaled_num = frac.numerator * (2 ** (scale - pow2)) * (5 ** (scale - pow5))
        dec = Decimal(scaled_num).scaleb(-scale)
        as_float = float(dec)
        if as_float.is_integer():
            return int(as_float)
        return as_float
    return float(frac)


def extract_turns_from_prompt(prompt_text: str) -> Tuple[TurnSpec, ...]:
    """Deterministically parse structured `Turn ... [node_id]: ...` lines from a prompt string."""
    if not isinstance(prompt_text, str):
        raise ValueError("Prompt text must be a string.")
    extracted: List[TurnSpec] = []
    for raw_line in prompt_text.splitlines():
        m_hdr = _TURN_HEADER_RE.match(raw_line)
        if not m_hdr:
            continue
        node_id, body = m_hdr.group(1), m_hdr.group(2)
        m_prem = _PREMISE_BODY_RE.match(body)
        if m_prem:
            val_tok, unit_tok = m_prem.group(1), m_prem.group(2)
            val_frac = to_exact_fraction(val_tok)
            val_clean: Any = int(val_frac) if val_frac.denominator == 1 else val_tok
            extracted.append(TurnSpec(node_id=node_id, kind="premise", value=val_clean, unit=unit_tok))
            continue
        m_der = _DERIVE_BODY_RE.match(body)
        if m_der:
            arith_op, left_id, right_id = m_der.group(1), m_der.group(2), m_der.group(3)
            extracted.append(
                TurnSpec(
                    node_id=node_id,
                    kind="derive",
                    arith_op=arith_op,
                    left_id=left_id,
                    right_id=right_id,
                )
            )
            continue
        m_ret = _RETRACT_BODY_RE.match(body)
        if m_ret:
            ret_op, target_id, val_tok, unit_tok = (
                m_ret.group(1),
                m_ret.group(2),
                m_ret.group(3),
                m_ret.group(4),
            )
            if ret_op not in RETRACTION_OPERATORS:
                raise ValueError(f"Unsupported retraction operator {ret_op!r} in line {raw_line!r}")
            val_frac = to_exact_fraction(val_tok)
            val_clean = int(val_frac) if val_frac.denominator == 1 else val_tok
            extracted.append(
                TurnSpec(
                    node_id=node_id,
                    kind="retract",
                    value=val_clean,
                    unit=unit_tok,
                    retraction_op=ret_op,
                    target_id=target_id,
                )
            )
            continue
        raise ValueError(f"Unrecognized structured turn body in line: {raw_line!r}")
    return tuple(extracted)


def extract_prompt_contract(prompt_text: str) -> Tuple[str, str, Tuple[Tuple[str, str], ...]]:
    """Extract `(target_node_id, expected_unit, preserved_targets)` from a Turn-1 or Turn-2 prompt."""
    m_target = _TARGET_UNIT_PROMPT_RE.search(prompt_text)
    if not m_target:
        raise ValueError("Could not extract target node ID and expected unit from prompt.")
    target_id, expected_unit = m_target.group(1), m_target.group(2)
    preserved: List[Tuple[str, str]] = []
    for m_pres in _PRESERVED_SCHEMA_RE.finditer(prompt_text):
        preserved.append((m_pres.group(1), m_pres.group(2)))
    return target_id, expected_unit, tuple(preserved)


def _salvage_envelope_lenient(raw_text: str) -> EnvelopeParseResult:
    """Attempt strict `response_adapter` first; if trailing punctuation/fences exist, salvage JSON span."""
    strict = response_adapter(raw_text)
    if strict.ok:
        return strict
    if not isinstance(raw_text, str):
        return strict
    m = _JSON_OBJECT_SPAN_RE.search(raw_text.strip())
    if not m:
        return strict
    candidate = m.group(0).strip()
    salvaged = response_adapter(candidate)
    if salvaged.ok:
        return salvaged
    # Check if only invalid extra keys inside `preserved` or non-critical schema noise occurred
    try:
        obj = json.loads(candidate)
    except Exception:
        return strict
    if not isinstance(obj, dict):
        return strict
    cleaned: Dict[str, Any] = {}
    if isinstance(obj.get("notes"), str):
        cleaned["notes"] = obj["notes"]
    if isinstance(obj.get("step_id"), str) and obj["step_id"].strip():
        cleaned["step_id"] = obj["step_id"].strip()
    if "value" in obj:
        cleaned["value"] = obj["value"]
    if isinstance(obj.get("unit"), str) and obj["unit"].strip():
        cleaned["unit"] = obj["unit"].strip()
    if isinstance(obj.get("retracted_ids"), list):
        cleaned["retracted_ids"] = [x for x in obj["retracted_ids"] if isinstance(x, str) and x.strip()]
    else:
        cleaned["retracted_ids"] = []
    if isinstance(obj.get("preserved"), dict):
        valid_p: Dict[str, Any] = {}
        for k, v in obj["preserved"].items():
            if isinstance(k, str) and isinstance(v, dict) and set(v.keys()) == {"value", "unit"}:
                valid_p[k] = v
        if valid_p:
            cleaned["preserved"] = valid_p
    try:
        return response_adapter(json.dumps(cleaned))
    except Exception:
        return strict


def _compute_ancestor_closure(
    turns: Sequence[TurnSpec],
    replacements: Mapping[str, str],
) -> Dict[str, FrozenSet[str]]:
    """Compute full transitive structural ancestor set for every node ID across Turn 1 + Turn 2.

    Includes both original structural IDs (e.g., `p1`) and any replacement chain IDs (`p1_r1`, `p1_r2`)
    so that retracting `p1` or `p1_r1` deterministically taints every downstream consumer.
    """
    defs: Dict[str, TurnSpec] = {t.node_id: t for t in turns}

    def resolve_chain(nid: str) -> Set[str]:
        chain: Set[str] = {nid}
        cur = nid
        seen: Set[str] = set()
        while cur in replacements and cur not in seen:
            seen.add(cur)
            nxt = replacements[cur]
            chain.add(nxt)
            cur = nxt
        return chain

    memo: Dict[str, FrozenSet[str]] = {}
    visiting: Set[str] = set()

    def ancestors_of(nid: str) -> FrozenSet[str]:
        if nid in memo:
            return memo[nid]
        if nid in visiting:
            raise ValueError(f"Dependency cycle while computing ancestors at {nid!r}")
        visiting.add(nid)
        acc: Set[str] = resolve_chain(nid)
        # Walk backwards along any retraction target chain without re-calling ancestors_of on chain peers
        retract_frontier = list(acc)
        while retract_frontier:
            cur_id = retract_frontier.pop()
            rspec = defs.get(cur_id)
            if rspec is not None and rspec.kind == "retract" and rspec.target_id and rspec.target_id not in acc:
                acc.add(rspec.target_id)
                acc.update(resolve_chain(rspec.target_id))
                retract_frontier.append(rspec.target_id)
        for member_id in list(acc):
            spec = defs.get(member_id)
            if spec is not None and spec.kind == "derive":
                if spec.left_id:
                    acc.add(spec.left_id)
                    acc.update(ancestors_of(spec.left_id))
                if spec.right_id:
                    acc.add(spec.right_id)
                    acc.update(ancestors_of(spec.right_id))
        visiting.remove(nid)
        frozen = frozenset(acc)
        memo[nid] = frozen
        return frozen

    for t in turns:
        ancestors_of(t.node_id)
    return memo


def build_arm3a_ledger(
    turn1_turns: Sequence[TurnSpec],
    turn2_turns: Sequence[TurnSpec],
    target_node_id: str = "d6",
    expected_unit: Optional[str] = None,
    preserved_targets: Sequence[Tuple[str, str]] = (),
    turn1_raw_output: str = "",
    turn1_target_id: str = "d4",
    turn1_expected_unit: Optional[str] = None,
) -> Arm3ALedgerState:
    """Build the deterministic Turn-2 dependency ledger from Turn-1/Turn-2 specs and `turn1_raw_output`."""
    t1_tuple = tuple(turn1_turns)
    t2_tuple = tuple(turn2_turns)
    all_turns = t1_tuple + t2_tuple

    # Evaluate Turn-1 oracle state
    o1 = ExactRationalOracle()
    for t in t1_tuple:
        o1.append_turn(t)
    snap1 = o1.evaluate()

    # Evaluate Turn-1 + Turn-2 active oracle state
    o2 = ExactRationalOracle()
    for t in all_turns:
        o2.append_turn(t)
    snap2 = o2.evaluate()

    if target_node_id not in snap2.active_values_base:
        raise ValueError(f"Target node {target_node_id!r} is not active in Turn-2 ledger.")

    t2_base, t2_dim, t2_base_unit = snap2.active_values_base[target_node_id]
    out_unit = expected_unit or t2_base_unit
    target_truth_frac = convert_from_base_fraction(t2_base, t2_dim, out_unit)

    superseded_set: Set[str] = set(snap2.retracted_ids)
    superseded_sorted = tuple(sorted(superseded_set))

    # Pre-retraction value (if any retractions exist) to guard against STALE_STATE_RETENTION
    pre_ret_frac: Optional[Fraction] = None
    if superseded_sorted:
        o_unret = ExactRationalOracle()
        for t in all_turns:
            if t.kind != "retract":
                o_unret.append_turn(t)
        snap_unret = o_unret.evaluate()
        if target_node_id in snap_unret.active_values_base:
            ub, udim, _ = snap_unret.active_values_base[target_node_id]
            if udim == t2_dim:
                pre_ret_frac = convert_from_base_fraction(ub, udim, out_unit)

    ancestor_map = _compute_ancestor_closure(all_turns, snap2.replacement_map)

    def resolve_latest(nid: str) -> str:
        cur = nid
        seen: Set[str] = set()
        while cur in snap2.replacement_map and cur not in seen:
            seen.add(cur)
            cur = snap2.replacement_map[cur]
        return cur

    t1_ids = {t.node_id for t in t1_tuple}
    node_bindings: Dict[str, DependencyNodeBinding] = {}
    tainted_t1: List[str] = []
    untainted_t1: List[str] = []

    for t in all_turns:
        nid = t.node_id
        if nid not in snap2.active_values_base:
            continue
        bval, bdim, bunit = snap2.active_values_base[nid]
        anc = ancestor_map.get(nid, frozenset({nid}))
        # Proper ancestors or self intersecting superseded_set marks node as TAINTED
        is_tainted = bool(anc & superseded_set)
        origin = 1 if nid in t1_ids else 2
        decl_u = t.unit if t.kind in {"premise", "retract"} else bunit
        binding = DependencyNodeBinding(
            node_id=nid,
            turn_origin=origin,
            kind=t.kind,
            effective_node_id=resolve_latest(nid),
            ancestors=tuple(sorted(anc)),
            tainted=is_tainted,
            base_value=bval,
            dimension=bdim,
            base_unit=bunit,
            declared_unit=decl_u,
        )
        node_bindings[nid] = binding
        if origin == 1 and t.kind == "derive":
            if is_tainted:
                tainted_t1.append(nid)
            else:
                untainted_t1.append(nid)

    # Parse and audit Turn-1 committed model state (`state_v1`)
    env1 = _salvage_envelope_lenient(turn1_raw_output) if turn1_raw_output else EnvelopeParseResult(
        ok=False, status="EMPTY"
    )
    t1_matches = False
    pres_t1_matches: Dict[str, bool] = {}
    committed_pres: Dict[str, Tuple[Fraction, str]] = {
        k: (v, u) for k, v, u in env1.preserved
    }

    if env1.ok and env1.value_fraction is not None and turn1_target_id in snap1.active_values_base:
        t1_b, t1_d, t1_bu = snap1.active_values_base[turn1_target_id]
        t1_u = turn1_expected_unit or env1.unit or t1_bu
        try:
            c1_b, c1_d, _ = normalize_to_base_fraction(env1.value_fraction, env1.unit)
            if c1_d == t1_d:
                c1_disp = convert_from_base_fraction(c1_b, c1_d, t1_u)
                t1_disp = convert_from_base_fraction(t1_b, t1_d, t1_u)
                t1_matches = mixed_tolerance_match(c1_disp, t1_disp)[0] and env1.retracted_ids == tuple(
                    sorted(snap1.retracted_ids)
                )
        except DimensionMismatchError:
            t1_matches = False

    # Compute verified preserved bindings:
    # For any UNTAINTED preserved node (`d_indep`), if Turn-1 committed a valid verified value,
    # lock that exact Turn-1 commitment; otherwise compute exact untainted truth from Turn-1 graph.
    pres_truth_bindings: Dict[str, Tuple[Fraction, str]] = {}
    for pid, punit in preserved_targets:
        if pid not in snap2.active_values_base:
            raise ValueError(f"Preserved target {pid!r} missing from active DAG.")
        pb, pdim, _ = snap2.active_values_base[pid]
        ptruth = convert_from_base_fraction(pb, pdim, punit)
        p_ok = False
        chosen_frac = ptruth
        if pid in committed_pres:
            cp_val, cp_unit = committed_pres[pid]
            try:
                cp_b, cp_d, _ = normalize_to_base_fraction(cp_val, cp_unit)
                if cp_d == pdim:
                    cp_in_pu = convert_from_base_fraction(cp_b, cp_d, punit)
                    if mixed_tolerance_match(cp_in_pu, ptruth)[0]:
                        p_ok = True
                        # Preserve verbatim Turn-1 committed value when unit matches requested preserved unit
                        if cp_unit == punit:
                            chosen_frac = cp_val
                        else:
                            chosen_frac = cp_in_pu
            except DimensionMismatchError:
                p_ok = False
        pres_t1_matches[pid] = p_ok
        pres_truth_bindings[pid] = (chosen_frac, punit)

    committed_state = CommittedTurn1State(
        parse_ok=env1.ok,
        step_id=env1.step_id,
        committed_value=env1.value_fraction,
        committed_unit=env1.unit,
        committed_retracted_ids=env1.retracted_ids,
        committed_preserved=committed_pres,
        matches_turn1_truth=t1_matches,
        preserved_matches_turn1_truth=pres_t1_matches,
    )

    return Arm3ALedgerState(
        turn1_turns=t1_tuple,
        turn2_turns=t2_tuple,
        target_node_id=target_node_id,
        expected_unit=out_unit,
        preserved_targets=tuple(preserved_targets),
        superseded_retracted_ids=superseded_sorted,
        replacement_map=dict(snap2.replacement_map),
        tainted_turn1_ids=tuple(tainted_t1),
        untainted_turn1_ids=tuple(untainted_t1),
        node_bindings=node_bindings,
        target_truth_fraction=target_truth_frac,
        target_dimension=t2_dim,
        pre_retraction_target_fraction=pre_ret_frac,
        preserved_truth_bindings=pres_truth_bindings,
        committed_turn1=committed_state,
    )


def build_arm3a_ledger_from_prompts(
    turn1_prompt: str,
    turn2_prompt: str,
    turn1_raw_output: str = "",
) -> Arm3ALedgerState:
    """Construct `Arm3ALedgerState` directly from raw Turn-1 and Turn-2 prompt strings."""
    t1_turns = extract_turns_from_prompt(turn1_prompt)
    t2_turns = extract_turns_from_prompt(turn2_prompt)
    t1_target, t1_unit, _ = extract_prompt_contract(turn1_prompt)
    t2_target, t2_unit, t2_preserved = extract_prompt_contract(turn2_prompt)
    return build_arm3a_ledger(
        turn1_turns=t1_turns,
        turn2_turns=t2_turns,
        target_node_id=t2_target,
        expected_unit=t2_unit,
        preserved_targets=t2_preserved,
        turn1_raw_output=turn1_raw_output,
        turn1_target_id=t1_target,
        turn1_expected_unit=t1_unit,
    )


def format_turn2_ledger_directive(ledger: Arm3ALedgerState) -> str:
    """Format a zero-numeric-leakage structural dependency-ledger block before Turn-2 generation.

    Strictly enforces anti-leakage for primary `K2` validity:
    - Emits ZERO pre-computed Turn-2 numeric values, ZERO rebuilt Turn-1 derived numbers, and ZERO
      target `d6` answers.
    - Passes ONLY structural graph lineage (`R`, `TAINTED` vs `UNTAINTED` Turn-1 node ID sets,
      symbolic replacement chain IDs `orig -> latest_id` without numbers) plus the model's own
      committed Turn-1 `state_v1` `preserved` binding when untainted.
    """
    ret_json = json.dumps(list(ledger.superseded_retracted_ids))
    lines: List[str] = [
        "[Arm3A Structural Dependency-Ledger Guard]",
        f"- Verified superseded retracted_ids R (exact LHS of ':=' only; exclude d* and active replacements): {ret_json}",
        f"- Tainted Turn-1 derived node IDs (ancestors intersect R; recompute from active premises): {list(ledger.tainted_turn1_ids)}",
        f"- Untainted Turn-1 derived node IDs (disjoint from R; keep Turn-1 commitment unchanged): {list(ledger.untainted_turn1_ids)}",
    ]

    # Symbolic replacement chain pointers ONLY (zero numeric values)
    if ledger.replacement_map:
        seen_roots: Set[str] = set()
        chain_parts: List[str] = []
        for orig_id in sorted(ledger.replacement_map):
            # Only emit root-to-effective pointer chains
            if orig_id in ledger.node_bindings:
                eff_id = ledger.node_bindings[orig_id].effective_node_id
            else:
                cur = orig_id
                visited: Set[str] = set()
                while cur in ledger.replacement_map and cur not in visited:
                    visited.add(cur)
                    cur = ledger.replacement_map[cur]
                eff_id = cur
            pair_key = f"{orig_id}->{eff_id}"
            if pair_key not in seen_roots:
                seen_roots.add(pair_key)
                chain_parts.append(f"{orig_id} -> {eff_id}")
        if chain_parts:
            lines.append("- Active symbolic replacement pointers (recompute downstream arithmetic yourself): " + "; ".join(chain_parts))

    # Lock ONLY the model's own committed Turn-1 `state_v1` preserved entry when untainted (zero oracle leakage)
    if ledger.preserved_targets:
        untainted_set = set(ledger.untainted_turn1_ids)
        state_v1_pres: Dict[str, Dict[str, Any]] = {}
        unresolved_pres: List[str] = []
        for pid, punit in ledger.preserved_targets:
            if pid in untainted_set and pid in ledger.committed_turn1.committed_preserved:
                c_val, c_unit = ledger.committed_turn1.committed_preserved[pid]
                state_v1_pres[pid] = {"value": format_exact_json_number(c_val), "unit": c_unit}
            else:
                unresolved_pres.append(f"{pid} ({punit})")
        if state_v1_pres:
            lines.append("- Untainted Turn-1 state_v1 preserved commitment lock: " + json.dumps(state_v1_pres))
        if unresolved_pres:
            lines.append(
                "- Untainted preserved nodes (re-derive from Turn-1 premises without tainting): "
                + ", ".join(unresolved_pres)
            )
        pres_clause = 'and "preserved" object matching the untainted lock'
    else:
        pres_clause = 'and DO NOT include a "preserved" key (omit "preserved" completely)'

    lines.append(
        f"Invariant: compute '{ledger.target_node_id}' in '{ledger.expected_unit}' yourself and emit ONLY "
        f"standalone JSON with exact retracted_ids R ({pres_clause}) and zero trailing punctuation."
    )
    return "\n".join(lines)


def augment_turn2_prompt_with_ledger(turn2_prompt: str, ledger: Arm3ALedgerState) -> str:
    """Append the zero-leakage structural Arm 3A dependency-ledger block to `turn2_prompt`."""
    directive = format_turn2_ledger_directive(ledger)
    return f"{turn2_prompt.rstrip()}\n\n{directive}"


def apply_arm3a_invariant_guard(
    ledger: Arm3ALedgerState,
    raw_turn2_output: str,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-5,
    mode: str = "structural_only",
) -> Arm3AGuardDecision:
    """Enforce deterministic Turn-2 DAG invariants before final JSON emission.

    Evaluation Hierarchy & Modes:
    - `"structural_only"` (default, Primary `K2` Co-Logged Structural Guard):
      Enforces `I1` (envelope/format hygiene), `I2` (exact superseded `retracted_ids`), and `I3`
      (untainted `state_v1` `preserved` commitment lock when committed in Turn 1, stripping spurious
      `preserved` keys on Condition A..D), while NEVER overwriting the model's Turn-2 numeric `value`
      or `unit` (and never injecting oracle `preserved` numbers when `state_v1` had no commitment).
    - `"full"` (Secondary Symbolic-Offload Ceiling ONLY — NEVER used for primary `K2` claims):
      Enforces all 4 invariants (`I1..I4`), substituting verified rational target and preserved bindings
      from the symbolic DAG evaluator as an explicit neuro-symbolic upper-bound reference.

    Invariants:
    - I1 (Envelope & Format Hygiene): Strips trailing periods/fences and rejects extra `preserved` keys
      on non-Condition-E episodes (e.g., Condition A `preserved` hallucination or Mistral trailing `.`).
    - I2 (AGM Inclusion / Zero Over-Retraction): `retracted_ids` MUST equal `ledger.superseded_retracted_ids`
      (`[]` on un-retracted Condition A; never derived `d*` IDs or un-superseded `p1_r2` replacement IDs).
    - I3 (AGM Preservation / Disjoint `d_indep` Lock): Untainted `preserved` subgraphs (`d_indep` in
      Condition E) are locked to `state_v1` in `structural_only` mode (or exact DAG truth in `full` ceiling mode).
    - I4 (Symbolic-Offload Ceiling Target Binding, `mode="full"` only): `step_id`, `unit` dimension, and
      converted `value` match the post-retraction rational DAG truth within `(abs_tol, rel_tol)`.
    """
    if mode not in {"structural_only", "full"}:
        raise ValueError(f"Unsupported Arm 3A guard mode {mode!r}; expected 'structural_only' or 'full'.")
    req_pres_keys = tuple(k for k, _ in ledger.preserved_targets)
    strict_env = response_adapter(raw_turn2_output, required_preserved=req_pres_keys)
    lenient_env = _salvage_envelope_lenient(raw_turn2_output)

    violations: List[str] = []
    notes_text = "arm3a verified ledger emission"

    # Extract original notes if present
    if isinstance(raw_turn2_output, str):
        m_span = _JSON_OBJECT_SPAN_RE.search(raw_turn2_output.strip())
        if m_span:
            try:
                parsed_raw_dict = json.loads(m_span.group(0))
                if isinstance(parsed_raw_dict, dict) and isinstance(parsed_raw_dict.get("notes"), str):
                    notes_text = parsed_raw_dict["notes"]
            except Exception:
                pass

    if not strict_env.ok:
        violations.append(f"I1_ENVELOPE_FORMAT:{strict_env.reason}")

    env = strict_env if strict_env.ok else lenient_env

    # Check I1 extra preserved keys on Condition A..D (where preserved_targets is empty)
    if not ledger.preserved_targets and env.ok and env.preserved:
        violations.append("I3_UNEXPECTED_PRESERVED_ON_NON_E")

    # Check step_id
    if not env.ok or env.step_id != ledger.target_node_id:
        violations.append(f"I1_STEP_ID_MISMATCH:{env.step_id!r}!={ledger.target_node_id!r}")

    # Check I2: retracted_ids exact equality with superseded LHS targets
    if not env.ok or env.retracted_ids != ledger.superseded_retracted_ids:
        extra_ret = sorted(set(env.retracted_ids) - set(ledger.superseded_retracted_ids)) if env.ok else []
        missing_ret = sorted(set(ledger.superseded_retracted_ids) - set(env.retracted_ids)) if env.ok else []
        if not ledger.superseded_retracted_ids and extra_ret:
            violations.append(f"I2_COND_A_FALSE_RETRACTION:{extra_ret}")
        elif extra_ret:
            violations.append(f"I2_SPURIOUS_OVER_RETRACTION:{extra_ret}")
        if missing_ret or not env.ok:
            violations.append(f"I2_MISSING_SUPERSEDED_IDS:{missing_ret}")

    # Check I3: Condition E untainted preserved `d_indep` lock
    # In `structural_only` mode, compare/lock strictly against the model's own Turn-1 `state_v1` commitment
    # (never injecting oracle-computed numbers if `state_v1` lacked a commitment).
    untainted_set = set(ledger.untainted_turn1_ids)
    obs_pres = {k: (v, u) for k, v, u in env.preserved} if env.ok else {}
    if ledger.preserved_targets:
        extra_pkeys = sorted(set(obs_pres.keys()) - set(req_pres_keys))
        if extra_pkeys:
            violations.append(f"I3_SPURIOUS_PRESERVED_KEYS:{extra_pkeys}")
        for pid, punit in ledger.preserved_targets:
            if mode == "structural_only":
                if pid in untainted_set and pid in ledger.committed_turn1.committed_preserved:
                    ref_pval, ref_punit = ledger.committed_turn1.committed_preserved[pid]
                else:
                    if pid not in obs_pres:
                        violations.append(f"I3_MISSING_PRESERVED:{pid}")
                    continue
            else:
                ref_pval, ref_punit = ledger.preserved_truth_bindings[pid]

            if pid not in obs_pres:
                violations.append(f"I3_MISSING_PRESERVED:{pid}")
                continue
            cand_pv, cand_pu = obs_pres[pid]
            try:
                cp_b, cp_dim, _ = normalize_to_base_fraction(cand_pv, cand_pu)
                tp_b, tp_dim, _ = normalize_to_base_fraction(ref_pval, ref_punit)
                if cp_dim != tp_dim:
                    violations.append(f"I3_PRESERVED_DIM_TAMPERED:{pid}")
                else:
                    cp_in_pu = convert_from_base_fraction(cp_b, cp_dim, ref_punit)
                    if not mixed_tolerance_match(cp_in_pu, ref_pval, abs_tol=abs_tol, rel_tol=rel_tol)[0]:
                        violations.append(f"I3_PRESERVED_VALUE_TAMPERED:{pid}")
            except DimensionMismatchError:
                violations.append(f"I3_PRESERVED_UNIT_INVALID:{pid}")

    # Check I4: Target value & SI/USD dimensional binding (blocking stale Turn-1 retention & arithmetic drift)
    final_val_frac = ledger.target_truth_fraction
    final_unit = ledger.expected_unit
    value_valid = False
    if env.ok and env.value_fraction is not None:
        try:
            cb, cdim, _ = normalize_to_base_fraction(env.value_fraction, env.unit)
            if cdim != ledger.target_dimension:
                violations.append(f"I4_TARGET_DIM_MISMATCH:{cdim}!={ledger.target_dimension}")
            else:
                c_disp = convert_from_base_fraction(cb, cdim, ledger.expected_unit)
                if mixed_tolerance_match(
                    c_disp, ledger.target_truth_fraction, abs_tol=abs_tol, rel_tol=rel_tol
                )[0]:
                    value_valid = True
                    final_val_frac = env.value_fraction
                    final_unit = env.unit
                else:
                    if (
                        ledger.pre_retraction_target_fraction is not None
                        and mixed_tolerance_match(
                            c_disp, ledger.pre_retraction_target_fraction, abs_tol=abs_tol, rel_tol=rel_tol
                        )[0]
                    ):
                        violations.append("I4_STALE_STATE_BLOCKED")
                    else:
                        violations.append("I4_TARGET_VALUE_MISMATCH")
        except DimensionMismatchError:
            violations.append(f"I4_UNKNOWN_UNIT:{env.unit!r}")
    else:
        violations.append("I4_MISSING_TARGET_VALUE")

    # Fast path: zero violations -> pass through exact raw JSON untouched
    active_violations = (
        [v for v in violations if not v.startswith("I4_")]
        if mode == "structural_only" and env.ok and env.value_fraction is not None
        else violations
    )
    if not active_violations and strict_env.ok and (value_valid or mode == "structural_only"):
        pres_clean = {k: (v, u) for k, v, u in strict_env.preserved}
        return Arm3AGuardDecision(
            emitted_json=raw_turn2_output.strip(),
            raw_parse_ok=True,
            intervened=False,
            violations_blocked=(),
            retracted_ids_verified=ledger.superseded_retracted_ids,
            target_value_fraction=env.value_fraction if mode == "structural_only" and env.value_fraction is not None else final_val_frac,
            target_unit=env.unit if mode == "structural_only" and env.unit else final_unit,
            preserved_verified=pres_clean,
        )

    # In `structural_only` mode, if Turn-2 failed to parse a numeric value/unit at all, do NOT fabricate one:
    # return the raw output so `PARSE_REJECT` is scored honestly without oracle replacement.
    if mode == "structural_only" and (not env.ok or env.value_fraction is None or not env.unit):
        return Arm3AGuardDecision(
            emitted_json=raw_turn2_output.strip() if isinstance(raw_turn2_output, str) else "",
            raw_parse_ok=strict_env.ok,
            intervened=False,
            violations_blocked=tuple(active_violations),
            retracted_ids_verified=ledger.superseded_retracted_ids,
            target_value_fraction=Fraction(0, 1),
            target_unit="",
            preserved_verified={},
        )

    use_model_value = (
        (mode == "structural_only" and env.ok and env.value_fraction is not None and bool(env.unit))
        or value_valid
    )
    chosen_val_frac = (
        env.value_fraction
        if (mode == "structural_only" and env.ok and env.value_fraction is not None)
        else (final_val_frac if value_valid else ledger.target_truth_fraction)
    )
    chosen_unit = (
        env.unit
        if (mode == "structural_only" and env.ok and env.unit)
        else (final_unit if value_valid else ledger.expected_unit)
    )

    # Deterministic repair emission
    repaired_payload: Dict[str, Any] = {
        "notes": notes_text,
        "step_id": ledger.target_node_id,
        "value": format_exact_json_number(chosen_val_frac),
        "unit": chosen_unit,
        "retracted_ids": list(ledger.superseded_retracted_ids),
    }
    verified_pres_out: Dict[str, Tuple[Fraction, str]] = {}
    if ledger.preserved_targets:
        pres_dict: Dict[str, Dict[str, Any]] = {}
        missing_uncommitted_pres = False
        for pid, punit in ledger.preserved_targets:
            if mode == "structural_only":
                if pid in untainted_set and pid in ledger.committed_turn1.committed_preserved:
                    pval, pu = ledger.committed_turn1.committed_preserved[pid]
                elif pid in obs_pres:
                    pval, pu = obs_pres[pid]
                else:
                    missing_uncommitted_pres = True
                    break
            else:
                pval, pu = ledger.preserved_truth_bindings[pid]
            pres_dict[pid] = {"value": format_exact_json_number(pval), "unit": pu}
            verified_pres_out[pid] = (pval, pu)
        if missing_uncommitted_pres:
            return Arm3AGuardDecision(
                emitted_json=raw_turn2_output.strip() if isinstance(raw_turn2_output, str) else "",
                raw_parse_ok=strict_env.ok,
                intervened=False,
                violations_blocked=tuple(active_violations),
                retracted_ids_verified=ledger.superseded_retracted_ids,
                target_value_fraction=chosen_val_frac,
                target_unit=chosen_unit,
                preserved_verified={},
            )
        repaired_payload["preserved"] = pres_dict

    emitted_str = json.dumps(repaired_payload, ensure_ascii=False)
    # Self-verify through unmodified response_adapter before returning
    post_check = response_adapter(emitted_str, required_preserved=req_pres_keys)
    if not post_check.ok:
        raise RuntimeError(f"Arm 3A internal invariant failure: emitted JSON failed response_adapter: {post_check.reason}")

    return Arm3AGuardDecision(
        emitted_json=emitted_str,
        raw_parse_ok=strict_env.ok,
        intervened=True,
        violations_blocked=tuple(active_violations if use_model_value and mode == "structural_only" else violations),
        retracted_ids_verified=ledger.superseded_retracted_ids,
        target_value_fraction=chosen_val_frac,
        target_unit=chosen_unit,
        preserved_verified=verified_pres_out,
    )


__all__ = [
    "Arm3AGuardDecision",
    "Arm3ALedgerState",
    "CommittedTurn1State",
    "DependencyNodeBinding",
    "apply_arm3a_invariant_guard",
    "augment_turn2_prompt_with_ledger",
    "build_arm3a_ledger",
    "build_arm3a_ledger_from_prompts",
    "extract_prompt_contract",
    "extract_turns_from_prompt",
    "format_exact_json_number",
    "format_turn2_ledger_directive",
]
