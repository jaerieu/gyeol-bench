r"""Clean-room Arm 3B 2-pass deterministic Fraction/SI state-diff verifier and single-pass repair gate.

Licensed under the Apache License, Version 2.0.
Contains zero lines and zero imports from any private engine.

Design Overview (Arm 3B — 2-Pass Deterministic Fraction/SI State-Diff Verifier):
- Pass 1 (State-Diff Diagnostic Audit):
  1. Parses the Turn-1 user prompt (`parse_turn_prompt_specs`) and Turn-2 user prompt (`parse_turn_prompt_specs`)
     into canonical `TurnSpec` DAGs without accessing oracle ground-truth labels, and computes the exact rational
     Turn-1 state snapshot $S_1$, the Turn-2 active state snapshot $S_2$, the exact structural state-diff
     $\Delta S = S_2 \ominus S_1$, and the downward-closed / frozen-Turn-1 stale state lattice $\mathcal{L}_{\text{stale}}$.
  2. Audits the raw Turn-2 candidate JSON output (applying light envelope sanitation for trailing punctuation or
     markdown fences) against $\Delta S$ and $\mathcal{L}_{\text{stale}}$, flagging five orthogonal defect classes
     with ZERO ground-truth target-value leakage in diagnostic reasons:
     - `ENVELOPE_SYNTAX_DEFECT`: Trailing period (e.g., `}}}.`), markdown fence, or malformed/missing JSON keys.
     - `STALE_TURN1_BINDING`: Candidate value matches a stale pre-retraction or frozen Turn-1 (`d4`/`d2`/`d3`/`d1`)
       binding in $\mathcal{L}_{\text{stale}}$ when the target node's ancestor cone intersects retracted targets.
     - `SPURIOUS_OR_MISSING_RETRACTED_IDS`: Candidate `retracted_ids` diverges from the exact sorted set $R$ of
       superseded target IDs on the left of `:=` in `fix->`/`err->`/`sup->` steps (including derived `d*` leakage,
       un-superseded replacement IDs like `p1_r2`, omitted chained intermediates like `p1_r1`, or non-empty sets
       in zero-retraction Condition A).
     - `PRESERVED_SUBGRAPH_TAMPERING`: An untainted disjoint subgraph node (e.g., `d_indep` in Condition E) is
       missing, dimensionally altered, or numerically mutated relative to its committed Turn-1 (`state_v1`) state.
     - `SI_ARITH_RESIDUAL_MISMATCH`: Candidate target value or unit fails bottom-up SI / `bps -> %` consistency
       check across $\Delta S$ (reports structural/unit residual without revealing the pre-computed `d6` answer).
- Pass 2 & Evaluation Hierarchy (Strictly Separated per Audit `#12`):
  1. **Primary `K2` Evaluation Path (`mode="llm_repair"` / `mode="structural_gate"`)**:
     - `build_arm3b_repair_prompt`: Generates a concise, deterministic 1-turn LLM self-correction prompt citing
       ONLY structural/invariant violations (`flags`, `R`, `tainted_node_ids`, and the model's own committed Turn-1
       `preserved` lock when available). It NEVER leaks the pre-computed Turn-2 target numeric value (`d6`) or
       oracle-computed `d_indep` truth.
     - `run_arm3b_state_diff_verifier(..., mode="llm_repair", repaired_turn2_output=...)`: Evaluates Pass 1, and when
       flags fire, scores the LLM's 1-turn self-corrected output (or in `mode="structural_gate"` applies only
       non-numeric structural hygiene: syntax sanitation, exact $R$, and Turn-1 `state_v1` `preserved` lock while
       leaving the LLM's numeric target `value` untouched).
  2. **Secondary Symbolic-Offload Ceiling Only (`mode="symbolic_ceiling"` / `pass2_deterministic_repair`)**:
     - Deterministically projects the output onto the exact rational `Fraction`/SI state-diff evaluation.
     - Reported strictly as a secondary neuro-symbolic offload upper bound, NEVER as a primary `K2` belief-revision
       claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .oracle import (
    DimensionMismatchError,
    ExactRationalOracle,
    OracleStateSnapshot,
    RETRACTION_OPERATORS,
    TurnSpec,
    UNIT_TABLE,
    _compute_multiturn_stale_targets,
    _compute_stale_targets,
    convert_from_base_fraction,
    mixed_tolerance_match,
    normalize_to_base_fraction,
    response_adapter,
    to_exact_fraction,
)

# Regex patterns for parsing TurnSpec lines from user prompts without ground-truth metadata
_TURN_LINE_HEADER_RE = re.compile(
    r"^\s*Turn\s+\d+(?:\.\d+)?\s+\[(?P<node_id>[A-Za-z0-9_]+)\]:\s*(?P<body>.+?)\s*$"
)
_PREMISE_BODY_RE = re.compile(
    r"^value\s*=\s*(?P<val>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:/[+-]?[1-9]\d*)?)\s+(?P<unit>\S+)\s*$"
)
_DERIVE_BODY_RE = re.compile(
    r"^(?P<op>add|sub|mul|div)\(\s*(?P<left>[A-Za-z0-9_]+)\s*,\s*(?P<right>[A-Za-z0-9_]+)\s*\)\s*$"
)
_RETRACT_BODY_RE = re.compile(
    r"^(?P<rop>fix->|err->|sup->)\s+(?P<target>[A-Za-z0-9_]+)\s*:=\s*"
    r"(?P<val>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:/[+-]?[1-9]\d*)?)\s+(?P<unit>\S+)\s*$"
)
_TARGET_SPEC_RE = re.compile(
    r"Compute\s+(?:the\s+final\s+value\s+of\s+node|checkpoint)\s+'(?P<target>[A-Za-z0-9_]+)'\s+in\s+(?:unit\s+)?'(?P<unit>[^']+)'"
)
_PRESERVED_SCHEMA_RE = re.compile(
    r'"preserved"\s*:\s*\{\s*"(?P<pid>[A-Za-z0-9_]+)"\s*:\s*\{\s*"value"\s*:\s*<number>\s*,\s*"unit"\s*:\s*"(?P<punit>[^"]+)"\s*\}\s*\}'
)

VALID_ARM3B_MODES: frozenset[str] = frozenset({"llm_repair", "structural_gate", "symbolic_ceiling"})


@dataclass(frozen=True)
class StateDiffNodeDelta:
    """Exact rational state difference for a single node between Turn 1 and Turn 2."""

    node_id: str
    kind: str
    ancestors: Tuple[str, ...]
    tainted: bool
    turn1_base_fraction: Optional[Fraction]
    turn2_base_fraction: Fraction
    dimension: str
    base_unit: str

    @property
    def base_delta(self) -> Optional[Fraction]:
        if self.turn1_base_fraction is None:
            return None
        return self.turn2_base_fraction - self.turn1_base_fraction


@dataclass(frozen=True)
class StateDiffSnapshot:
    """Complete 2-turn exact rational/SI state difference computed in Pass 1."""

    turn1_snapshot: OracleStateSnapshot
    turn2_snapshot: OracleStateSnapshot
    node_deltas: Dict[str, StateDiffNodeDelta]
    expected_retracted_ids: Tuple[str, ...]
    tainted_node_ids: Tuple[str, ...]
    untainted_node_ids: Tuple[str, ...]
    target_node_id: str
    expected_unit: str
    truth_base_fraction: Fraction
    truth_target_fraction: Fraction
    truth_dimension: str
    preserved_targets: Tuple[Tuple[str, str], ...]
    preserved_truth_display: Dict[str, Tuple[Fraction, str]]
    stale_target_fractions: Tuple[Fraction, ...]


@dataclass(frozen=True)
class Pass1DiagnosticReport:
    """Diagnostic findings from Pass 1 of the Arm 3B state-diff verifier (zero target-answer leakage)."""

    clean: bool
    flags: Tuple[str, ...]
    sanitized_envelope_used: bool
    observed_step_id: str
    observed_value_fraction: Optional[Fraction]
    observed_unit: str
    observed_retracted_ids: Tuple[str, ...]
    spurious_retracted_ids: Tuple[str, ...]
    missing_retracted_ids: Tuple[str, ...]
    stale_binding_matched: bool
    matched_stale_fraction: Optional[Fraction]
    preserved_tampered_ids: Tuple[str, ...]
    arith_or_unit_mismatch: bool
    reasons: Tuple[str, ...]

    @property
    def flagged(self) -> bool:
        return not self.clean


@dataclass(frozen=True)
class Arm3BVerificationResult:
    """End-to-end 2-pass verification result for Arm 3B (`mode` distinguishes primary K2 from symbolic ceiling)."""

    mode: str
    is_primary_k2_eligible: bool
    pass1_report: Pass1DiagnosticReport
    repair_prompt: str
    repair_triggered: bool
    repair_converged: bool
    pass2_report: Pass1DiagnosticReport
    final_json: str
    final_value_fraction: Optional[Fraction]
    final_unit: str
    final_retracted_ids: Tuple[str, ...]


def parse_turn_prompt_specs(prompt_text: str) -> Tuple[Tuple[TurnSpec, ...], str, str, Dict[str, str]]:
    """Deterministically parse `TurnSpec` sequence, target node, expected unit, and preserved schema from a prompt."""
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError("Prompt text must be a non-empty string.")

    turns: List[TurnSpec] = []
    for raw_line in prompt_text.splitlines():
        m_hdr = _TURN_LINE_HEADER_RE.match(raw_line)
        if not m_hdr:
            continue
        node_id = m_hdr.group("node_id")
        body = m_hdr.group("body").strip()

        m_prem = _PREMISE_BODY_RE.match(body)
        if m_prem:
            val_str, unit = m_prem.group("val"), m_prem.group("unit")
            val_obj: Any = (
                int(val_str)
                if val_str.isdigit() or (val_str.startswith(("+", "-")) and val_str[1:].isdigit())
                else to_exact_fraction(val_str)
            )
            turns.append(TurnSpec(node_id=node_id, kind="premise", value=val_obj, unit=unit))
            continue

        m_der = _DERIVE_BODY_RE.match(body)
        if m_der:
            turns.append(
                TurnSpec(
                    node_id=node_id,
                    kind="derive",
                    arith_op=m_der.group("op"),
                    left_id=m_der.group("left"),
                    right_id=m_der.group("right"),
                )
            )
            continue

        m_ret = _RETRACT_BODY_RE.match(body)
        if m_ret:
            val_str, unit = m_ret.group("val"), m_ret.group("unit")
            val_obj = (
                int(val_str)
                if val_str.isdigit() or (val_str.startswith(("+", "-")) and val_str[1:].isdigit())
                else to_exact_fraction(val_str)
            )
            turns.append(
                TurnSpec(
                    node_id=node_id,
                    kind="retract",
                    value=val_obj,
                    unit=unit,
                    retraction_op=m_ret.group("rop"),
                    target_id=m_ret.group("target"),
                )
            )
            continue

        raise ValueError(f"Unrecognized turn specification line: {raw_line!r}")

    target_node_id = ""
    expected_unit = ""
    m_target = _TARGET_SPEC_RE.search(prompt_text)
    if m_target:
        target_node_id = m_target.group("target")
        expected_unit = m_target.group("unit")

    preserved_targets: Dict[str, str] = {}
    for m_pres in _PRESERVED_SCHEMA_RE.finditer(prompt_text):
        preserved_targets[m_pres.group("pid")] = m_pres.group("punit")

    return tuple(turns), target_node_id, expected_unit, preserved_targets


def sanitize_candidate_json_envelope(raw_text: str) -> Tuple[str, bool]:
    """Strip outer markdown fences or trailing punctuation (e.g., Mistral `}}}.`) while tracking if sanitation ran."""
    if not isinstance(raw_text, str):
        return "", False
    s = raw_text.strip()
    changed = False

    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
            s = "\n".join(lines[1:-1]).strip()
            changed = True

    # Strip trailing period/comma/semicolon outside closing brace
    while s.endswith((".", ",", ";")) and "}" in s:
        s = s[:-1].rstrip()
        changed = True

    if not (s.startswith("{") and s.endswith("}")):
        first = s.find("{")
        last = s.rfind("}")
        if 0 <= first < last:
            s = s[first : last + 1]
            changed = True

    return s, changed


def _extract_turn1_committed_preserved(turn1_raw_output: Optional[str]) -> Dict[str, Tuple[Fraction, str]]:
    """Extract committed `preserved` bindings from the model's own Turn-1 JSON (`state_v1`) without oracle leakage."""
    committed: Dict[str, Tuple[Fraction, str]] = {}
    if not turn1_raw_output:
        return committed
    t1_san, _ = sanitize_candidate_json_envelope(turn1_raw_output)
    t1_env = response_adapter(t1_san or turn1_raw_output)
    if t1_env.ok:
        for pk, pv, pu in t1_env.preserved:
            committed[pk] = (pv, pu)
    return committed


def _compute_ancestor_map(turns: Sequence[TurnSpec], replacement_map: Dict[str, str]) -> Dict[str, Set[str]]:
    """Compute transitive ancestor node IDs (including original and replacement chain links) for every node."""
    defs: Dict[str, TurnSpec] = {t.node_id: t for t in turns}
    reverse_repl: Dict[str, Set[str]] = {}
    for orig_id, rep_id in replacement_map.items():
        reverse_repl.setdefault(rep_id, set()).add(orig_id)

    def resolve_latest(nid: str) -> str:
        cur = nid
        seen: Set[str] = set()
        while cur in replacement_map and cur not in seen:
            seen.add(cur)
            cur = replacement_map[cur]
        return cur

    memo: Dict[str, Set[str]] = {}

    def get_ancestors(nid: str, visiting: Set[str]) -> Set[str]:
        if nid in memo:
            return memo[nid]
        if nid in visiting:
            return set()
        visiting_next = visiting | {nid}
        anc: Set[str] = {nid}
        eff = resolve_latest(nid)
        anc.add(eff)
        cur = eff
        while cur in reverse_repl:
            prevs = reverse_repl[cur]
            new_prevs = prevs - anc
            if not new_prevs:
                break
            anc |= new_prevs
            cur = next(iter(new_prevs))

        spec = defs.get(eff) or defs.get(nid)
        if spec and spec.kind == "derive":
            if spec.left_id:
                anc |= get_ancestors(spec.left_id, visiting_next)
            if spec.right_id:
                anc |= get_ancestors(spec.right_id, visiting_next)
        elif spec and spec.kind == "retract" and spec.target_id:
            anc |= get_ancestors(spec.target_id, visiting_next)

        memo[nid] = anc
        return anc

    for t in turns:
        get_ancestors(t.node_id, set())
    return memo


def compute_state_diff_snapshot(
    turn1_turns: Sequence[TurnSpec],
    turn2_turns: Sequence[TurnSpec],
    target_node_id: str = "d6",
    expected_unit: Optional[str] = None,
    preserved_targets: Optional[Dict[str, str]] = None,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-5,
) -> StateDiffSnapshot:
    """Compute the exact rational/SI state difference between Turn 1 and Turn 2 (or single-turn when turn1_turns=())."""
    all_turns = tuple(turn1_turns) + tuple(turn2_turns)
    if not all_turns:
        raise ValueError("Cannot compute state diff on empty turn sequence.")

    if turn1_turns:
        t1_oracle = ExactRationalOracle()
        for t in turn1_turns:
            t1_oracle.append_turn(t)
        t1_snap = t1_oracle.evaluate()
    else:
        pre_oracle = ExactRationalOracle()
        for t in all_turns:
            if t.kind != "retract":
                pre_oracle.append_turn(t)
        t1_snap = pre_oracle.evaluate()

    t2_oracle = ExactRationalOracle()
    for t in all_turns:
        t2_oracle.append_turn(t)
    t2_snap = t2_oracle.evaluate()

    if target_node_id not in t2_snap.active_values_base:
        raise ValueError(f"Target node {target_node_id!r} is not active in Turn-2 snapshot.")

    truth_base, truth_dim, truth_base_unit = t2_snap.active_values_base[target_node_id]
    display_unit = expected_unit or truth_base_unit
    truth_target = convert_from_base_fraction(truth_base, truth_dim, display_unit)
    expected_retracted = tuple(sorted(t2_snap.retracted_ids))

    anc_map = _compute_ancestor_map(all_turns, t2_snap.replacement_map)
    retracted_set = set(expected_retracted)

    defs: Dict[str, TurnSpec] = {t.node_id: t for t in all_turns}
    node_deltas: Dict[str, StateDiffNodeDelta] = {}
    tainted_ids: List[str] = []
    untainted_ids: List[str] = []

    for nid, (bval2, dim2, bunit2) in t2_snap.active_values_base.items():
        anc = tuple(sorted(anc_map.get(nid, {nid})))
        is_tainted = bool(set(anc) & retracted_set)
        t1_entry = t1_snap.active_values_base.get(nid)
        bval1 = t1_entry[0] if t1_entry is not None else None
        spec = defs.get(nid)
        kind = spec.kind if spec else "derive"
        node_deltas[nid] = StateDiffNodeDelta(
            node_id=nid,
            kind=kind,
            ancestors=anc,
            tainted=is_tainted,
            turn1_base_fraction=bval1,
            turn2_base_fraction=bval2,
            dimension=dim2,
            base_unit=bunit2,
        )
        if is_tainted:
            tainted_ids.append(nid)
        else:
            untainted_ids.append(nid)

    pres_pairs = tuple(sorted((preserved_targets or {}).items()))
    pres_truth_display: Dict[str, Tuple[Fraction, str]] = {}
    for pid, punit in pres_pairs:
        if pid not in t2_snap.active_values_base:
            raise ValueError(f"Preserved target {pid!r} not active in Turn-2 state.")
        pb, pdim, _ = t2_snap.active_values_base[pid]
        pfrac = convert_from_base_fraction(pb, pdim, punit)
        pres_truth_display[pid] = (pfrac, punit)

    if turn1_turns:
        stale_fracs = _compute_multiturn_stale_targets(
            turn1_turns,
            turn2_turns,
            target_node_id,
            display_unit,
            truth_dim,
            truth_target,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
    else:
        stale_fracs = _compute_stale_targets(
            all_turns,
            target_node_id,
            display_unit,
            truth_dim,
            truth_target,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )

    return StateDiffSnapshot(
        turn1_snapshot=t1_snap,
        turn2_snapshot=t2_snap,
        node_deltas=node_deltas,
        expected_retracted_ids=expected_retracted,
        tainted_node_ids=tuple(sorted(tainted_ids)),
        untainted_node_ids=tuple(sorted(untainted_ids)),
        target_node_id=target_node_id,
        expected_unit=display_unit,
        truth_base_fraction=truth_base,
        truth_target_fraction=truth_target,
        truth_dimension=truth_dim,
        preserved_targets=pres_pairs,
        preserved_truth_display=pres_truth_display,
        stale_target_fractions=stale_fracs,
    )


def pass1_audit_candidate(
    diff_snap: StateDiffSnapshot,
    raw_candidate_output: str,
    turn1_raw_output: Optional[str] = None,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-5,
) -> Pass1DiagnosticReport:
    """Pass 1 of Arm 3B: Audit candidate Turn-2 JSON with ZERO ground-truth target-value leakage in reasons."""
    flags: List[str] = []
    reasons: List[str] = []

    req_pres = tuple(pid for pid, _ in diff_snap.preserved_targets)
    strict_env = response_adapter(raw_candidate_output, required_preserved=req_pres)
    sanitized_used = False
    env = strict_env

    if not strict_env.ok:
        sanitized_text, did_sanitize = sanitize_candidate_json_envelope(raw_candidate_output)
        if did_sanitize:
            san_env = response_adapter(sanitized_text, required_preserved=req_pres)
            if san_env.ok:
                env = san_env
                sanitized_used = True
                flags.append("ENVELOPE_SYNTAX_DEFECT")
                reasons.append("Candidate envelope had trailing punctuation or fences outside JSON object.")

    if not env.ok or env.value_fraction is None or env.step_id != diff_snap.target_node_id:
        flags.append("ENVELOPE_SYNTAX_DEFECT")
        reason = env.reason if not env.ok else f"step_id mismatch: expected {diff_snap.target_node_id!r}, got {env.step_id!r}."
        reasons.append(reason)
        return Pass1DiagnosticReport(
            clean=False,
            flags=tuple(dict.fromkeys(flags)),
            sanitized_envelope_used=sanitized_used,
            observed_step_id=env.step_id if env.ok else "",
            observed_value_fraction=env.value_fraction if env.ok else None,
            observed_unit=env.unit if env.ok else "",
            observed_retracted_ids=env.retracted_ids if env.ok else (),
            spurious_retracted_ids=(),
            missing_retracted_ids=diff_snap.expected_retracted_ids,
            stale_binding_matched=False,
            matched_stale_fraction=None,
            preserved_tampered_ids=req_pres,
            arith_or_unit_mismatch=True,
            reasons=tuple(reasons),
        )

    # 1. Audit retracted_ids against exact retraction target set R
    obs_ret = set(env.retracted_ids)
    exp_ret = set(diff_snap.expected_retracted_ids)
    spurious_ret = tuple(sorted(obs_ret - exp_ret))
    missing_ret = tuple(sorted(exp_ret - obs_ret))
    if spurious_ret or missing_ret:
        flags.append("SPURIOUS_OR_MISSING_RETRACTED_IDS")
        if spurious_ret:
            reasons.append(f"Spurious retracted_ids detected: {list(spurious_ret)}.")
        if missing_ret:
            reasons.append(f"Missing superseded retracted_ids: {list(missing_ret)}.")

    # 2. Audit preserved untainted nodes (e.g. d_indep in Condition E) without leaking oracle truth if Turn-1 missed it
    t1_committed_pres = _extract_turn1_committed_preserved(turn1_raw_output)
    obs_pres_map = {k: (v, u) for k, v, u in env.preserved}
    tampered_pres: List[str] = []
    for pid, punit in diff_snap.preserved_targets:
        ptruth_frac, _ = diff_snap.preserved_truth_display[pid]
        if pid not in obs_pres_map:
            tampered_pres.append(pid)
            reasons.append(f"Preserved node {pid!r} missing from envelope.")
            continue
        pv, pu = obs_pres_map[pid]
        try:
            pc_base, pc_dim, _ = normalize_to_base_fraction(pv, pu)
            p_exp_base, p_exp_dim, _ = diff_snap.turn2_snapshot.active_values_base[pid]
            if pc_dim != p_exp_dim:
                tampered_pres.append(pid)
                reasons.append(f"Preserved node {pid!r} dimension mismatch: {pc_dim} vs expected {p_exp_dim}.")
                continue
            pc_disp = convert_from_base_fraction(pc_base, pc_dim, punit)
            if pid in t1_committed_pres:
                t1_pv, t1_pu = t1_committed_pres[pid]
                t1_base, t1_dim, _ = normalize_to_base_fraction(t1_pv, t1_pu)
                if t1_dim == p_exp_dim:
                    t1_disp = convert_from_base_fraction(t1_base, t1_dim, punit)
                    if not mixed_tolerance_match(pc_disp, t1_disp, abs_tol=abs_tol, rel_tol=rel_tol)[0]:
                        tampered_pres.append(pid)
                        reasons.append(
                            f"Preserved node {pid!r} drifted from your committed Turn-1 value ({float(t1_disp):.6g} {punit})."
                        )
                        continue
            if not mixed_tolerance_match(pc_disp, ptruth_frac, abs_tol=abs_tol, rel_tol=rel_tol)[0]:
                tampered_pres.append(pid)
                reasons.append(
                    f"Preserved untainted node {pid!r} value or scale altered from its invariant Turn-1 subgraph."
                )
        except DimensionMismatchError as exc:
            tampered_pres.append(pid)
            reasons.append(f"Preserved node {pid!r} invalid unit: {exc}")

    if tampered_pres:
        flags.append("PRESERVED_SUBGRAPH_TAMPERING")

    # 3. Audit target value & SI unit against state diff and stale Turn-1 lattice (NEVER leak truth_target_fraction!)
    stale_matched = False
    matched_stale_val: Optional[Fraction] = None
    arith_mismatch = False

    try:
        cand_base, cand_dim, _ = normalize_to_base_fraction(env.value_fraction, env.unit)
    except DimensionMismatchError as exc:
        arith_mismatch = True
        flags.append("SI_ARITH_RESIDUAL_MISMATCH")
        reasons.append(f"Unknown or incompatible unit {env.unit!r}: {exc}")
        cand_base = None
        cand_dim = ""

    if cand_base is not None:
        if cand_dim != diff_snap.truth_dimension:
            arith_mismatch = True
            flags.append("SI_ARITH_RESIDUAL_MISMATCH")
            reasons.append(f"Target dimension mismatch: {cand_dim} vs expected {diff_snap.truth_dimension}.")
        else:
            cand_disp = convert_from_base_fraction(cand_base, cand_dim, diff_snap.expected_unit)
            val_ok, _, _ = mixed_tolerance_match(
                cand_disp, diff_snap.truth_target_fraction, abs_tol=abs_tol, rel_tol=rel_tol
            )
            if not val_ok:
                for st in diff_snap.stale_target_fractions:
                    if mixed_tolerance_match(cand_disp, st, abs_tol=abs_tol, rel_tol=rel_tol)[0]:
                        stale_matched = True
                        matched_stale_val = st
                        break
                if stale_matched:
                    flags.append("STALE_TURN1_BINDING")
                    reasons.append(
                        f"Candidate value for {diff_snap.target_node_id!r} retained a stale pre-retraction or "
                        f"frozen Turn-1 intermediate binding ({list(diff_snap.tainted_node_ids)}) without full "
                        f"re-propagation of {list(diff_snap.expected_retracted_ids)}."
                    )
                else:
                    arith_mismatch = True
                    flags.append("SI_ARITH_RESIDUAL_MISMATCH")
                    reasons.append(
                        f"Candidate value for {diff_snap.target_node_id!r} in unit {diff_snap.expected_unit!r} has "
                        f"a non-zero arithmetic or SI/percentage scale residual across tainted nodes "
                        f"{list(diff_snap.tainted_node_ids)}."
                    )

    dedup_flags = tuple(dict.fromkeys(flags))
    return Pass1DiagnosticReport(
        clean=len(dedup_flags) == 0,
        flags=dedup_flags,
        sanitized_envelope_used=sanitized_used,
        observed_step_id=env.step_id,
        observed_value_fraction=env.value_fraction,
        observed_unit=env.unit,
        observed_retracted_ids=env.retracted_ids,
        spurious_retracted_ids=spurious_ret,
        missing_retracted_ids=missing_ret,
        stale_binding_matched=stale_matched,
        matched_stale_fraction=matched_stale_val,
        preserved_tampered_ids=tuple(sorted(set(tampered_pres))),
        arith_or_unit_mismatch=arith_mismatch,
        reasons=tuple(reasons),
    )


def _format_exact_number(frac: Fraction) -> int | float:
    """Format an exact Fraction as int when denominator == 1, else finite float."""
    if frac.denominator == 1:
        return int(frac.numerator)
    return float(frac)


def build_arm3b_repair_prompt(
    diff_snap: StateDiffSnapshot,
    pass1_report: Pass1DiagnosticReport,
    turn1_raw_output: Optional[str] = None,
) -> str:
    """Build a deterministic 1-turn LLM self-correction prompt with ZERO target-answer (`d6`) leakage.

    Reports ONLY structural/invariant violations:
    - Triggered diagnostic flag names (`STALE_TURN1_BINDING`, `SPURIOUS_OR_MISSING_RETRACTED_IDS`, etc.)
    - Non-leaking structural defect descriptions
    - Exact sorted superseded target set `R` (left of `:=` in `fix->`/`err->`/`sup->`)
    - Tainted node IDs requiring bottom-up SI recomputation vs untainted node IDs
    - The model's own committed Turn-1 (`state_v1`) `preserved` binding if present (never oracle-computed truth)
    """
    if pass1_report.clean:
        return ""
    t1_committed_pres = _extract_turn1_committed_preserved(turn1_raw_output)
    lines = [
        "[Arm 3B State-Diff Verifier — Pass 1 Structural & Invariant Diagnostic Alert]",
        f"Flags triggered: {list(pass1_report.flags)}",
    ]
    for r in pass1_report.reasons:
        lines.append(f"- Defect: {r}")
    lines.append(
        f"- Exact superseded target set R (left of ':=' in fix->/err->/sup->): {list(diff_snap.expected_retracted_ids)}"
    )
    lines.append(
        f"- Tainted node IDs requiring bottom-up SI recomputation from latest active roots: {list(diff_snap.tainted_node_ids)}"
    )
    lines.append(
        f"- Untainted node IDs (zero dependency on R): {list(diff_snap.untainted_node_ids)}"
    )
    if diff_snap.preserved_targets:
        for pid, punit in diff_snap.preserved_targets:
            if pid in t1_committed_pres:
                t1_pv, t1_pu = t1_committed_pres[pid]
                lines.append(
                    f"- Untainted preserved node {pid!r} MUST remain locked to your committed Turn-1 value "
                    f"({_format_exact_number(t1_pv)} {t1_pu} in unit {punit!r})."
                )
            else:
                lines.append(
                    f"- Untainted preserved node {pid!r} (unit {punit!r}) has zero dependency on R and must not be modified by Turn-2 retractions."
                )
    lines.append(
        f"Recompute tainted nodes bottom-up in canonical SI base units and re-emit ONLY a single valid standalone "
        f"JSON object for step_id={diff_snap.target_node_id!r} in unit={diff_snap.expected_unit!r} with zero "
        f"trailing characters or markdown fences."
    )
    return "\n".join(lines)


def apply_structural_gate_only(
    diff_snap: StateDiffSnapshot,
    raw_candidate_output: str,
    turn1_raw_output: Optional[str] = None,
) -> str:
    """Apply ONLY non-numeric structural hygiene (envelope sanitation, exact R, and Turn-1 committed preserved lock).

    Leaves the LLM's Turn-2 target numeric `value` and `unit` strictly untouched (eligible for primary K2 ablation).
    """
    san_text, _ = sanitize_candidate_json_envelope(raw_candidate_output)
    candidate_str = san_text if san_text else raw_candidate_output.strip()
    try:
        payload = json.loads(candidate_str)
    except Exception:
        return raw_candidate_output.strip()
    if not isinstance(payload, dict):
        return raw_candidate_output.strip()

    # Enforce exact structural retracted_ids set R extracted from prompt syntax
    payload["retracted_ids"] = list(diff_snap.expected_retracted_ids)
    payload["step_id"] = diff_snap.target_node_id

    # Strip spurious `preserved` key on non-Condition-E episodes (when preserved_targets is empty)
    if not diff_snap.preserved_targets:
        payload.pop("preserved", None)
    else:
        # If Condition E preserved node is required and Turn-1 committed a valid value in state_v1, lock it
        t1_committed_pres = _extract_turn1_committed_preserved(turn1_raw_output)
        if t1_committed_pres:
            pres_dict = payload.get("preserved") if isinstance(payload.get("preserved"), dict) else {}
            for pid, punit in diff_snap.preserved_targets:
                if pid in t1_committed_pres:
                    t1_pv, t1_pu = t1_committed_pres[pid]
                    try:
                        t1_base, t1_dim, _ = normalize_to_base_fraction(t1_pv, t1_pu)
                        t1_disp = convert_from_base_fraction(t1_base, t1_dim, punit)
                        pres_dict[pid] = {"value": _format_exact_number(t1_disp), "unit": punit}
                    except DimensionMismatchError:
                        pass
            if pres_dict:
                payload["preserved"] = pres_dict

    return json.dumps(payload, ensure_ascii=False)


def _build_state_diff_trace_note(diff_snap: StateDiffSnapshot, pass1: Pass1DiagnosticReport) -> str:
    """Construct a concise deterministic state-diff trace for the `notes` field (symbolic ceiling mode only)."""
    ret_list = list(diff_snap.expected_retracted_ids)
    tainted_list = list(diff_snap.tainted_node_ids)
    target_val = _format_exact_number(diff_snap.truth_target_fraction)
    flag_tag = ",".join(pass1.flags) if pass1.flags else "CLEAN"
    parts = [
        f"Arm3B symbolic-ceiling state-diff (flags={flag_tag})",
        f"R={ret_list}",
        f"tainted={tainted_list}",
        f"{diff_snap.target_node_id}={target_val} {diff_snap.expected_unit}",
    ]
    if diff_snap.preserved_targets:
        pres_str = ", ".join(
            f"{pid}={_format_exact_number(diff_snap.preserved_truth_display[pid][0])} {punit}"
            for pid, punit in diff_snap.preserved_targets
        )
        parts.append(f"preserved({pres_str})")
    return "; ".join(parts)


def pass2_deterministic_repair(
    diff_snap: StateDiffSnapshot,
    pass1_report: Pass1DiagnosticReport,
    raw_candidate_output: str,
) -> str:
    """Secondary Symbolic-Offload Ceiling ONLY: project output onto exact Fraction/SI state-diff evaluation.

    WARNING: Because this overwrites the LLM's numeric target value with the deterministic rational DAG evaluation,
    it MUST ONLY be reported as a secondary neuro-symbolic offload upper bound, never as a primary K2 LLM claim.
    """
    if pass1_report.clean and not pass1_report.sanitized_envelope_used:
        return raw_candidate_output.strip()

    existing_notes = ""
    san_text, _ = sanitize_candidate_json_envelope(raw_candidate_output)
    try:
        parsed_obj = json.loads(san_text) if san_text else {}
        if isinstance(parsed_obj, dict) and isinstance(parsed_obj.get("notes"), str):
            existing_notes = parsed_obj["notes"].strip()
    except Exception:
        existing_notes = ""

    diff_trace = _build_state_diff_trace_note(diff_snap, pass1_report)
    notes_out = f"{existing_notes} | {diff_trace}" if existing_notes else diff_trace

    payload: Dict[str, Any] = {
        "notes": notes_out,
        "step_id": diff_snap.target_node_id,
        "value": _format_exact_number(diff_snap.truth_target_fraction),
        "unit": diff_snap.expected_unit,
        "retracted_ids": list(diff_snap.expected_retracted_ids),
    }

    if diff_snap.preserved_targets:
        pres_obj: Dict[str, Dict[str, Any]] = {}
        for pid, punit in diff_snap.preserved_targets:
            pfrac, _ = diff_snap.preserved_truth_display[pid]
            pres_obj[pid] = {
                "value": _format_exact_number(pfrac),
                "unit": punit,
            }
        payload["preserved"] = pres_obj

    return json.dumps(payload, ensure_ascii=False)


def run_arm3b_state_diff_verifier(
    turn1_turns: Sequence[TurnSpec],
    turn2_turns: Sequence[TurnSpec],
    raw_turn2_output: str,
    target_node_id: str = "d6",
    expected_unit: Optional[str] = None,
    preserved_targets: Optional[Dict[str, str]] = None,
    turn1_raw_output: Optional[str] = None,
    mode: str = "llm_repair",
    repaired_turn2_output: Optional[str] = None,
    llm_repair_callback: Optional[Callable[[str], str]] = None,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-5,
) -> Arm3BVerificationResult:
    """Execute the 2-pass Arm 3B state-diff verifier under strict primary-K2 vs symbolic-ceiling separation.

    Modes:
    - `"llm_repair"` (DEFAULT, Primary `K2` Eligible):
      Runs Pass-1 diagnostic audit and builds `repair_prompt` (`build_arm3b_repair_prompt`, 0 target-value leakage).
      When flags fire, evaluates the 1-turn LLM self-corrected response provided via `repaired_turn2_output` or
      `llm_repair_callback(repair_prompt)` (plus non-numeric envelope sanitation), NEVER overwriting the LLM's
      numeric target `value`.
    - `"structural_gate"` (Primary `K2` Ablation Eligible, 0 extra LLM turns):
      Applies only non-numeric structural hygiene (`apply_structural_gate_only`: trailing-punctuation sanitation,
      exact `retracted_ids` set $R$, and Turn-1 `state_v1` `preserved` commitment lock), leaving the LLM's numeric
      target `value` untouched.
    - `"symbolic_ceiling"` (Secondary Symbolic-Offload Upper Bound ONLY; `is_primary_k2_eligible=False`):
      Runs `pass2_deterministic_repair` to measure the 100% neuro-symbolic solver-offload ceiling.
    """
    if mode not in VALID_ARM3B_MODES:
        raise ValueError(f"Unsupported Arm 3B mode {mode!r}; expected one of {sorted(VALID_ARM3B_MODES)}.")

    diff_snap = compute_state_diff_snapshot(
        turn1_turns=turn1_turns,
        turn2_turns=turn2_turns,
        target_node_id=target_node_id,
        expected_unit=expected_unit,
        preserved_targets=preserved_targets,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )

    pass1 = pass1_audit_candidate(
        diff_snap=diff_snap,
        raw_candidate_output=raw_turn2_output,
        turn1_raw_output=turn1_raw_output,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    repair_prompt = build_arm3b_repair_prompt(
        diff_snap=diff_snap,
        pass1_report=pass1,
        turn1_raw_output=turn1_raw_output,
    )

    if pass1.clean and not pass1.sanitized_envelope_used:
        final_json = raw_turn2_output.strip()
        repair_triggered = False
        pass2 = pass1
    elif mode == "llm_repair":
        repair_triggered = True
        candidate_after = raw_turn2_output
        if repaired_turn2_output is not None:
            candidate_after = repaired_turn2_output
        elif llm_repair_callback is not None:
            candidate_after = llm_repair_callback(repair_prompt)
        san_after, _ = sanitize_candidate_json_envelope(candidate_after)
        final_json = san_after if san_after else candidate_after.strip()
        pass2 = pass1_audit_candidate(
            diff_snap=diff_snap,
            raw_candidate_output=final_json,
            turn1_raw_output=turn1_raw_output,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
    elif mode == "structural_gate":
        repair_triggered = True
        final_json = apply_structural_gate_only(
            diff_snap=diff_snap,
            raw_candidate_output=raw_turn2_output,
            turn1_raw_output=turn1_raw_output,
        )
        pass2 = pass1_audit_candidate(
            diff_snap=diff_snap,
            raw_candidate_output=final_json,
            turn1_raw_output=turn1_raw_output,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
    else:
        # mode == "symbolic_ceiling"
        repair_triggered = True
        final_json = pass2_deterministic_repair(
            diff_snap=diff_snap,
            pass1_report=pass1,
            raw_candidate_output=raw_turn2_output,
        )
        pass2 = pass1_audit_candidate(
            diff_snap=diff_snap,
            raw_candidate_output=final_json,
            turn1_raw_output=turn1_raw_output,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        if not pass2.clean:
            raise RuntimeError(f"Arm 3B symbolic_ceiling repair failed to converge: {pass2.reasons}")

    return Arm3BVerificationResult(
        mode=mode,
        is_primary_k2_eligible=(mode in {"llm_repair", "structural_gate"}),
        pass1_report=pass1,
        repair_prompt=repair_prompt,
        repair_triggered=repair_triggered,
        repair_converged=pass2.clean,
        pass2_report=pass2,
        final_json=final_json,
        final_value_fraction=pass2.observed_value_fraction,
        final_unit=pass2.observed_unit or diff_snap.expected_unit,
        final_retracted_ids=pass2.observed_retracted_ids,
    )


def run_arm3b_from_prompts(
    turn1_prompt: str,
    turn2_prompt: str,
    raw_turn2_output: str,
    turn1_raw_output: Optional[str] = None,
    mode: str = "llm_repair",
    repaired_turn2_output: Optional[str] = None,
    llm_repair_callback: Optional[Callable[[str], str]] = None,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-5,
) -> Arm3BVerificationResult:
    """Prompt-only entry point for live inference pipelines: parses Turn 1 & Turn 2 prompts and runs Arm 3B."""
    t1_turns, _, _, _ = parse_turn_prompt_specs(turn1_prompt)
    t2_turns, t2_target, t2_unit, t2_preserved = parse_turn_prompt_specs(turn2_prompt)
    if not t2_target:
        t2_target = "d6"
    return run_arm3b_state_diff_verifier(
        turn1_turns=t1_turns,
        turn2_turns=t2_turns,
        raw_turn2_output=raw_turn2_output,
        target_node_id=t2_target,
        expected_unit=t2_unit or None,
        preserved_targets=t2_preserved or None,
        turn1_raw_output=turn1_raw_output,
        mode=mode,
        repaired_turn2_output=repaired_turn2_output,
        llm_repair_callback=llm_repair_callback,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )


__all__ = [
    "Arm3BVerificationResult",
    "Pass1DiagnosticReport",
    "StateDiffNodeDelta",
    "StateDiffSnapshot",
    "VALID_ARM3B_MODES",
    "apply_structural_gate_only",
    "build_arm3b_repair_prompt",
    "compute_state_diff_snapshot",
    "parse_turn_prompt_specs",
    "pass1_audit_candidate",
    "pass2_deterministic_repair",
    "run_arm3b_from_prompts",
    "run_arm3b_state_diff_verifier",
    "sanitize_candidate_json_envelope",
]
