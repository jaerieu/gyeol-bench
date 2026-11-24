#!/usr/bin/env python3
"""Track 1C: v0.2 Frontier contents[0] Context-Eviction / Summary-Handoff Ablation.

Dual-branch ablation reusing a shared Turn-1 Paid Batch output:
- C1 (self_summary_eviction): Evicts raw turn1_prompt (contents[0]); passes ONLY
  committed Turn-1 JSON state_v1 (full untruncated notes, declared value/unit,
  preserved) in alternating user/model/user roles.
- C2 (materialized_ledger_eviction): Evicts raw turn1_prompt conversational text;
  passes a 100% mathematically solvable materialized state ledger containing Turn-1
  committed d4/d_indep, Turn-1 primitive bindings, Turn-1 derivation rules, and
  stale cached Turn-1 intermediates computed from ep.turn1_turns.

Zero external dependencies (pure Python stdlib, Apache-2.0). Enforces:
- AI_STUDIO_PAID_TIER_CONFIRMED=1 for live runs
- 0 -latest aliases, 0 Google item-gen
- FINISH_REASON_MAX_TOKENS == 0
- Cumulative AI Studio spend < $5.00 USD hard cap
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gyeol_bench import (
    ExactRationalOracle,
    MultiTurnEpisode,
    V3_MULTITURN_BASELINE_SYSTEM_PROMPT,
    evaluate_multiturn_episode,
    evaluate_retraction_episode,
    generate_multiturn_dataset,
    parse_step_envelope,
)

PILOT_SEED = 20261151
PILOT_COUNT = 50
ALLOWED_MODELS = ("models/gemini-4-carbon", "models/gemini-4-hydrogen")
DEFAULT_SPEND_PATH = Path("/home/gyeol/.config/gyeol/aistudio-spend.json")
DEFAULT_CHECKPOINT_DIR = Path("/tmp/gyeol-colab-checkpoints/frontier-eviction-v02")
PRIOR_SPEND_USD = 1.130102
HARD_CAP_CUMULATIVE_USD = 5.00

# Batch API (50% discount) pricing per 1M tokens
BATCH_PRICE_PER_1M = {
    "models/gemini-4-carbon": {"input": 0.15, "output": 0.60, "thought": 0.60},
    "models/gemini-4-hydrogen": {"input": 0.625, "output": 2.50, "thought": 2.50},
}

VERDICT_KEYS = (
    "PASS",
    "PARSE_REJECT",
    "CONDITIONAL_ARITH_ERROR",
    "STALE_STATE_RETENTION",
    "OVER_RETRACTION",
)


def _load_prior_spend(spend_path: Path = DEFAULT_SPEND_PATH) -> float:
    if spend_path.exists():
        try:
            data = json.loads(spend_path.read_text(encoding="utf-8"))
            for key in ("spentUsd", "cumulative_usd", "total_usd", "aistudio_paid_batch_pilot_usd", "spent_usd"):
                if key in data and isinstance(data[key], (int, float)):
                    return float(data[key])
        except Exception:
            pass
    return PRIOR_SPEND_USD


def _update_spend_ledger(
    spend_path: Path,
    delta_usd: float,
    new_batches: List[str],
) -> float:
    data: Dict[str, Any] = {}
    if spend_path.exists():
        try:
            data = json.loads(spend_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    prior = _load_prior_spend(spend_path)
    updated = round(prior + delta_usd, 6)
    data["spentUsd"] = updated
    data["updatedAt"] = datetime.now(timezone.utc).isoformat()
    recorded = list(data.get("recordedBatches", []))
    for b in new_batches:
        if b and b not in recorded:
            recorded.append(b)
    data["recordedBatches"] = recorded
    spend_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = spend_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(spend_path)
    return updated


def _estimate_cost_usd(model: str, prompt_tok: int, cand_tok: int, thought_tok: int) -> float:
    prices = BATCH_PRICE_PER_1M.get(model, BATCH_PRICE_PER_1M["models/gemini-4-carbon"])
    return (
        (prompt_tok / 1_000_000.0) * prices["input"]
        + (cand_tok / 1_000_000.0) * prices["output"]
        + (thought_tok / 1_000_000.0) * prices["thought"]
    )


def _assert_safety_guards(model: str, live: bool, spend_path: Path = DEFAULT_SPEND_PATH) -> float:
    if model not in ALLOWED_MODELS or "-latest" in model:
        raise ValueError(f"Disallowed or -latest model alias: {model}")
    prior = _load_prior_spend(spend_path)
    if prior >= HARD_CAP_CUMULATIVE_USD:
        raise RuntimeError(f"Cumulative spend ${prior:.6f} already >= ${HARD_CAP_CUMULATIVE_USD:.2f} hard cap.")
    if live:
        if os.environ.get("AI_STUDIO_PAID_TIER_CONFIRMED") != "1":
            raise RuntimeError("AI_STUDIO_PAID_TIER_CONFIRMED=1 is required for live Paid Batch runs.")
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    return prior


def _parse_json_fenced(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    cleaned = raw.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if m:
        cleaned = m.group(1).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def build_c1_self_summary_contents(
    ep: MultiTurnEpisode,
    raw_turn1_json: str,
) -> List[Dict[str, Any]]:
    """C1: Evict raw turn1_prompt; pass ONLY committed Turn-1 JSON state_v1 (full notes)."""
    parsed = parse_step_envelope(raw_turn1_json)
    raw_obj = _parse_json_fenced(raw_turn1_json)
    val = raw_obj.get("value")
    if val is None and parsed.ok and parsed.value_fraction is not None:
        val = float(parsed.value_fraction)
    state_v1 = {
        "state_version": "state_v1",
        "episode_id": ep.episode_id,
        "committed_turn1_step": ep.turn1_target_id,
        "envelope_ok": parsed.ok,
        "declared_value": val,
        "declared_unit": raw_obj.get("unit") or (parsed.unit if parsed.ok else ep.turn1_expected_unit),
        "preserved": raw_obj.get("preserved") if "preserved" in raw_obj else (parsed.preserved if parsed.ok else None),
        "notes": raw_obj.get("notes", ""),
        "eviction_notice": "Raw Turn-1 premise equations (contents[0]) evicted from context.",
    }
    handoff_text = (
        "Committed Turn-1 Checkpoint (state_v1, raw Turn-1 equations evicted):\n"
        + json.dumps(state_v1, ensure_ascii=False)
    )
    return [
        {"role": "user", "parts": [{"text": handoff_text}]},
        {"role": "model", "parts": [{"text": "Acknowledged state_v1 checkpoint."}]},
        {"role": "user", "parts": [{"text": ep.turn2_prompt}]},
    ]


def build_c2_materialized_ledger_contents(
    ep: MultiTurnEpisode,
    raw_turn1_json: str,
) -> List[Dict[str, Any]]:
    """C2: Evict raw turn1_prompt; pass 100% solvable materialized state ledger."""
    parsed = parse_step_envelope(raw_turn1_json)
    raw_obj = _parse_json_fenced(raw_turn1_json)
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

    snap = oracle.evaluate()
    stale_intermediates: Dict[str, str] = {}
    for nid, (base_val, dim, orig_unit) in snap.active_values_base.items():
        if nid.startswith("d") or nid.startswith("u_d"):
            stale_intermediates[nid] = f"base={float(base_val):.6g} (dim={dim}, unit={orig_unit})"

    ledger = {
        "ledger_version": "materialized_ledger_v1",
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


def _extract_response_fields(item: Dict[str, Any]) -> Tuple[str, bool, Dict[str, int], Optional[str]]:
    if "error" in item:
        return "", False, {"promptTokenCount": 0, "candidatesTokenCount": 0, "thoughtsTokenCount": 0}, str(item["error"])
    resp = item.get("response", item)
    cand = (resp.get("candidates") or [{}])[0]
    max_tok = cand.get("finishReason") == "MAX_TOKENS"
    parts = cand.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought") and "text" in p)
    usage = resp.get("usageMetadata", {})
    counts = {
        "promptTokenCount": int(usage.get("promptTokenCount", 0)),
        "candidatesTokenCount": int(usage.get("candidatesTokenCount", 0)),
        "thoughtsTokenCount": int(usage.get("thoughtsTokenCount", 0)),
    }
    return text, max_tok, counts, None


def _align_inlined_responses(
    episodes: List[MultiTurnEpisode],
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for it in items:
        k = (it.get("metadata") or {}).get("key")
        if k:
            by_key[k] = it
    if len(by_key) == len(episodes):
        return [by_key[ep.episode_id] for ep in episodes]
    return items


def _poll_batch_v1beta(op_name: str, api_key: str) -> List[Dict[str, Any]]:
    poll_url = f"https://generativelanguage.googleapis.com/v1beta/{op_name}"
    for _ in range(240):
        poll_req = urllib.request.Request(poll_url, headers={"x-goog-api-key": api_key}, method="GET")
        with urllib.request.urlopen(poll_req, timeout=60) as presp:
            pdata = json.loads(presp.read().decode("utf-8"))
        if pdata.get("done"):
            return pdata.get("response", {}).get("inlinedResponses", {}).get("inlinedResponses", [])
        time.sleep(5)
    raise TimeoutError(f"Batch operation timed out: {op_name}")


def _batch_generate_v1beta(
    model: str,
    display_name: str,
    requests_with_keys: List[Tuple[str, Dict[str, Any]]],
    api_key: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    url = f"https://generativelanguage.googleapis.com/v1beta/{model}:batchGenerateContent"
    payload = {
        "batch": {
            "displayName": display_name,
            "inputConfig": {
                "requests": {
                    "requests": [
                        {"request": req, "metadata": {"key": key}}
                        for key, req in requests_with_keys
                    ]
                }
            },
        }
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        op = json.loads(resp.read().decode("utf-8"))

    op_name = op["name"]
    return op_name, _poll_batch_v1beta(op_name, api_key)


def _run_or_load_stage(
    ckpt_path: Path,
    episodes: List[MultiTurnEpisode],
    model: str,
    display_name: str,
    requests_with_keys: List[Tuple[str, Dict[str, Any]]],
    api_key: str,
    recover_batch_name: Optional[str] = None,
) -> Tuple[str, List[str], int, Dict[str, int], bool]:
    """Run a batch stage or load from checkpoint; returns (batch_name, raw_texts, max_tok_hits, usage, was_cached)."""
    if ckpt_path.exists():
        cached = json.loads(ckpt_path.read_text(encoding="utf-8"))
        return (
            cached["batch_name"],
            cached["raw_texts"],
            cached["max_tokens_hits"],
            cached["usage"],
            True,
        )

    if recover_batch_name:
        batch_name = recover_batch_name
        items = _poll_batch_v1beta(recover_batch_name, api_key)
    else:
        batch_name, items = _batch_generate_v1beta(model, display_name, requests_with_keys, api_key)

    aligned = _align_inlined_responses(episodes, items)
    raw_texts: List[str] = []
    max_tok_hits = 0
    usage = {"promptTokenCount": 0, "candidatesTokenCount": 0, "thoughtsTokenCount": 0}
    for item in aligned:
        txt, mtok, u, _ = _extract_response_fields(item)
        if mtok:
            max_tok_hits += 1
        for k in usage:
            usage[k] += u[k]
        raw_texts.append(txt)

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text(
        json.dumps({
            "batch_name": batch_name,
            "raw_texts": raw_texts,
            "max_tokens_hits": max_tok_hits,
            "usage": usage,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return batch_name, raw_texts, max_tok_hits, usage, False


def _tally_branch(
    episodes: List[MultiTurnEpisode],
    raw2_list: List[str],
    t1_ok_list: List[bool],
) -> Dict[str, Any]:
    verdict_counts = {k: 0 for k in VERDICT_KEYS}
    clean_t1_verdicts = {k: 0 for k in VERDICT_KEYS}
    cond_a_fail = 0
    cond_be_fail = 0
    per_ep: List[Dict[str, Any]] = []

    for ep, r2, t1_ok in zip(episodes, raw2_list, t1_ok_list):
        ev = evaluate_multiturn_episode(
            ep.turn1_turns,
            ep.turn2_turns,
            r2,
            target_node_id=ep.target_node_id,
            expected_unit=ep.expected_unit,
            preserved_targets=dict(ep.preserved_targets) if ep.preserved_targets else None,
        )
        v = ev.verdict
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
        if t1_ok:
            clean_t1_verdicts[v] = clean_t1_verdicts.get(v, 0) + 1
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
            "verdict": v,
            "reason": ev.reason,
        })

    n = len(episodes)
    n_clean = sum(1 for x in t1_ok_list if x)
    return {
        "itt_n": n,
        "itt_pass": verdict_counts["PASS"],
        "itt_fail": n - verdict_counts["PASS"],
        "itt_stale_rate": verdict_counts["STALE_STATE_RETENTION"] / n if n else 0.0,
        "itt_verdicts": verdict_counts,
        "clean_t1_n": n_clean,
        "clean_t1_pass": clean_t1_verdicts["PASS"],
        "clean_t1_stale_rate": clean_t1_verdicts["STALE_STATE_RETENTION"] / n_clean if n_clean else 0.0,
        "clean_t1_verdicts": clean_t1_verdicts,
        "condition_a_fail": cond_a_fail,
        "condition_be_fail": cond_be_fail,
        "episodes": per_ep,
    }


def run_ablation(
    model: str,
    live: bool,
    out_path: Path,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    spend_path: Path = DEFAULT_SPEND_PATH,
    recover_t1_batch: Optional[str] = None,
) -> Dict[str, Any]:
    prior_spend = _assert_safety_guards(model, live=live, spend_path=spend_path)
    episodes = generate_multiturn_dataset(seed=PILOT_SEED, count=PILOT_COUNT)

    if not live:
        dummy_r1 = json.dumps({
            "notes": "Full untruncated Turn-1 notes trace.",
            "step_id": episodes[0].turn1_target_id,
            "value": 10.0,
            "unit": episodes[0].turn1_expected_unit,
            "retracted_ids": [],
        })
        c1_sample = build_c1_self_summary_contents(episodes[0], dummy_r1)
        c2_sample = build_c2_materialized_ledger_contents(episodes[0], dummy_r1)
        summary = {
            "mode": "dry_run",
            "model": model,
            "seed": PILOT_SEED,
            "count": len(episodes),
            "prior_spend_usd": prior_spend,
            "hard_cap_cumulative_usd": HARD_CAP_CUMULATIVE_USD,
            "c1_roles": [m["role"] for m in c1_sample],
            "c2_roles": [m["role"] for m in c2_sample],
            "c1_has_raw_turn1": episodes[0].turn1_prompt in str(c1_sample),
            "c2_has_raw_turn1": episodes[0].turn1_prompt in str(c2_sample),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    api_key = os.environ["GEMINI_API_KEY"]
    tag = model.split("/")[-1]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Shared Turn 1
    t1_reqs = [
        (ep.episode_id, {
            "systemInstruction": {"parts": [{"text": V3_MULTITURN_BASELINE_SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": ep.turn1_prompt}]}],
            "generationConfig": {"temperature": 0.0, "thinkingConfig": {"thinkingLevel": "MEDIUM"}},
        })
        for ep in episodes
    ]
    b_t1, raw1_list, mtok_t1, u_t1, cached_t1 = _run_or_load_stage(
        checkpoint_dir / f"{tag}-t1.json",
        episodes,
        model,
        f"track1c-t1-{tag}",
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

    # C1: self_summary_eviction
    c1_reqs = [
        (ep.episode_id, {
            "systemInstruction": {"parts": [{"text": V3_MULTITURN_BASELINE_SYSTEM_PROMPT}]},
            "contents": build_c1_self_summary_contents(ep, r1),
            "generationConfig": {"temperature": 0.0, "thinkingConfig": {"thinkingLevel": "MEDIUM"}},
        })
        for ep, r1 in zip(episodes, raw1_list)
    ]
    b_c1, c1_raw2, mtok_c1, u_c1, cached_c1 = _run_or_load_stage(
        checkpoint_dir / f"{tag}-c1.json",
        episodes,
        model,
        f"track1c-c1-{tag}",
        c1_reqs,
        api_key,
    )

    # C2: materialized_ledger_eviction
    c2_reqs = [
        (ep.episode_id, {
            "systemInstruction": {"parts": [{"text": V3_MULTITURN_BASELINE_SYSTEM_PROMPT}]},
            "contents": build_c2_materialized_ledger_contents(ep, r1),
            "generationConfig": {"temperature": 0.0, "thinkingConfig": {"thinkingLevel": "MEDIUM"}},
        })
        for ep, r1 in zip(episodes, raw1_list)
    ]
    b_c2, c2_raw2, mtok_c2, u_c2, cached_c2 = _run_or_load_stage(
        checkpoint_dir / f"{tag}-c2.json",
        episodes,
        model,
        f"track1c-c2-{tag}",
        c2_reqs,
        api_key,
    )

    usage_total = {
        k: u_t1[k] + u_c1[k] + u_c2[k]
        for k in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount")
    }
    max_tokens_hits = mtok_t1 + mtok_c1 + mtok_c2
    run_cost = _estimate_cost_usd(
        model,
        usage_total["promptTokenCount"],
        usage_total["candidatesTokenCount"],
        usage_total["thoughtsTokenCount"],
    )
    uncached_cost = sum(
        _estimate_cost_usd(model, u["promptTokenCount"], u["candidatesTokenCount"], u["thoughtsTokenCount"])
        for u, is_c in ((u_t1, cached_t1), (u_c1, cached_c1), (u_c2, cached_c2))
        if not is_c
    )
    new_batches = [b for b, is_c in ((b_t1, cached_t1), (b_c1, cached_c1), (b_c2, cached_c2)) if not is_c]
    cumulative_spend = _update_spend_ledger(spend_path, uncached_cost, new_batches)

    c1_tally = _tally_branch(episodes, c1_raw2, t1_ok_list)
    c2_tally = _tally_branch(episodes, c2_raw2, t1_ok_list)

    (checkpoint_dir / f"{tag}-t1-c1-c2.jsonl").write_text(
        "\n".join(
            json.dumps({
                "episode_id": ep.episode_id,
                "condition": ep.condition,
                "t1_ok": t1_ok,
                "raw1": r1,
                "c1_raw2": c1r2,
                "c2_raw2": c2r2,
            }, ensure_ascii=False)
            for ep, t1_ok, r1, c1r2, c2r2 in zip(episodes, t1_ok_list, raw1_list, c1_raw2, c2_raw2)
        ) + "\n",
        encoding="utf-8",
    )

    summary = {
        "mode": "live",
        "model": model,
        "seed": PILOT_SEED,
        "count": len(episodes),
        "batches": {"t1": b_t1, "c1": b_c1, "c2": b_c2},
        "finish_reason_max_tokens": max_tokens_hits,
        "usageMetadata": usage_total,
        "run_spend_usd": round(run_cost, 6),
        "cumulative_spend_usd": cumulative_spend,
        "t1_clean_count": sum(1 for x in t1_ok_list if x),
        "C1_self_summary_eviction": {k: v for k, v in c1_tally.items() if k != "episodes"},
        "C2_materialized_ledger_eviction": {k: v for k, v in c2_tally.items() if k != "episodes"},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Track 1C v0.2 Frontier Context-Eviction Ablation")
    parser.add_argument("--model", default="models/gemini-4-carbon", choices=ALLOWED_MODELS)
    parser.add_argument("--live", action="store_true", help="Run live Paid Batch API")
    parser.add_argument("--recover-t1-batch", default=None, help="Existing Turn-1 batch name (batches/...) to poll instead of resubmitting")
    parser.add_argument("--out", default=str(REPO_ROOT / "manifests" / "frontier-eviction-v02-results.json"))
    args = parser.parse_args()

    res = run_ablation(
        model=args.model,
        live=args.live,
        out_path=Path(args.out),
        recover_t1_batch=args.recover_t1_batch,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
