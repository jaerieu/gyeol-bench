#!/usr/bin/env python3
"""Colab L4 24 GB Runtime Smoke Probe & v3.0 N=50 Pilot for 2026 Open-Weight Models."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, "/content")
sys.path.insert(0, "/home/gyeol/gyeol-bench")

try:
    import oracle  # Colab /content/oracle.py
except ImportError:
    from gyeol_bench import oracle  # local repo


def gpu_mem_gb() -> Tuple[float, float]:
    import torch
    if not torch.cuda.is_available():
        return 0.0, 0.0
    alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
    res = torch.cuda.max_memory_reserved() / (1024 ** 3)
    return round(alloc, 3), round(res, 3)


def clear_gpu() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_hf_4bit(repo_id: str, trust_remote_code: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=trust_remote_code)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    load_s = round(time.perf_counter() - t0, 2)
    alloc_gb, res_gb = gpu_mem_gb()
    print(f"[load] {repo_id} loaded in {load_s}s | VRAM alloc={alloc_gb}GB res={res_gb}GB", flush=True)
    return tok, model, load_s


def generate_hf_turn(
    tok: Any,
    model: Any,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 512,
    enable_thinking: Optional[bool] = None,
) -> Tuple[str, float, int]:
    import torch

    kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        prompt_text = tok.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        prompt_text = tok.apply_chat_template(messages, **kwargs)

    inputs = tok(prompt_text, return_tensors="pt").to(model.device)
    in_len = inputs["input_ids"].shape[1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    dt = time.perf_counter() - t0
    gen_ids = out[0][in_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    return text, dt, int(gen_ids.shape[0])


def run_hf_model(
    model_key: str,
    repo_id: str,
    mode: str,
    arm: str,
    enable_thinking: Optional[bool],
    max_new_tokens: int,
    out_dir: Path,
    trust_remote_code: bool = False,
) -> Dict[str, Any]:
    clear_gpu()
    sys_prompt = (
        oracle.V3_MULTITURN_LINEAGE_K2_SYSTEM_PROMPT
        if arm == "lineage_k2"
        else oracle.V3_MULTITURN_BASELINE_SYSTEM_PROMPT
    )
    episodes = oracle.generate_multiturn_dataset(seed=20261151, count=50)
    target_eps = [episodes[1]] if mode == "smoke" else episodes  # index 1 = Condition B retraction

    tok, model, load_s = load_hf_4bit(repo_id, trust_remote_code=trust_remote_code)

    out_jsonl = out_dir / f"{model_key}_{arm}_{mode}.jsonl"
    out_summary = out_dir / f"{model_key}_{arm}_{mode}_summary.json"
    records: List[Dict[str, Any]] = []
    verdict_counts: Dict[str, int] = {}
    t_start = time.perf_counter()

    with out_jsonl.open("w", encoding="utf-8") as fh:
        for i, ep in enumerate(target_eps):
            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": ep.turn1_prompt},
            ]
            t1_raw, t1_dt, t1_toks = generate_hf_turn(
                tok, model, msgs, max_new_tokens=max_new_tokens, enable_thinking=enable_thinking
            )
            msgs.append({"role": "assistant", "content": t1_raw})
            msgs.append({"role": "user", "content": ep.turn2_prompt})
            t2_raw, t2_dt, t2_toks = generate_hf_turn(
                tok, model, msgs, max_new_tokens=max_new_tokens, enable_thinking=enable_thinking
            )

            ev = oracle.evaluate_multiturn_episode(
                ep.turn1_turns,
                ep.turn2_turns,
                t2_raw,
                target_node_id=ep.target_node_id,
                expected_unit=ep.expected_unit,
                preserved_targets=dict(ep.preserved_targets) if ep.preserved_targets else None,
            )
            verdict_counts[ev.verdict] = verdict_counts.get(ev.verdict, 0) + 1
            rec = {
                "episode_id": ep.episode_id,
                "condition": ep.condition,
                "ok": ev.ok,
                "verdict": ev.verdict,
                "reason": ev.reason,
                "t1_sec": round(t1_dt, 3),
                "t2_sec": round(t2_dt, 3),
                "total_sec": round(t1_dt + t2_dt, 3),
                "t1_tokens": t1_toks,
                "t2_tokens": t2_toks,
                "t1_raw": t1_raw,
                "t2_raw": t2_raw,
            }
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if (i + 1) % 5 == 0 or mode == "smoke":
                alloc_gb, res_gb = gpu_mem_gb()
                print(
                    f"[{model_key}] {i+1}/{len(target_eps)} {ep.episode_id}({ep.condition}) "
                    f"verdict={ev.verdict} dt={rec['total_sec']}s toks={t1_toks}+{t2_toks} "
                    f"vram_res={res_gb}GB",
                    flush=True,
                )

    total_s = round(time.perf_counter() - t_start, 2)
    alloc_gb, res_gb = gpu_mem_gb()
    avg_s = round(sum(r["total_sec"] for r in records) / max(1, len(records)), 3)
    pass_count = sum(1 for r in records if r["ok"])
    valid_json_count = sum(1 for r in records if r["verdict"] != "PARSE_REJECT")

    smoke_pass = (
        res_gb <= 22.0
        and avg_s < 15.0
        and valid_json_count == len(records)
    )
    summary = {
        "model_key": model_key,
        "repo_id": repo_id,
        "mode": mode,
        "arm": arm,
        "enable_thinking": enable_thinking,
        "max_new_tokens": max_new_tokens,
        "load_sec": load_s,
        "total_eval_sec": total_s,
        "avg_sec_per_item": avg_s,
        "peak_vram_alloc_gb": alloc_gb,
        "peak_vram_reserved_gb": res_gb,
        "n_episodes": len(records),
        "pass_count": pass_count,
        "pass_rate": round(pass_count / max(1, len(records)), 4),
        "valid_json_count": valid_json_count,
        "verdict_counts": verdict_counts,
        "smoke_gate_pass": smoke_pass,
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[SUMMARY]", json.dumps(summary, ensure_ascii=False), flush=True)
    del model
    clear_gpu()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--mode", choices=["smoke", "n50"], default="smoke")
    ap.add_argument("--arm", choices=["baseline", "lineage_k2"], default="baseline")
    ap.add_argument("--enable-thinking", choices=["none", "false", "true"], default="none")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--out-dir", default="/content/probe_out")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eth = None if args.enable_thinking == "none" else (args.enable_thinking == "true")
    run_hf_model(
        model_key=args.model_key,
        repo_id=args.repo_id,
        mode=args.mode,
        arm=args.arm,
        enable_thinking=eth,
        max_new_tokens=args.max_new_tokens,
        out_dir=out_dir,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
