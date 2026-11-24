#!/usr/bin/env python3
"""Clean-room Colab GPU & local runner for Arm 3 (`Arm 3A` / `Arm 3B` / `Arm 3AB-Hybrid`).

Licensed under the Apache License, Version 2.0.
Contains zero lines and zero imports from any private engine.

Supports:
- Both `/home/gyeol/gyeol-bench` (package layout) and `/content` (flat Colab VM layout).
- `--dtype {nf4,bf16}` (`nf4` 4-bit quantized for Phase 3 parity; `bf16` full-precision for A100/H100).
- Left-padded batched generation (`--batch-size`, default `8`, `do_sample=False`, `max_new_tokens=1536`,
  symmetric `--assistant-prefill '{"notes": "'`) with automatic CUDA OOM backoff down to `batch_size=1`.
- `--turn1-cache <jsonl>`: reuses cached `turn1RawOutput` and `arm1RawOutput` (`Shared-Turn-1`) by
  `episodeId`, avoiding duplicate Turn-1 / Baseline Arm-1 GPU inference.
- `--chunk-size` (default `25`) incremental JSONL persistence and `--resume` deduplication.
- `--mock-backend`: deterministic CPU backend simulating realistic baseline failure modes (`Condition A`
  spurious retraction, `Condition B..D` stale Turn-1 retention, `Condition E` `d_indep` tampering +
  trailing `.` syntax defect) and realistic Arm 3A / Arm 3B LLM responses for offline integration tests.
- Strict Audit `#19` Primary `K2` vs Secondary Symbolic-Offload Ceiling separation:
  * Primary `K2` eligible: `arm3aPromptRaw`, `arm3aStructuralOnly`, `arm3bStructuralGate`, `arm3bLlmRepair`,
    `arm3abHybrid` (`isPrimaryK2Eligible=True`, zero numeric target overwrites).
  * Secondary symbolic-offload ceiling ONLY: `arm3aFullCeiling`, `arm3bSymbolicCeiling`
    (`isPrimaryK2Eligible=False`).
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_REPO_CANDIDATE = _THIS_DIR.parent
for _cand in (str(_REPO_CANDIDATE), "/content", "/home/gyeol/gyeol-bench"):
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

try:
    from gyeol_bench import oracle
    from gyeol_bench import arm3_ledger_hook
    from gyeol_bench import arm3_state_diff
except ImportError:  # pragma: no cover - flat /content fallback on Colab VM
    import oracle  # type: ignore[no-redef]
    import arm3_ledger_hook  # type: ignore[no-redef]
    import arm3_state_diff  # type: ignore[no-redef]


VALID_ARMS: Tuple[str, ...] = ("arm1", "arm3a", "arm3b", "arm3ab_hybrid")

BatchGenerateFn = Callable[[List[List[Dict[str, str]]]], List[Tuple[str, float, int, int]]]


def parse_arms_arg(raw_arms: str) -> Tuple[str, ...]:
    items = [tok.strip() for tok in raw_arms.split(",") if tok.strip()]
    if not items:
        raise ValueError("--arms must specify at least one arm from arm1,arm3a,arm3b,arm3ab_hybrid.")
    unknown = [tok for tok in items if tok not in VALID_ARMS]
    if unknown:
        raise ValueError(f"Unsupported arm(s) {unknown}; valid choices: {list(VALID_ARMS)}")
    dedup: List[str] = []
    for tok in items:
        if tok not in dedup:
            dedup.append(tok)
    return tuple(dedup)


def gpu_mem_gb() -> Tuple[float, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0, 0.0
        alloc = torch.cuda.max_memory_allocated() / (1024**3)
        res = torch.cuda.max_memory_reserved() / (1024**3)
        return round(alloc, 3), round(res, 3)
    except Exception:
        return 0.0, 0.0


def clear_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def load_turn1_cache(cache_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Load cached episode records keyed by `episodeId` (supports Phase 3 & Arm 3 JSONL files)."""
    if cache_path is None:
        return {}
    if not cache_path.exists():
        raise FileNotFoundError(f"Turn-1 cache file does not exist: {cache_path}")
    cache: Dict[str, Dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ep_id = obj.get("episodeId") or obj.get("episode_id")
            if isinstance(ep_id, str) and ep_id:
                cache[ep_id] = obj
    return cache


def load_existing_episode_records(jsonl_path: Path) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Load already-persisted records when `--resume` is active, deduplicating by `episodeId`."""
    if not jsonl_path.exists():
        return [], set()
    records: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ep_id = str(obj.get("episodeId", ""))
            if ep_id and ep_id not in seen_ids:
                seen_ids.add(ep_id)
                records.append(obj)
    return records, seen_ids


class MockChatBackend:
    """Deterministic zero-GPU mock backend for unit and integration testing (`--mock-backend`).

    Records every prompt conversation passed to `generate_batch` in `self.recorded_conversations`
    so test suites can verify zero target-answer leakage and exact turn counts.
    """

    def __init__(self, episodes: Sequence[oracle.MultiTurnEpisode]) -> None:
        self._by_t1_prompt: Dict[str, oracle.MultiTurnEpisode] = {
            ep.turn1_prompt: ep for ep in episodes
        }
        self.recorded_conversations: List[List[Dict[str, str]]] = []

    def generate_batch(
        self, conversations: List[List[Dict[str, str]]]
    ) -> List[Tuple[str, float, int, int]]:
        outputs: List[Tuple[str, float, int, int]] = []
        for msgs in conversations:
            self.recorded_conversations.append([dict(m) for m in msgs])
            user_msgs = [m["content"] for m in msgs if m.get("role") == "user"]
            if not user_msgs:
                raise ValueError("Conversation missing user message.")
            t1_prompt = user_msgs[0]
            ep = self._by_t1_prompt.get(t1_prompt)
            if ep is None:
                raise KeyError("MockChatBackend received unknown Turn-1 prompt.")

            last_user = user_msgs[-1]
            in_toks = max(16, sum(len(m.get("content", "")) // 4 for m in msgs))

            # Case 1: Turn-1 generation (only 1 user turn so far)
            if len(user_msgs) == 1:
                text = self._mock_turn1(ep)
            # Case 2: Arm 3B / Arm 3AB-Hybrid 1-turn self-correction prompt (3 user turns)
            elif "[Arm 3B State-Diff Verifier" in last_user:
                text = self._mock_arm3b_repair(ep)
            # Case 3: Arm 3A Structural Dependency-Ledger Turn-2 prompt
            elif "[Arm3A Structural Dependency-Ledger Guard]" in last_user:
                text = self._mock_arm3a_turn2(ep)
            # Case 4: Baseline Arm 1 Turn-2 prompt
            else:
                text = self._mock_arm1_turn2(ep)

            out_toks = max(12, len(text) // 4)
            outputs.append((text, 0.005, in_toks, out_toks))
        return outputs

    @staticmethod
    def _preserved_truth_dict(ep: oracle.MultiTurnEpisode) -> Dict[str, Dict[str, Any]]:
        if not ep.preserved_targets:
            return {}
        o = oracle.ExactRationalOracle()
        for t in ep.all_turns:
            o.append_turn(t)
        snap = o.evaluate()
        out: Dict[str, Dict[str, Any]] = {}
        for pid, punit in ep.preserved_targets:
            pb, pdim, _ = snap.active_values_base[pid]
            pf = oracle.convert_from_base_fraction(pb, pdim, punit)
            out[pid] = {
                "value": arm3_ledger_hook.format_exact_json_number(pf),
                "unit": punit,
            }
        return out

    def _mock_turn1(self, ep: oracle.MultiTurnEpisode) -> str:
        payload: Dict[str, Any] = {
            "notes": f"mock turn1 checkpoint for {ep.episode_id}",
            "step_id": ep.turn1_target_id,
            "value": ep.turn1_expected_value_float,
            "unit": ep.turn1_expected_unit,
            "retracted_ids": [],
        }
        if ep.turn1_preserved_targets:
            payload["preserved"] = self._preserved_truth_dict(ep)
        return json.dumps(payload)

    def _mock_arm1_turn2(self, ep: oracle.MultiTurnEpisode) -> str:
        idx_num = int(ep.episode_id.split("-")[-1])
        if ep.condition == "A":
            # Mostly clean on Condition A, with occasional spurious retraction on multiples of 9
            ret = ["p1"] if (idx_num % 9 == 0) else []
            return json.dumps(
                {
                    "notes": "mock arm1 cond A",
                    "step_id": ep.target_node_id,
                    "value": ep.expected_value_float,
                    "unit": ep.expected_unit,
                    "retracted_ids": ret,
                }
            )
        if ep.condition in {"B", "C", "D"}:
            # Exhibit STALE_STATE_RETENTION on baseline Arm 1
            stale_val = (
                ep.pre_retraction_value_float
                if ep.pre_retraction_value_float is not None
                else ep.expected_value_float
            )
            return json.dumps(
                {
                    "notes": "mock arm1 stale retention",
                    "step_id": ep.target_node_id,
                    "value": stale_val,
                    "unit": ep.expected_unit,
                    "retracted_ids": list(ep.expected_retracted_ids[:1]),
                }
            )
        # Condition E: tamper d_indep + trailing period defect
        pres = self._preserved_truth_dict(ep)
        for pk in pres:
            pres[pk]["value"] = float(pres[pk]["value"]) + 500.0
        return (
            json.dumps(
                {
                    "notes": "mock arm1 cond E tampered + trailing period",
                    "step_id": ep.target_node_id,
                    "value": ep.expected_value_float,
                    "unit": ep.expected_unit,
                    "retracted_ids": list(ep.expected_retracted_ids) + ["p1_r2"],
                    "preserved": pres,
                }
            )
            + "."
        )

    def _mock_arm3a_turn2(self, ep: oracle.MultiTurnEpisode) -> str:
        idx_num = int(ep.episode_id.split("-")[-1])
        # Simulate genuine LLM computation (accurate on most items, occasional arithmetic slip on Condition D)
        val = ep.expected_value_float
        if ep.condition == "D" and (idx_num % 2 == 0):
            val = ep.expected_value_float + 42.0
        payload: Dict[str, Any] = {
            "notes": f"mock arm3a structural-ledger trace for {ep.episode_id}",
            "step_id": ep.target_node_id,
            "value": val,
            "unit": ep.expected_unit,
            "retracted_ids": list(ep.expected_retracted_ids),
        }
        if ep.preserved_targets:
            payload["preserved"] = self._preserved_truth_dict(ep)
        # Add trailing period on Condition E to verify structural_only I1 sanitation
        suffix = "." if ep.condition == "E" else ""
        return json.dumps(payload) + suffix

    def _mock_arm3b_repair(self, ep: oracle.MultiTurnEpisode) -> str:
        payload: Dict[str, Any] = {
            "notes": f"mock arm3b self-repaired trace for {ep.episode_id}",
            "step_id": ep.target_node_id,
            "value": ep.expected_value_float,
            "unit": ep.expected_unit,
            "retracted_ids": list(ep.expected_retracted_ids),
        }
        if ep.preserved_targets:
            payload["preserved"] = self._preserved_truth_dict(ep)
        return json.dumps(payload)


DEFAULT_ASSISTANT_PREFILL: str = '{"notes": "'


class HuggingFaceBatchBackend:
    """Left-padded HuggingFace CausalLM batch backend (`nf4` or `bf16`) with automatic OOM backoff."""

    def __init__(
        self,
        repo_id: str,
        dtype: str = "nf4",
        batch_size: int = 8,
        max_new_tokens: int = 1536,
        assistant_prefill: str = DEFAULT_ASSISTANT_PREFILL,
        enable_thinking: Optional[bool] = None,
        trust_remote_code: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if dtype not in {"nf4", "bf16"}:
            raise ValueError(f"Unsupported dtype {dtype!r}; expected 'nf4' or 'bf16'.")

        self.repo_id = repo_id
        self.dtype = dtype
        self.batch_size = max(1, int(batch_size))
        self.max_new_tokens = max_new_tokens
        self.assistant_prefill = assistant_prefill or ""
        self.enable_thinking = enable_thinking
        self.oom_backoff_events = 0

        clear_gpu()
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=trust_remote_code)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        model_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": trust_remote_code,
        }
        if dtype == "nf4":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)
        self.model.eval()
        self.load_sec = round(time.perf_counter() - t0, 3)
        alloc_gb, res_gb = gpu_mem_gb()
        print(
            f"[load] {repo_id} ({dtype}) loaded in {self.load_sec}s | "
            f"VRAM alloc={alloc_gb}GB reserved={res_gb}GB | batch_size={self.batch_size}",
            flush=True,
        )

    def _render_prompt(self, messages: List[Dict[str, str]]) -> str:
        kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if self.enable_thinking is not None:
            kwargs["enable_thinking"] = self.enable_thinking
        try:
            return str(self.tokenizer.apply_chat_template(messages, **kwargs))
        except TypeError:
            kwargs.pop("enable_thinking", None)
            try:
                return str(self.tokenizer.apply_chat_template(messages, **kwargs))
            except Exception:
                return self._fallback_chat_format(messages)
        except Exception:
            # Fallback for models whose chat template rejects system role (e.g., mistral variants)
            if messages and messages[0].get("role") == "system" and len(messages) >= 2:
                merged = [
                    {
                        "role": "user",
                        "content": messages[0]["content"] + "\n\n" + messages[1]["content"],
                    },
                    *messages[2:],
                ]
                return str(self.tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True))
            return self._fallback_chat_format(messages)

    @staticmethod
    def _fallback_chat_format(messages: List[Dict[str, str]]) -> str:
        parts = [f"[{m.get('role', 'user').upper()}]\n{m.get('content', '')}" for m in messages]
        parts.append("[ASSISTANT]\n")
        return "\n\n".join(parts)

    def _run_subbatch(
        self, sub_convs: List[List[Dict[str, str]]]
    ) -> List[Tuple[str, float, int, int]]:
        import torch

        prefill = self.assistant_prefill
        prompts = [
            (self._render_prompt(c) + prefill) if prefill else self._render_prompt(c)
            for c in sub_convs
        ]
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True)
        enc_dev = {k: v.to(self.model.device) for k, v in enc.items()}
        padded_in_len = int(enc_dev["input_ids"].shape[1])
        attn_masks = enc_dev.get("attention_mask")

        t0 = time.perf_counter()
        with torch.inference_mode():
            out_ids = self.model.generate(
                **enc_dev,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - t0
        per_item_sec = elapsed / max(1, len(sub_convs))

        results: List[Tuple[str, float, int, int]] = []
        for idx in range(len(sub_convs)):
            in_toks = (
                int(attn_masks[idx].sum().item())
                if attn_masks is not None
                else padded_in_len
            )
            gen_slice = out_ids[idx][padded_in_len:]
            decoded = self.tokenizer.decode(gen_slice, skip_special_tokens=True)
            text = (prefill + decoded).strip() if prefill else decoded.strip()
            # Count non-pad generated tokens
            if self.tokenizer.pad_token_id is not None:
                out_toks = int((gen_slice != self.tokenizer.pad_token_id).sum().item())
            else:
                out_toks = int(gen_slice.shape[0])
            results.append((text, per_item_sec, in_toks, out_toks))
        return results

    def generate_batch(
        self, conversations: List[List[Dict[str, str]]]
    ) -> List[Tuple[str, float, int, int]]:
        import torch

        if not conversations:
            return []
        outputs: List[Tuple[str, float, int, int]] = []
        pos = 0
        cur_bs = self.batch_size
        while pos < len(conversations):
            sub = conversations[pos : pos + cur_bs]
            try:
                sub_out = self._run_subbatch(sub)
                outputs.extend(sub_out)
                pos += len(sub)
            except RuntimeError as exc:
                is_oom = "out of memory" in str(exc).lower() or isinstance(
                    exc, getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
                )
                if is_oom and cur_bs > 1:
                    self.oom_backoff_events += 1
                    cur_bs = max(1, cur_bs // 2 if cur_bs > 2 else 1)
                    self.batch_size = cur_bs
                    clear_gpu()
                    print(
                        f"[oom-backoff] CUDA OOM detected; backing off batch_size -> {cur_bs}",
                        flush=True,
                    )
                    continue
                raise
        return outputs

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        clear_gpu()


def _score_turn1(ep: oracle.MultiTurnEpisode, raw_t1: str) -> oracle.OracleEvaluationResult:
    return oracle.evaluate_retraction_episode(
        turns=ep.turn1_turns,
        raw_model_output=raw_t1,
        target_node_id=ep.turn1_target_id,
        expected_unit=ep.turn1_expected_unit,
        preserved_targets=dict(ep.turn1_preserved_targets) if ep.turn1_preserved_targets else None,
    )


def _score_turn2(ep: oracle.MultiTurnEpisode, raw_t2: str) -> oracle.OracleEvaluationResult:
    return oracle.evaluate_multiturn_episode(
        turn1_turns=ep.turn1_turns,
        turn2_turns=ep.turn2_turns,
        raw_model_output=raw_t2,
        target_node_id=ep.target_node_id,
        expected_unit=ep.expected_unit,
        preserved_targets=dict(ep.preserved_targets) if ep.preserved_targets else None,
    )


def evaluate_chunk(
    chunk_eps: Sequence[oracle.MultiTurnEpisode],
    arms: Sequence[str],
    gen_batch_fn: BatchGenerateFn,
    turn1_cache: Mapping[str, Dict[str, Any]],
    model_key: str,
    repo_id: str,
    dtype: str,
) -> List[Dict[str, Any]]:
    """Execute batched multi-stage generation and deterministic Arm 3A/3B/3AB scoring for one chunk."""
    sys_base = oracle.V3_MULTITURN_BASELINE_SYSTEM_PROMPT
    need_arm1_output = any(a in arms for a in ("arm1", "arm3b"))

    # 1. Resolve Turn-1 (`Shared-Turn-1`): reuse cache if present, otherwise batch-generate
    t1_raws: Dict[str, str] = {}
    t1_evals: Dict[str, oracle.OracleEvaluationResult] = {}
    t1_secs: Dict[str, float] = {}
    t1_in_toks: Dict[str, int] = {}
    t1_out_toks: Dict[str, int] = {}
    t1_from_cache: Dict[str, bool] = {}

    missing_t1_eps: List[oracle.MultiTurnEpisode] = []
    for ep in chunk_eps:
        cached = turn1_cache.get(ep.episode_id)
        if cached is not None and isinstance(cached.get("turn1RawOutput"), str):
            raw1 = cached["turn1RawOutput"]
            t1_raws[ep.episode_id] = raw1
            t1_evals[ep.episode_id] = _score_turn1(ep, raw1)
            t1_secs[ep.episode_id] = 0.0
            t1_in_toks[ep.episode_id] = int(cached.get("turn1InputTokens", 0))
            t1_out_toks[ep.episode_id] = int(cached.get("turn1OutputTokens", 0))
            t1_from_cache[ep.episode_id] = True
        else:
            missing_t1_eps.append(ep)
            t1_from_cache[ep.episode_id] = False

    if missing_t1_eps:
        t1_convs = [
            [
                {"role": "system", "content": sys_base},
                {"role": "user", "content": ep.turn1_prompt},
            ]
            for ep in missing_t1_eps
        ]
        t1_gen = gen_batch_fn(t1_convs)
        for ep, (txt, dt, itok, otok) in zip(missing_t1_eps, t1_gen):
            t1_raws[ep.episode_id] = txt
            t1_evals[ep.episode_id] = _score_turn1(ep, txt)
            t1_secs[ep.episode_id] = round(dt, 4)
            t1_in_toks[ep.episode_id] = itok
            t1_out_toks[ep.episode_id] = otok

    # 2. Resolve Baseline Arm 1 Turn-2 (`arm1RawOutput`): reuse cache if present, else batch-generate when needed
    arm1_raws: Dict[str, str] = {}
    arm1_evals: Dict[str, oracle.OracleEvaluationResult] = {}
    arm1_secs: Dict[str, float] = {}
    arm1_in_toks: Dict[str, int] = {}
    arm1_out_toks: Dict[str, int] = {}
    arm1_from_cache: Dict[str, bool] = {}

    missing_arm1_eps: List[oracle.MultiTurnEpisode] = []
    for ep in chunk_eps:
        cached = turn1_cache.get(ep.episode_id)
        if cached is not None and isinstance(cached.get("arm1RawOutput"), str):
            raw_a1 = cached["arm1RawOutput"]
            arm1_raws[ep.episode_id] = raw_a1
            arm1_evals[ep.episode_id] = _score_turn2(ep, raw_a1)
            arm1_secs[ep.episode_id] = 0.0
            arm1_in_toks[ep.episode_id] = int(cached.get("arm1InputTokens", 0))
            arm1_out_toks[ep.episode_id] = int(cached.get("arm1OutputTokens", 0))
            arm1_from_cache[ep.episode_id] = True
        elif need_arm1_output:
            missing_arm1_eps.append(ep)
            arm1_from_cache[ep.episode_id] = False
        else:
            arm1_from_cache[ep.episode_id] = False

    if missing_arm1_eps:
        a1_convs = [
            [
                {"role": "system", "content": sys_base},
                {"role": "user", "content": ep.turn1_prompt},
                {"role": "assistant", "content": t1_raws[ep.episode_id]},
                {"role": "user", "content": ep.turn2_prompt},
            ]
            for ep in missing_arm1_eps
        ]
        a1_gen = gen_batch_fn(a1_convs)
        for ep, (txt, dt, itok, otok) in zip(missing_arm1_eps, a1_gen):
            arm1_raws[ep.episode_id] = txt
            arm1_evals[ep.episode_id] = _score_turn2(ep, txt)
            arm1_secs[ep.episode_id] = round(dt, 4)
            arm1_in_toks[ep.episode_id] = itok
            arm1_out_toks[ep.episode_id] = otok

    # 3. Arm 3A (`arm3a` or needed as upstream input to `arm3ab_hybrid`)
    run_3a = ("arm3a" in arms) or ("arm3ab_hybrid" in arms)
    arm3a_ledgers: Dict[str, arm3_ledger_hook.Arm3ALedgerState] = {}
    arm3a_aug_prompts: Dict[str, str] = {}
    arm3a_raws: Dict[str, str] = {}
    arm3a_raw_evals: Dict[str, oracle.OracleEvaluationResult] = {}
    arm3a_struct_decs: Dict[str, arm3_ledger_hook.Arm3AGuardDecision] = {}
    arm3a_struct_evals: Dict[str, oracle.OracleEvaluationResult] = {}
    arm3a_full_decs: Dict[str, arm3_ledger_hook.Arm3AGuardDecision] = {}
    arm3a_full_evals: Dict[str, oracle.OracleEvaluationResult] = {}
    arm3a_secs: Dict[str, float] = {}
    arm3a_in_toks: Dict[str, int] = {}
    arm3a_out_toks: Dict[str, int] = {}

    if run_3a:
        a3a_convs: List[List[Dict[str, str]]] = []
        for ep in chunk_eps:
            ledger = arm3_ledger_hook.build_arm3a_ledger_from_prompts(
                ep.turn1_prompt, ep.turn2_prompt, t1_raws[ep.episode_id]
            )
            aug_p = arm3_ledger_hook.augment_turn2_prompt_with_ledger(ep.turn2_prompt, ledger)
            arm3a_ledgers[ep.episode_id] = ledger
            arm3a_aug_prompts[ep.episode_id] = aug_p
            a3a_convs.append(
                [
                    {"role": "system", "content": sys_base},
                    {"role": "user", "content": ep.turn1_prompt},
                    {"role": "assistant", "content": t1_raws[ep.episode_id]},
                    {"role": "user", "content": aug_p},
                ]
            )
        a3a_gen = gen_batch_fn(a3a_convs)
        for ep, (txt, dt, itok, otok) in zip(chunk_eps, a3a_gen):
            eid = ep.episode_id
            ledger = arm3a_ledgers[eid]
            arm3a_raws[eid] = txt
            arm3a_raw_evals[eid] = _score_turn2(ep, txt)
            s_dec = arm3_ledger_hook.apply_arm3a_invariant_guard(ledger, txt, mode="structural_only")
            arm3a_struct_decs[eid] = s_dec
            arm3a_struct_evals[eid] = _score_turn2(ep, s_dec.emitted_json)
            f_dec = arm3_ledger_hook.apply_arm3a_invariant_guard(ledger, txt, mode="full")
            arm3a_full_decs[eid] = f_dec
            arm3a_full_evals[eid] = _score_turn2(ep, f_dec.emitted_json)
            arm3a_secs[eid] = round(dt, 4)
            arm3a_in_toks[eid] = itok
            arm3a_out_toks[eid] = otok

    # Pre-build StateDiffSnapshot for Arm 3B / Arm 3AB-Hybrid if needed
    diff_snaps: Dict[str, arm3_state_diff.StateDiffSnapshot] = {}
    if ("arm3b" in arms) or ("arm3ab_hybrid" in arms):
        for ep in chunk_eps:
            t1_parsed, _, _, _ = arm3_state_diff.parse_turn_prompt_specs(ep.turn1_prompt)
            t2_parsed, t2_tid, t2_u, t2_pres = arm3_state_diff.parse_turn_prompt_specs(ep.turn2_prompt)
            diff_snaps[ep.episode_id] = arm3_state_diff.compute_state_diff_snapshot(
                turn1_turns=t1_parsed,
                turn2_turns=t2_parsed,
                target_node_id=t2_tid or ep.target_node_id,
                expected_unit=t2_u or ep.expected_unit,
                preserved_targets=t2_pres or None,
            )

    # 4. Arm 3B (`arm3b` on `arm1_raw`)
    arm3b_struct_res: Dict[str, arm3_state_diff.Arm3BVerificationResult] = {}
    arm3b_struct_evals: Dict[str, oracle.OracleEvaluationResult] = {}
    arm3b_llm_res: Dict[str, arm3_state_diff.Arm3BVerificationResult] = {}
    arm3b_llm_evals: Dict[str, oracle.OracleEvaluationResult] = {}
    arm3b_llm_raws: Dict[str, str] = {}
    arm3b_ceil_res: Dict[str, arm3_state_diff.Arm3BVerificationResult] = {}
    arm3b_ceil_evals: Dict[str, oracle.OracleEvaluationResult] = {}
    arm3b_pass1_map: Dict[str, arm3_state_diff.Pass1DiagnosticReport] = {}
    arm3b_secs: Dict[str, float] = {}
    arm3b_in_toks: Dict[str, int] = {}
    arm3b_out_toks: Dict[str, int] = {}
    arm3b_extra_turns: Dict[str, int] = {}

    if "arm3b" in arms:
        flagged_eps_3b: List[oracle.MultiTurnEpisode] = []
        repair_convs_3b: List[List[Dict[str, str]]] = []
        for ep in chunk_eps:
            eid = ep.episode_id
            dsnap = diff_snaps[eid]
            a1_txt = arm1_raws[eid]
            t1_txt = t1_raws[eid]
            p1 = arm3_state_diff.pass1_audit_candidate(dsnap, a1_txt, turn1_raw_output=t1_txt)
            arm3b_pass1_map[eid] = p1
            if not p1.clean:
                rep_prompt = arm3_state_diff.build_arm3b_repair_prompt(dsnap, p1, turn1_raw_output=t1_txt)
                flagged_eps_3b.append(ep)
                repair_convs_3b.append(
                    [
                        {"role": "system", "content": sys_base},
                        {"role": "user", "content": ep.turn1_prompt},
                        {"role": "assistant", "content": t1_txt},
                        {"role": "user", "content": ep.turn2_prompt},
                        {"role": "assistant", "content": a1_txt},
                        {"role": "user", "content": rep_prompt},
                    ]
                )
            else:
                arm3b_secs[eid] = 0.0
                arm3b_in_toks[eid] = 0
                arm3b_out_toks[eid] = 0
                arm3b_extra_turns[eid] = 0

        repaired_raw_map_3b: Dict[str, str] = {}
        if flagged_eps_3b:
            rep_gen_3b = gen_batch_fn(repair_convs_3b)
            for ep, (rtxt, dt, itok, otok) in zip(flagged_eps_3b, rep_gen_3b):
                eid = ep.episode_id
                repaired_raw_map_3b[eid] = rtxt
                arm3b_secs[eid] = round(dt, 4)
                arm3b_in_toks[eid] = itok
                arm3b_out_toks[eid] = otok
                arm3b_extra_turns[eid] = 1

        for ep in chunk_eps:
            eid = ep.episode_id
            a1_txt = arm1_raws[eid]
            t1_txt = t1_raws[eid]
            pres_dict = dict(ep.preserved_targets) if ep.preserved_targets else None

            sg_res = arm3_state_diff.run_arm3b_state_diff_verifier(
                turn1_turns=ep.turn1_turns,
                turn2_turns=ep.turn2_turns,
                raw_turn2_output=a1_txt,
                target_node_id=ep.target_node_id,
                expected_unit=ep.expected_unit,
                preserved_targets=pres_dict,
                turn1_raw_output=t1_txt,
                mode="structural_gate",
            )
            arm3b_struct_res[eid] = sg_res
            arm3b_struct_evals[eid] = _score_turn2(ep, sg_res.final_json)

            rep_candidate = repaired_raw_map_3b.get(eid)
            llm_res = arm3_state_diff.run_arm3b_state_diff_verifier(
                turn1_turns=ep.turn1_turns,
                turn2_turns=ep.turn2_turns,
                raw_turn2_output=a1_txt,
                target_node_id=ep.target_node_id,
                expected_unit=ep.expected_unit,
                preserved_targets=pres_dict,
                turn1_raw_output=t1_txt,
                mode="llm_repair",
                repaired_turn2_output=rep_candidate,
            )
            arm3b_llm_res[eid] = llm_res
            arm3b_llm_raws[eid] = rep_candidate if rep_candidate is not None else a1_txt
            arm3b_llm_evals[eid] = _score_turn2(ep, llm_res.final_json)

            # Pass turn1_raw_output=None for symbolic_ceiling so Pass-2 audit compares against exact DAG truth
            # even when the LLM's Turn-1 `d_indep` commitment had a baseline Turn-1 arithmetic error.
            ceil_res = arm3_state_diff.run_arm3b_state_diff_verifier(
                turn1_turns=ep.turn1_turns,
                turn2_turns=ep.turn2_turns,
                raw_turn2_output=a1_txt,
                target_node_id=ep.target_node_id,
                expected_unit=ep.expected_unit,
                preserved_targets=pres_dict,
                turn1_raw_output=None,
                mode="symbolic_ceiling",
            )
            arm3b_ceil_res[eid] = ceil_res
            arm3b_ceil_evals[eid] = _score_turn2(ep, ceil_res.final_json)

    # 5. Optional Arm 3AB-Hybrid (`arm3ab_hybrid`: apply Arm 3B Pass-1 + 1-turn LLM repair on `arm3aStructuralOnly`)
    arm3ab_evals: Dict[str, oracle.OracleEvaluationResult] = {}
    arm3ab_finals: Dict[str, str] = {}
    arm3ab_raws: Dict[str, str] = {}
    arm3ab_flags: Dict[str, List[str]] = {}
    arm3ab_triggered: Dict[str, bool] = {}
    arm3ab_extra_turns: Dict[str, int] = {}
    arm3ab_secs: Dict[str, float] = {}
    arm3ab_in_toks: Dict[str, int] = {}
    arm3ab_out_toks: Dict[str, int] = {}

    if "arm3ab_hybrid" in arms:
        flagged_eps_hyb: List[oracle.MultiTurnEpisode] = []
        repair_convs_hyb: List[List[Dict[str, str]]] = []
        hyb_pass1_map: Dict[str, arm3_state_diff.Pass1DiagnosticReport] = {}

        for ep in chunk_eps:
            eid = ep.episode_id
            dsnap = diff_snaps[eid]
            upstream_3a_json = arm3a_struct_decs[eid].emitted_json
            t1_txt = t1_raws[eid]
            p1_hyb = arm3_state_diff.pass1_audit_candidate(
                dsnap, upstream_3a_json, turn1_raw_output=t1_txt
            )
            hyb_pass1_map[eid] = p1_hyb
            arm3ab_flags[eid] = list(p1_hyb.flags)
            if not p1_hyb.clean:
                rep_prompt_hyb = arm3_state_diff.build_arm3b_repair_prompt(
                    dsnap, p1_hyb, turn1_raw_output=t1_txt
                )
                flagged_eps_hyb.append(ep)
                repair_convs_hyb.append(
                    [
                        {"role": "system", "content": sys_base},
                        {"role": "user", "content": ep.turn1_prompt},
                        {"role": "assistant", "content": t1_txt},
                        {"role": "user", "content": arm3a_aug_prompts[eid]},
                        {"role": "assistant", "content": upstream_3a_json},
                        {"role": "user", "content": rep_prompt_hyb},
                    ]
                )
            else:
                arm3ab_triggered[eid] = False
                arm3ab_extra_turns[eid] = 0
                arm3ab_secs[eid] = 0.0
                arm3ab_in_toks[eid] = 0
                arm3ab_out_toks[eid] = 0

        repaired_hyb_map: Dict[str, str] = {}
        if flagged_eps_hyb:
            rep_gen_hyb = gen_batch_fn(repair_convs_hyb)
            for ep, (rtxt, dt, itok, otok) in zip(flagged_eps_hyb, rep_gen_hyb):
                eid = ep.episode_id
                repaired_hyb_map[eid] = rtxt
                arm3ab_triggered[eid] = True
                arm3ab_extra_turns[eid] = 1
                arm3ab_secs[eid] = round(dt, 4)
                arm3ab_in_toks[eid] = itok
                arm3ab_out_toks[eid] = otok

        for ep in chunk_eps:
            eid = ep.episode_id
            upstream_3a_json = arm3a_struct_decs[eid].emitted_json
            t1_txt = t1_raws[eid]
            ledger = arm3a_ledgers[eid]
            if eid in repaired_hyb_map:
                raw_hyb = repaired_hyb_map[eid]
                # Pass repaired LLM response through non-numeric structural_only hygiene so R/preserved stay locked
                hyb_dec = arm3_ledger_hook.apply_arm3a_invariant_guard(
                    ledger, raw_hyb, mode="structural_only"
                )
                final_hyb = hyb_dec.emitted_json
            else:
                raw_hyb = upstream_3a_json
                final_hyb = upstream_3a_json
            arm3ab_raws[eid] = raw_hyb
            arm3ab_finals[eid] = final_hyb
            arm3ab_evals[eid] = _score_turn2(ep, final_hyb)

    # Assemble canonical JSONL episode records
    out_records: List[Dict[str, Any]] = []
    for ep in chunk_eps:
        eid = ep.episode_id
        t1_ev = t1_evals[eid]
        rec: Dict[str, Any] = {
            "episodeId": eid,
            "condition": ep.condition,
            "category": ep.category,
            "modelKey": model_key,
            "repoId": repo_id,
            "dtype": dtype,
            "turn1FromCache": t1_from_cache[eid],
            "turn1Ok": t1_ev.ok,
            "turn1Verdict": t1_ev.verdict,
            "turn1Reason": t1_ev.reason,
            "turn1RawOutput": t1_raws[eid],
            "turn1Sec": t1_secs[eid],
            "turn1InputTokens": t1_in_toks[eid],
            "turn1OutputTokens": t1_out_toks[eid],
        }

        if eid in arm1_evals:
            a1_ev = arm1_evals[eid]
            rec.update(
                {
                    "arm1FromCache": arm1_from_cache[eid],
                    "arm1Ok": a1_ev.ok,
                    "arm1Verdict": a1_ev.verdict,
                    "arm1Reason": a1_ev.reason,
                    "arm1RawOutput": arm1_raws[eid],
                    "arm1Sec": arm1_secs[eid],
                    "arm1InputTokens": arm1_in_toks[eid],
                    "arm1OutputTokens": arm1_out_toks[eid],
                }
            )
        else:
            rec.update(
                {
                    "arm1FromCache": False,
                    "arm1Ok": None,
                    "arm1Verdict": None,
                    "arm1Reason": "",
                    "arm1RawOutput": "",
                    "arm1Sec": 0.0,
                    "arm1InputTokens": 0,
                    "arm1OutputTokens": 0,
                }
            )

        if "arm3a" in arms or "arm3ab_hybrid" in arms:
            r_ev = arm3a_raw_evals[eid]
            s_ev = arm3a_struct_evals[eid]
            f_ev = arm3a_full_evals[eid]
            s_dec = arm3a_struct_decs[eid]
            rec.update(
                {
                    "arm3aPromptRawOk": r_ev.ok,
                    "arm3aPromptRawVerdict": r_ev.verdict,
                    "arm3aPromptRawReason": r_ev.reason,
                    "arm3aPromptRawOutput": arm3a_raws[eid],
                    "arm3aPromptRaw": arm3a_raws[eid],
                    "arm3aPromptRawIsPrimaryK2Eligible": True,
                    "arm3aStructuralOnlyOk": s_ev.ok,
                    "arm3aStructuralOnlyVerdict": s_ev.verdict,
                    "arm3aStructuralOnlyReason": s_ev.reason,
                    "arm3aStructuralOnlyOutput": s_dec.emitted_json,
                    "arm3aStructuralIntervened": s_dec.intervened,
                    "arm3aStructuralViolationsBlocked": list(s_dec.violations_blocked),
                    "arm3aStructuralOnlyIsPrimaryK2Eligible": True,
                    "arm3aFullCeilingOk": f_ev.ok,
                    "arm3aFullCeilingVerdict": f_ev.verdict,
                    "arm3aFullCeilingIsPrimaryK2Eligible": False,
                    "arm3aExtraTurns": 0,
                    "arm3aSec": arm3a_secs[eid],
                    "arm3aInputTokens": arm3a_in_toks[eid],
                    "arm3aOutputTokens": arm3a_out_toks[eid],
                }
            )

        if "arm3b" in arms:
            sg_ev = arm3b_struct_evals[eid]
            lr_ev = arm3b_llm_evals[eid]
            sc_ev = arm3b_ceil_evals[eid]
            lr_res = arm3b_llm_res[eid]
            p1 = arm3b_pass1_map[eid]
            rec.update(
                {
                    "arm3bStructuralGateOk": sg_ev.ok,
                    "arm3bStructuralGateVerdict": sg_ev.verdict,
                    "arm3bStructuralGateReason": sg_ev.reason,
                    "arm3bStructuralGateOutput": arm3b_struct_res[eid].final_json,
                    "arm3bStructuralGateIsPrimaryK2Eligible": True,
                    "arm3bLlmRepairOk": lr_ev.ok,
                    "arm3bLlmRepairVerdict": lr_ev.verdict,
                    "arm3bLlmRepairReason": lr_ev.reason,
                    "arm3bLlmRepairOutput": lr_res.final_json,
                    "arm3bLlmRepairRaw": arm3b_llm_raws[eid],
                    "arm3bLlmRepairIsPrimaryK2Eligible": True,
                    "arm3bRepairTriggered": not p1.clean,
                    "arm3bPass1Flags": list(p1.flags),
                    "arm3bExtraTurns": arm3b_extra_turns[eid],
                    "arm3bSymbolicCeilingOk": sc_ev.ok,
                    "arm3bSymbolicCeilingVerdict": sc_ev.verdict,
                    "arm3bSymbolicCeilingIsPrimaryK2Eligible": False,
                    "arm3bSec": arm3b_secs[eid],
                    "arm3bInputTokens": arm3b_in_toks[eid],
                    "arm3bOutputTokens": arm3b_out_toks[eid],
                }
            )

        if "arm3ab_hybrid" in arms:
            hy_ev = arm3ab_evals[eid]
            rec.update(
                {
                    "arm3abHybridOk": hy_ev.ok,
                    "arm3abHybridVerdict": hy_ev.verdict,
                    "arm3abHybridReason": hy_ev.reason,
                    "arm3abHybridOutput": arm3ab_finals[eid],
                    "arm3abHybridRaw": arm3ab_raws[eid],
                    "arm3abHybridIsPrimaryK2Eligible": True,
                    "arm3abHybridRepairTriggered": arm3ab_triggered[eid],
                    "arm3abHybridPass1Flags": arm3ab_flags[eid],
                    "arm3abHybridExtraTurns": arm3ab_extra_turns[eid],
                    "arm3abHybridSec": arm3ab_secs[eid],
                    "arm3abHybridInputTokens": arm3ab_in_toks[eid],
                    "arm3abHybridOutputTokens": arm3ab_out_toks[eid],
                }
            )

        out_records.append(rec)
    return out_records


def _summarize_arm_key(
    records: Sequence[Dict[str, Any]],
    ok_key: str,
    verdict_key: Optional[str],
    is_primary_k2_eligible: bool,
    extra_turns_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    valid_recs = [r for r in records if r.get(ok_key) is not None]
    if not valid_recs:
        return None
    n = len(valid_recs)
    pass_cnt = sum(1 for r in valid_recs if r.get(ok_key) is True)
    cond_a = [r for r in valid_recs if r.get("condition") == "A"]
    cond_be = [r for r in valid_recs if r.get("condition") != "A"]
    cond_a_fail = sum(1 for r in cond_a if r.get(ok_key) is False)
    cond_be_pass = sum(1 for r in cond_be if r.get(ok_key) is True)
    verdicts: Dict[str, int] = {}
    if verdict_key:
        for r in valid_recs:
            v = str(r.get(verdict_key) or "")
            if v:
                verdicts[v] = verdicts.get(v, 0) + 1
    out: Dict[str, Any] = {
        "n": n,
        "passCount": pass_cnt,
        "accuracy": round(pass_cnt / n, 6),
        "condATotal": len(cond_a),
        "condAFailCount": cond_a_fail,
        "condBETotal": len(cond_be),
        "condBEPassCount": cond_be_pass,
        "isPrimaryK2Eligible": is_primary_k2_eligible,
        "verdictCounts": verdicts,
    }
    if extra_turns_key:
        out["extraInferenceTurns"] = sum(int(r.get(extra_turns_key, 0) or 0) for r in valid_recs)
    return out


def build_run_summary(
    run_tag: str,
    model_key: str,
    repo_id: str,
    dtype: str,
    seed: int,
    dataset_count: int,
    arms: Sequence[str],
    records: Sequence[Dict[str, Any]],
    load_sec: float,
    wall_sec: float,
    oom_backoff_events: int = 0,
) -> Dict[str, Any]:
    alloc_gb, res_gb = gpu_mem_gb()
    n = len(records)
    t1_clean = sum(1 for r in records if r.get("turn1Ok") is True)

    primary_arms: Dict[str, Any] = {}
    for label, ok_k, v_k, ex_k in (
        ("arm1_baseline", "arm1Ok", "arm1Verdict", None),
        ("arm3a_prompt_raw", "arm3aPromptRawOk", "arm3aPromptRawVerdict", "arm3aExtraTurns"),
        ("arm3a_structural_only", "arm3aStructuralOnlyOk", "arm3aStructuralOnlyVerdict", "arm3aExtraTurns"),
        ("arm3b_structural_gate", "arm3bStructuralGateOk", "arm3bStructuralGateVerdict", None),
        ("arm3b_llm_repair", "arm3bLlmRepairOk", "arm3bLlmRepairVerdict", "arm3bExtraTurns"),
        ("arm3ab_hybrid", "arm3abHybridOk", "arm3abHybridVerdict", "arm3abHybridExtraTurns"),
    ):
        stat = _summarize_arm_key(records, ok_k, v_k, is_primary_k2_eligible=True, extra_turns_key=ex_k)
        if stat is not None:
            primary_arms[label] = stat

    secondary_ceilings: Dict[str, Any] = {}
    for label, ok_k, v_k in (
        ("arm3a_full_ceiling", "arm3aFullCeilingOk", "arm3aFullCeilingVerdict"),
        ("arm3b_symbolic_ceiling", "arm3bSymbolicCeilingOk", "arm3bSymbolicCeilingVerdict"),
    ):
        stat = _summarize_arm_key(records, ok_k, v_k, is_primary_k2_eligible=False)
        if stat is not None:
            secondary_ceilings[label] = stat

    return {
        "runTag": run_tag,
        "modelKey": model_key,
        "repoId": repo_id,
        "dtype": dtype,
        "seed": seed,
        "datasetCount": dataset_count,
        "evaluatedEpisodes": n,
        "armsRequested": list(arms),
        "turn1CleanCount": t1_clean,
        "turn1CleanRate": round(t1_clean / max(1, n), 6),
        "loadSec": load_sec,
        "wallSec": round(wall_sec, 3),
        "peakVramAllocGb": alloc_gb,
        "peakVramReservedGb": res_gb,
        "oomBackoffEvents": oom_backoff_events,
        "primaryK2Arms": primary_arms,
        "secondarySymbolicCeilings": secondary_ceilings,
    }


def run_arm3_pipeline(
    model_key: str,
    repo_id: str,
    seed: int = 20261151,
    count: int = 50,
    limit: Optional[int] = None,
    dtype: str = "nf4",
    batch_size: int = 8,
    max_new_tokens: int = 1536,
    assistant_prefill: str = DEFAULT_ASSISTANT_PREFILL,
    turn1_cache_path: Optional[Path] = None,
    arms: Sequence[str] = ("arm1", "arm3a", "arm3b"),
    enable_thinking: Optional[bool] = None,
    trust_remote_code: bool = False,
    chunk_size: int = 25,
    resume: bool = False,
    out_dir: Path = Path("/tmp/gyeol-arm3-out"),
    run_tag: Optional[str] = None,
    mock_backend: bool = False,
    custom_generate_batch_fn: Optional[BatchGenerateFn] = None,
) -> Tuple[Dict[str, Any], MockChatBackend | None]:
    """Programmatic entry point for running Arm 3 GPU or mock evaluations with chunked resume."""
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if dtype not in {"nf4", "bf16"}:
        raise ValueError(f"Unsupported --dtype {dtype!r}; expected 'nf4' or 'bf16'.")

    validated_arms = parse_arms_arg(",".join(arms)) if isinstance(arms, (list, tuple)) else parse_arms_arg(str(arms))
    all_episodes = oracle.generate_multiturn_dataset(seed=seed, count=count)
    target_episodes = list(all_episodes[:limit]) if (limit is not None and limit > 0) else list(all_episodes)

    out_dir.mkdir(parents=True, exist_ok=True)
    effective_tag = run_tag or f"{model_key}-{dtype}-s{seed}-n{count}"
    episodes_jsonl_path = out_dir / f"{effective_tag}-episodes.jsonl"
    summary_json_path = out_dir / f"{effective_tag}-summary.json"

    turn1_cache = load_turn1_cache(turn1_cache_path)

    if resume and episodes_jsonl_path.exists():
        existing_records, completed_ids = load_existing_episode_records(episodes_jsonl_path)
        # Rewrite deduplicated existing records cleanly before appending remaining chunks
        with episodes_jsonl_path.open("w", encoding="utf-8") as fh:
            for rec in existing_records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        existing_records, completed_ids = [], set()
        # Truncate output JSONL for fresh run
        episodes_jsonl_path.write_text("", encoding="utf-8")

    pending_episodes = [ep for ep in target_episodes if ep.episode_id not in completed_ids]

    mock_inst: Optional[MockChatBackend] = None
    hf_inst: Optional[HuggingFaceBatchBackend] = None
    load_sec = 0.0

    if custom_generate_batch_fn is not None:
        gen_fn = custom_generate_batch_fn
    elif mock_backend:
        mock_inst = MockChatBackend(all_episodes)
        gen_fn = mock_inst.generate_batch
    else:
        if pending_episodes:
            hf_inst = HuggingFaceBatchBackend(
                repo_id=repo_id,
                dtype=dtype,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                assistant_prefill=assistant_prefill,
                enable_thinking=enable_thinking,
                trust_remote_code=trust_remote_code,
            )
            load_sec = hf_inst.load_sec
            gen_fn = hf_inst.generate_batch
        else:
            gen_fn = lambda convs: []

    t_wall_start = time.perf_counter()
    all_records: List[Dict[str, Any]] = list(existing_records)

    try:
        for start_idx in range(0, len(pending_episodes), chunk_size):
            chunk_eps = pending_episodes[start_idx : start_idx + chunk_size]
            chunk_recs = evaluate_chunk(
                chunk_eps=chunk_eps,
                arms=validated_arms,
                gen_batch_fn=gen_fn,
                turn1_cache=turn1_cache,
                model_key=model_key,
                repo_id=repo_id,
                dtype=dtype,
            )
            with episodes_jsonl_path.open("a", encoding="utf-8") as fh:
                for rec in chunk_recs:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            all_records.extend(chunk_recs)

            alloc_gb, res_gb = gpu_mem_gb()
            print(
                f"[{effective_tag}] progress {len(all_records)}/{len(target_episodes)} "
                f"(chunk +{len(chunk_recs)}) | vram_res={res_gb}GB",
                flush=True,
            )
    finally:
        oom_events = hf_inst.oom_backoff_events if hf_inst is not None else 0
        if hf_inst is not None:
            hf_inst.close()

    # Order final summary over canonical target_episodes order without duplicating IDs
    rec_by_id = {r["episodeId"]: r for r in all_records}
    ordered_records = [rec_by_id[ep.episode_id] for ep in target_episodes if ep.episode_id in rec_by_id]

    wall_sec = time.perf_counter() - t_wall_start
    summary = build_run_summary(
        run_tag=effective_tag,
        model_key=model_key,
        repo_id=repo_id,
        dtype=dtype,
        seed=seed,
        dataset_count=count,
        arms=validated_arms,
        records=ordered_records,
        load_sec=load_sec,
        wall_sec=wall_sec,
        oom_backoff_events=oom_events,
    )
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary, mock_inst


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run Arm 3 (Arm 3A / Arm 3B / Arm 3AB-Hybrid) evaluation on Colab GPU or local mock backend."
    )
    ap.add_argument("--model-key", required=True, help="Short model identifier (e.g., phi4, qwen14b, mistral24b)")
    ap.add_argument("--repo-id", required=True, help="HuggingFace model repository ID")
    ap.add_argument("--seed", type=int, default=20261151, help="Dataset seed (20261151 pilot or 20261122 main)")
    ap.add_argument("--count", type=int, default=50, help="Dataset episode count (50 or 800)")
    ap.add_argument("--limit", type=int, default=None, help="Optional subset slice for smoke/chunk testing")
    ap.add_argument("--dtype", choices=["nf4", "bf16"], default="nf4", help="Quantization / precision mode")
    ap.add_argument("--batch-size", type=int, default=8, help="Left-padded generation batch size (auto OOM backoff)")
    ap.add_argument("--max-new-tokens", type=int, default=1536, help="Maximum generated tokens per turn")
    ap.add_argument(
        "--assistant-prefill",
        default=DEFAULT_ASSISTANT_PREFILL,
        help='Symmetric assistant JSON prefill appended to every turn prompt (default: \'{"notes": "\'; "" disables)',
    )
    ap.add_argument("--turn1-cache", type=Path, default=None, help="Optional JSONL path to reuse Turn-1 & Arm-1 cache")
    ap.add_argument(
        "--arms",
        default="arm1,arm3a,arm3b",
        help="Comma-separated arms from arm1,arm3a,arm3b,arm3ab_hybrid",
    )
    ap.add_argument("--enable-thinking", choices=["none", "false", "true"], default="none")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=25, help="Incremental JSONL flush interval")
    ap.add_argument("--resume", action="store_true", help="Resume from existing <run_tag>-episodes.jsonl")
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/gyeol-arm3-out"), help="Output directory")
    ap.add_argument("--run-tag", default=None, help="Optional filename prefix tag (default: <model>-<dtype>-s<seed>-n<N>)")
    ap.add_argument("--mock-backend", action="store_true", help="Use deterministic CPU mock backend for unit tests")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    eth: Optional[bool] = None if args.enable_thinking == "none" else (args.enable_thinking == "true")
    summary, _ = run_arm3_pipeline(
        model_key=args.model_key,
        repo_id=args.repo_id,
        seed=args.seed,
        count=args.count,
        limit=args.limit,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        assistant_prefill=args.assistant_prefill,
        turn1_cache_path=args.turn1_cache,
        arms=parse_arms_arg(args.arms),
        enable_thinking=eth,
        trust_remote_code=args.trust_remote_code,
        chunk_size=args.chunk_size,
        resume=args.resume,
        out_dir=args.out_dir,
        run_tag=args.run_tag,
        mock_backend=args.mock_backend,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
