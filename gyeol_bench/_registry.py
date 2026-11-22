"""UK AISI Inspect AI external registry entry point for gyeol-bench (`2026-05-08` format).

Supports both `inspect eval gyeol_bench/retraction_dag` (when `inspect_ai >= 0.3.0` is installed)
and zero-dependency offline inspection in pure Python stdlib environments.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .oracle import (
    TurnSpec,
    evaluate_retraction_episode,
    generate_procedural_dataset,
)

try:
    from inspect_ai import Task, task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.scorer import CORRECT, INCORRECT, Metric, SampleScore, Score, Target, Value, accuracy, grouped, metric, scorer, stderr
    from inspect_ai.solver import TaskState, generate

    _INSPECT_AVAILABLE = True
except ImportError:  # pragma: no cover - clean stdlib fallback when inspect_ai is not installed
    _INSPECT_AVAILABLE = False


def _serialize_turns(turns: Any) -> List[Dict[str, Any]]:
    return [
        {
            "node_id": t.node_id, "kind": t.kind, "value": t.value, "unit": t.unit,
            "retraction_op": t.retraction_op, "target_id": t.target_id,
            "arith_op": t.arith_op, "left_id": t.left_id, "right_id": t.right_id,
        }
        for t in turns
    ]


def _deserialize_turns(items: List[Dict[str, Any]]) -> List[TurnSpec]:
    return [TurnSpec(**item) for item in items]


if _INSPECT_AVAILABLE:

    @metric
    def error_split_metrics() -> Metric:
        def calc(scores: List[SampleScore]) -> Value:
            n = len(scores)
            if not n:
                return {}
            verdicts = [(s.score.metadata or {}).get("verdict", "") for s in scores]
            valid = sum(1 for v in verdicts if v != "PARSE_REJECT")
            return {
                "conditional_accuracy": sum(1 for v in verdicts if v == "PASS") / valid if valid else 0.0,
                "PARSE_REJECT": sum(1 for v in verdicts if v == "PARSE_REJECT") / n,
                "CONDITIONAL_ARITH_ERROR": sum(1 for v in verdicts if v == "CONDITIONAL_ARITH_ERROR") / n,
                "STALE_STATE_RETENTION": sum(1 for v in verdicts if v == "STALE_STATE_RETENTION") / n,
                "OVER_RETRACTION": sum(1 for v in verdicts if v == "OVER_RETRACTION") / n,
            }
        return calc

    @scorer(metrics=[accuracy(), stderr(), grouped(accuracy(), "condition"), error_split_metrics()])
    def retraction_dag_scorer() -> Any:
        async def score(state: TaskState, target: Target) -> Score:
            meta = state.metadata or {}
            res = evaluate_retraction_episode(
                turns=_deserialize_turns(meta["turns"]),
                raw_model_output=state.output.completion if state.output else "",
                target_node_id=meta["target_node_id"],
                expected_unit=meta["expected_unit"],
                preserved_targets=meta.get("preserved_targets") or None,
            )
            return Score(
                value=CORRECT if res.ok else INCORRECT,
                answer=state.output.completion if state.output else "",
                explanation=res.verdict if res.ok else f"{res.verdict}: {res.reason}",
                reason="invalid_response_format" if res.verdict == "PARSE_REJECT" else None,
                metadata={"verdict": res.verdict, "condition": meta.get("condition", ""), "category": meta.get("category", "")},
            )
        return score

    @task
    def retraction_dag(seed: int = 20261122, sample_count: int = 800) -> Task:
        episodes = generate_procedural_dataset(seed=seed, count=sample_count)
        samples = [
            Sample(
                id=ep.episode_id,
                input=ep.prompt,
                target=json.dumps({
                    "step_id": ep.target_node_id,
                    "value": ep.expected_value_float,
                    "unit": ep.expected_unit,
                    "retracted_ids": list(ep.expected_retracted_ids),
                    **({"preserved": {k: {"unit": u} for k, u in ep.preserved_targets}} if ep.preserved_targets else {}),
                }),
                metadata={
                    "condition": ep.condition,
                    "category": ep.category,
                    "turns": _serialize_turns(ep.turns),
                    "target_node_id": ep.target_node_id,
                    "expected_unit": ep.expected_unit,
                    "preserved_targets": dict(ep.preserved_targets),
                },
            )
            for ep in episodes
        ]
        return Task(
            dataset=MemoryDataset(samples=samples, name="gyeol_bench_retraction_dag"),
            solver=[generate()],
            scorer=retraction_dag_scorer(),
        )

else:

    def retraction_dag(seed: int = 20261122, sample_count: int = 800) -> Dict[str, Any]:
        """Pure-stdlib task specification descriptor when inspect_ai is not installed locally."""
        episodes = generate_procedural_dataset(seed=seed, count=sample_count)
        conds = {c: sum(1 for e in episodes if e.condition == c) for c in ("A", "B", "C", "D", "E")}
        return {
            "name": "gyeol_bench/retraction_dag",
            "registry_spec_date": "2026-05-08",
            "inspect_ai_installed": False,
            "seed": seed,
            "sample_count": len(episodes),
            "conditions": conds,
            "categories": {"normal": conds["A"], "perturbed": len(episodes) - conds["A"]},
        }


__all__ = ["retraction_dag"]
