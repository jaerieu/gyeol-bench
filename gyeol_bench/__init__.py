"""Gyeol-Bench: Clean-room quantitative premise retraction and SI unit cascade benchmark."""

from .oracle import (
    DimensionMismatchError,
    EnvelopeParseResult,
    ExactRationalOracle,
    MultiTurnEpisode,
    OracleEvaluationResult,
    RetractionEpisode,
    TurnSpec,
    V3_MULTITURN_BASELINE_SYSTEM_PROMPT,
    V3_MULTITURN_LINEAGE_K2_SYSTEM_PROMPT,
    evaluate_multiturn_episode,
    evaluate_retraction_episode,
    generate_multiturn_dataset,
    generate_procedural_dataset,
    mixed_tolerance_match,
    normalize_to_base_fraction,
    parse_step_envelope,
    response_adapter,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DimensionMismatchError",
    "EnvelopeParseResult",
    "ExactRationalOracle",
    "MultiTurnEpisode",
    "OracleEvaluationResult",
    "RetractionEpisode",
    "TurnSpec",
    "V3_MULTITURN_BASELINE_SYSTEM_PROMPT",
    "V3_MULTITURN_LINEAGE_K2_SYSTEM_PROMPT",
    "evaluate_multiturn_episode",
    "evaluate_retraction_episode",
    "generate_multiturn_dataset",
    "generate_procedural_dataset",
    "mixed_tolerance_match",
    "normalize_to_base_fraction",
    "parse_step_envelope",
    "response_adapter",
]
