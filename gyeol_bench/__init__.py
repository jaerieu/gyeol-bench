"""Gyeol-Bench: Clean-room quantitative premise retraction and SI unit cascade benchmark."""

from .oracle import (
    DimensionMismatchError,
    EnvelopeParseResult,
    ExactRationalOracle,
    OracleEvaluationResult,
    RetractionEpisode,
    TurnSpec,
    evaluate_retraction_episode,
    generate_procedural_dataset,
    mixed_tolerance_match,
    normalize_to_base_fraction,
    parse_step_envelope,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DimensionMismatchError",
    "EnvelopeParseResult",
    "ExactRationalOracle",
    "OracleEvaluationResult",
    "RetractionEpisode",
    "TurnSpec",
    "evaluate_retraction_episode",
    "generate_procedural_dataset",
    "mixed_tolerance_match",
    "normalize_to_base_fraction",
    "parse_step_envelope",
]
