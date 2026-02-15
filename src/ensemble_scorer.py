"""Ensemble scoring module that combines multiple detection engines.

Combines Isolation Forest, LSTM Autoencoder, and rule-based detection
into a single confidence score using configurable weighted averaging.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EngineResult:
    """Result from a single detection engine."""
    name: str
    raw_score: float
    normalized_score: float
    prediction: int  # -1=anomaly, 1=normal, 0=no data


@dataclass
class EnsembleResult:
    """Combined result from all detection engines."""
    confidence_score: float  # 0-1 where 1=most anomalous
    is_anomaly: bool
    engines: Dict[str, EngineResult] = field(default_factory=dict)

    @property
    def engine_summary(self) -> Dict[str, float]:
        """Summary of engine scores for logging/persistence."""
        return {
            name: {
                "raw_score": er.raw_score,
                "normalized_score": er.normalized_score,
                "prediction": er.prediction,
            }
            for name, er in self.engines.items()
        }


class EnsembleScorer:
    """Combines scores from multiple anomaly detection engines.

    Weights are configurable via environment variables:
      ENSEMBLE_WEIGHT_IF   - Isolation Forest weight (default 0.4)
      ENSEMBLE_WEIGHT_LSTM - LSTM Autoencoder weight (default 0.4)
      ENSEMBLE_WEIGHT_RULES - Rule-based weight (default 0.2)
      ENSEMBLE_ANOMALY_THRESHOLD - Threshold for anomaly (default 0.6)
    """

    def __init__(self):
        self.weights = {
            "isolation_forest": float(os.getenv("ENSEMBLE_WEIGHT_IF", "0.4")),
            "lstm": float(os.getenv("ENSEMBLE_WEIGHT_LSTM", "0.4")),
            "rules": float(os.getenv("ENSEMBLE_WEIGHT_RULES", "0.2")),
        }
        self.threshold = float(os.getenv("ENSEMBLE_ANOMALY_THRESHOLD", "0.6"))

    @staticmethod
    def normalize_isolation_forest(score: float) -> float:
        """Normalize IF score to [0, 1] where 1 = most anomalous.

        IF decision_function scores: negative = anomalous, positive = normal.
        Typical range is roughly [-0.5, 0.5].
        """
        # Map [-0.5, 0.5] -> [1.0, 0.0]
        normalized = 0.5 - score
        return max(0.0, min(1.0, normalized))

    @staticmethod
    def normalize_lstm(reconstruction_error: float, threshold: float) -> float:
        """Normalize LSTM reconstruction error to [0, 1] where 1 = most anomalous.

        Divides by 2x threshold so that the threshold maps to 0.5.
        """
        if threshold <= 0:
            return 0.0
        normalized = reconstruction_error / (2.0 * threshold)
        return max(0.0, min(1.0, normalized))

    @staticmethod
    def normalize_rules(triggered: bool) -> float:
        """Normalize rule-based result: 1.0 if any rule triggered, 0.0 otherwise."""
        return 1.0 if triggered else 0.0

    def combine(self, results: List[EngineResult]) -> EnsembleResult:
        """Combine engine results into a single ensemble score.

        Engines with prediction=0 (no data) are excluded and their weight
        is redistributed proportionally to the remaining engines.
        """
        engines_dict: Dict[str, EngineResult] = {}
        active_results: List[EngineResult] = []
        total_active_weight = 0.0

        for result in results:
            engines_dict[result.name] = result
            if result.prediction != 0:
                active_results.append(result)
                total_active_weight += self.weights.get(result.name, 0.0)

        if not active_results or total_active_weight == 0.0:
            return EnsembleResult(
                confidence_score=0.0,
                is_anomaly=False,
                engines=engines_dict,
            )

        # Weighted average with redistribution
        weighted_sum = 0.0
        for result in active_results:
            weight = self.weights.get(result.name, 0.0)
            redistributed_weight = weight / total_active_weight
            weighted_sum += result.normalized_score * redistributed_weight

        confidence = max(0.0, min(1.0, weighted_sum))

        return EnsembleResult(
            confidence_score=confidence,
            is_anomaly=confidence >= self.threshold,
            engines=engines_dict,
        )
