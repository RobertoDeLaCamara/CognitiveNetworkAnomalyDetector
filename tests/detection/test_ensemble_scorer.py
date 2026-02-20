"""Tests for the ensemble scoring module."""

import os
import pytest
from src.detection.ensemble_scorer import EnsembleScorer, EngineResult, EnsembleResult


class TestNormalization:
    """Test normalization functions for each engine."""

    def test_normalize_isolation_forest_anomalous(self):
        # Negative IF score = anomalous → should map near 1.0
        assert EnsembleScorer.normalize_isolation_forest(-0.5) == 1.0

    def test_normalize_isolation_forest_normal(self):
        # Positive IF score = normal → should map near 0.0
        assert EnsembleScorer.normalize_isolation_forest(0.5) == 0.0

    def test_normalize_isolation_forest_zero(self):
        assert EnsembleScorer.normalize_isolation_forest(0.0) == 0.5

    def test_normalize_isolation_forest_clipping(self):
        # Beyond expected range
        assert EnsembleScorer.normalize_isolation_forest(-1.0) == 1.0
        assert EnsembleScorer.normalize_isolation_forest(1.0) == 0.0

    def test_normalize_lstm_at_threshold(self):
        # Error == threshold → should be 0.5
        assert EnsembleScorer.normalize_lstm(0.1, 0.1) == 0.5

    def test_normalize_lstm_zero_error(self):
        assert EnsembleScorer.normalize_lstm(0.0, 0.1) == 0.0

    def test_normalize_lstm_high_error(self):
        # Error == 2*threshold → should be 1.0
        assert EnsembleScorer.normalize_lstm(0.2, 0.1) == 1.0

    def test_normalize_lstm_clipping(self):
        # Beyond 2*threshold
        assert EnsembleScorer.normalize_lstm(0.5, 0.1) == 1.0

    def test_normalize_lstm_zero_threshold(self):
        assert EnsembleScorer.normalize_lstm(0.5, 0.0) == 0.0

    def test_normalize_rules_triggered(self):
        assert EnsembleScorer.normalize_rules(True) == 1.0

    def test_normalize_rules_not_triggered(self):
        assert EnsembleScorer.normalize_rules(False) == 0.0


class TestCombine:
    """Test the combine method for ensemble scoring."""

    def setup_method(self):
        self.scorer = EnsembleScorer()

    def test_combine_all_engines_anomalous(self):
        results = [
            EngineResult("isolation_forest", -0.5, 1.0, -1),
            EngineResult("lstm", 0.2, 1.0, -1),
            EngineResult("rules", 1.0, 1.0, -1),
        ]
        ensemble = self.scorer.combine(results)
        assert ensemble.confidence_score == 1.0
        assert ensemble.is_anomaly is True

    def test_combine_all_engines_normal(self):
        results = [
            EngineResult("isolation_forest", 0.5, 0.0, 1),
            EngineResult("lstm", 0.0, 0.0, 1),
            EngineResult("rules", 0.0, 0.0, 1),
        ]
        ensemble = self.scorer.combine(results)
        assert ensemble.confidence_score == 0.0
        assert ensemble.is_anomaly is False

    def test_combine_missing_lstm_engine(self):
        """When LSTM has prediction=0, its weight is redistributed."""
        results = [
            EngineResult("isolation_forest", -0.3, 0.8, -1),
            EngineResult("lstm", 0.0, 0.0, 0),  # no data
            EngineResult("rules", 1.0, 1.0, -1),
        ]
        ensemble = self.scorer.combine(results)
        # Only IF (0.4) and rules (0.2) active → weights: 0.4/0.6, 0.2/0.6
        expected = (0.8 * (0.4 / 0.6)) + (1.0 * (0.2 / 0.6))
        assert abs(ensemble.confidence_score - expected) < 1e-6
        assert ensemble.is_anomaly is True

    def test_combine_no_engines(self):
        ensemble = self.scorer.combine([])
        assert ensemble.confidence_score == 0.0
        assert ensemble.is_anomaly is False

    def test_combine_all_engines_no_data(self):
        results = [
            EngineResult("isolation_forest", 0.0, 0.0, 0),
            EngineResult("lstm", 0.0, 0.0, 0),
            EngineResult("rules", 0.0, 0.0, 0),
        ]
        ensemble = self.scorer.combine(results)
        assert ensemble.confidence_score == 0.0
        assert ensemble.is_anomaly is False

    def test_combine_single_engine(self):
        results = [
            EngineResult("isolation_forest", -0.3, 0.8, -1),
        ]
        ensemble = self.scorer.combine(results)
        assert ensemble.confidence_score == 0.8
        assert ensemble.is_anomaly is True

    def test_threshold_boundary_below(self):
        """Score just below threshold should not be anomaly."""
        self.scorer.threshold = 0.6
        results = [
            EngineResult("isolation_forest", 0.0, 0.59, -1),
        ]
        ensemble = self.scorer.combine(results)
        assert ensemble.is_anomaly is False

    def test_threshold_boundary_at(self):
        """Score exactly at threshold should be anomaly."""
        self.scorer.threshold = 0.6
        results = [
            EngineResult("isolation_forest", 0.0, 0.6, -1),
        ]
        ensemble = self.scorer.combine(results)
        assert ensemble.is_anomaly is True

    def test_weight_redistribution_correctness(self):
        """Redistributed weights should sum to 1.0."""
        results = [
            EngineResult("isolation_forest", -0.3, 0.8, -1),
            EngineResult("rules", 1.0, 1.0, -1),
        ]
        # IF weight=0.4, rules weight=0.2 → total=0.6
        # Redistributed: IF=0.4/0.6=0.667, rules=0.2/0.6=0.333
        ensemble = self.scorer.combine(results)
        expected = 0.8 * (0.4 / 0.6) + 1.0 * (0.2 / 0.6)
        assert abs(ensemble.confidence_score - expected) < 1e-6


class TestEngineSummary:
    """Test the engine_summary property."""

    def test_engine_summary_format(self):
        result = EnsembleResult(
            confidence_score=0.7,
            is_anomaly=True,
            engines={
                "isolation_forest": EngineResult("isolation_forest", -0.3, 0.8, -1),
                "rules": EngineResult("rules", 1.0, 1.0, -1),
            },
        )
        summary = result.engine_summary
        assert "isolation_forest" in summary
        assert summary["isolation_forest"]["raw_score"] == -0.3
        assert summary["isolation_forest"]["normalized_score"] == 0.8
        assert summary["isolation_forest"]["prediction"] == -1


class TestEnvConfiguration:
    """Test that environment variables configure the scorer."""

    def test_custom_weights(self, monkeypatch):
        monkeypatch.setenv("ENSEMBLE_WEIGHT_IF", "0.5")
        monkeypatch.setenv("ENSEMBLE_WEIGHT_LSTM", "0.3")
        monkeypatch.setenv("ENSEMBLE_WEIGHT_RULES", "0.2")
        scorer = EnsembleScorer()
        assert scorer.weights["isolation_forest"] == 0.5
        assert scorer.weights["lstm"] == 0.3
        assert scorer.weights["rules"] == 0.2

    def test_custom_threshold(self, monkeypatch):
        monkeypatch.setenv("ENSEMBLE_ANOMALY_THRESHOLD", "0.8")
        scorer = EnsembleScorer()
        assert scorer.threshold == 0.8
