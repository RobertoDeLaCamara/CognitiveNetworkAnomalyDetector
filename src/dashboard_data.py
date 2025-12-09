"""Data loading utilities for the Streamlit dashboard."""

import re
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

from .dashboard_config import (
    LOG_FILE, MODEL_DIR, LOG_PATTERNS, MAX_LOG_LINES,
    FEATURE_NAMES, DATE_FORMAT, get_mlflow_uri
)
from .logger_setup import logger


@dataclass
class AnomalyRecord:
    """Represents a single anomaly detection."""
    timestamp: datetime
    ip_address: str
    anomaly_score: float
    alert_type: str  # "ML" or "RULE"
    description: str
    features: Optional[Dict[str, float]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame compatibility."""
        d = asdict(self)
        d['timestamp'] = self.timestamp
        return d


class AnomalyDataLoader:
    """Load and parse anomaly detection logs."""

    def __init__(self, log_file: Path = LOG_FILE):
        """Initialize the data loader.
        
        Args:
            log_file: Path to anomaly detection log file
        """
        self.log_file = Path(log_file)
        self._cache: Optional[List[AnomalyRecord]] = None
        self._cache_time: Optional[datetime] = None

    def load_anomalies(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_records: int = MAX_LOG_LINES
    ) -> pd.DataFrame:
        """Load anomaly records from log file.
        
        Args:
            start_date: Filter anomalies after this date
            end_date: Filter anomalies before this date
            max_records: Maximum number of records to load
            
        Returns:
            DataFrame with anomaly records
        """
        records = self._parse_log_file(max_records)
        
        # Filter by date range
        if start_date:
            records = [r for r in records if r.timestamp >= start_date]
        if end_date:
            records = [r for r in records if r.timestamp <= end_date]
        
        if not records:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=[
                'timestamp', 'ip_address', 'anomaly_score', 
                'alert_type', 'description'
            ])
        
        # Convert to DataFrame
        df = pd.DataFrame([r.to_dict() for r in records])
        df = df.sort_values('timestamp', ascending=False)
        
        return df

    def _parse_log_file(self, max_lines: int) -> List[AnomalyRecord]:
        """Parse the anomaly log file.
        
        Args:
            max_lines: Maximum number of lines to read
            
        Returns:
            List of AnomalyRecord objects
        """
        if not self.log_file.exists():
            logger.warning(f"Log file not found: {self.log_file}")
            return []
        
        records = []
        
        try:
            with open(self.log_file, 'r') as f:
                # Read last N lines (most recent)
                lines = f.readlines()[-max_lines:]
            
            for line in lines:
                record = self._parse_log_line(line)
                if record:
                    records.append(record)
        
        except Exception as e:
            logger.error(f"Error parsing log file: {e}")
        
        return records

    def _parse_log_line(self, line: str) -> Optional[AnomalyRecord]:
        """Parse a single log line into an AnomalyRecord.
        
        Args:
            line: Log line to parse
            
        Returns:
            AnomalyRecord or None if line doesn't contain alert
        """
        try:
            # Extract timestamp
            timestamp_match = re.search(LOG_PATTERNS["timestamp"], line)
            if not timestamp_match:
                return None
            
            timestamp_str = timestamp_match.group(1)
            timestamp = datetime.strptime(timestamp_str, DATE_FORMAT)
            
            # Check for ML anomaly
            ml_match = re.search(LOG_PATTERNS["ml_alert"], line)
            if ml_match:
                ip_address = ml_match.group(1)
                anomaly_score = float(ml_match.group(2))
                
                return AnomalyRecord(
                    timestamp=timestamp,
                    ip_address=ip_address,
                    anomaly_score=anomaly_score,
                    alert_type="ML",
                    description=f"ML-detected anomaly (score: {anomaly_score:.3f})"
                )
            
            # Check for rule-based anomaly
            rule_match = re.search(LOG_PATTERNS["rule_alert"], line)
            if rule_match:
                description = rule_match.group(1)
                
                # Try to extract IP from description using the "from X.X.X.X" pattern
                ip_match = re.search(LOG_PATTERNS["ip_address"], description)
                if ip_match:
                    ip_address = ip_match.group(1)
                else:
                    # Fallback: try any IP pattern
                    ip_match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', description)
                    ip_address = ip_match.group(1) if ip_match else "unknown"
                
                return AnomalyRecord(
                    timestamp=timestamp,
                    ip_address=ip_address,
                    anomaly_score=0.0,  # Rule-based don't have scores
                    alert_type="RULE",
                    description=description
                )
        
        except Exception as e:
            logger.debug(f"Failed to parse log line: {e}")
        
        return None

    def get_recent_anomalies(self, minutes: int = 5) -> pd.DataFrame:
        """Get anomalies from the last N minutes.
        
        Args:
            minutes: Number of minutes to look back
            
        Returns:
            DataFrame with recent anomalies
        """
        start_date = datetime.now() - timedelta(minutes=minutes)
        return self.load_anomalies(start_date=start_date)

    def get_anomaly_stats(self) -> Dict[str, any]:
        """Get summary statistics about anomalies.
        
        Returns:
            Dictionary with statistics
        """
        df = self.load_anomalies()
        
        if df.empty:
            return {
                "total_anomalies": 0,
                "ml_anomalies": 0,
                "rule_anomalies": 0,
                "unique_ips": 0,
                "avg_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
            }
        
        ml_anomalies = df[df['alert_type'] == 'ML']
        
        return {
            "total_anomalies": len(df),
            "ml_anomalies": len(ml_anomalies),
            "rule_anomalies": len(df) - len(ml_anomalies),
            "unique_ips": df['ip_address'].nunique(),
            "avg_score": ml_anomalies['anomaly_score'].mean() if len(ml_anomalies) > 0 else 0.0,
            "min_score": ml_anomalies['anomaly_score'].min() if len(ml_anomalies) > 0 else 0.0,
            "max_score": ml_anomalies['anomaly_score'].max() if len(ml_anomalies) > 0 else 0.0,
        }


class ModelMetricsLoader:
    """Load model metadata and metrics."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        """Initialize the model metrics loader.
        
        Args:
            model_dir: Path to model directory
        """
        self.model_dir = Path(model_dir)

    def get_model_info(self) -> Optional[Dict[str, any]]:
        """Get information about the current model.
        
        Returns:
            Dictionary with model metadata or None if not found
        """
        model_path = self.model_dir / "isolation_forest_model.pkl"
        
        if not model_path.exists():
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Extract metadata if available
            metadata = model_data.get('metadata', {})
            
            return {
                "model_type": metadata.get('model_type', 'IsolationForest'),
                "n_features": metadata.get('n_features', len(FEATURE_NAMES)),
                "contamination": metadata.get('contamination', 'unknown'),
                "n_estimators": metadata.get('n_estimators', 'unknown'),
                "trained_date": metadata.get('trained_date', 'unknown'),
                "model_version": metadata.get('model_version', 'unknown'),
                "file_size": model_path.stat().st_size / 1024,  # KB
            }
        
        except Exception as e:
            logger.error(f"Error loading model metadata: {e}")
            return None

    def is_model_loaded(self) -> bool:
        """Check if a trained model exists.
        
        Returns:
            True if model file exists
        """
        model_path = self.model_dir / "isolation_forest_model.pkl"
        return model_path.exists()


class MLflowDataLoader:
    """Load data from MLflow tracking server."""

    def __init__(self):
        """Initialize MLflow data loader."""
        self.tracking_uri = get_mlflow_uri()
        self.enabled = self.tracking_uri is not None

    def get_experiments(self, limit: int = 10) -> List[Dict[str, any]]:
        """Get list of MLflow experiments.
        
        Args:
            limit: Maximum number of experiments to return
            
        Returns:
            List of experiment dictionaries
        """
        if not self.enabled:
            return []
        
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            
            experiments = mlflow.search_experiments(max_results=limit)
            
            return [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage,
                }
                for exp in experiments
            ]
        
        except Exception as e:
            logger.error(f"Error loading MLflow experiments: {e}")
            return []

    def get_recent_runs(self, experiment_name: str = "cognitive-anomaly-detector", limit: int = 20) -> pd.DataFrame:
        """Get recent MLflow runs.
        
        Args:
            experiment_name: Name of the experiment
            limit: Maximum number of runs to return
            
        Returns:
            DataFrame with run information
        """
        if not self.enabled:
            return pd.DataFrame()
        
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Get experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                return pd.DataFrame()
            
            # Search runs
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=limit,
                order_by=["start_time DESC"]
            )
            
            return runs
        
        except Exception as e:
            logger.error(f"Error loading MLflow runs: {e}")
            return pd.DataFrame()
