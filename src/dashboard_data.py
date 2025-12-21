"""Data loading utilities for the Streamlit dashboard."""

import re
import json
import joblib
import glob
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

from .dashboard_config import (
    LOG_FILE, MODEL_DIR, LOG_PATTERNS, MAX_LOG_LINES,
    FEATURE_NAMES, DATE_FORMAT, get_mlflow_uri, DB_FILE
)
from .logger_setup import logger
from .db_manager import DBManager


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
            log_file: Path to anomaly detection log file (kept for backward compatibility)
        """
        self.log_file = Path(log_file)
        # Initialize DB Manager
        try:
            self.db_manager = DBManager(str(DB_FILE))
        except Exception as e:
            logger.error(f"Failed to init DB manager for dashboard: {e}")
            self.db_manager = None
            
        self._cache: Optional[List[AnomalyRecord]] = None
        self._cache_time: Optional[datetime] = None

    def load_anomalies(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_records: int = MAX_LOG_LINES
    ) -> pd.DataFrame:
        """Load anomaly records from database.
        
        Args:
            start_date: Filter anomalies after this date
            end_date: Filter anomalies before this date
            max_records: Maximum number of records to load
            
        Returns:
            DataFrame with anomaly records
        """
        if not self.db_manager:
            return pd.DataFrame()
            
        # Fetch from DB
        records = self.db_manager.get_anomalies(
            start_date=start_date,
            end_date=end_date,
            limit=max_records
        )
        
        if not records:
             # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=[
                'timestamp', 'ip_address', 'anomaly_score', 
                'alert_type', 'description'
            ])

        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
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
        """Get summary statistics about anomalies."""
        if not self.db_manager:
            return {
                "total_anomalies": 0, "ml_anomalies": 0, "rule_anomalies": 0,
                "unique_ips": 0, "avg_score": 0.0, "min_score": 0.0, "max_score": 0.0
            }
            
        # Get basic counts from DB (fast)
        stats = self.db_manager.get_stats()
        
        # Get score stats (requires querying entries)
        # Optimization: Create a specific DB query for this later if needed
        # For now, just query ML entries to calc stats
        ml_df = self.load_anomalies(max_records=1000) # Only recent ones for speed? Or all?
        ml_df = ml_df[ml_df['alert_type'] == 'ML']
        
        if not ml_df.empty:
            stats.update({
                "avg_score": ml_df['anomaly_score'].mean(),
                "min_score": ml_df['anomaly_score'].min(),
                "max_score": ml_df['anomaly_score'].max(),
            })
        else:
             stats.update({
                "avg_score": 0.0, "min_score": 0.0, "max_score": 0.0
            })
            
        return stats

    def get_raw_logs(self, lines: int = 100) -> str:
        """Get the last N lines of the log file as a string.

        Args:
            lines: Number of lines to read

        Returns:
            String containing the log lines
        """
        if not self.log_file.exists():
            return "Log file not found."

        try:
            with open(self.log_file, 'r') as f:
                content = f.readlines()
                return "".join(content[-lines:])
        except Exception as e:
            return f"Error reading log file: {e}"

class ModelMetricsLoader:
    """Load model metadata and metrics."""

    def __init__(self, model_dir: Path = MODEL_DIR, mlflow_loader: Optional['MLflowDataLoader'] = None):
        """Initialize the model metrics loader.
        
        Args:
            model_dir: Path to model directory
            mlflow_loader: Optional MLflow data loader
        """
        self.model_dir = Path(model_dir)
        self.mlflow_loader = mlflow_loader

    def _get_latest_model_path(self) -> Optional[Path]:
        """Get path to the latest model file."""
        # Look for joblib files matching pattern
        pattern = str(self.model_dir / "isolation_forest_v*.joblib")
        files = glob.glob(pattern)
        
        if not files:
            # Fallback to old name just in case
            old_path = self.model_dir / "isolation_forest_model.pkl"
            return old_path if old_path.exists() else None
            
        # Parse versions to find latest
        # Filename format: isolation_forest_v{version}.joblib
        latest_file = None
        max_version = -1
        
        for f_path in files:
            try:
                # Extract version number using regex
                match = re.search(r'v(\d+)\.joblib$', f_path)
                if match:
                    version = int(match.group(1))
                    if version > max_version:
                        max_version = version
                        latest_file = Path(f_path)
            except Exception:
                continue
                
        # If regex matching failed for all, just take the last byte-ordered file
        if latest_file is None and files:
            files.sort()
            latest_file = Path(files[-1])
            
        return latest_file

    def get_model_info(self) -> Optional[Dict[str, any]]:
        """Get information about the current model.
        
        Returns:
            Dictionary with model metadata or None if not found
        """
        # Try MLflow first if available
        if self.mlflow_loader and self.mlflow_loader.enabled:
             mlflow_info = self.mlflow_loader.get_production_model_info()
             if mlflow_info:
                 return mlflow_info

        model_path = self._get_latest_model_path()
        
        if not model_path:
            return None
        
        try:
            # Load using joblib instead of pickle
            model_data = joblib.load(model_path)
            
            # Handle dictionary format (new) or direct model object (old)
            if isinstance(model_data, dict) and 'metadata' in model_data:
                metadata = model_data.get('metadata', {})
                
                return {
                    "model_type": metadata.get('model_type', 'IsolationForest'),
                    "n_features": metadata.get('n_features', len(FEATURE_NAMES)),
                    "contamination": metadata.get('contamination', 'unknown'),
                    "n_estimators": metadata.get('n_estimators', 'unknown'),
                    "trained_date": metadata.get('training_date', 'unknown'),
                    "model_version": metadata.get('version', 'unknown'),
                    "file_size": model_path.stat().st_size / 1024,  # KB
                }
            elif hasattr(model_data, 'get_params'):
                # It's a raw sklearn model (old format)
                params = model_data.get_params()
                return {
                    "model_type": type(model_data).__name__,
                    "n_features": getattr(model_data, 'n_features_in_', 'unknown'),
                    "contamination": params.get('contamination', 'unknown'),
                    "n_estimators": params.get('n_estimators', 'unknown'),
                    "trained_date": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(),
                    "model_version": "1.0 (legacy)",
                    "file_size": model_path.stat().st_size / 1024,  # KB
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Error loading model metadata from {model_path}: {e}")
            return None

    def is_model_loaded(self) -> bool:
        """Check if a trained model exists.
        
        Returns:
            True if model file exists
        """
        return self._get_latest_model_path() is not None


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
    def get_production_model_info(self) -> Optional[Dict[str, any]]:
        """Get information about the production model from MLflow."""
        if not self.enabled:
            return None
            
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            from .dashboard_config import REGISTERED_MODEL_NAME
            
            mlflow.set_tracking_uri(self.tracking_uri)
            client = MlflowClient()
            
            # Get latest versions
            versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
            
            if not versions:
                return None
                
            # Find Production version
            prod_version = next((v for v in versions if v.current_stage == "Production"), None)
            
            # Fallback to latest version if no Production
            if not prod_version:
                # Sort by version number desc
                versions.sort(key=lambda x: int(x.version), reverse=True)
                target_version = versions[0]
                stage = "None (Latest)"
            else:
                target_version = prod_version
                stage = "Production"
                
            run = mlflow.get_run(target_version.run_id)
            params = run.data.params
            metrics = run.data.metrics
            tags = run.data.tags
            
            # Try to get file size (approximate from artifact)
            # This is hard to get without downloading, so we'll skip or estimate
            
            return {
                "model_type": tags.get("model_type", "IsolationForest"),
                "n_features": int(params.get("n_features", len(FEATURE_NAMES))),
                "contamination": params.get("contamination", "unknown"),
                "n_estimators": params.get("n_estimators", "unknown"),
                "trained_date": datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
                "model_version": f"{target_version.version} ({stage})",
                "file_size": 0.0,  # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Error fetching model info from MLflow: {e}")
            return None
