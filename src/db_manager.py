"""Database manager for storing anomaly detection results."""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

from .config import LOG_FILE

logger = logging.getLogger(__name__)

class DBManager:
    """Manages SQLite database operations for anomaly storage."""
    
    def __init__(self, db_path: str = "anomalies.db"):
        """Initialize database manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        # Ensure path is absolute if not already
        if not Path(db_path).is_absolute():
            self.db_path = str(Path(__file__).parent.parent / db_path)
        else:
            self.db_path = db_path
            
        self._init_db()
    
    def _get_connection(self):
        """Create a database connection."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def _init_db(self):
        """Initialize database schema."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ip_address TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            description TEXT,
            anomaly_score REAL,
            raw_data TEXT,
            is_reviewed BOOLEAN DEFAULT 0
        );
        """
        try:
            with self._get_connection() as conn:
                conn.execute(create_table_sql)
                # Create index for faster querying by timestamp and IP
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON anomalies(timestamp);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ip ON anomalies(ip_address);")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {e}")

    def add_anomaly(self, 
                   ip_address: str, 
                   alert_type: str, 
                   description: str, 
                   anomaly_score: float = 0.0,
                   raw_data: Optional[Dict] = None):
        """Add a new anomaly record.
        
        Args:
            ip_address: Source IP address
            alert_type: Type of alert (ML, RULE, etc.)
            description: Human readable description
            anomaly_score: Score (for ML alerts)
            raw_data: Dictionary of additional data (features, etc.)
        """
        sql = """
        INSERT INTO anomalies (timestamp, ip_address, alert_type, description, anomaly_score, raw_data)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            with self._get_connection() as conn:
                conn.execute(sql, (
                    current_time,
                    ip_address,
                    alert_type,
                    description,
                    anomaly_score,
                    json.dumps(raw_data) if raw_data else None
                ))
        except sqlite3.Error as e:
            logger.error(f"Failed to save anomaly: {e}")

    def get_anomalies(self, 
                     start_date: Optional[datetime] = None, 
                     end_date: Optional[datetime] = None,
                     limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve anomalies with fitering.
        
        Args:
            start_date: Filter start datetime
            end_date: Filter end datetime
            limit: Max records to return
            
        Returns:
            List of anomaly dictionaries
        """
        query = "SELECT * FROM anomalies WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.strftime("%Y-%m-%d %H:%M:%S"))
            
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.strftime("%Y-%m-%d %H:%M:%S"))
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch anomalies: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics directly from DB."""
        stats = {}
        try:
            with self._get_connection() as conn:
                # Total count
                stats['total_anomalies'] = conn.execute(
                    "SELECT COUNT(*) FROM anomalies"
                ).fetchone()[0]
                
                # ML vs Rule
                stats['ml_anomalies'] = conn.execute(
                    "SELECT COUNT(*) FROM anomalies WHERE alert_type = 'ML'"
                ).fetchone()[0]
                
                stats['rule_anomalies'] = conn.execute(
                    "SELECT COUNT(*) FROM anomalies WHERE alert_type != 'ML'"
                ).fetchone()[0]
                
                # Unique IPs
                stats['unique_ips'] = conn.execute(
                    "SELECT COUNT(DISTINCT ip_address) FROM anomalies"
                ).fetchone()[0]
                
                return stats
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch stats: {e}")
            return {
                'total_anomalies': 0, 
                'ml_anomalies': 0, 
                'rule_anomalies': 0, 
                'unique_ips': 0
            }
