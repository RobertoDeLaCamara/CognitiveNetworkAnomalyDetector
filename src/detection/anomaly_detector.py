"""Main anomaly detection module combining rule-based and ML-based detection."""

import time
import statistics
import os
import ipaddress
from collections import defaultdict, deque
from pathlib import Path
from typing import Tuple
from scapy.all import IP, ICMP, TCP, UDP, Raw
from src.core.logger_setup import logger
from src.detection.payload_analyzer import detect_malicious_payload
from src.config.config import THRESHOLD_MULTIPLIER, HIGH_TRAFFIC_PORTS, ICMP_THRESHOLD, PAYLOAD_THRESHOLD, TRUSTED_SUBNETS
from src.core.resource_monitor import resource_monitor
from src.core.db_manager import DBManager

# ML imports
try:
    from src.ml.feature_extractor import FeatureExtractor
    from src.ml.isolation_forest_detector import IsolationForestDetector
    from src.config.ml_config import (
        ML_ENABLED,
        MIN_PACKETS_FOR_ML,
        ISOLATION_FOREST_MODEL_PATH,
        SCALER_MODEL_PATH
    )
    from src.detection.ensemble_scorer import EnsembleScorer, EngineResult
    ML_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML components not available: {e}")
    ML_AVAILABLE = False
    ML_ENABLED = False

# LSTM imports (optional — model may not be trained yet)
try:
    from src.ml.lstm_autoencoder_detector import LSTMAutoencoderDetector, LSTM_MODEL_PATH
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

# Build absolute paths for model files


def _get_model_path(relative_path: str) -> str:
    """Convert relative model path to absolute path with security validation."""
    # Validate path to prevent traversal attacks
    if '..' in relative_path or relative_path.startswith('/'):
        raise ValueError(f"Invalid model path: {relative_path}")

    if os.path.isabs(relative_path):
        # Only allow paths within project directory
        project_root = Path(__file__).parent.parent
        abs_path = Path(relative_path)
        try:
            abs_path.relative_to(project_root)
            return relative_path
        except ValueError:
            raise ValueError(f"Path outside project directory: {relative_path}")

    project_root = Path(__file__).parent.parent
    return str(project_root / relative_path)


# Convert relative paths to absolute
if ML_AVAILABLE:
    ISOLATION_FOREST_MODEL_PATH = _get_model_path(ISOLATION_FOREST_MODEL_PATH)
    SCALER_MODEL_PATH = _get_model_path(SCALER_MODEL_PATH)


def _validate_ip_address(ip_str: str) -> bool:
    """Validate IP address format and check if it's not a reserved address.

    Args:
        ip_str: IP address string to validate

    Returns:
        True if IP is valid and not reserved
    """
    try:
        ip = ipaddress.ip_address(ip_str)
        # Skip reserved/private addresses that might be noise
        if ip.is_loopback or ip.is_link_local:
            return False

        # Check against trusted subnets
        for subnet in TRUSTED_SUBNETS:
            try:
                if ip in ipaddress.ip_network(subnet):
                    return False  # Ignore trusted subnet
            except ValueError:
                pass

        return True
    except (ipaddress.AddressValueError, ValueError):
        return False


def _sanitize_ip_for_logging(ip_str: str) -> str:
    """Sanitize IP address for safe logging.

    Args:
        ip_str: IP address string

    Returns:
        Sanitized IP address string
    """
    try:
        # Validate and return the IP if valid
        ipaddress.ip_address(ip_str)
        return ip_str
    except (ipaddress.AddressValueError, ValueError):
        return "<invalid_ip>"


class PacketAnalyzer:
    """Encapsulates anomaly detection logic for network packets."""

    def __init__(self, enable_ml: bool = True, max_ips: int = 1000):
        """Initialize the packet analyzer.

        Args:
            enable_ml: Whether to enable ML-based detection
            max_ips: Maximum number of IPs to track (prevents memory exhaustion)
        """
        # Enforce stricter limits to prevent DoS
        if max_ips > 5000:
            max_ips = 5000
            logger.warning("IP limit capped at 5000 for security")

        self.db_manager = DBManager()
        self.packet_count_per_ip = defaultdict(int)
        self.packet_rate_per_ip = defaultdict(lambda: deque(maxlen=10))
        self.start_time = time.time()
        self.max_ips = max_ips
        self.ml_enabled = enable_ml and ML_AVAILABLE and ML_ENABLED
        self.packet_count = 0
        self.max_packets = 100000  # Prevent memory exhaustion

        # Rate limiting for alerts
        self.alert_timestamps = defaultdict(lambda: deque(maxlen=5))
        self.alert_cooldown = 60  # seconds

        # Hot-reloading state
        self.last_model_check = time.time()
        self.model_check_interval = 60  # Check every 60 seconds
        self.lstm_model_mtime = 0.0

        # ML components
        self.feature_extractor = None
        self.ml_detector = None
        self.lstm_detector = None
        self.ensemble_scorer = None

        if self.ml_enabled:
            try:
                self.feature_extractor = FeatureExtractor()
                self.ml_detector = IsolationForestDetector()

                # Try to load pre-trained model
                model_loaded = False

                # Option 1: Load from MLflow (if enabled)
                if ML_ENABLED and os.getenv('MLFLOW_ENABLE_REMOTE_LOADING', 'false').lower() == 'true':
                    try:
                        logger.info("Attempting to load model from MLflow...")
                        stage = os.getenv('MLFLOW_MODEL_STAGE', 'Production')
                        version = os.getenv('MLFLOW_MODEL_VERSION')

                        if version:
                            self.ml_detector.load_from_mlflow(version=int(version))
                        else:
                            self.ml_detector.load_from_mlflow(stage=stage)

                        model_loaded = True
                        logger.info("ML detector loaded successfully from MLflow")
                    except Exception as mlflow_e:
                        logger.error(f"Failed to load model from MLflow: {mlflow_e}")
                        logger.info("Falling back to local model file...")

                # Option 2: Load from local file (default or fallback)
                if not model_loaded:
                    if os.path.exists(ISOLATION_FOREST_MODEL_PATH) and os.path.exists(SCALER_MODEL_PATH):
                        try:
                            self.ml_detector.load(ISOLATION_FOREST_MODEL_PATH, SCALER_MODEL_PATH)
                            logger.info("ML detector loaded successfully from local file")
                        except Exception as load_e:
                            logger.warning(f"Failed to load pre-trained model: {load_e}")
                            self.ml_enabled = False
                    else:
                        logger.warning(
                            f"No pre-trained model found at {ISOLATION_FOREST_MODEL_PATH}. "
                            "ML detection will be disabled."
                        )
                        self.ml_enabled = False
                # Initialize LSTM detector (optional)
                if LSTM_AVAILABLE:
                    try:
                        self.lstm_detector = LSTMAutoencoderDetector()
                        if os.path.exists(LSTM_MODEL_PATH):
                            loaded = self.lstm_detector.load()
                            if not loaded:
                                logger.warning("LSTM model failed to load, disabling LSTM engine")
                                self.lstm_detector = None
                        else:
                            logger.info("No LSTM model found, LSTM engine disabled")
                            self.lstm_detector = None
                    except Exception as lstm_e:
                        logger.warning(f"Failed to initialize LSTM detector: {lstm_e}")
                        self.lstm_detector = None

                # Initialize ensemble scorer
                self.ensemble_scorer = EnsembleScorer()

            except Exception as e:
                logger.error(f"Failed to initialize ML detector: {e}")
                self.ml_enabled = False

    def _check_model_updates(self) -> None:
        """Check if models have changed on disk and reload them."""
        if not LSTM_AVAILABLE or not self.lstm_detector:
            return

        current_time = time.time()
        if (current_time - self.last_model_check) < self.model_check_interval:
            return

        self.last_model_check = current_time

        try:
            # Check LSTM model
            if os.path.exists(LSTM_MODEL_PATH):
                mtime = os.path.getmtime(LSTM_MODEL_PATH)
                # First run, just initialize state
                if self.lstm_model_mtime == 0.0:
                    self.lstm_model_mtime = mtime
                    return

                if mtime > self.lstm_model_mtime:
                    logger.info("LSTM model file changed. Reloading...")
                    if self.lstm_detector.load():
                        logger.info("LSTM model reloaded successfully.")
                    else:
                        logger.error("Failed to reload LSTM model.")
                    self.lstm_model_mtime = mtime
        except Exception as e:
            logger.error(f"Error checking for model updates: {e}")

    def _should_alert(self, ip_src: str, alert_type: str) -> bool:
        """Check if we should send an alert based on rate limiting.

        Args:
            ip_src: Source IP address
            alert_type: Type of alert

        Returns:
            True if alert should be sent
        """
        current_time = time.time()
        alert_key = f"{ip_src}:{alert_type}"

        # Check recent alerts for this IP and type
        recent_alerts = self.alert_timestamps[alert_key]

        # Remove old alerts outside cooldown period
        while recent_alerts and current_time - recent_alerts[0] > self.alert_cooldown:
            recent_alerts.popleft()

        # Allow alert if not too many recent alerts
        if len(recent_alerts) < 3:  # Max 3 alerts per cooldown period
            recent_alerts.append(current_time)
            return True

        return False

    def _cleanup_old_data(self) -> None:
        """Clean up old data to prevent memory exhaustion."""
        if len(self.packet_count_per_ip) > self.max_ips:
            # Remove oldest IPs (simple cleanup strategy)
            sorted_ips = sorted(
                self.packet_count_per_ip.items(),
                key=lambda x: x[1]
            )
            # Remove bottom 10% of IPs by packet count
            to_remove = len(sorted_ips) // 10
            for ip, _ in sorted_ips[:to_remove]:
                del self.packet_count_per_ip[ip]
                if ip in self.packet_rate_per_ip:
                    del self.packet_rate_per_ip[ip]

    def calculate_packet_rate(self, ip_src: str) -> Tuple[float, float]:
        """Calculate the packet rate per IP with input validation.

        Args:
            ip_src: Source IP address

        Returns:
            Tuple of (current_rate, average_rate)
        """
        if not _validate_ip_address(ip_src):
            return 0.0, 0.0

        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if elapsed_time <= 0:
            return 0.0, 0.0

        # Update the packet rate per second
        rate = self.packet_count_per_ip[ip_src] / elapsed_time
        self.packet_rate_per_ip[ip_src].append(rate)

        # Calculate the average of the latest rates
        rates = list(self.packet_rate_per_ip[ip_src])
        avg_rate = statistics.mean(rates) if rates else 0.0
        return rate, avg_rate

    def log_alert(self, subject: str, body: str, alert_type: str = "RULE",
                  ip_src: str = None, anomaly_score: float = 0.0,
                  raw_data: dict = None) -> None:
        """Log alerts to file and console with rate limiting.

        Args:
            subject: Alert subject line
            body: Alert details
            alert_type: Type of alert ("RULE", "ML", or "ML_ENSEMBLE")
            ip_src: Source IP address (for rate limiting)
            anomaly_score: Score from the detection engine
            raw_data: Optional raw data dict to persist
        """
        # Rate limit alerts if IP is provided
        if ip_src and not self._should_alert(ip_src, alert_type):
            return

        # Sanitize inputs for logging
        safe_subject = subject[:200] if subject else "Unknown Alert"
        safe_body = body[:500] if body else "No details"
        valid_types = {"RULE", "ML", "ML_ENSEMBLE", "ICMP_FLOOD"}
        safe_type = alert_type if alert_type in valid_types else "UNKNOWN"

        log_message = f"[{safe_type}] {safe_subject} - {safe_body}"
        logger.info(log_message)
        print(f"[ALERT] {log_message}")

        # Persist to database
        try:
            kwargs = dict(
                ip_address=ip_src if ip_src else "unknown",
                alert_type=safe_type,
                description=f"{safe_subject} - {safe_body}",
                anomaly_score=anomaly_score,
            )
            if raw_data is not None:
                kwargs["raw_data"] = raw_data
            self.db_manager.add_anomaly(**kwargs)
        except Exception as e:
            logger.error(f"Failed to persist alert to DB: {e}")

    def log_ml_alert(self, ip: str, anomaly_score: float, features) -> None:
        """Log ML-detected anomalies with input validation.

        Args:
            ip: Source IP address
            anomaly_score: Anomaly score from ML model
            features: Feature array from extraction
        """
        safe_ip = _sanitize_ip_for_logging(ip)

        # Validate anomaly score
        if not isinstance(anomaly_score, (int, float)) or abs(anomaly_score) > 10:
            logger.warning(f"Invalid anomaly score for {safe_ip}: {anomaly_score}")
            return

        alert_subject = f"ML ANOMALY DETECTED: {safe_ip}"
        alert_body = (
            f"IP {safe_ip} flagged as anomalous by ML model. "
            f"Anomaly score: {anomaly_score:.4f} (lower = more anomalous). "
            f"Total packets: {self.packet_count_per_ip.get(ip, 0)}"
        )
        self.log_alert(alert_subject, alert_body, alert_type="ML", ip_src=ip)

        # Update DB with specific ML score
        # Note: redundant insert here but handled for simplicity.
        # Ideally, refactor log_alert to accept optional score.
        # For now, we will just let log_alert save it with score 0.0 and update it here?
        # Better: Pass score through log_alert if we refactored it.
        # Minimal change approach: Overwrite the last insert? No, easier to just accept the 0.0 for now
        # or separate the DB call.

        # Actually, let's just do a specific DB insert for ML here to get the score right
        try:
            self.db_manager.add_anomaly(
                ip_address=safe_ip,
                alert_type="ML",
                description=alert_body,
                anomaly_score=anomaly_score,
                raw_data={"features": features.tolist() if hasattr(features, 'tolist') else features}
            )
        except Exception as e:
            logger.error(f"Failed to persist ML alert to DB: {e}")

    def analyze_packet(self, packet) -> None:
        """Analyze a single packet for anomalies with comprehensive validation.

        Args:
            packet: Scapy packet object
        """
        try:
            if not packet or IP not in packet:
                return

            # Global packet limit to prevent DoS
            self.packet_count += 1
            if self.packet_count > self.max_packets:
                logger.warning("Packet limit reached, dropping packets")
                return

            # Check resource usage
            if resource_monitor.should_throttle():
                logger.warning("Resource usage high, throttling packet processing")
                return

            # Check for model updates periodically
            self._check_model_updates()

            ip_src = packet[IP].src

            # Validate IP address
            if not _validate_ip_address(ip_src):
                return

            # Prevent memory exhaustion
            if len(self.packet_count_per_ip) > self.max_ips:
                self._cleanup_old_data()

            # Increment packet count first
            self.packet_count_per_ip[ip_src] += 1

            # Per-IP packet limit check
            if self.packet_count_per_ip[ip_src] > 10000:
                logger.warning(f"IP {ip_src} exceeded packet limit")
                return

            # ML-based detection (if enabled)
            rule_fired = False  # Track whether any rule fires for this packet
            if self.ml_enabled and self.feature_extractor is not None and self.ml_detector is not None:
                try:
                    # Process packet for feature extraction
                    self.feature_extractor.process_packet(packet)

                    # Only run ML inference if we have enough packets
                    current_packet_count = self.packet_count_per_ip[ip_src]

                    if current_packet_count >= MIN_PACKETS_FOR_ML:
                        features = self.feature_extractor.extract_features(ip_src)

                        if features is not None and len(features) > 0:
                            engine_results = []

                            # Engine 1: Isolation Forest
                            if_prediction, if_score = self.ml_detector.predict(features)
                            if_normalized = EnsembleScorer.normalize_isolation_forest(if_score)
                            engine_results.append(EngineResult(
                                name="isolation_forest",
                                raw_score=float(if_score),
                                normalized_score=if_normalized,
                                prediction=int(if_prediction),
                            ))

                            # Engine 2: LSTM Autoencoder (if available)
                            if self.lstm_detector is not None:
                                try:
                                    lstm_pred, lstm_error = self.lstm_detector.predict_single(features)
                                    lstm_normalized = EnsembleScorer.normalize_lstm(
                                        lstm_error, self.lstm_detector.threshold
                                    )
                                    engine_results.append(EngineResult(
                                        name="lstm",
                                        raw_score=float(lstm_error),
                                        normalized_score=lstm_normalized,
                                        prediction=int(lstm_pred),
                                    ))
                                except Exception as lstm_e:
                                    logger.debug(f"LSTM prediction skipped: {lstm_e}")

                            # Engine 3: Rules (set after rule evaluation below)
                            # Will be combined after rules run — see below

                            # Store for post-rule ensemble combination
                            self._pending_ensemble = {
                                "ip_src": ip_src,
                                "features": features,
                                "engine_results": engine_results,
                                "if_score": if_score,
                                "if_prediction": if_prediction,
                            }
                except Exception as e:
                    # Don't let ML errors break rule-based detection
                    logger.error(f"ML detection error for {_sanitize_ip_for_logging(ip_src)}: {e}")

            current_rate, avg_rate = self.calculate_packet_rate(ip_src)
            safe_ip = _sanitize_ip_for_logging(ip_src)

            # Detect traffic spikes
            if avg_rate > 0 and current_rate > avg_rate * THRESHOLD_MULTIPLIER:
                rule_fired = True
                alert_subject = f"ALERT: Traffic spike from {safe_ip}"
                alert_body = (f"IP {safe_ip} has a traffic rate of {current_rate:.2f} packets/sec, "
                              f"which is significantly higher than the average of {avg_rate:.2f} packets/sec.")
                self.log_alert(alert_subject, alert_body, ip_src=ip_src)

            # Detect ICMP traffic (e.g., ping flood attack)
            if ICMP in packet:
                if self.packet_count_per_ip[ip_src] > ICMP_THRESHOLD:
                    rule_fired = True
                    alert_subject = f"ALERT: Possible ICMP attack (ping flood) from {safe_ip}"
                    alert_body = f"IP {safe_ip} has sent more than {ICMP_THRESHOLD} ICMP packets."
                    self.log_alert(alert_subject, alert_body, alert_type="ICMP_FLOOD", ip_src=ip_src)

            # Detect unusual TCP/UDP traffic on sensitive or uncommon ports
            if TCP in packet or UDP in packet:
                try:
                    dport = packet[TCP].dport if TCP in packet else packet[UDP].dport

                    # Validate port number
                    if not isinstance(dport, int) or not (0 <= dport <= 65535):
                        logger.warning(f"Invalid port number from {safe_ip}: {dport}")
                        return

                    if dport not in HIGH_TRAFFIC_PORTS:
                        rule_fired = True
                        alert_subject = f"ALERT: Traffic on uncommon port {dport} from {safe_ip}"
                        alert_body = f"Traffic detected from IP {safe_ip} to port {dport}, which is unusual."
                        self.log_alert(alert_subject, alert_body, ip_src=ip_src)

                    # Payload analysis for unusual behavior
                    if Raw in packet:
                        payload = packet[Raw].load

                        if not isinstance(payload, bytes):
                            logger.warning(f"Invalid payload type from {safe_ip}")
                            return

                        payload_size = len(payload)

                        # Detect unusually large payloads
                        if payload_size > PAYLOAD_THRESHOLD:
                            rule_fired = True
                            alert_subject = f"ALERT: Unusually large payload from {safe_ip}"
                            alert_body = (f"A payload of {payload_size} bytes was detected"
                                          f" from {safe_ip} to port {dport}.")
                            self.log_alert(alert_subject, alert_body, ip_src=ip_src)

                        # Detect malicious patterns in the payload
                        try:
                            is_malicious, pattern = detect_malicious_payload(payload)
                            if is_malicious and pattern:
                                rule_fired = True
                                alert_subject = f"ALERT: Malicious payload detected from {safe_ip}"
                                alert_body = (
                                    f"The pattern '{pattern}' was detected in traffic"
                                    f" from {safe_ip} to port {dport}."
                                )
                                self.log_alert(alert_subject, alert_body, ip_src=ip_src)
                        except Exception as payload_e:
                            logger.error(f"Error analyzing payload from {safe_ip}: {payload_e}")

                except (AttributeError, KeyError) as e:
                    logger.error(f"Error processing packet from {safe_ip}: {e}")

            # Finalize ensemble scoring (after rules have been evaluated)
            pending = getattr(self, '_pending_ensemble', None)
            if pending and self.ensemble_scorer is not None:
                try:
                    engine_results = pending["engine_results"]
                    # Add rules engine result
                    engine_results.append(EngineResult(
                        name="rules",
                        raw_score=1.0 if rule_fired else 0.0,
                        normalized_score=EnsembleScorer.normalize_rules(rule_fired),
                        prediction=-1 if rule_fired else 1,
                    ))

                    ensemble_result = self.ensemble_scorer.combine(engine_results)
                    if ensemble_result.is_anomaly:
                        alert_subject = f"ML ENSEMBLE ANOMALY: {safe_ip}"
                        alert_body = (
                            f"Ensemble confidence: {
                                ensemble_result.confidence_score:.3f}. " f"Engines: {
                                ', '.join(
                                    f'{n}={
                                        r.normalized_score:.3f}' for n,
                                    r in ensemble_result.engines.items())}")
                        self.log_alert(
                            alert_subject, alert_body,
                            alert_type="ML_ENSEMBLE",
                            ip_src=ip_src,
                            anomaly_score=ensemble_result.confidence_score,
                            raw_data=ensemble_result.engine_summary,
                        )
                except Exception as ens_e:
                    logger.error(f"Ensemble scoring error: {ens_e}")
                finally:
                    self._pending_ensemble = None

        except Exception as e:
            logger.error(f"Error in packet analysis: {e}", exc_info=True)


# Backward compatibility: module-level instance and function
packet_analyzer = PacketAnalyzer()
packet_count_per_ip = packet_analyzer.packet_count_per_ip
analyze_packet = packet_analyzer.analyze_packet
