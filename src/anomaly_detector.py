import time
import statistics
import os
from collections import defaultdict, deque
from pathlib import Path
from scapy.all import IP, ICMP, TCP, UDP, Raw
from .logger_setup import logger
from .payload_analyzer import detect_malicious_payload
from .config import THRESHOLD_MULTIPLIER, HIGH_TRAFFIC_PORTS, ICMP_THRESHOLD, PAYLOAD_THRESHOLD

# ML imports
try:
    from .feature_extractor import FeatureExtractor
    from .isolation_forest_detector import IsolationForestDetector
    from .ml_config import (
        ML_ENABLED,
        MIN_PACKETS_FOR_ML,
        ML_ANOMALY_THRESHOLD,
        ISOLATION_FOREST_MODEL_PATH,
        SCALER_MODEL_PATH,
        MODEL_DIR
    )
    ML_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML components not available: {e}")
    ML_AVAILABLE = False
    ML_ENABLED = False

# Build absolute paths for model files
def _get_model_path(relative_path: str) -> str:
    """Convert relative model path to absolute path."""
    if os.path.isabs(relative_path):
        return relative_path
    project_root = Path(__file__).parent.parent
    return str(project_root / relative_path)

# Convert relative paths to absolute
if ML_AVAILABLE:
    ISOLATION_FOREST_MODEL_PATH = _get_model_path(ISOLATION_FOREST_MODEL_PATH)
    SCALER_MODEL_PATH = _get_model_path(SCALER_MODEL_PATH)


class PacketAnalyzer:
    """Encapsulates anomaly detection logic for network packets."""
    
    def __init__(self, enable_ml: bool = True):
        """Initialize the packet analyzer.
        
        Args:
            enable_ml: Whether to enable ML-based detection
        """
        self.packet_count_per_ip = defaultdict(int)
        self.packet_rate_per_ip = defaultdict(lambda: deque(maxlen=10))
        self.start_time = time.time()
        self.ml_enabled = enable_ml and ML_AVAILABLE and ML_ENABLED
        
        # ML components
        self.feature_extractor = None
        self.ml_detector = None
        
        if self.ml_enabled:
            try:
                self.feature_extractor = FeatureExtractor()
                self.ml_detector = IsolationForestDetector()
                
                # Try to load pre-trained model
                if os.path.exists(ISOLATION_FOREST_MODEL_PATH) and os.path.exists(SCALER_MODEL_PATH):
                    try:
                        self.ml_detector.load(ISOLATION_FOREST_MODEL_PATH, SCALER_MODEL_PATH)
                        logger.info("ML detector loaded successfully")
                    except Exception as load_e:
                        logger.warning(f"Failed to load pre-trained model: {load_e}")
                        self.ml_enabled = False
                else:
                    logger.warning(
                        f"No pre-trained model found at {ISOLATION_FOREST_MODEL_PATH}. "
                        "ML detection will be disabled."
                    )
                    self.ml_enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize ML detector: {e}")
                self.ml_enabled = False
    
    def calculate_packet_rate(self, ip_src: str) -> tuple:
        """Calculate the packet rate per IP.
        
        Args:
            ip_src: Source IP address
        
        Returns:
            Tuple of (current_rate, average_rate)
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if elapsed_time == 0:
            return 0, 0

        # Update the packet rate per second
        rate = self.packet_count_per_ip[ip_src] / elapsed_time
        self.packet_rate_per_ip[ip_src].append(rate)

        # Calculate the average of the latest rates
        avg_rate = statistics.mean(self.packet_rate_per_ip[ip_src])
        return rate, avg_rate
    
    @staticmethod
    def log_alert(subject: str, body: str, alert_type: str = "RULE") -> None:
        """Log alerts to file and console.
        
        Args:
            subject: Alert subject line
            body: Alert details
            alert_type: Type of alert ("RULE" or "ML")
        """
        log_message = f"[{alert_type}] {subject} - {body}"
        logger.info(log_message)
        print(f"[ALERT] {log_message}")
    
    def log_ml_alert(self, ip: str, anomaly_score: float, features) -> None:
        """Log ML-detected anomalies.
        
        Args:
            ip: Source IP address
            anomaly_score: Anomaly score from ML model
            features: Feature array from extraction
        """
        alert_subject = f"ML ANOMALY DETECTED: {ip}"
        alert_body = (
            f"IP {ip} flagged as anomalous by ML model. "
            f"Anomaly score: {anomaly_score:.4f} (lower = more anomalous). "
            f"Total packets: {self.packet_count_per_ip.get(ip, 0)}"
        )
        self.log_alert(alert_subject, alert_body, alert_type="ML")
    
    def analyze_packet(self, packet) -> None:
        """Analyze a single packet for anomalies.
        
        Args:
            packet: Scapy packet object
        """
        if IP not in packet:
            return

        ip_src = packet[IP].src
        self.packet_count_per_ip[ip_src] += 1

        # ML-based detection (if enabled)
        if self.ml_enabled and self.feature_extractor is not None and self.ml_detector is not None:
            try:
                # Process packet for feature extraction
                self.feature_extractor.process_packet(packet)
                
                # Only run ML inference if we have enough packets
                if self.packet_count_per_ip[ip_src] >= MIN_PACKETS_FOR_ML:
                    features = self.feature_extractor.extract_features(ip_src)
                    
                    if features is not None:
                        # Make prediction
                        prediction, anomaly_score = self.ml_detector.predict(features)
                        
                        # Alert if anomalous (prediction == -1 or score below threshold)
                        if prediction == -1 or anomaly_score < ML_ANOMALY_THRESHOLD:
                            self.log_ml_alert(ip_src, anomaly_score, features)
            except Exception as e:
                # Don't let ML errors break rule-based detection
                logger.error(f"ML detection error for {ip_src}: {e}")

        current_rate, avg_rate = self.calculate_packet_rate(ip_src)

        # Detect traffic spikes
        if avg_rate > 0 and current_rate > avg_rate * THRESHOLD_MULTIPLIER:
            alert_subject = f"ALERT: Traffic spike from {ip_src}"
            alert_body = (f"IP {ip_src} has a traffic rate of {current_rate:.2f} packets/sec, "
                          f"which is significantly higher than the average of {avg_rate:.2f} packets/sec.")
            self.log_alert(alert_subject, alert_body)

        # Detect ICMP traffic (e.g., ping flood attack)
        if ICMP in packet:
            if self.packet_count_per_ip[ip_src] > ICMP_THRESHOLD:
                alert_subject = f"ALERT: Possible ICMP attack (ping flood) from {ip_src}"
                alert_body = f"IP {ip_src} has sent more than {ICMP_THRESHOLD} ICMP packets."
                self.log_alert(alert_subject, alert_body)

        # Detect unusual TCP/UDP traffic on sensitive or uncommon ports
        if TCP in packet or UDP in packet:
            dport = packet[TCP].dport if TCP in packet else packet[UDP].dport
            if dport not in HIGH_TRAFFIC_PORTS:
                alert_subject = f"ALERT: Traffic on uncommon port {dport} from {ip_src}"
                alert_body = f"Traffic detected from IP {ip_src} to port {dport}, which is unusual."
                self.log_alert(alert_subject, alert_body)

            # Payload analysis for unusual behavior
            if Raw in packet:
                payload = packet[Raw].load
                payload_size = len(payload)

                # Detect unusually large payloads
                if payload_size > PAYLOAD_THRESHOLD:
                    alert_subject = f"ALERT: Unusually large payload from {ip_src}"
                    alert_body = f"A payload of {payload_size} bytes was detected from {ip_src} to port {dport}."
                    self.log_alert(alert_subject, alert_body)

                # Detect malicious patterns in the payload
                is_malicious, pattern = detect_malicious_payload(payload)
                if is_malicious:
                    alert_subject = f"ALERT: Malicious payload detected from {ip_src}"
                    alert_body = f"The pattern '{pattern}' was detected in traffic from {ip_src} to port {dport}."
                    self.log_alert(alert_subject, alert_body)


# Backward compatibility: module-level instance and function
packet_analyzer = PacketAnalyzer()
packet_count_per_ip = packet_analyzer.packet_count_per_ip
analyze_packet = packet_analyzer.analyze_packet
