import time
import statistics
import os
from collections import defaultdict, deque
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
        SCALER_MODEL_PATH
    )
    ML_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML components not available: {e}")
    ML_AVAILABLE = False
    ML_ENABLED = False

# Tracking variables
packet_count_per_ip = defaultdict(int)
packet_rate_per_ip = defaultdict(lambda: deque(maxlen=10))
start_time = time.time()

# ML components (initialized if enabled)
feature_extractor = None
ml_detector = None

if ML_ENABLED and ML_AVAILABLE:
    try:
        feature_extractor = FeatureExtractor()
        ml_detector = IsolationForestDetector()
        
        # Try to load pre-trained model
        if os.path.exists(ISOLATION_FOREST_MODEL_PATH) and os.path.exists(SCALER_MODEL_PATH):
            ml_detector.load()
            logger.info("ML detector loaded successfully")
        else:
            logger.warning(
                "No pre-trained model found. ML detection will be disabled. "
                "Run model training first."
            )
            ML_ENABLED = False
    except Exception as e:
        logger.error(f"Failed to initialize ML detector: {e}")
        ML_ENABLED = False

def calculate_packet_rate(ip_src):
    """Calculates the packet rate per IP, based on recent intervals."""
    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time == 0:
        return 0, 0

    # Update the packet rate per second
    rate = packet_count_per_ip[ip_src] / elapsed_time
    packet_rate_per_ip[ip_src].append(rate)

    # Calculate the average of the latest rates
    avg_rate = statistics.mean(packet_rate_per_ip[ip_src])
    return rate, avg_rate

def log_alert(subject, body, alert_type="RULE"):
    """Logs alerts to the rotating log file and prints them to the console.
    
    Args:
        subject: Alert subject line
        body: Alert details
        alert_type: Type of alert ("RULE" or "ML")
    """
    log_message = f"[{alert_type}] {subject} - {body}"
    logger.info(log_message)
    print(f"[ALERT] {log_message}")

def log_ml_alert(ip: str, anomaly_score: float, features: dict):
    """Logs ML-detected anomalies with feature context.
    
    Args:
        ip: Source IP address
        anomaly_score: Anomaly score from ML model (lower = more anomalous)
        features: Dictionary or array of extracted features
    """
    alert_subject = f"ML ANOMALY DETECTED: {ip}"
    alert_body = (
        f"IP {ip} flagged as anomalous by ML model. "
        f"Anomaly score: {anomaly_score:.4f} (lower = more anomalous). "
        f"Total packets: {packet_count_per_ip.get(ip, 0)}"
    )
    log_alert(alert_subject, alert_body, alert_type="ML")

def analyze_packet(packet):
    """Analyzes packets to detect advanced network anomalies (rule-based + ML)."""
    global packet_count_per_ip, feature_extractor, ml_detector

    if IP not in packet:
        return

    ip_src = packet[IP].src
    packet_count_per_ip[ip_src] += 1

    # ML-based detection (if enabled)
    if ML_ENABLED and feature_extractor is not None and ml_detector is not None:
        try:
            # Process packet for feature extraction
            feature_extractor.process_packet(packet)
            
            # Only run ML inference if we have enough packets
            if packet_count_per_ip[ip_src] >= MIN_PACKETS_FOR_ML:
                features = feature_extractor.extract_features(ip_src)
                
                if features is not None:
                    # Make prediction
                    prediction, anomaly_score = ml_detector.predict(features)
                    
                    # Alert if anomalous (prediction == -1 or score below threshold)
                    if prediction == -1 or anomaly_score < ML_ANOMALY_THRESHOLD:
                        log_ml_alert(ip_src, anomaly_score, features)
        except Exception as e:
            # Don't let ML errors break rule-based detection
            logger.error(f"ML detection error for {ip_src}: {e}")

    current_rate, avg_rate = calculate_packet_rate(ip_src)

    # Detect traffic spikes
    if avg_rate > 0 and current_rate > avg_rate * THRESHOLD_MULTIPLIER:
        alert_subject = f"ALERT: Traffic spike from {ip_src}"
        alert_body = (f"IP {ip_src} has a traffic rate of {current_rate:.2f} packets/sec, "
                      f"which is significantly higher than the average of {avg_rate:.2f} packets/sec.")
        log_alert(alert_subject, alert_body)

    # Detect ICMP traffic (e.g., ping flood attack)
    if ICMP in packet:
        if packet_count_per_ip[ip_src] > ICMP_THRESHOLD:
            alert_subject = f"ALERT: Possible ICMP attack (ping flood) from {ip_src}"
            alert_body = f"IP {ip_src} has sent more than {ICMP_THRESHOLD} ICMP packets."
            log_alert(alert_subject, alert_body)

    # Detect unusual TCP/UDP traffic on sensitive or uncommon ports
    if TCP in packet or UDP in packet:
        dport = packet[TCP].dport if TCP in packet else packet[UDP].dport
        if dport not in HIGH_TRAFFIC_PORTS:
            alert_subject = f"ALERT: Traffic on uncommon port {dport} from {ip_src}"
            alert_body = f"Traffic detected from IP {ip_src} to port {dport}, which is unusual."
            log_alert(alert_subject, alert_body)

        # Payload analysis for unusual behavior
        if Raw in packet:
            payload = packet[Raw].load
            payload_size = len(payload)

            # Detect unusually large payloads
            if payload_size > PAYLOAD_THRESHOLD:
                alert_subject = f"ALERT: Unusually large payload from {ip_src}"
                alert_body = f"A payload of {payload_size} bytes was detected from {ip_src} to port {dport}."
                log_alert(alert_subject, alert_body)

            # Detect malicious patterns in the payload
            is_malicious, pattern = detect_malicious_payload(payload)
            if is_malicious:
                alert_subject = f"ALERT: Malicious payload detected from {ip_src}"
                alert_body = f"The pattern '{pattern}' was detected in traffic from {ip_src} to port {dport}."
                log_alert(alert_subject, alert_body)
