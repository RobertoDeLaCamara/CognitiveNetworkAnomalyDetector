# Configuration for the anomaly detection system.

# Logging settings
LOG_FILE = 'anomaly_detection.log'
LOG_MAX_SIZE = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 3

# Dynamic threshold settings
THRESHOLD_MULTIPLIER = 2  # Multiplier of the average to define a traffic anomaly
MONITORING_INTERVAL = 60  # Seconds
HIGH_TRAFFIC_PORTS = [22, 53, 80, 443]  # Common ports (SSH, DNS, HTTP, HTTPS)
ICMP_THRESHOLD = 50  # ICMP packet threshold to identify possible ping flood attacks
PAYLOAD_THRESHOLD = 100  # Payload size to detect unusual traffic

# Extended list of malicious patterns
MALICIOUS_PATTERNS = [
    # SQL Injection patterns
    b"SELECT", b"UNION", b"INSERT", b"DELETE", b"UPDATE", b"' OR '1'='1'", b"DROP", b"ALTER", b"CREATE", b"TRUNCATE", b"exec", b"xp_cmdshell",
    b"UNION SELECT", b"--", b"' OR 1=1",  # Common in SQLi

    # Malicious shell command patterns
    b"/bin/bash", b"/bin/sh", b"wget", b"curl", b"chmod", b"&&", b"|", b"sudo", b"scp", b"ftp", b"nc", b"nmap",

    # PHP/Web Server injections
    b"<?php", b"eval(", b"system(", b"passthru(", b"shell_exec(", b"exec(", b"base64_decode(", b"$_GET", b"$_POST",

    # XSS - Cross-Site Scripting
    b"<script>", b"alert(", b"document.cookie", b"onerror=", b"onload=",

    # Data exfiltration or critical file access patterns
    b"/etc/passwd", b"/etc/shadow", b"C:\\Windows\\System32\\", b".htpasswd", b"../../",

    # Common web application attack patterns
    b"admin'--", b"' OR 1=1 --", b"' OR 'x'='x", b"' OR 'a'='a", b"'='", b"' AND 'a'='a"
]

# ========== ML Configuration ==========

# Enable/disable ML-based detection (requires trained model)
ML_ENABLED = True

# Path to trained Isolation Forest model (relative to project root)
ML_MODEL_PATH = 'models/isolation_forest_v1.joblib'
ML_SCALER_PATH = 'models/scaler_v1.joblib'

# Minimum packets before running ML inference
MIN_PACKETS_FOR_ML = 10

# Anomaly score threshold for ML alerts
# Isolation Forest scores are typically in range [-0.5, 0.5]
# Lower scores indicate anomalies
ML_ANOMALY_THRESHOLD = -0.1
