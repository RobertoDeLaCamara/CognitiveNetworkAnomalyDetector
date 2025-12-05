"""Configuration for the anomaly detection system."""

import os
from pathlib import Path

# Logging settings
LOG_FILE = os.getenv('ANOMALY_LOG_FILE', 'anomaly_detection.log')
LOG_MAX_SIZE = int(os.getenv('LOG_MAX_SIZE', 5 * 1024 * 1024))  # 5 MB
LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', 3))

# Dynamic threshold settings
THRESHOLD_MULTIPLIER = float(os.getenv('THRESHOLD_MULTIPLIER', 2.0))  # Multiplier for traffic anomaly
MONITORING_INTERVAL = int(os.getenv('MONITORING_INTERVAL', 60))  # Seconds
HIGH_TRAFFIC_PORTS = [22, 53, 80, 443, 993, 995, 587, 465]  # Common ports
ICMP_THRESHOLD = int(os.getenv('ICMP_THRESHOLD', 50))  # ICMP packet threshold
PAYLOAD_THRESHOLD = int(os.getenv('PAYLOAD_THRESHOLD', 100))  # Payload size threshold

# Security settings
MAX_PAYLOAD_SCAN_SIZE = 1024  # Maximum payload size to scan for patterns
MAX_PATTERN_MATCHES = 10  # Maximum pattern matches to log per packet

# Extended list of malicious patterns
# Note: Organized from most specific to least specific to minimize false positives
# Patterns are case-sensitive and should be used with proper context validation
MALICIOUS_PATTERNS = [
    # SQL Injection patterns (high confidence)
    b"UNION SELECT", b"' OR '1'='1'", b"' OR 1=1--", b"' OR 'x'='x'", 
    b"admin'--", b"' UNION SELECT", b"'; DROP TABLE", b"'; DELETE FROM",
    b"xp_cmdshell", b"sp_executesql",
    
    # Command injection (high confidence)
    b"; wget http", b"; curl http", b"&& wget", b"| wget",
    b"; chmod 777", b"&& chmod", b"/bin/bash -c", b"/bin/sh -c",
    b"nc -l -p", b"ncat -l",
    
    # Web application attacks (high confidence)
    b"<?php system(", b"<?php exec(", b"<?php shell_exec(",
    b"eval(base64_decode(", b"system($_GET[", b"exec($_POST[",
    
    # XSS patterns (high confidence)
    b"<script>alert(", b"javascript:alert(", b"<img src=x onerror=",
    b"<svg onload=", b"<iframe src=",
    
    # Directory traversal (high confidence)
    b"../../../etc/passwd", b"..\\..\\..\\windows",
    b"../../../../etc/shadow", b"..\\..\\..\\boot.ini",
    
    # File inclusion attacks
    b"php://filter", b"php://input", b"data://text/plain",
    b"file:///etc/passwd", b"file:///c:/windows",
]

def validate_config():
    """Validate configuration values."""
    if THRESHOLD_MULTIPLIER <= 1.0:
        raise ValueError("THRESHOLD_MULTIPLIER must be greater than 1.0")
    if MONITORING_INTERVAL <= 0:
        raise ValueError("MONITORING_INTERVAL must be positive")
    if ICMP_THRESHOLD <= 0:
        raise ValueError("ICMP_THRESHOLD must be positive")
    if PAYLOAD_THRESHOLD <= 0:
        raise ValueError("PAYLOAD_THRESHOLD must be positive")

# Validate configuration on import
validate_config()
