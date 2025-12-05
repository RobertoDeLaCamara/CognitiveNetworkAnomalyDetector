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
# Note: Organized from most specific to least specific
# Longer patterns are checked before shorter ones to avoid false positives
MALICIOUS_PATTERNS = [
    # SQL Injection patterns (specific multi-word patterns first)
    b"UNION SELECT", b"' OR '1'='1'", b"' OR 1=1", b"' OR 'x'='x", b"' OR 'a'='a", b"' AND 'a'='a",
    b"admin'--", b"' OR 1=1 --",
    b"xp_cmdshell", b"base64_decode(", b"<!--", b"-->",
    
    # More specific SQL keywords (less prone to false positives)
    b"DROP TABLE", b"DELETE FROM", b"INSERT INTO", b"UPDATE SET", b"ALTER TABLE",
    b"'; DROP", b"';DELETE",

    # Malicious shell command patterns (specific commands)
    b"wget http", b"curl http", b"chmod 777", b"sudo su", b"/bin/bash -c", b"/bin/sh -c",
    b"nc -l", b"nmap -sV",
    
    # PHP/Web Server injections (specific patterns)
    b"<?php", b"eval(", b"system(", b"passthru(", b"shell_exec(", b"exec(",
    b"$_GET[", b"$_POST[", b"$_REQUEST[", b"$_SERVER[",

    # XSS - Cross-Site Scripting (complete patterns)
    b"<script>", b"</script>", b"alert(", b"document.cookie", b"onerror=", b"onload=",
    b"onclick=", b"onmouseover=",

    # Data exfiltration or critical file access patterns
    b"/etc/passwd", b"/etc/shadow", b".htpasswd", b"../../etc", b"..\\..\\windows",
    b"C:\\Windows\\System32\\",
]
