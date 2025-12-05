"""Security configuration for the anomaly detection system."""

import os
from typing import List, Dict, Any

# ========== Input Validation Settings ==========

# Maximum sizes to prevent DoS attacks
MAX_IP_TRACKING = int(os.getenv('MAX_IP_TRACKING', 10000))
MAX_PAYLOAD_SIZE = int(os.getenv('MAX_PAYLOAD_SIZE', 1024))
MAX_PATTERN_MATCHES = int(os.getenv('MAX_PATTERN_MATCHES', 10))

# Rate limiting settings
ALERT_COOLDOWN_SECONDS = int(os.getenv('ALERT_COOLDOWN_SECONDS', 60))
MAX_ALERTS_PER_IP = int(os.getenv('MAX_ALERTS_PER_IP', 3))

# ========== File Security Settings ==========

# Secure file permissions (octal)
LOG_FILE_PERMISSIONS = 0o640  # Owner read/write, group read
MODEL_FILE_PERMISSIONS = 0o644  # Owner read/write, group/others read
DATA_FILE_PERMISSIONS = 0o640  # Owner read/write, group read

# Maximum file sizes (bytes)
MAX_LOG_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_MODEL_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_TRAINING_FILE_SIZE = 500 * 1024 * 1024  # 500 MB

# ========== Network Security Settings ==========

# IP address validation
ALLOWED_IP_RANGES: List[str] = [
    '10.0.0.0/8',
    '172.16.0.0/12', 
    '192.168.0.0/16',
    '127.0.0.0/8'
]

# Blocked IP ranges (known malicious or unwanted)
BLOCKED_IP_RANGES: List[str] = [
    '0.0.0.0/8',
    '224.0.0.0/4',  # Multicast
    '240.0.0.0/4',  # Reserved
]

# ========== Logging Security Settings ==========

# Sensitive data patterns to sanitize in logs
SENSITIVE_PATTERNS = [
    r'password[=:]\s*\S+',
    r'token[=:]\s*\S+',
    r'key[=:]\s*\S+',
    r'secret[=:]\s*\S+',
    r'auth[=:]\s*\S+',
]

# Maximum log message length
MAX_LOG_MESSAGE_LENGTH = 1000

# ========== ML Security Settings ==========

# Feature validation ranges
FEATURE_VALIDATION_RANGES: Dict[str, Dict[str, float]] = {
    'packets_per_second': {'min': 0.0, 'max': 10000.0},
    'bytes_per_second': {'min': 0.0, 'max': 1e9},
    'avg_packet_size': {'min': 0.0, 'max': 65535.0},
    'session_duration': {'min': 0.0, 'max': 86400.0},  # Max 24 hours
}

# Model validation settings
MAX_FEATURE_VALUES = 1e6  # Maximum allowed feature value
MIN_TRAINING_SAMPLES = 50  # Minimum samples for reliable training

# ========== Environment Security ==========

# Required environment variables for production
REQUIRED_ENV_VARS = [
    'ANOMALY_LOG_FILE',
    'MONITORING_INTERVAL',
]

# Dangerous environment variables to check
DANGEROUS_ENV_VARS = [
    'LD_PRELOAD',
    'LD_LIBRARY_PATH',
    'PYTHONPATH',
]

def validate_security_config() -> List[str]:
    """Validate security configuration and return any issues.
    
    Returns:
        List of security issues found
    """
    issues = []
    
    # Check file size limits
    if MAX_LOG_FILE_SIZE > 1024 * 1024 * 1024:  # 1GB
        issues.append("Log file size limit too large (>1GB)")
    
    # Check rate limiting
    if ALERT_COOLDOWN_SECONDS < 1:
        issues.append("Alert cooldown too short (<1 second)")
    
    if MAX_ALERTS_PER_IP > 100:
        issues.append("Max alerts per IP too high (>100)")
    
    # Check tracking limits
    if MAX_IP_TRACKING > 100000:
        issues.append("IP tracking limit too high (>100k)")
    
    return issues

def get_secure_temp_dir() -> str:
    """Get a secure temporary directory path.
    
    Returns:
        Path to secure temporary directory
    """
    import tempfile
    return tempfile.mkdtemp(prefix='anomaly_detector_')

def sanitize_for_logging(text: str) -> str:
    """Sanitize text for safe logging.
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized text safe for logging
    """
    import re
    
    if not isinstance(text, str):
        text = str(text)
    
    # Limit length
    if len(text) > MAX_LOG_MESSAGE_LENGTH:
        text = text[:MAX_LOG_MESSAGE_LENGTH] + "..."
    
    # Remove sensitive patterns
    for pattern in SENSITIVE_PATTERNS:
        text = re.sub(pattern, '[REDACTED]', text, flags=re.IGNORECASE)
    
    # Replace non-printable characters
    text = ''.join(c if c.isprintable() else '?' for c in text)
    
    return text

# Validate configuration on import
_security_issues = validate_security_config()
if _security_issues:
    import warnings
    for issue in _security_issues:
        warnings.warn(f"Security configuration issue: {issue}")