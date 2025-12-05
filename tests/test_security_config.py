import pytest
from src.security_config import (
    validate_security_config, 
    sanitize_for_logging,
    MAX_IP_TRACKING,
    ALERT_COOLDOWN_SECONDS,
    MAX_ALERTS_PER_IP
)

def test_security_config_validation():
    """Test that security configuration validation works."""
    issues = validate_security_config()
    # Should return empty list if no issues
    assert isinstance(issues, list)

def test_sanitize_for_logging():
    """Test that log sanitization works correctly."""
    # Test normal text
    result = sanitize_for_logging("Normal log message")
    assert result == "Normal log message"
    
    # Test sensitive data removal
    result = sanitize_for_logging("password=secret123")
    assert "secret123" not in result
    assert "[REDACTED]" in result
    
    # Test length limiting
    long_text = "A" * 2000
    result = sanitize_for_logging(long_text)
    assert len(result) <= 1003  # 1000 + "..."
    
    # Test non-string input
    result = sanitize_for_logging(12345)
    assert result == "12345"

def test_security_constants():
    """Test that security constants are reasonable."""
    assert MAX_IP_TRACKING > 0
    assert MAX_IP_TRACKING <= 100000
    
    assert ALERT_COOLDOWN_SECONDS >= 1
    assert ALERT_COOLDOWN_SECONDS <= 3600
    
    assert MAX_ALERTS_PER_IP >= 1
    assert MAX_ALERTS_PER_IP <= 100