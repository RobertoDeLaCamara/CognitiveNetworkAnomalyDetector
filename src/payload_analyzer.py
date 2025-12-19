"""Payload analysis module for detecting malicious patterns in network traffic."""

import re
import threading
import time
from typing import Tuple, Optional, List
from .config import MALICIOUS_PATTERNS, MAX_PAYLOAD_SCAN_SIZE, MAX_PATTERN_MATCHES
from .logger_setup import logger
from .utils import calculate_entropy

# Timeout for pattern matching to prevent ReDoS attacks
PATTERN_MATCH_TIMEOUT = 1  # 1 second

# Sort patterns by length (longest first) to match more specific patterns first
# This ensures "UNION SELECT" is matched before "SELECT"
_SORTED_PATTERNS = sorted(MALICIOUS_PATTERNS, key=len, reverse=True)

class PatternMatchTimeoutError(Exception):
    pass

def _sanitize_pattern_for_logging(pattern: str) -> str:
    """Sanitize pattern string for safe logging.
    
    Args:
        pattern: Pattern string to sanitize
        
    Returns:
        Sanitized pattern string safe for logging
    """
    # Limit length and remove potentially dangerous characters
    sanitized = pattern[:50]  # Limit to 50 characters
    # Replace non-printable characters with placeholder
    sanitized = ''.join(c if c.isprintable() else '?' for c in sanitized)
    return sanitized

def _validate_payload(payload: bytes) -> bool:
    """Validate payload input.
    
    Args:
        payload: Payload bytes to validate
        
    Returns:
        True if payload is valid for analysis
    """
    if not isinstance(payload, bytes):
        return False
    if len(payload) == 0:
        return False
    if len(payload) > MAX_PAYLOAD_SCAN_SIZE:
        logger.debug(f"Payload too large for scanning: {len(payload)} bytes")
        return False
    return True

def detect_malicious_payload(payload: bytes) -> Tuple[bool, Optional[str]]:
    """Analyzes the payload content to detect suspicious patterns.
    
    Uses length-sorted pattern matching to detect longer patterns before shorter ones,
    reducing false positives (e.g., "UNION SELECT" before "SELECT").
    
    Args:
        payload: Raw bytes payload to analyze (max MAX_PAYLOAD_SCAN_SIZE bytes)
    
    Returns:
        Tuple of (is_malicious: bool, pattern_found: str or None)
    """
    if not _validate_payload(payload):
        return False, None
    
    # Limit payload size for performance and security
    scan_payload = payload[:MAX_PAYLOAD_SCAN_SIZE]
    
    try:
        # Thread-safe timeout implementation
        result = [None]
        exception = [None]
        
        def pattern_match():
            try:
                # Convert to lowercase for case-insensitive matching
                payload_lower = scan_payload.lower()
                
                for pattern in _SORTED_PATTERNS:
                    try:
                        # Ensure pattern is bytes and convert to lowercase
                        if isinstance(pattern, bytes):
                            pattern_lower = pattern.lower()
                            if pattern_lower in payload_lower:
                                # Return the original pattern (not lowercase) for logging
                                safe_pattern = _sanitize_pattern_for_logging(
                                    pattern.decode('utf-8', errors='replace')
                                )
                                result[0] = (True, safe_pattern)
                                return
                                
                    except (UnicodeDecodeError, TypeError, AttributeError) as e:
                        logger.debug(f"Error processing pattern: {e}")
                        continue
                
                result[0] = (False, None)
            except Exception as e:
                exception[0] = e
        
        # Run pattern matching in thread with timeout
        thread = threading.Thread(target=pattern_match)
        thread.daemon = True
        thread.start()
        thread.join(timeout=PATTERN_MATCH_TIMEOUT)
        
        if thread.is_alive():
            logger.warning("Pattern matching timeout - potential ReDoS attack")
            return True, "TIMEOUT_ATTACK"
        
        if exception[0]:
            raise exception[0]
            
        return result[0] if result[0] else (False, None)
    except Exception as e:
        logger.error(f"Error in payload analysis: {e}")
        return False, None

def get_payload_statistics(payload: bytes) -> dict:
    """Get basic statistics about a payload.
    
    Args:
        payload: Payload bytes to analyze
        
    Returns:
        Dictionary with payload statistics
    """
    if not _validate_payload(payload):
        return {}
    
    try:
        stats = {
            'size': len(payload),
            'printable_chars': sum(1 for b in payload if 32 <= b <= 126),
            'null_bytes': payload.count(b'\x00'),
            'entropy': calculate_entropy(payload)
        }
        stats['printable_ratio'] = stats['printable_chars'] / len(payload) if len(payload) > 0 else 0
        return stats
    except Exception as e:
        logger.error(f"Error calculating payload statistics: {e}")
        return {}
