import re
from typing import Tuple, Optional
from .config import MALICIOUS_PATTERNS

# Sort patterns by length (longest first) to match more specific patterns first
# This ensures "UNION SELECT" is matched before "SELECT"
_SORTED_PATTERNS = sorted(MALICIOUS_PATTERNS, key=len, reverse=True)

def detect_malicious_payload(payload: bytes) -> Tuple[bool, Optional[str]]:
    """Analyzes the payload content to detect suspicious patterns.
    
    Uses length-sorted pattern matching to detect longer patterns before shorter ones,
    reducing false positives (e.g., "UNION SELECT" before "SELECT").
    
    Args:
        payload: Raw bytes payload to analyze
    
    Returns:
        Tuple of (is_malicious: bool, pattern_found: str or None)
    """
    if not payload:
        return False, None
    
    for pattern in _SORTED_PATTERNS:
        try:
            # Search for bytes pattern directly in payload
            if isinstance(pattern, bytes):
                if pattern in payload:
                    return True, pattern.decode('utf-8', errors='ignore')
        except (UnicodeDecodeError, TypeError):
            # Skip patterns that can't be decoded
            continue
    
    return False, None
