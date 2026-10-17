"""Payloads larger than the scan limit are truncated, not skipped.

Regression: _validate_payload used to reject payloads over MAX_PAYLOAD_SCAN_SIZE,
so an attack at the start of a normal 1460-byte TCP segment was never scanned.
"""

import pytest

from src.config.config import MAX_PAYLOAD_SCAN_SIZE
from src.detection.payload_analyzer import detect_malicious_payload, get_payload_statistics

ATTACK = b"GET /items?id=1 UNION SELECT password FROM users"


def pad(payload: bytes, size: int, filler: bytes = b" ") -> bytes:
    return payload + filler * (size - len(payload))


def test_scan_limit_is_what_the_tests_assume():
    assert MAX_PAYLOAD_SCAN_SIZE == 1024


@pytest.mark.parametrize("size", [
    len(ATTACK),
    MAX_PAYLOAD_SCAN_SIZE - 1,
    MAX_PAYLOAD_SCAN_SIZE,
    MAX_PAYLOAD_SCAN_SIZE + 1,   # was silently skipped
    1460,                        # a full-size TCP segment
    9000,                        # jumbo frame
])
def test_attack_at_the_start_is_detected_at_any_payload_size(size):
    is_malicious, pattern = detect_malicious_payload(pad(ATTACK, size))
    assert is_malicious is True
    assert pattern == "UNION SELECT"


def test_attack_beyond_the_scan_window_is_not_scanned():
    """Documented limit: only the first MAX_PAYLOAD_SCAN_SIZE bytes are inspected."""
    payload = b"A" * MAX_PAYLOAD_SCAN_SIZE + ATTACK
    assert detect_malicious_payload(payload) == (False, None)


def test_pattern_straddling_the_window_edge_is_not_matched():
    """A pattern cut by the truncation point does not match (no partial matches)."""
    cut = len(b"UNION SELECT") // 2
    payload = b"A" * (MAX_PAYLOAD_SCAN_SIZE - cut) + b"UNION SELECT"
    assert detect_malicious_payload(payload) == (False, None)


def test_large_benign_payload_is_still_clean():
    assert detect_malicious_payload(b"Normal data " * 500) == (False, None)


def test_non_bytes_and_empty_are_still_rejected():
    assert detect_malicious_payload("UNION SELECT") == (False, None)
    assert detect_malicious_payload(None) == (False, None)
    assert detect_malicious_payload(b"") == (False, None)


def test_statistics_work_on_large_payloads():
    stats = get_payload_statistics(b"A" * 2000)
    assert stats["size"] == 2000
    assert stats["printable_ratio"] == 1.0
