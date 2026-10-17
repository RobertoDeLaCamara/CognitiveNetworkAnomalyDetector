"""Offline packet sources for the detector: PCAP replay and a built-in sample.

Both feed the same ``PacketProcessor`` queue that live capture uses, so the
detection pipeline and its output are identical to a live run.
"""

import os
import time
from typing import Callable, Iterable, Iterator

from scapy.all import ICMP, IP, TCP, UDP, PcapReader, Raw

from src.core.logger_setup import logger

REPLAY_SPEEDS = ("fast", "original")

# In "original" speed, never wait longer than this for a single gap between
# packets, so a capture with hours of silence does not hang the replay.
MAX_ORIGINAL_GAP = 5.0


def read_pcap(path: str) -> Iterator:
    """Stream packets from a PCAP/PCAPNG file one at a time (constant memory)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PCAP file not found: {path}")
    reader = PcapReader(path)
    try:
        yield from reader
    finally:
        reader.close()


def _sleep(seconds: float, should_stop: Callable[[], bool]) -> None:
    """Sleep in short slices so a shutdown request is honoured promptly."""
    deadline = time.monotonic() + seconds
    while not should_stop():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.1, remaining))


def replay_packets(
    packets: Iterable,
    processor,
    speed: str = "fast",
    should_stop: Callable[[], bool] = lambda: False,
    max_gap: float = MAX_ORIGINAL_GAP,
) -> int:
    """Enqueue packets into ``processor``; returns how many were enqueued.

    ``speed="fast"`` enqueues as quickly as the workers drain the queue.
    ``speed="original"`` waits the capture's inter-packet gaps (each capped at
    ``max_gap`` seconds).
    """
    if speed not in REPLAY_SPEEDS:
        raise ValueError(f"speed must be one of {REPLAY_SPEEDS}, got {speed!r}")

    previous = None
    count = 0
    for packet in packets:
        if should_stop():
            break
        if speed == "original":
            current = float(packet.time)
            if previous is not None:
                _sleep(min(max(current - previous, 0.0), max_gap), should_stop)
            previous = current
        processor.enqueue(packet, block=True)
        count += 1
    logger.info(f"Replay finished: {count} packets enqueued")
    return count


def replay_pcap(path: str, processor, speed: str = "fast",
                should_stop: Callable[[], bool] = lambda: False) -> int:
    """Replay a PCAP file through ``processor``; returns the packet count."""
    return replay_packets(read_pcap(path), processor, speed, should_stop)


def synthetic_packets() -> Iterator:
    """Deterministic sample traffic that exercises the main rules, no root needed.

    - 10.0.0.5: ordinary HTTPS traffic (no alert expected)
    - 10.0.0.66: a SQL-injection payload on port 80 and traffic on an uncommon port
    - 10.0.0.77: an ICMP flood (more than the default ICMP threshold)
    """
    clock = 1_700_000_000.0

    def stamp(packet, offset):
        packet.time = clock + offset
        return packet

    offset = 0.0
    for i in range(20):
        offset += 0.2
        yield stamp(IP(src="10.0.0.5", dst="10.0.0.9") / TCP(sport=40000 + i, dport=443)
                    / Raw(load=b"GET /index.html HTTP/1.1\r\nHost: example.org\r\n\r\n"), offset)

    # Alerts are rate-limited to 3 per IP and type, so send one uncommon-port
    # packet first and then two payloads: all three rules show up in the output.
    offset += 0.2
    yield stamp(IP(src="10.0.0.66", dst="10.0.0.9") / UDP(sport=50010, dport=31337) / Raw(load=b"ping"), offset)
    for i in range(2):
        offset += 0.2
        yield stamp(IP(src="10.0.0.66", dst="10.0.0.9") / TCP(sport=50000 + i, dport=80)
                    / Raw(load=b"GET /items?id=1' UNION SELECT password FROM users-- HTTP/1.1\r\n\r\n"), offset)

    for i in range(60):
        offset += 0.05
        yield stamp(IP(src="10.0.0.77", dst="10.0.0.9") / ICMP() / Raw(load=b"x" * 32), offset)
