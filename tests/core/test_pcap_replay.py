"""Tests for PCAP replay, the built-in sample traffic and the offline CLI modes."""

import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from scapy.all import ICMP, IP, TCP, Raw, wrpcap

from src.core import pcap_replay
from src.core.packet_queue import PacketProcessor
from src.core.pcap_replay import read_pcap, replay_packets, replay_pcap, synthetic_packets

REPO_ROOT = Path(__file__).resolve().parents[2]


class RecordingProcessor:
    """Stand-in for PacketProcessor that just records what it is given."""

    def __init__(self):
        self.packets = []

    def enqueue(self, packet, block=False):
        self.packets.append(packet)


def make_packets(times):
    packets = []
    for i, t in enumerate(times):
        pkt = IP(src="10.1.1.1", dst="10.1.1.2") / TCP(sport=1000 + i, dport=80) / Raw(load=b"x")
        pkt.time = t
        packets.append(pkt)
    return packets


class TestSyntheticSample:
    def test_is_deterministic_and_covers_three_sources(self):
        first = [(p[IP].src, float(p.time)) for p in synthetic_packets()]
        second = [(p[IP].src, float(p.time)) for p in synthetic_packets()]
        assert first == second
        assert {src for src, _ in first} == {"10.0.0.5", "10.0.0.66", "10.0.0.77"}

    def test_timestamps_never_go_backwards(self):
        times = [float(p.time) for p in synthetic_packets()]
        assert times == sorted(times)

    def test_icmp_flood_exceeds_default_threshold(self):
        icmp = [p for p in synthetic_packets() if ICMP in p]
        assert len(icmp) > 50

    def test_contains_a_sql_injection_payload(self):
        payloads = [bytes(p[Raw].load) for p in synthetic_packets() if Raw in p]
        assert any(b"UNION SELECT" in load for load in payloads)


class TestReadPcap:
    def test_round_trip_through_a_file(self, tmp_path):
        path = tmp_path / "sample.pcap"
        wrpcap(str(path), list(synthetic_packets()))
        packets = list(read_pcap(str(path)))
        assert len(packets) == len(list(synthetic_packets()))
        assert packets[0][IP].src == "10.0.0.5"

    def test_is_lazy(self, tmp_path):
        """Packets are streamed, not loaded all at once."""
        path = tmp_path / "sample.pcap"
        wrpcap(str(path), make_packets([1.0, 2.0, 3.0]))
        stream = read_pcap(str(path))
        assert next(stream)[IP].src == "10.1.1.1"
        stream.close()

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(read_pcap(str(tmp_path / "nope.pcap")))


class TestReplayPackets:
    def test_fast_does_not_wait(self, monkeypatch):
        waits = []
        monkeypatch.setattr(pcap_replay, "_sleep", lambda s, stop: waits.append(s))
        proc = RecordingProcessor()
        count = replay_packets(make_packets([0.0, 10.0, 20.0]), proc, speed="fast")
        assert count == 3 and len(proc.packets) == 3
        assert waits == []

    def test_original_waits_the_capture_gaps(self, monkeypatch):
        waits = []
        monkeypatch.setattr(pcap_replay, "_sleep", lambda s, stop: waits.append(s))
        replay_packets(make_packets([0.0, 0.5, 2.0]), RecordingProcessor(), speed="original")
        assert waits == [0.5, 1.5]

    def test_original_caps_long_gaps(self, monkeypatch):
        waits = []
        monkeypatch.setattr(pcap_replay, "_sleep", lambda s, stop: waits.append(s))
        replay_packets(make_packets([0.0, 3600.0]), RecordingProcessor(), speed="original", max_gap=5.0)
        assert waits == [5.0]

    def test_out_of_order_timestamps_do_not_wait_negative(self, monkeypatch):
        waits = []
        monkeypatch.setattr(pcap_replay, "_sleep", lambda s, stop: waits.append(s))
        replay_packets(make_packets([5.0, 1.0]), RecordingProcessor(), speed="original")
        assert waits == [0.0]

    def test_stops_when_asked(self):
        proc = RecordingProcessor()
        stop_after = {"n": 0}

        def should_stop():
            stop_after["n"] += 1
            return stop_after["n"] > 2

        count = replay_packets(make_packets([0, 1, 2, 3, 4]), proc, should_stop=should_stop)
        assert count == 2

    def test_rejects_unknown_speed(self):
        with pytest.raises(ValueError):
            replay_packets(make_packets([0.0]), RecordingProcessor(), speed="warp")

    def test_replay_pcap_reads_the_file(self, tmp_path):
        path = tmp_path / "t.pcap"
        wrpcap(str(path), make_packets([1.0, 2.0]))
        proc = RecordingProcessor()
        assert replay_pcap(str(path), proc) == 2


class TestPacketProcessorOfflineSupport:
    def test_blocking_enqueue_loses_nothing_with_a_tiny_queue(self):
        seen = []
        lock = threading.Lock()

        def slow(pkt):
            time.sleep(0.002)
            with lock:
                seen.append(pkt)

        proc = PacketProcessor(callback=slow, num_workers=1, max_queue_size=2)
        proc.start()
        try:
            for i in range(30):
                proc.enqueue(i, block=True)
            assert proc.drain(timeout=10)
            assert proc.stats["dropped_packets"] == 0
            assert sorted(seen) == list(range(30))
        finally:
            proc.stop()

    def test_non_blocking_enqueue_still_drops_when_full(self):
        proc = PacketProcessor(callback=lambda p: None, num_workers=1, max_queue_size=1)
        # workers are not started, so nothing drains the queue
        proc.enqueue(1)
        proc.enqueue(2)
        assert proc.stats["dropped_packets"] == 1

    def test_drain_times_out_when_nothing_processes_the_queue(self):
        proc = PacketProcessor(callback=lambda p: None, num_workers=1, max_queue_size=5)
        proc.enqueue(1)
        assert proc.drain(timeout=0.2) is False

    def test_drain_on_an_empty_queue(self):
        proc = PacketProcessor(callback=lambda p: None, num_workers=1, max_queue_size=5)
        assert proc.drain(timeout=0.2) is True


def run_main(*args, timeout=180):
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
    )


def anomaly_rows():
    db = REPO_ROOT / "anomalies.db"
    if not db.exists():
        return 0
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT COUNT(*) FROM anomalies").fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


class TestOfflineCli:
    """End-to-end: the real main.py, no root and no live capture."""

    def test_dry_run_runs_the_pipeline_and_leaves_the_real_db_alone(self):
        before = anomaly_rows()
        result = run_main("--dry-run")
        assert result.returncode == 0, result.stdout + result.stderr
        out = result.stdout
        assert "Dry run" in out
        assert "[ICMP_FLOOD]" in out            # rule engine ran
        assert "Malicious payload" in out        # payload analysis ran
        assert "Traffic summary:" in out         # same summary as live mode
        assert "IP: 10.0.0.77" in out
        assert anomaly_rows() == before

    def test_pcap_replay_from_a_file(self, tmp_path):
        path = tmp_path / "attack.pcap"
        packets = []
        for i in range(60):
            pkt = IP(src="10.9.9.9", dst="10.9.9.1") / ICMP() / Raw(load=b"y" * 16)
            pkt.time = 1_700_000_000.0 + i * 0.01
            packets.append(pkt)
        wrpcap(str(path), packets)

        result = run_main("--dry-run", "--pcap", str(path), "--replay-speed", "fast")
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"Replaying {path}" in result.stdout
        assert "60 packets read" in result.stdout
        assert "[ICMP_FLOOD]" in result.stdout
        assert "IP: 10.9.9.9" in result.stdout

    def test_pcap_with_original_speed(self, tmp_path):
        path = tmp_path / "slow.pcap"
        wrpcap(str(path), make_packets([1_700_000_000.0, 1_700_000_000.3]))
        started = time.monotonic()
        result = run_main("--dry-run", "--pcap", str(path), "--replay-speed", "original")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "speed: original" in result.stdout
        assert "2 packets read" in result.stdout
        assert time.monotonic() - started >= 0.3

    @pytest.mark.parametrize("args, message", [
        (["--pcap", "/definitely/not/here.pcap"], "PCAP file not found"),
        (["--dry-run", "--interface", "eth0"], "--interface cannot be combined"),
        (["--pcap", "x.pcap", "--interface", "eth0"], "--interface cannot be combined"),
    ])
    def test_invalid_combinations_fail_fast(self, args, message):
        result = run_main(*args)
        assert result.returncode == 1
        assert message in result.stdout

    def test_unknown_replay_speed_is_rejected_by_argparse(self):
        result = run_main("--dry-run", "--replay-speed", "warp")
        assert result.returncode == 1
        assert "invalid choice" in result.stderr
