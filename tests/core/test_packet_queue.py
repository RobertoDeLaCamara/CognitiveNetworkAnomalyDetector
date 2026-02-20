"""Tests for the async packet processing queue."""

import threading
import time
import pytest
from src.core.packet_queue import PacketProcessor


class TestPacketProcessor:
    """Test the PacketProcessor queue and worker threads."""

    def test_enqueue_dequeue(self):
        """Packets enqueued should be processed by workers."""
        results = []
        lock = threading.Lock()

        def callback(pkt):
            with lock:
                results.append(pkt)

        proc = PacketProcessor(callback=callback, num_workers=1, max_queue_size=100)
        proc.start()
        try:
            for i in range(10):
                proc.enqueue(i)
            # Wait for processing
            time.sleep(1)
            with lock:
                assert sorted(results) == list(range(10))
        finally:
            proc.stop()

    def test_queue_full_drops(self):
        """When queue is full, enqueue should increment dropped count."""
        # Use a blocking callback to fill the queue
        block_event = threading.Event()

        def blocking_callback(pkt):
            block_event.wait()

        proc = PacketProcessor(callback=blocking_callback, num_workers=1, max_queue_size=5)
        proc.start()
        try:
            # Fill the queue (1 item taken by worker, 5 in queue = 6 total)
            for i in range(20):
                proc.enqueue(i)
            stats = proc.stats
            assert stats["dropped_packets"] > 0
        finally:
            block_event.set()
            proc.stop()

    def test_stop_with_pending(self):
        """Stopping with pending items should not hang."""
        proc = PacketProcessor(callback=lambda p: time.sleep(0.01), num_workers=1, max_queue_size=100)
        proc.start()
        for i in range(5):
            proc.enqueue(i)
        proc.stop(timeout=2.0)
        assert len(proc._workers) == 0

    def test_multiple_workers(self):
        """Multiple workers should process concurrently."""
        worker_ids = set()
        lock = threading.Lock()

        def callback(pkt):
            with lock:
                worker_ids.add(threading.current_thread().name)
            time.sleep(0.05)

        proc = PacketProcessor(callback=callback, num_workers=3, max_queue_size=100)
        proc.start()
        try:
            for i in range(30):
                proc.enqueue(i)
            time.sleep(2)
            with lock:
                assert len(worker_ids) > 1, f"Expected multiple workers, got {worker_ids}"
        finally:
            proc.stop()

    def test_stats_property(self):
        """Stats should report correct values."""
        proc = PacketProcessor(callback=lambda p: None, num_workers=2, max_queue_size=100)
        assert proc.stats["dropped_packets"] == 0
        assert proc.stats["worker_count"] == 0
        proc.start()
        assert proc.stats["worker_count"] == 2
        proc.stop()

    def test_callback_exception_doesnt_crash_worker(self):
        """Exceptions in callback should not kill the worker thread."""
        call_count = 0
        lock = threading.Lock()

        def flaky_callback(pkt):
            nonlocal call_count
            with lock:
                call_count += 1
            if pkt == 0:
                raise ValueError("test error")

        proc = PacketProcessor(callback=flaky_callback, num_workers=1, max_queue_size=100)
        proc.start()
        try:
            proc.enqueue(0)  # will raise
            proc.enqueue(1)  # should still be processed
            proc.enqueue(2)
            time.sleep(1)
            with lock:
                assert call_count == 3
        finally:
            proc.stop()
