"""Asynchronous packet processing with a bounded queue and worker threads.

Decouples Scapy's sniff callback from ML inference to prevent blocking
the capture loop on expensive per-packet analysis.
"""

import os
import queue
import threading
from typing import Callable

from .logger_setup import logger


class PacketProcessor:
    """Queued packet processor with configurable worker threads.

    Args:
        callback: Function to call for each packet (e.g. analyze_packet).
        num_workers: Number of worker threads (env: PACKET_WORKERS, default 2).
        max_queue_size: Max items before dropping (env: PACKET_QUEUE_SIZE, default 10000).
    """

    def __init__(
        self,
        callback: Callable,
        num_workers: int = None,
        max_queue_size: int = None,
    ):
        self.callback = callback
        self.num_workers = num_workers or int(os.getenv("PACKET_WORKERS", "2"))
        self.max_queue_size = max_queue_size or int(os.getenv("PACKET_QUEUE_SIZE", "10000"))
        self._queue: queue.Queue = queue.Queue(maxsize=self.max_queue_size)
        self._stop_event = threading.Event()
        self._workers: list[threading.Thread] = []
        self._dropped_count = 0
        self._dropped_lock = threading.Lock()

    def enqueue(self, packet) -> None:
        """Non-blocking enqueue; increments dropped count on queue full."""
        try:
            self._queue.put_nowait(packet)
        except queue.Full:
            with self._dropped_lock:
                self._dropped_count += 1

    def _worker_loop(self) -> None:
        """Worker loop: pull packets from the queue and process them."""
        while not self._stop_event.is_set():
            try:
                packet = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.callback(packet)
            except Exception as e:
                logger.error(f"Packet processing error: {e}")
            finally:
                self._queue.task_done()

    def start(self) -> None:
        """Spawn worker daemon threads."""
        self._stop_event.clear()
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True, name=f"pkt-worker-{i}")
            t.start()
            self._workers.append(t)
        logger.info(f"PacketProcessor started with {self.num_workers} workers (queue size: {self.max_queue_size})")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal workers to stop and wait for them to finish."""
        self._stop_event.set()
        for t in self._workers:
            t.join(timeout=timeout)
        self._workers.clear()
        logger.info("PacketProcessor stopped")

    @property
    def stats(self) -> dict:
        """Current processor statistics."""
        with self._dropped_lock:
            dropped = self._dropped_count
        return {
            "queue_size": self._queue.qsize(),
            "dropped_packets": dropped,
            "worker_count": len(self._workers),
        }
