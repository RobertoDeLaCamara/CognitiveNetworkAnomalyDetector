#!/usr/bin/env python3
"""Main entry point for the cognitive anomaly detector."""

import sys
import os
import signal
import argparse
from dotenv import load_dotenv

load_dotenv()

from scapy.all import sniff
from src.detection.anomaly_detector import analyze_packet, packet_analyzer, packet_count_per_ip
from src.config.config import MONITORING_INTERVAL
from src.core.logger_setup import logger
from src.core.packet_queue import PacketProcessor
from src.core.pcap_replay import REPLAY_SPEEDS, replay_packets, replay_pcap, synthetic_packets

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    print("\nShutdown requested, stopping monitoring...")
    shutdown_requested = True

def print_summary(processor):
    """Print the per-IP traffic summary and the packet processor statistics."""
    print("\nTraffic summary:")
    if not packet_count_per_ip:
        print("No traffic was captured.")
        logger.info("No traffic captured during monitoring period")
    else:
        logger.info(f"Captured traffic from {len(packet_count_per_ip)} unique IPs")
        # Print the number of packets sent by each IP address
        for ip, count in sorted(packet_count_per_ip.items(), key=lambda x: x[1], reverse=True):
            print(f"IP: {ip}, Packets sent: {count}")

    stats = processor.stats
    print(f"\nPacket processor: queue_size={stats['queue_size']}, "
          f"dropped={stats['dropped_packets']}, workers={stats['worker_count']}")

def run_offline(processor, pcap_path=None, speed="fast", dry_run=False):
    """Run the detection pipeline on a PCAP file or the built-in sample (no capture, no root).

    In dry-run mode alerts go to a temporary database instead of anomalies.db.
    Returns the number of packets read.
    """
    temp_dir = None
    if dry_run:
        import tempfile
        from src.core.db_manager import DBManager
        temp_dir = tempfile.mkdtemp(prefix="cad-dry-run-")
        packet_analyzer.db_manager = DBManager(os.path.join(temp_dir, "dry-run.db"))
        print("Dry run: alerts are stored in a temporary database, not in anomalies.db.")

    try:
        source = pcap_path or "the built-in synthetic sample"
        print(f"\nReplaying {source} (speed: {speed})...")
        processor.start()
        def stop():
            return shutdown_requested

        if pcap_path:
            count = replay_pcap(pcap_path, processor, speed, stop)
        else:
            count = replay_packets(synthetic_packets(), processor, speed, stop)
        if not processor.drain():
            logger.warning("Timed out waiting for queued packets to be processed")
        print(f"\nReplay finished: {count} packets read.")
        print_summary(processor)
        return count
    finally:
        if temp_dir:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

def check_privileges():
    """Check if running with sufficient privileges for packet capture."""
    if os.name == 'posix':  # Unix-like systems
        if os.geteuid() != 0:
            logger.warning("Running without root privileges. Packet capture may be limited.")
            print("Warning: Not running as root. Some network interfaces may not be accessible.")
            return False
    return True

def main():
    """Main function that monitors the network and detects anomalies.

    This function starts the local network monitoring process by using Scapy's sniff
    function to capture packets. The analyze_packet function is used as the callback
    function for sniff, which is called with each captured packet.

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Network anomaly detector")
    parser.add_argument(
        '--duration',
        type=int,
        default=MONITORING_INTERVAL,
        help=f'Monitoring duration in seconds (default: {MONITORING_INTERVAL})'
    )
    parser.add_argument(
        '--interface',
        type=str,
        default=None,
        help='Network interface to monitor (default: all)'
    )
    parser.add_argument(
        '--pcap',
        type=str,
        default=None,
        metavar='FILE',
        help='Replay a PCAP file through the detector instead of capturing live traffic (no root needed)'
    )
    parser.add_argument(
        '--replay-speed',
        choices=REPLAY_SPEEDS,
        default='fast',
        help="With --pcap/--dry-run: 'fast' = no delay (default), 'original' = keep the capture's timing "
             "(each gap capped at 5 s)"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run the full pipeline without live capture or root, on --pcap or a built-in synthetic sample. '
             'Alerts go to a temporary database, not to anomalies.db'
    )

    try:
        args = parser.parse_args()

        # Validate arguments with strict security checks
        if args.duration <= 0 or args.duration > 3600:
            print("Error: Duration must be between 1 and 3600 seconds")
            return 1

        if (args.pcap or args.dry_run) and args.interface:
            print("Error: --interface cannot be combined with --pcap or --dry-run (they do not capture live traffic)")
            return 1

        if args.pcap and not os.path.isfile(args.pcap):
            print(f"Error: PCAP file not found: {args.pcap}")
            return 1

        if args.interface:
            # Strict interface name validation to prevent injection
            import re
            if (len(args.interface) > 15 or 
                not re.match(r'^[a-zA-Z0-9_-]+$', args.interface) or
                args.interface.startswith('-') or
                '..' in args.interface):
                print("Error: Invalid interface name. Use only alphanumeric, underscore, and hyphen.")
                return 1
        
        monitoring_duration = args.duration
        interface = args.interface
        
    except SystemExit:
        return 1
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    processor = PacketProcessor(callback=analyze_packet)

    try:
        logger.info("Starting anomaly detector")

        # Offline mode: replay a PCAP file or the built-in sample, then exit.
        if args.pcap or args.dry_run:
            run_offline(processor, args.pcap, args.replay_speed, args.dry_run)
            print("Replay complete.")
            logger.info("Offline replay finished successfully")
            return 0

        # Check privileges
        check_privileges()

        processor.start()

        while not shutdown_requested:
            print(f"\nStarting local network monitoring for {monitoring_duration} seconds...")

            # Start packet capture with error handling
            try:
                # Additional security: validate interface exists if specified
                if interface:
                    try:
                        import psutil
                        available_interfaces = list(psutil.net_if_addrs().keys())
                        if interface not in available_interfaces:
                            logger.error(f"Interface '{interface}' not found")
                            print(f"Error: Interface '{interface}' not available")
                            return 1
                    except ImportError:
                        logger.warning("psutil not available - skipping interface validation")
                        print(f"Warning: Cannot validate interface '{interface}' - psutil not installed")

                sniff(
                    prn=processor.enqueue,
                    timeout=monitoring_duration,
                    iface=interface,
                    store=False,
                    stop_filter=lambda x: shutdown_requested
                )
            except PermissionError:
                logger.error("Permission denied: Cannot capture packets. Run as root or check network permissions.")
                print("Error: Permission denied. Try running with sudo or check network interface permissions.")
                return 1
            except OSError as e:
                logger.error(f"Network interface error: {e}")
                print(f"Error: Network interface problem - {e}")
                return 1

            # Display summary of captured traffic and packet processor stats
            print_summary(processor)

            if shutdown_requested:
                break

        print("Monitoring finished.")
        logger.info("Anomaly detector finished successfully")
        return 0

    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
        logger.info("Monitoring interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1
    finally:
        processor.stop()

if __name__ == "__main__":
    sys.exit(main())
