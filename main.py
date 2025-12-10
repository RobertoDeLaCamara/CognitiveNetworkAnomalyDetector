#!/usr/bin/env python3
"""Main entry point for the cognitive anomaly detector."""

import sys
import os
import signal
import argparse
from scapy.all import sniff
from src.anomaly_detector import analyze_packet, packet_count_per_ip
from src.config import MONITORING_INTERVAL
from src.logger_setup import logger

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    print("\nShutdown requested, stopping monitoring...")
    shutdown_requested = True

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
    
    try:
        args = parser.parse_args()
        
        # Validate arguments
        if args.duration <= 0 or args.duration > 3600:
            print("Error: Duration must be between 1 and 3600 seconds")
            return 1
        
        if args.interface:
            if len(args.interface) > 20 or not args.interface.replace('-', '').replace('_', '').isalnum():
                print("Error: Invalid interface name")
                return 1
        
        monitoring_duration = args.duration
        interface = args.interface
        
    except SystemExit:
        return 1
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logger.info("Starting anomaly detector")
        
        # Check privileges
        check_privileges()
        
        while not shutdown_requested:
            print(f"\nStarting local network monitoring for {monitoring_duration} seconds...")
            
            # Start packet capture with error handling
            try:
                sniff(
                    prn=analyze_packet, 
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
            
            # Display summary of captured traffic
            print("\nTraffic summary:")
            if not packet_count_per_ip:
                print("No traffic was captured.")
                logger.info("No traffic captured during monitoring period")
            else:
                logger.info(f"Captured traffic from {len(packet_count_per_ip)} unique IPs")
                # Print the number of packets sent by each IP address
                for ip, count in sorted(packet_count_per_ip.items(), key=lambda x: x[1], reverse=True):
                    print(f"IP: {ip}, Packets sent: {count}")
            
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

if __name__ == "__main__":
    sys.exit(main())
