#!/usr/bin/env python3
"""Generate synthetic network traffic to trigger anomaly detections.

This script creates various types of network activity that should
trigger both ML-based and rule-based anomaly detection.
"""

import time
import random
import subprocess
import sys
import sys
from datetime import datetime
from scapy.all import IP, ICMP, send


def print_status(message):
    """Print status message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def generate_icmp_flood():
    """Generate ICMP flood (should trigger rule-based detection)."""
    print_status("🌊 Generating ICMP flood...")
    target = "8.8.8.8"
    
    for i in range(15):
        try:
            subprocess.run(
                ["ping", "-c", "1", "-W", "1", target],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2
            )
        except:
            pass
    
    print_status("✓ ICMP flood complete")


def generate_port_scan():
    """Generate port scanning activity (uncommon ports)."""
    print_status("🔍 Generating port scan activity...")
    target = "scanme.nmap.org"
    
    # Scan some uncommon ports
    uncommon_ports = [8888, 9999, 31337, 12345, 54321]
    
    for port in uncommon_ports:
        try:
            subprocess.run(
                ["timeout", "1", "nc", "-zv", target, str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2
            )
        except:
            pass
        time.sleep(0.2)
    
    print_status("✓ Port scan complete")


def generate_high_bandwidth():
    """Generate high bandwidth traffic."""
    print_status("📡 Generating high bandwidth traffic...")
    
    # Multiple rapid requests
    urls = [
        "http://example.com",
        "http://httpbin.org/get",
        "http://www.google.com",
        "http://www.cloudflare.com"
    ]
    
    for _ in range(20):
        url = random.choice(urls)
        try:
            subprocess.run(
                ["curl", "-s", "-m", "2", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3
            )
        except:
            pass
        time.sleep(0.1)
    
    print_status("✓ High bandwidth traffic complete")


def generate_dns_queries():
    """Generate rapid DNS queries."""
    print_status("🌐 Generating DNS query burst...")
    
    domains = [
        "example.com",
        "google.com", 
        "github.com",
        "stackoverflow.com",
        "reddit.com"
    ]
    
    for _ in range(10):
        domain = random.choice(domains)
        try:
            subprocess.run(
                ["nslookup", domain],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2
            )
        except:
            pass
        time.sleep(0.1)
    
    print_status("✓ DNS queries complete")


def generate_mixed_traffic():
    """Generate mixed normal and anomalous traffic."""
    print_status("🔀 Generating mixed traffic pattern...")
    
    actions = [
        lambda: subprocess.run(["ping", "-c", "1", "8.8.8.8"], 
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2),
        lambda: subprocess.run(["curl", "-s", "-m", "2", "http://example.com"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3),
        lambda: subprocess.run(["nslookup", "google.com"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2),
    ]
    
    for _ in range(15):
        try:
            random.choice(actions)()
        except:
            pass
        time.sleep(random.uniform(0.1, 0.5))
    
    print_status("✓ Mixed traffic complete")


def generate_diverse_ips():
    """Generate traffic with spoofed source IPs to ensure enough samples."""
    print_status("🌍 Generating traffic with SPOOFED source IPs using Scapy...")
    
    # Generate 120 random source IPs (need > 100 for training)
    for i in range(120):
        fake_ip = f"10.0.{100 + i // 254}.{i % 254 + 1}"
        
        # Create packet with spoofed source
        # We send it to a non-existent destination or local
        pkt = IP(src=fake_ip, dst="8.8.8.8")/ICMP()
        
        try:
            # Send multiple packets to meet minimum packet requirements for feature extraction
            for _ in range(4):
                send(pkt, verbose=False)
                time.sleep(0.002)
        except Exception as e:
            # Requires root usually, but container runs as root
            # On host (traffic injector), might fail if not root.
            # But the user is 'roberto', might have sudoless docker but not raw socket access?
            print(f"Warning: Failed to send spoofed packet: {e}")
            pass
        
        time.sleep(0.05)
    
    print_status("✓ Spoofed IP traffic complete")


def main():
    """Main function to run synthetic traffic generation."""
    print("=" * 60)
    print("🚀 SYNTHETIC TRAFFIC GENERATOR")
    print("=" * 60)
    print("\nThis will generate network traffic to trigger anomaly detection.")
    print("Make sure the anomaly detector is running!\n")
    
    # Wait a moment for user to see the message
    time.sleep(2)
    
    try:
        # Pattern 1: ICMP Flood (rule-based detection)
        generate_icmp_flood()
        time.sleep(2)
        
        # Pattern 2: Port Scanning (uncommon ports)
        generate_port_scan()
        time.sleep(2)
        
        # Pattern 3: High bandwidth (traffic spike)
        generate_high_bandwidth()
        time.sleep(2)
        
        # Pattern 4: DNS query burst
        generate_dns_queries()
        time.sleep(2)
        
        # Pattern 5: Mixed traffic
        generate_mixed_traffic()
        
        # Pattern 6: Diverse IPs (ensure >10 samples)
        generate_diverse_ips()
        
        print("\n" + "=" * 60)
        print_status("✅ Synthetic traffic generation complete!")
        print("=" * 60)
        print("\n📊 Check your dashboard for detected anomalies!")
        print("   Dashboard: http://localhost:8501")
        print("\n💡 Tip: Enable auto-refresh on the Home page to see")
        print("   anomalies appear in real-time.\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Traffic generation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
