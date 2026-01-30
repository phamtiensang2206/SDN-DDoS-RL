#!/usr/bin/env python3
"""
label_data.py - Tự động gán nhãn cho dữ liệu traffic
Sử dụng threshold-based labeling cho attack detection
"""

import pandas as pd
import argparse
import numpy as np


def auto_label_data(filepath='data.csv', 
                   packet_rate_threshold=None,
                   byte_rate_threshold=None,
                   flow_count_threshold=None,
                   output_file=None):
    """
    Tự động label data dựa trên threshold
    
    Args:
        filepath: Path to input CSV
        packet_rate_threshold: Threshold cho packet rate (packets/s)
        byte_rate_threshold: Threshold cho byte rate (bytes/s)
        flow_count_threshold: Threshold cho flow count
        output_file: Output file path (None = overwrite input)
    """
    print(f"[+] Loading data from {filepath}...")
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"[!] Error: File {filepath} not found!")
        return
    
    print(f"[+] Total records: {len(df)}")
    
    # Analyze current distribution
    if 'label' in df.columns:
        print(f"\n[+] Current label distribution:")
        print(f"    Normal (0): {len(df[df['label']==0])}")
        print(f"    Attack (1): {len(df[df['label']==1])}")
    
    # Calculate statistics
    print(f"\n[+] Traffic statistics:")
    print(f"    Packet Rate - Mean: {df['packet_rate'].mean():.2f}, "
          f"Std: {df['packet_rate'].std():.2f}, "
          f"Max: {df['packet_rate'].max():.2f}")
    print(f"    Byte Rate   - Mean: {df['byte_rate'].mean():.2f}, "
          f"Std: {df['byte_rate'].std():.2f}, "
          f"Max: {df['byte_rate'].max():.2f}")
    print(f"    Flow Count  - Mean: {df['flow_count'].mean():.2f}, "
          f"Std: {df['flow_count'].std():.2f}, "
          f"Max: {df['flow_count'].max():.2f}")
    
    # Auto-calculate thresholds if not provided
    if packet_rate_threshold is None:
        # Use 95th percentile + 2*std as threshold
        packet_rate_threshold = df['packet_rate'].quantile(0.95) + 2 * df['packet_rate'].std()
    
    if byte_rate_threshold is None:
        byte_rate_threshold = df['byte_rate'].quantile(0.95) + 2 * df['byte_rate'].std()
    
    if flow_count_threshold is None:
        flow_count_threshold = df['flow_count'].quantile(0.95) + 2 * df['flow_count'].std()
    
    print(f"\n[+] Using thresholds:")
    print(f"    Packet Rate: > {packet_rate_threshold:.2f}")
    print(f"    Byte Rate:   > {byte_rate_threshold:.2f}")
    print(f"    Flow Count:  > {flow_count_threshold:.2f}")
    
    # Label attacks
    print(f"\n[+] Labeling data...")
    
    # Initialize all as normal
    df['label'] = 0
    
    # Mark as attack if ANY threshold is exceeded
    attack_mask = (
        (df['packet_rate'] > packet_rate_threshold) |
        (df['byte_rate'] > byte_rate_threshold) |
        (df['flow_count'] > flow_count_threshold)
    )
    
    df.loc[attack_mask, 'label'] = 1
    
    # New distribution
    normal_count = len(df[df['label'] == 0])
    attack_count = len(df[df['label'] == 1])
    
    print(f"\n[+] New label distribution:")
    print(f"    Normal (0): {normal_count} ({normal_count/len(df)*100:.1f}%)")
    print(f"    Attack (1): {attack_count} ({attack_count/len(df)*100:.1f}%)")
    
    if attack_count == 0:
        print("\n[!] WARNING: No attacks detected! Thresholds might be too high.")
        print("    Consider lowering thresholds or checking if attack traffic was generated.")
    elif attack_count / len(df) > 0.8:
        print("\n[!] WARNING: Too many attacks detected! Thresholds might be too low.")
        print("    Consider raising thresholds.")
    
    # Save
    output_path = output_file if output_file else filepath
    df.to_csv(output_path, index=False)
    print(f"\n[+] Labeled data saved to: {output_path}")


def manual_label_time_range(filepath='data.csv',
                            start_time=None,
                            end_time=None,
                            label=1,
                            output_file=None):
    """
    Label dữ liệu trong một khoảng thời gian cụ thể
    Hữu ích khi bạn biết chính xác khi nào attack xảy ra
    
    Args:
        filepath: Input CSV path
        start_time: Start time (format: 'YYYY-MM-DD HH:MM:SS')
        end_time: End time
        label: Label to assign (0 or 1)
        output_file: Output path
    """
    print(f"[+] Loading data from {filepath}...")
    
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if start_time:
        start_time = pd.to_datetime(start_time)
        print(f"[+] Start time: {start_time}")
    
    if end_time:
        end_time = pd.to_datetime(end_time)
        print(f"[+] End time: {end_time}")
    
    # Initialize labels if not exist
    if 'label' not in df.columns:
        df['label'] = 0
    
    # Label time range
    if start_time and end_time:
        mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
    elif start_time:
        mask = df['timestamp'] >= start_time
    elif end_time:
        mask = df['timestamp'] <= end_time
    else:
        print("[!] No time range specified!")
        return
    
    df.loc[mask, 'label'] = label
    
    print(f"[+] Labeled {mask.sum()} records as {label}")
    
    # Save
    output_path = output_file if output_file else filepath
    df.to_csv(output_path, index=False)
    print(f"[+] Saved to: {output_path}")


def interactive_label():
    """
    Interactive labeling mode
    """
    print("\n" + "="*60)
    print("INTERACTIVE DATA LABELING")
    print("="*60)
    
    filepath = input("Enter CSV file path [data.csv]: ").strip() or 'data.csv'
    
    print("\nLabeling method:")
    print("1. Auto-label using thresholds")
    print("2. Label by time range")
    
    choice = input("Choose method (1 or 2): ").strip()
    
    if choice == '1':
        print("\nEnter thresholds (press Enter to use auto-calculated):")
        
        pr = input("Packet Rate threshold: ").strip()
        packet_rate_threshold = float(pr) if pr else None
        
        br = input("Byte Rate threshold: ").strip()
        byte_rate_threshold = float(br) if br else None
        
        fc = input("Flow Count threshold: ").strip()
        flow_count_threshold = float(fc) if fc else None
        
        auto_label_data(filepath, packet_rate_threshold, byte_rate_threshold, flow_count_threshold)
    
    elif choice == '2':
        print("\nEnter time range (format: YYYY-MM-DD HH:MM:SS):")
        start_time = input("Start time: ").strip()
        end_time = input("End time: ").strip()
        label = int(input("Label (0=normal, 1=attack): ").strip())
        
        manual_label_time_range(filepath, start_time, end_time, label)
    
    else:
        print("Invalid choice!")


def main():
    parser = argparse.ArgumentParser(description='Auto-label traffic data for DDoS detection')
    parser.add_argument('--input', type=str, default='data.csv',
                       help='Input CSV file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (default: overwrite input)')
    parser.add_argument('--packet-threshold', type=float, default=None,
                       help='Packet rate threshold (auto if not specified)')
    parser.add_argument('--byte-threshold', type=float, default=None,
                       help='Byte rate threshold (auto if not specified)')
    parser.add_argument('--flow-threshold', type=float, default=None,
                       help='Flow count threshold (auto if not specified)')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_label()
    else:
        auto_label_data(
            filepath=args.input,
            packet_rate_threshold=args.packet_threshold,
            byte_rate_threshold=args.byte_threshold,
            flow_count_threshold=args.flow_threshold,
            output_file=args.output
        )


if __name__ == '__main__':
    main()
