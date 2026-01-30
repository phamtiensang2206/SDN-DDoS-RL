#!/usr/bin/env python3
"""
detect_realtime.py - Real-time DDoS Detection
Monitor traffic và phát hiện DDoS attacks real-time sử dụng trained Q-Learning model
"""

import time
import pandas as pd
import argparse
import os
from datetime import datetime
from rl_agent import QLearningAgent


class RealtimeDetector:
    """Real-time DDoS detector"""
    
    def __init__(self, model_path='rl_model.pkl', data_path='data.csv'):
        """
        Initialize detector
        
        Args:
            model_path: Path to trained model
            data_path: Path to data.csv being updated by controller
        """
        self.model_path = model_path
        self.data_path = data_path
        self.agent = None
        self.last_processed_index = 0
        self.attack_count = 0
        self.normal_count = 0
        self.alerts = []
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load trained Q-Learning model"""
        print(f"[+] Loading model from {self.model_path}...")
        
        self.agent = QLearningAgent()
        if not self.agent.load(self.model_path):
            print("[!] Failed to load model!")
            print("[!] Please train a model first using: python3 train.py")
            exit(1)
        
        print("[+] Model loaded successfully!")
    
    def _read_new_records(self):
        """
        Đọc các records mới từ data.csv
        
        Returns:
            List of new records (DataFrame rows)
        """
        if not os.path.exists(self.data_path):
            return []
        
        try:
            df = pd.read_csv(self.data_path)
            
            if len(df) > self.last_processed_index:
                new_records = df.iloc[self.last_processed_index:]
                self.last_processed_index = len(df)
                return new_records
            
            return []
        
        except Exception as e:
            print(f"[!] Error reading data: {e}")
            return []
    
    def _process_record(self, record):
        """
        Process một record và detect attack
        
        Args:
            record: DataFrame row
            
        Returns:
            is_attack (bool), confidence (float)
        """
        # Extract features
        features = {
            'packet_rate': record.get('packet_rate', 0),
            'byte_rate': record.get('byte_rate', 0),
            'flow_count': record.get('flow_count', 0)
        }
        
        # Predict
        action, confidence = self.agent.predict(features)
        
        return (action == 1), confidence, features
    
    def _alert(self, record, confidence, features):
        """
        Raise alert khi detect attack
        
        Args:
            record: Traffic record
            confidence: Detection confidence
            features: Extracted features
        """
        alert_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        alert_info = {
            'time': alert_time,
            'src_ip': record.get('src_ip', 'unknown'),
            'dst_ip': record.get('dst_ip', 'unknown'),
            'protocol': record.get('protocol', 'unknown'),
            'packet_rate': features['packet_rate'],
            'byte_rate': features['byte_rate'],
            'flow_count': features['flow_count'],
            'confidence': confidence
        }
        
        self.alerts.append(alert_info)
        
        # Print alert
        print("\n" + "="*70)
        print("⚠️  [ALERT] DDoS ATTACK DETECTED!")
        print("="*70)
        print(f"Time:         {alert_time}")
        print(f"Source:       {alert_info['src_ip']} → {alert_info['dst_ip']}")
        print(f"Protocol:     {alert_info['protocol']}")
        print(f"Packet Rate:  {alert_info['packet_rate']:.2f} packets/s")
        print(f"Byte Rate:    {alert_info['byte_rate']:.2f} bytes/s")
        print(f"Flow Count:   {alert_info['flow_count']}")
        print(f"Confidence:   {confidence:.4f}")
        print("="*70 + "\n")
        
        # Optional: Implement mitigation actions here
        # e.g., block IP, rate limit, notify admin
    
    def monitor(self, interval=5, verbose=True):
        """
        Monitor traffic và detect attacks
        
        Args:
            interval: Check interval in seconds
            verbose: Print verbose output
        """
        print("\n[+] Starting real-time DDoS detection...")
        print(f"[+] Monitoring: {self.data_path}")
        print(f"[+] Check interval: {interval} seconds")
        print("[+] Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Read new records
                new_records = self._read_new_records()
                
                if len(new_records) > 0:
                    if verbose:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                              f"Processing {len(new_records)} new records...")
                    
                    # Process each record
                    for _, record in new_records.iterrows():
                        is_attack, confidence, features = self._process_record(record)
                        
                        if is_attack:
                            self.attack_count += 1
                            self._alert(record, confidence, features)
                        else:
                            self.normal_count += 1
                            if verbose and self.normal_count % 10 == 0:
                                print(f"    ✓ {self.normal_count} normal flows processed")
                
                # Sleep
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n\n[+] Stopping detector...")
            self._print_summary()
    
    def _print_summary(self):
        """Print detection summary"""
        print("\n" + "="*70)
        print("DETECTION SUMMARY")
        print("="*70)
        print(f"Total Records Processed: {self.normal_count + self.attack_count}")
        print(f"Normal Traffic:          {self.normal_count}")
        print(f"Attacks Detected:        {self.attack_count}")
        
        if self.attack_count > 0:
            print(f"\nAttack Rate:             {self.attack_count/(self.normal_count+self.attack_count)*100:.2f}%")
            
            print(f"\nTop {min(5, len(self.alerts))} Recent Alerts:")
            for i, alert in enumerate(self.alerts[-5:], 1):
                print(f"\n  {i}. Time: {alert['time']}")
                print(f"     {alert['src_ip']} → {alert['dst_ip']}")
                print(f"     Packet Rate: {alert['packet_rate']:.2f}, "
                      f"Confidence: {alert['confidence']:.4f}")
        
        print("="*70 + "\n")
    
    def save_alerts(self, filepath='alerts.csv'):
        """
        Save alerts to CSV
        
        Args:
            filepath: Output file path
        """
        if not self.alerts:
            print("[!] No alerts to save")
            return
        
        df = pd.DataFrame(self.alerts)
        df.to_csv(filepath, index=False)
        print(f"[+] {len(self.alerts)} alerts saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Real-time DDoS Detection')
    parser.add_argument('--model', type=str, default='rl_model.pkl',
                       help='Path to trained model')
    parser.add_argument('--data', type=str, default='data.csv',
                       help='Path to data.csv file')
    parser.add_argument('--interval', type=int, default=5,
                       help='Check interval in seconds')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--save-alerts', type=str, default=None,
                       help='Save alerts to CSV file')
    
    args = parser.parse_args()
    
    # Create detector
    detector = RealtimeDetector(
        model_path=args.model,
        data_path=args.data
    )
    
    # Start monitoring
    try:
        detector.monitor(interval=args.interval, verbose=args.verbose)
    except KeyboardInterrupt:
        pass
    finally:
        # Save alerts if requested
        if args.save_alerts:
            detector.save_alerts(args.save_alerts)


if __name__ == '__main__':
    main()
