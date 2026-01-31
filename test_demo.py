#!/usr/bin/env python3
"""
test_demo.py - Demo script để test Q-Learning agent với synthetic data
Không cần Mininet/Ryu, chỉ test thuật toán Q-Learning
"""

import numpy as np
import pandas as pd
from rl_agent import QLearningAgent


def generate_synthetic_data(n_normal=500, n_attack=200):
    """
    Tạo synthetic data để test
    
    Normal traffic characteristics:
    - Packet rate: 50-200 packets/s
    - Byte rate: 5000-20000 bytes/s
    - Flow count: 5-15
    
    Attack traffic characteristics:
    - Packet rate: 500-2000 packets/s
    - Byte rate: 50000-200000 bytes/s
    - Flow count: 20-50
    """
    print(f"[+] Generating synthetic data...")
    print(f"    Normal samples: {n_normal}")
    print(f"    Attack samples: {n_attack}")
    
    # Normal traffic
    normal_data = []
    for _ in range(n_normal):
        features = {
            'packet_rate': np.random.uniform(50, 200),
            'byte_rate': np.random.uniform(5000, 20000),
            'flow_count': np.random.randint(5, 15)
        }
        normal_data.append((features, 0))
    
    # Attack traffic
    attack_data = []
    for _ in range(n_attack):
        features = {
            'packet_rate': np.random.uniform(500, 2000),
            'byte_rate': np.random.uniform(50000, 200000),
            'flow_count': np.random.randint(20, 50)
        }
        attack_data.append((features, 1))
    
    # Combine and shuffle
    all_data = normal_data + attack_data
    np.random.shuffle(all_data)
    
    print(f"[+] Generated {len(all_data)} total samples")
    return all_data


def split_data(data, train_ratio=0.8):
    """Split data into train and test sets"""
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    return train_data, test_data


def evaluate_model(agent, test_data):
    """Evaluate model on test data"""
    print("\n[+] Evaluating model on test set...")
    
    predictions = []
    true_labels = []
    
    for features, label in test_data:
        action, confidence = agent.predict(features)
        predictions.append(action)
        true_labels.append(label)
    
    # Calculate metrics
    true_positive = sum(1 for p, l in zip(predictions, true_labels) if p == 1 and l == 1)
    true_negative = sum(1 for p, l in zip(predictions, true_labels) if p == 0 and l == 0)
    false_positive = sum(1 for p, l in zip(predictions, true_labels) if p == 1 and l == 0)
    false_negative = sum(1 for p, l in zip(predictions, true_labels) if p == 0 and l == 1)
    
    accuracy = (true_positive + true_negative) / len(true_labels)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    print(f"Accuracy:       {accuracy:.2%}")
    print(f"Precision:      {precision:.2%}")
    print(f"Recall:         {recall:.2%}")
    print(f"F1-Score:       {f1_score:.2%}")
    print("-"*60)
    print(f"True Positive:  {true_positive}")
    print(f"True Negative:  {true_negative}")
    print(f"False Positive: {false_positive}")
    print(f"False Negative: {false_negative}")
    print("="*60)
    
    return accuracy, precision, recall, f1_score


def demo_training():
    """Run demo training with detailed reporting table"""
    print("\n" + "="*70)
    print("Q-LEARNING DDOS DETECTION - DEMO (500 EPISODES)")
    print("="*70)
    
    # 1. Tạo dữ liệu
    data = generate_synthetic_data(n_normal=1000, n_attack=500) # Tăng dữ liệu lên chút
    train_data, test_data = split_data(data, train_ratio=0.8)
    
    # 2. Khởi tạo Agent
    print("\n[+] Initializing Q-Learning agent...")
    agent = QLearningAgent(state_bins=[10, 10, 10], learning_rate=0.1, discount_factor=0.95, epsilon=1.0) # Epsilon bắt đầu = 1.0
    
    # Cập nhật feature ranges
    features_list = [f for f, l in data]
    agent.update_feature_ranges(features_list)
    
    # 3. Bắt đầu huấn luyện
    print("\n[+] Starting training...")
    
    # --- CẤU HÌNH ĐỂ RA BẢNG SỐ LIỆU ---
    episodes = 500             # Chạy 500 vòng như yêu cầu
    batch_size = 50
    
    # In tiêu đề bảng
    print("\n" + "-"*65)
    print(f"{'Episode':<12} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 65)

    for episode in range(episodes):
        # Lấy mẫu ngẫu nhiên (Mini-batch)
        batch_indices = np.random.choice(len(train_data), min(batch_size, len(train_data)), replace=False)
        batch_data = [train_data[i] for i in batch_indices]
        
        # Train
        agent.train_episode(batch_data)
        
        # --- CỨ MỖI 100 EPISODE THÌ TÍNH TOÁN VÀ IN RA BẢNG ---
        if (episode + 1) % 100 == 0:
            # Đánh giá trên tập TEST để có số liệu khách quan nhất
            preds = []
            true_lbls = []
            for f, l in test_data:
                a, _ = agent.predict(f)
                preds.append(a)
                true_lbls.append(l)
            
            # Tính toán thủ công các chỉ số
            tp = sum(1 for p, l in zip(preds, true_lbls) if p==1 and l==1)
            fp = sum(1 for p, l in zip(preds, true_lbls) if p==1 and l==0)
            fn = sum(1 for p, l in zip(preds, true_lbls) if p==0 and l==1)
            tn = sum(1 for p, l in zip(preds, true_lbls) if p==0 and l==0)
            
            acc = (tp + tn) / len(true_lbls)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
            # In dòng dữ liệu vào bảng
            interval = f"{episode-98}-{episode+1}"
            print(f"{interval:<12} | {acc:<10.1%} | {prec:<10.1%} | {rec:<10.1%} | {f1:<10.1%}")

    print("-" * 65)
    print("\n[+] Training completed!")
    
    # Lưu model
    agent.save('demo_model.pkl')
    return agent


def demo_inference():
    """Demo inference with saved model"""
    print("\n" + "="*70)
    print("INFERENCE DEMO - LOADING SAVED MODEL")
    print("="*70)
    
    model_file = 'demo_model.pkl'
    
    # Load model
    agent = QLearningAgent()
    if not agent.load(model_file):
        print("[!] Model not found! Run training demo first.")
        return
    
    # Test samples
    test_samples = [
        {'packet_rate': 80, 'byte_rate': 8000, 'flow_count': 6},      # Normal
        {'packet_rate': 1800, 'byte_rate': 180000, 'flow_count': 45}, # Attack
        {'packet_rate': 150, 'byte_rate': 15000, 'flow_count': 10},   # Normal
        {'packet_rate': 900, 'byte_rate': 90000, 'flow_count': 28},   # Attack
    ]
    
    print("\n[+] Testing with sample data:\n")
    
    for i, sample in enumerate(test_samples, 1):
        action, conf = agent.predict(sample)
        result = "ATTACK" if action == 1 else "NORMAL"
        
        print(f"Sample {i}:")
        print(f"  Packet Rate: {sample['packet_rate']:.0f} pkt/s")
        print(f"  Byte Rate:   {sample['byte_rate']:.0f} bytes/s")
        print(f"  Flow Count:  {sample['flow_count']}")
        print(f"  → Prediction: {result} (confidence: {conf:.4f})")
        print()


def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--inference':
        demo_inference()
    else:
        # Run training demo
        demo_training()
        
        # Ask if user wants to test inference
        print("\n" + "-"*70)
        response = input("Do you want to test inference with saved model? (y/n): ")
        if response.lower() == 'y':
            demo_inference()


if __name__ == '__main__':
    main()
