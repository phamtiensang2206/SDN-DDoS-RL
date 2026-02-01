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
    """Run demo training"""
    print("\n" + "="*70)
    print("Q-LEARNING DDOS DETECTION - DEMO")
    print("="*70)
    
    # Generate data
    data = generate_synthetic_data(n_normal=800, n_attack=400)
    
    # Split data
    train_data, test_data = split_data(data, train_ratio=0.8)
    print(f"\n[+] Data split:")
    print(f"    Training: {len(train_data)} samples")
    print(f"    Testing:  {len(test_data)} samples")
    
    # Initialize agent
    print("\n[+] Initializing Q-Learning agent...")
    agent = QLearningAgent(
        state_bins=[15, 15, 10],
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.3,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Update feature ranges
    features_list = [f for f, l in data]
    agent.update_feature_ranges(features_list)
    
    print("[+] Feature ranges:")
    for feat, (min_val, max_val) in agent.feature_ranges.items():
        print(f"    {feat}: [{min_val:.2f}, {max_val:.2f}]")
    
    # Training
    print("\n[+] Starting training...\n")
    
    episodes = 500
    batch_size = 50
    
    # Store metrics for each episode range
    metrics_data = {
        '0-100': [],
        '101-200': [],
        '201-300': [],
        '301-400': [],
        '401-500': []
    }
    episode_range_map = {
        range(0, 100): '0-100',
        range(100, 200): '101-200',
        range(200, 300): '201-300',
        range(300, 400): '301-400',
        range(400, 500): '401-500'
    }
    
    for episode in range(episodes):
        # Sample batch
        batch = np.random.choice(len(train_data), min(batch_size, len(train_data)), replace=False)
        batch_data = [train_data[i] for i in batch]
        
        # Train
        reward, accuracy = agent.train_episode(batch_data)
        
        # Evaluate on test set for this episode
        predictions = []
        true_labels = []
        for features, label in test_data:
            action, _ = agent.predict(features)
            predictions.append(action)
            true_labels.append(label)
        
        # Calculate metrics
        true_positive = sum(1 for p, l in zip(predictions, true_labels) if p == 1 and l == 1)
        true_negative = sum(1 for p, l in zip(predictions, true_labels) if p == 0 and l == 0)
        false_positive = sum(1 for p, l in zip(predictions, true_labels) if p == 1 and l == 0)
        false_negative = sum(1 for p, l in zip(predictions, true_labels) if p == 0 and l == 1)
        
        ep_accuracy = (true_positive + true_negative) / len(true_labels)
        ep_precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        ep_recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        ep_f1 = 2 * (ep_precision * ep_recall) / (ep_precision + ep_recall) if (ep_precision + ep_recall) > 0 else 0
        
        # Store metrics
        for ep_range, range_name in episode_range_map.items():
            if episode in ep_range:
                metrics_data[range_name].append({
                    'accuracy': ep_accuracy,
                    'precision': ep_precision,
                    'recall': ep_recall,
                    'f1': ep_f1
                })
                break
        
        # Print progress
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1:3d}/{episodes}: "
                  f"Reward={reward:6.2f}, "
                  f"Accuracy={ep_accuracy:.2%}, "
                  f"ε={agent.epsilon:.4f}")
    
    print("\n[+] Training completed!")
    
    # Print statistics
    agent.print_statistics()
    
    # Print performance table
    print("\n" + "="*80)
    print("4.2.1. Hiệu suất huấn luyện")
    print("="*80)
    print(f"{'Episode':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*80)
    
    for range_name in ['0-100', '101-200', '201-300', '301-400', '401-500']:
        if metrics_data[range_name]:
            # Calculate average for this range
            avg_accuracy = np.mean([m['accuracy'] for m in metrics_data[range_name]])
            avg_precision = np.mean([m['precision'] for m in metrics_data[range_name]])
            avg_recall = np.mean([m['recall'] for m in metrics_data[range_name]])
            avg_f1 = np.mean([m['f1'] for m in metrics_data[range_name]])
            
            print(f"{range_name:<12} {avg_accuracy:>10.1%}  {avg_precision:>10.1%}  {avg_recall:>10.1%}  {avg_f1:>10.1%}")
    
    print("="*80)
    
    # Evaluate
    evaluate_model(agent, test_data)
    
    # Test with specific samples
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    # Normal traffic sample
    normal_sample = {
        'packet_rate': 100,
        'byte_rate': 10000,
        'flow_count': 8
    }
    action, conf = agent.predict(normal_sample)
    print(f"\n1. Normal Traffic Sample:")
    print(f"   Features: {normal_sample}")
    print(f"   Prediction: {'ATTACK' if action == 1 else 'NORMAL'} (confidence: {conf:.4f})")
    
    # Attack sample
    attack_sample = {
        'packet_rate': 1500,
        'byte_rate': 150000,
        'flow_count': 35
    }
    action, conf = agent.predict(attack_sample)
    print(f"\n2. Attack Traffic Sample:")
    print(f"   Features: {attack_sample}")
    print(f"   Prediction: {'ATTACK' if action == 1 else 'NORMAL'} (confidence: {conf:.4f})")
    
    # Borderline case
    borderline_sample = {
        'packet_rate': 350,
        'byte_rate': 35000,
        'flow_count': 18
    }
    action, conf = agent.predict(borderline_sample)
    print(f"\n3. Borderline Sample:")
    print(f"   Features: {borderline_sample}")
    print(f"   Prediction: {'ATTACK' if action == 1 else 'NORMAL'} (confidence: {conf:.4f})")
    
    print("\n" + "="*60)
    
    # Save model
    model_file = 'demo_model.pkl'
    agent.save(model_file)
    print(f"\n[+] Demo completed! Model saved to {model_file}")
    
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
