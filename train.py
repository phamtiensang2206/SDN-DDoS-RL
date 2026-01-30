#!/usr/bin/env python3
"""
train.py - Train Q-Learning Agent for DDoS Detection
Đọc data từ data.csv, train agent, và evaluate performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from rl_agent import QLearningAgent


def load_and_preprocess_data(filepath='data.csv'):
    """
    Load và preprocess data từ CSV
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        features_list, labels_list
    """
    print(f"[+] Loading data from {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"[!] Error: File {filepath} not found!")
        return None, None
    
    # Read CSV
    df = pd.read_csv(filepath)
    print(f"[+] Loaded {len(df)} records")
    
    # Check columns
    required_cols = ['packet_rate', 'byte_rate', 'flow_count', 'label']
    for col in required_cols:
        if col not in df.columns:
            print(f"[!] Missing required column: {col}")
            return None, None
    
    # Remove rows with unknown IPs
    df = df[(df['src_ip'] != 'unknown') & (df['dst_ip'] != 'unknown')]
    
    # Handle missing values
    df = df.fillna(0)
    
    # Remove outliers (optional)
    for col in ['packet_rate', 'byte_rate', 'flow_count']:
        q99 = df[col].quantile(0.99)
        df = df[df[col] <= q99]
    
    print(f"[+] After preprocessing: {len(df)} records")
    
    # Distribution
    print(f"[+] Label distribution:")
    print(f"    Normal (0): {len(df[df['label'] == 0])}")
    print(f"    Attack (1): {len(df[df['label'] == 1])}")
    
    # Extract features and labels
    features_list = []
    labels_list = []
    
    for _, row in df.iterrows():
        features = {
            'packet_rate': row['packet_rate'],
            'byte_rate': row['byte_rate'],
            'flow_count': row['flow_count']
        }
        features_list.append(features)
        labels_list.append(int(row['label']))
    
    return features_list, labels_list


def balance_dataset(features, labels, method='undersample'):
    """
    Balance dataset để tránh bias
    
    Args:
        features: List of feature dicts
        labels: List of labels
        method: 'undersample' or 'oversample'
        
    Returns:
        Balanced features, labels
    """
    print(f"[+] Balancing dataset using {method}...")
    
    # Separate by class
    normal_features = [f for f, l in zip(features, labels) if l == 0]
    attack_features = [f for f, l in zip(features, labels) if l == 1]
    
    print(f"    Normal: {len(normal_features)}, Attack: {len(attack_features)}")
    
    if method == 'undersample':
        # Undersample majority class
        min_size = min(len(normal_features), len(attack_features))
        normal_features = normal_features[:min_size]
        attack_features = attack_features[:min_size]
    
    elif method == 'oversample':
        # Oversample minority class
        max_size = max(len(normal_features), len(attack_features))
        
        if len(normal_features) < max_size:
            normal_features = normal_features * (max_size // len(normal_features)) + \
                            normal_features[:max_size % len(normal_features)]
        if len(attack_features) < max_size:
            attack_features = attack_features * (max_size // len(attack_features)) + \
                            attack_features[:max_size % len(attack_features)]
    
    # Combine and shuffle
    balanced_features = normal_features + attack_features
    balanced_labels = [0] * len(normal_features) + [1] * len(attack_features)
    
    # Shuffle
    indices = np.random.permutation(len(balanced_features))
    balanced_features = [balanced_features[i] for i in indices]
    balanced_labels = [balanced_labels[i] for i in indices]
    
    print(f"    Balanced size: {len(balanced_features)}")
    
    return balanced_features, balanced_labels


def train_agent(features, labels, episodes=100, batch_size=50):
    """
    Train Q-Learning agent
    
    Args:
        features: List of feature dicts
        labels: List of labels
        episodes: Number of training episodes
        batch_size: Batch size for each episode
        
    Returns:
        Trained agent
    """
    print(f"\n[+] Training Q-Learning Agent...")
    print(f"    Episodes: {episodes}")
    print(f"    Batch size: {batch_size}")
    
    # Initialize agent
    agent = QLearningAgent(
        state_bins=[15, 15, 10],  # More bins for better granularity
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.3,  # Higher initial exploration
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Update feature ranges
    agent.update_feature_ranges(features)
    print(f"[+] Feature ranges updated:")
    for feat, (min_val, max_val) in agent.feature_ranges.items():
        print(f"    {feat}: [{min_val:.2f}, {max_val:.2f}]")
    
    # Training loop
    episode_rewards = []
    episode_accuracies = []
    
    print("\n[+] Starting training...\n")
    
    for episode in range(episodes):
        # Sample batch
        indices = np.random.choice(len(features), min(batch_size, len(features)), replace=False)
        batch_features = [features[i] for i in indices]
        batch_labels = [labels[i] for i in indices]
        
        # Create experiences
        experiences = list(zip(batch_features, batch_labels))
        
        # Train episode
        reward, accuracy = agent.train_episode(experiences)
        
        episode_rewards.append(reward)
        episode_accuracies.append(accuracy)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_accuracy = np.mean(episode_accuracies[-10:])
            print(f"Episode {episode+1:3d}/{episodes}: "
                  f"Reward={reward:6.2f}, "
                  f"Accuracy={accuracy:.2%}, "
                  f"Avg(10)={avg_accuracy:.2%}, "
                  f"ε={agent.epsilon:.4f}")
    
    print("\n[+] Training completed!")
    
    return agent, episode_rewards, episode_accuracies


def evaluate_agent(agent, features, labels):
    """
    Evaluate agent performance
    
    Args:
        agent: Trained agent
        features: Test features
        labels: True labels
        
    Returns:
        Accuracy, precision, recall, f1_score
    """
    print("\n[+] Evaluating agent...")
    
    predictions = []
    confidences = []
    
    for feat in features:
        action, conf = agent.predict(feat)
        predictions.append(action)
        confidences.append(conf)
    
    # Calculate metrics
    true_positive = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    true_negative = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    false_positive = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    false_negative = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    
    accuracy = (true_positive + true_negative) / len(labels)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:   {accuracy:.2%}")
    print(f"Precision:  {precision:.2%}")
    print(f"Recall:     {recall:.2%}")
    print(f"F1-Score:   {f1_score:.2%}")
    print("-"*60)
    print(f"True Positive:  {true_positive}")
    print(f"True Negative:  {true_negative}")
    print(f"False Positive: {false_positive}")
    print(f"False Negative: {false_negative}")
    print("="*60)
    
    return accuracy, precision, recall, f1_score


def plot_training_history(rewards, accuracies, save_path='training_history.png'):
    """
    Plot training history
    
    Args:
        rewards: List of episode rewards
        accuracies: List of episode accuracies
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot rewards
    ax1.plot(rewards, alpha=0.6, label='Episode Reward')
    ax1.plot(pd.Series(rewards).rolling(10).mean(), linewidth=2, label='Moving Average (10)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Rewards over Episodes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(accuracies, alpha=0.6, label='Episode Accuracy')
    ax2.plot(pd.Series(accuracies).rolling(10).mean(), linewidth=2, label='Moving Average (10)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy over Episodes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[+] Training history plot saved to {save_path}")
    plt.close()


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Q-Learning Agent for DDoS Detection')
    parser.add_argument('--data', type=str, default='data.csv',
                       help='Path to CSV data file')
    parser.add_argument('--episodes', type=int, default=200,
                       help='Number of training episodes')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for each episode')
    parser.add_argument('--balance', type=str, default='undersample',
                       choices=['none', 'undersample', 'oversample'],
                       help='Dataset balancing method')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Test set split ratio')
    parser.add_argument('--model-output', type=str, default='rl_model.pkl',
                       help='Output path for trained model')
    parser.add_argument('--plot-output', type=str, default='training_history.png',
                       help='Output path for training plot')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Q-LEARNING AGENT TRAINING FOR DDOS DETECTION")
    print("="*60)
    
    # Load data
    features, labels = load_and_preprocess_data(args.data)
    
    if features is None or labels is None:
        print("[!] Failed to load data. Exiting.")
        return
    
    # Balance dataset
    if args.balance != 'none':
        features, labels = balance_dataset(features, labels, method=args.balance)
    
    # Split train/test
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=args.test_split, random_state=42, stratify=labels
    )
    
    print(f"\n[+] Dataset split:")
    print(f"    Training: {len(train_features)}")
    print(f"    Testing:  {len(test_features)}")
    
    # Train agent
    agent, rewards, accuracies = train_agent(
        train_features, 
        train_labels, 
        episodes=args.episodes, 
        batch_size=args.batch_size
    )
    
    # Print agent statistics
    agent.print_statistics()
    
    # Evaluate on test set
    evaluate_agent(agent, test_features, test_labels)
    
    # Plot training history
    plot_training_history(rewards, accuracies, save_path=args.plot_output)
    
    # Save model
    agent.save(args.model_output)
    
    print("\n[+] Training pipeline completed successfully!")
    print(f"[+] Model saved to: {args.model_output}")
    print(f"[+] Plot saved to: {args.plot_output}")


if __name__ == '__main__':
    main()
