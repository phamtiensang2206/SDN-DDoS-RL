#!/usr/bin/env python3
"""
rl_agent.py - Q-Learning Agent for DDoS Detection
Sử dụng Q-Learning để học cách phát hiện DDoS attack dựa trên flow statistics
"""

import numpy as np
import pickle
import os
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning Agent cho DDoS Detection
    
    State: Discretized traffic features (packet_rate, byte_rate, flow_count)
    Actions: 0 (NORMAL), 1 (BLOCK/ALERT)
    Reward: +10 for correct detection, -10 for false positive/negative
    """
    
    def __init__(self, 
                 state_bins=[10, 10, 10],  # Bins cho mỗi feature
                 learning_rate=0.1,
                 discount_factor=0.95,
                 epsilon=0.1,
                 epsilon_decay=0.995,
                 epsilon_min=0.01):
        """
        Initialize Q-Learning Agent
        
        Args:
            state_bins: Số bins để discretize mỗi state feature
            learning_rate: Alpha - tốc độ học
            discount_factor: Gamma - discount factor cho future rewards
            epsilon: Epsilon cho epsilon-greedy exploration
            epsilon_decay: Decay rate cho epsilon
            epsilon_min: Minimum epsilon value
        """
        self.state_bins = state_bins
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: dictionary mapping (state, action) -> Q-value
        self.q_table = defaultdict(lambda: np.zeros(2))  # 2 actions: NORMAL, BLOCK
        
        # Feature ranges cho normalization (sẽ update khi train)
        self.feature_ranges = {
            'packet_rate': (0, 1000),
            'byte_rate': (0, 100000),
            'flow_count': (0, 50)
        }
        
        # Statistics
        self.total_episodes = 0
        self.total_rewards = 0
        self.detection_accuracy = []
        
    def discretize_state(self, features):
        """
        Chuyển continuous features thành discrete state
        
        Args:
            features: Dict chứa packet_rate, byte_rate, flow_count
            
        Returns:
            Tuple representing discretized state
        """
        packet_rate = features.get('packet_rate', 0)
        byte_rate = features.get('byte_rate', 0)
        flow_count = features.get('flow_count', 0)
        
        # Normalize và discretize
        pr_bin = min(int(packet_rate / (self.feature_ranges['packet_rate'][1] / self.state_bins[0])), 
                     self.state_bins[0] - 1)
        br_bin = min(int(byte_rate / (self.feature_ranges['byte_rate'][1] / self.state_bins[1])), 
                     self.state_bins[1] - 1)
        fc_bin = min(int(flow_count / (self.feature_ranges['flow_count'][1] / self.state_bins[2])), 
                     self.state_bins[2] - 1)
        
        return (pr_bin, br_bin, fc_bin)
    
    def choose_action(self, state, training=True):
        """
        Chọn action dựa trên epsilon-greedy policy
        
        Args:
            state: Current state (tuple)
            training: Nếu True, sử dụng epsilon-greedy; False thì greedy
            
        Returns:
            Action (0 or 1)
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(2)
        else:
            # Exploitation: best action
            q_values = self.q_table[state]
            return np.argmax(q_values)
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value sử dụng Q-learning update rule
        
        Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    def get_reward(self, action, true_label):
        """
        Tính reward dựa trên action và true label
        
        Args:
            action: Action được chọn (0: NORMAL, 1: BLOCK)
            true_label: True label (0: normal, 1: attack)
            
        Returns:
            Reward value
        """
        if action == true_label:
            # Correct prediction
            if true_label == 1:
                return 10  # Correctly detected attack
            else:
                return 5   # Correctly identified normal traffic
        else:
            # Incorrect prediction
            if action == 1 and true_label == 0:
                return -10  # False positive (blocked normal traffic)
            else:
                return -15  # False negative (missed attack) - worse!
    
    def train_episode(self, experiences):
        """
        Train agent trên một episode (batch of experiences)
        
        Args:
            experiences: List of (features, label) tuples
            
        Returns:
            Total reward for episode, accuracy
        """
        episode_reward = 0
        correct_predictions = 0
        
        for i, (features, label) in enumerate(experiences):
            state = self.discretize_state(features)
            action = self.choose_action(state, training=True)
            reward = self.get_reward(action, label)
            
            # Get next state
            if i < len(experiences) - 1:
                next_features, _ = experiences[i + 1]
                next_state = self.discretize_state(next_features)
            else:
                next_state = state  # Terminal state
            
            # Update Q-value
            self.update_q_value(state, action, reward, next_state)
            
            episode_reward += reward
            if action == label:
                correct_predictions += 1
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update statistics
        self.total_episodes += 1
        self.total_rewards += episode_reward
        accuracy = correct_predictions / len(experiences) if experiences else 0
        self.detection_accuracy.append(accuracy)
        
        return episode_reward, accuracy
    
    def predict(self, features):
        """
        Predict action cho given features (không training)
        
        Args:
            features: Dict chứa traffic features
            
        Returns:
            action: 0 (NORMAL) or 1 (ATTACK)
            confidence: Q-value difference
        """
        state = self.discretize_state(features)
        q_values = self.q_table[state]
        action = np.argmax(q_values)
        
        # Confidence based on Q-value difference
        confidence = abs(q_values[1] - q_values[0])
        
        return action, confidence
    
    def update_feature_ranges(self, data):
        """
        Update feature ranges dựa trên training data
        
        Args:
            data: List of feature dictionaries
        """
        packet_rates = [d.get('packet_rate', 0) for d in data]
        byte_rates = [d.get('byte_rate', 0) for d in data]
        flow_counts = [d.get('flow_count', 0) for d in data]
        
        self.feature_ranges['packet_rate'] = (
            min(packet_rates), max(packet_rates) * 1.1  # 10% buffer
        )
        self.feature_ranges['byte_rate'] = (
            min(byte_rates), max(byte_rates) * 1.1
        )
        self.feature_ranges['flow_count'] = (
            min(flow_counts), max(flow_counts) * 1.1
        )
    
    def save(self, filepath='rl_model.pkl'):
        """
        Lưu trained model
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'q_table': dict(self.q_table),  # Convert defaultdict to dict
            'feature_ranges': self.feature_ranges,
            'state_bins': self.state_bins,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'total_episodes': self.total_episodes,
            'total_rewards': self.total_rewards,
            'detection_accuracy': self.detection_accuracy
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"[+] Model saved to {filepath}")
    
    def load(self, filepath='rl_model.pkl'):
        """
        Load trained model
        
        Args:
            filepath: Path to model file
        """
        if not os.path.exists(filepath):
            print(f"[!] Model file not found: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore Q-table as defaultdict
        self.q_table = defaultdict(lambda: np.zeros(2))
        self.q_table.update(model_data['q_table'])
        
        self.feature_ranges = model_data['feature_ranges']
        self.state_bins = model_data['state_bins']
        self.learning_rate = model_data['learning_rate']
        self.discount_factor = model_data['discount_factor']
        self.epsilon = model_data['epsilon']
        self.total_episodes = model_data.get('total_episodes', 0)
        self.total_rewards = model_data.get('total_rewards', 0)
        self.detection_accuracy = model_data.get('detection_accuracy', [])
        
        print(f"[+] Model loaded from {filepath}")
        print(f"    Episodes trained: {self.total_episodes}")
        print(f"    Average accuracy: {np.mean(self.detection_accuracy):.2%}")
        
        return True
    
    def get_statistics(self):
        """
        Get training statistics
        
        Returns:
            Dict of statistics
        """
        return {
            'total_episodes': self.total_episodes,
            'total_rewards': self.total_rewards,
            'avg_reward_per_episode': self.total_rewards / max(1, self.total_episodes),
            'current_epsilon': self.epsilon,
            'avg_accuracy': np.mean(self.detection_accuracy) if self.detection_accuracy else 0,
            'recent_accuracy': np.mean(self.detection_accuracy[-10:]) if len(self.detection_accuracy) >= 10 else 0,
            'q_table_size': len(self.q_table)
        }
    
    def print_statistics(self):
        """Print training statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("Q-LEARNING AGENT STATISTICS")
        print("="*60)
        print(f"Total Episodes:        {stats['total_episodes']}")
        print(f"Total Rewards:         {stats['total_rewards']:.2f}")
        print(f"Avg Reward/Episode:    {stats['avg_reward_per_episode']:.2f}")
        print(f"Current Epsilon:       {stats['current_epsilon']:.4f}")
        print(f"Average Accuracy:      {stats['avg_accuracy']:.2%}")
        print(f"Recent Accuracy (10):  {stats['recent_accuracy']:.2%}")
        print(f"Q-Table Size:          {stats['q_table_size']} states")
        print("="*60 + "\n")


if __name__ == '__main__':
    # Test agent
    print("Testing Q-Learning Agent...")
    
    agent = QLearningAgent()
    
    # Simulate some training data
    print("\n[+] Simulating training...")
    for episode in range(5):
        experiences = []
        for i in range(20):
            # Random features
            features = {
                'packet_rate': np.random.uniform(0, 500),
                'byte_rate': np.random.uniform(0, 50000),
                'flow_count': np.random.randint(0, 30)
            }
            # Random label (biased toward normal)
            label = 1 if np.random.random() > 0.7 else 0
            experiences.append((features, label))
        
        reward, accuracy = agent.train_episode(experiences)
        print(f"Episode {episode+1}: Reward={reward:.2f}, Accuracy={accuracy:.2%}")
    
    # Test prediction
    print("\n[+] Testing prediction...")
    test_features = {
        'packet_rate': 800,  # High rate - potential attack
        'byte_rate': 80000,
        'flow_count': 45
    }
    action, confidence = agent.predict(test_features)
    print(f"Prediction: {'ATTACK' if action == 1 else 'NORMAL'} (confidence: {confidence:.2f})")
    
    # Print statistics
    agent.print_statistics()
    
    # Test save/load
    agent.save('test_model.pkl')
    new_agent = QLearningAgent()
    new_agent.load('test_model.pkl')
