# Phát Hiện Tấn Công DDoS trong Mạng SDN Dựa Trên Học Tăng Cường (Q-Learning)

## 📋 Mô tả đề tài

Hệ thống phát hiện tấn công DDoS (Distributed Denial of Service) trong mạng SDN (Software-Defined Networking) sử dụng thuật toán Reinforcement Learning - cụ thể là Q-Learning.

### Các thành phần chính:
- **Mạng SDN**: Mô phỏng bằng Mininet với topology 3 switches, 9 hosts
- **SDN Controller**: Ryu controller thu thập flow statistics
- **Traffic Generator**: Sinh traffic bình thường và các loại tấn công DDoS
- **Q-Learning Agent**: Học phát hiện DDoS dựa trên flow features

### Kiến trúc hệ thống:

```
┌─────────────────────────────────────────────────────────┐
│                    Mininet Network                       │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐         │
│  │   SW1   │──────│   SW2   │──────│   SW3   │         │
│  └────┬────┘      └────┬────┘      └────┬────┘         │
│       │                │                │               │
│   h1 h2 h3         h4 h5 h6         h7 h8 h9(target)   │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                 Ryu SDN Controller                       │
│            (Thu thập Flow Statistics)                    │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Data Collection                        │
│        (packet_rate, byte_rate, flow_count)             │
│                   → data.csv                             │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Q-Learning Agent Training                   │
│   State: (discretized traffic features)                 │
│   Actions: [NORMAL, BLOCK]                              │
│   Reward: +10 correct, -10 false positive/negative      │
└─────────────────────────────────────────────────────────┘
```

## 🔧 Cài đặt

### Yêu cầu hệ thống:
- **OS**: Ubuntu 20.04/22.04 hoặc Debian-based Linux
- **Python**: 3.7+
- **RAM**: Tối thiểu 4GB
- **Disk**: 10GB trống

### Bước 1: Cài đặt dependencies cơ bản

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Cài đặt Python và pip
sudo apt-get install -y python3 python3-pip python3-dev

# Cài đặt Git
sudo apt-get install -y git

# Cài đặt các tools cần thiết
sudo apt-get install -y build-essential
```

### Bước 2: Cài đặt Mininet

```bash
# Cài đặt Mininet
sudo apt-get install -y mininet

# Hoặc cài từ source (khuyến nghị)
cd ~
git clone https://github.com/mininet/mininet
cd mininet
git checkout 2.3.0
sudo PYTHON=python3 ./util/install.sh -a

# Test Mininet
sudo mn --version
sudo mn --test pingall
```

### Bước 3: Cài đặt Ryu Controller

```bash
# Cài đặt Ryu SDN Framework
sudo pip3 install ryu

# Verify installation
ryu-manager --version
```

### Bước 4: Cài đặt Open vSwitch

```bash
# Cài đặt OpenvSwitch
sudo apt-get install -y openvswitch-switch

# Start OVS service
sudo systemctl start openvswitch-switch
sudo systemctl enable openvswitch-switch

# Verify
sudo ovs-vsctl show
```

### Bước 5: Cài đặt Python packages

```bash
# Cài đặt các thư viện Python
sudo pip3 install numpy pandas matplotlib scikit-learn

# Hoặc dùng requirements (nếu có)
# sudo pip3 install -r requirements.txt
```

### Bước 6: Cài đặt traffic tools (optional)

```bash
# hping3 cho SYN/UDP flood
sudo apt-get install -y hping3

# iperf cho bandwidth testing
sudo apt-get install -y iperf

# Apache bench cho HTTP flood
sudo apt-get install -y apache2-utils
```

### Bước 7: Clone project

```bash
# Tạo thư mục project
mkdir -p ~/SDN-DDoS-RL
cd ~/SDN-DDoS-RL

# Copy các file đã tạo vào đây
# (topo.py, controller.py, traffic.py, rl_agent.py, train.py)
```

## 🚀 Hướng dẫn sử dụng

### Workflow tổng quát:

```
1. Khởi động Ryu Controller → Thu thập flow stats
2. Khởi động Mininet Network → Tạo topology
3. Generate Traffic → Normal + Attack traffic
4. Thu thập Data → Lưu vào data.csv
5. Train Q-Learning Agent → Học phát hiện attack
6. Evaluate Model → Kiểm tra performance
```

### Bước 1: Khởi động Ryu Controller

Mở **Terminal 1**:

```bash
cd ~/SDN-DDoS-RL

# Khởi động Ryu controller
sudo ryu-manager controller.py --verbose

# Hoặc chạy trực tiếp
sudo python3 controller.py
```

Bạn sẽ thấy output:
```
loading app controller.py
instantiating app controller.py of DDoSController
DDoS Detection Controller Started
```

### Bước 2: Khởi động Mininet Network

Mở **Terminal 2**:

```bash
cd ~/SDN-DDoS-RL

# Khởi động mạng SDN
sudo python3 topo.py
```

Sau khi mạng khởi động thành công, bạn sẽ vào Mininet CLI:
```
*** Creating network
*** Adding switches
*** Adding hosts
*** Creating switch-to-switch links
*** Creating host-to-switch links
*** Starting network
*** Testing connectivity
*** Network is ready
*** h9 (10.0.0.9) is the target server
*** h1-h6 are normal users, h7-h8 can be attackers

mininet>
```

### Bước 3: Generate Traffic và Thu thập Data

Trong Mininet CLI (Terminal 2), có nhiều cách để generate traffic:

#### Cách 1: Sử dụng script traffic.py trong Mininet

```python
# Trong mininet CLI
mininet> py exec(open('traffic.py').read())
mininet> py run_traffic_scenario(net, 'normal')
```

#### Cách 2: Chạy từng scenario riêng lẻ

```python
# Normal traffic (120 giây)
mininet> py exec(open('traffic.py').read())
mininet> py generator = TrafficGenerator(net)
mininet> py generator.generate_normal_traffic(duration=120, interval=1)
```

```python
# ICMP Flood attack (60 giây)
mininet> py generator.generate_icmp_flood(duration=60)
```

```python
# SYN Flood attack (60 giây)
mininet> py generator.generate_syn_flood(duration=60)
```

```python
# UDP Flood attack (60 giây)
mininet> py generator.generate_udp_flood(duration=60)
```

```python
# Mixed attack (tất cả các loại)
mininet> py generator.generate_mixed_attack(duration=90)
```

#### Cách 3: Full test scenario (Khuyến nghị)

Chạy full test để thu thập đủ data cho training:

```python
mininet> py exec(open('traffic.py').read())
mininet> py run_traffic_scenario(net, 'full')
```

Scenario 'full' sẽ chạy:
- 120s normal traffic
- 60s ICMP flood
- 60s normal traffic
- 60s SYN flood
- 60s normal traffic
- 60s UDP flood
- 60s normal traffic

**Tổng thời gian: ~8-10 phút**

#### Cách 4: Manual traffic từ CLI

```bash
# Ping từ h1 đến h9
mininet> h1 ping -c 10 h9

# HTTP request
mininet> h2 wget -O /dev/null http://10.0.0.9/

# ICMP flood manual
mininet> h7 ping -f -s 1024 10.0.0.9 &

# Kill attack
mininet> h7 pkill ping
```

### Bước 4: Kiểm tra Data đã thu thập

Mở **Terminal 3**:

```bash
cd ~/SDN-DDoS-RL

# Xem số dòng data
wc -l data.csv

# Xem 10 dòng đầu
head -n 10 data.csv

# Xem realtime (mỗi 5s)
watch -n 5 'tail -n 20 data.csv'
```

### Bước 5: Label dữ liệu

**QUAN TRỌNG**: File `data.csv` mặc định có `label=0` (normal). Bạn cần update label thành `1` cho các flow attack.

#### Cách 1: Manual edit (nhỏ)

```bash
nano data.csv
# Tìm các dòng có packet_rate hoặc byte_rate cao bất thường
# Thay đổi label từ 0 → 1
```

#### Cách 2: Python script tự động

Tạo file `label_data.py`:

```python
import pandas as pd

df = pd.read_csv('data.csv')

# Threshold cho attack (điều chỉnh dựa trên data của bạn)
PACKET_RATE_THRESHOLD = 500
BYTE_RATE_THRESHOLD = 50000
FLOW_COUNT_THRESHOLD = 30

# Label attack nếu vượt threshold
df.loc[
    (df['packet_rate'] > PACKET_RATE_THRESHOLD) |
    (df['byte_rate'] > BYTE_RATE_THRESHOLD) |
    (df['flow_count'] > FLOW_COUNT_THRESHOLD),
    'label'
] = 1

# Save
df.to_csv('data.csv', index=False)
print(f"Total records: {len(df)}")
print(f"Normal (0): {len(df[df['label']==0])}")
print(f"Attack (1): {len(df[df['label']==1])}")
```

Chạy:
```bash
python3 label_data.py
```

### Bước 6: Train Q-Learning Agent

```bash
cd ~/SDN-DDoS-RL

# Train với default parameters
python3 train.py

# Train với custom parameters
python3 train.py --episodes 300 --batch-size 128 --balance oversample

# Xem tất cả options
python3 train.py --help
```

**Output mẫu:**
```
============================================================
Q-LEARNING AGENT TRAINING FOR DDOS DETECTION
============================================================
[+] Loading data from data.csv...
[+] Loaded 2500 records
[+] After preprocessing: 2450 records
[+] Label distribution:
    Normal (0): 1800
    Attack (1): 650

[+] Training Q-Learning Agent...
    Episodes: 200
    Batch size: 64

Episode  10/200: Reward= 45.00, Accuracy=75.00%, Avg(10)=72.50%, ε=0.2850
Episode  20/200: Reward= 62.00, Accuracy=82.14%, Avg(10)=78.93%, ε=0.2707
...
Episode 200/200: Reward= 85.00, Accuracy=92.86%, Avg(10)=91.25%, ε=0.0100

[+] Training completed!

============================================================
EVALUATION RESULTS
============================================================
Accuracy:   91.84%
Precision:  89.23%
Recall:     94.12%
F1-Score:   91.61%
============================================================

[+] Model saved to: rl_model.pkl
[+] Plot saved to: training_history.png
```

### Bước 7: Test Model Real-time

Tạo file `detect_realtime.py`:

```python
#!/usr/bin/env python3
"""Real-time DDoS detection"""

import time
import pandas as pd
from rl_agent import QLearningAgent

# Load trained model
agent = QLearningAgent()
if not agent.load('rl_model.pkl'):
    print("Failed to load model!")
    exit(1)

print("[+] Monitoring traffic for DDoS attacks...")
print("[+] Press Ctrl+C to stop")

last_size = 0

try:
    while True:
        # Đọc data.csv
        df = pd.read_csv('data.csv')
        
        if len(df) > last_size:
            # Process new records
            new_records = df.iloc[last_size:]
            
            for _, row in new_records.iterrows():
                features = {
                    'packet_rate': row['packet_rate'],
                    'byte_rate': row['byte_rate'],
                    'flow_count': row['flow_count']
                }
                
                action, confidence = agent.predict(features)
                
                if action == 1:  # Attack detected
                    print(f"\n⚠️  [ALERT] DDoS Attack Detected!")
                    print(f"    Source: {row['src_ip']} → {row['dst_ip']}")
                    print(f"    Packet Rate: {row['packet_rate']:.2f}")
                    print(f"    Byte Rate: {row['byte_rate']:.2f}")
                    print(f"    Confidence: {confidence:.2f}")
            
            last_size = len(df)
        
        time.sleep(5)  # Check every 5 seconds

except KeyboardInterrupt:
    print("\n[+] Monitoring stopped")
```

Chạy:
```bash
python3 detect_realtime.py
```

## 📊 Phân tích kết quả

### 1. Xem Training History

```bash
# Xem plot
xdg-open training_history.png

# Hoặc dùng image viewer
eog training_history.png
```

### 2. Analyze Data Distribution

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

# Plot distribution
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, col in enumerate(['packet_rate', 'byte_rate', 'flow_count']):
    ax = axes[idx // 2, idx % 2]
    
    normal = df[df['label'] == 0][col]
    attack = df[df['label'] == 1][col]
    
    ax.hist(normal, bins=30, alpha=0.5, label='Normal')
    ax.hist(attack, bins=30, alpha=0.5, label='Attack')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax.set_title(f'{col} Distribution')
    ax.legend()

plt.tight_layout()
plt.savefig('data_distribution.png')
print("Saved to data_distribution.png")
```

### 3. Confusion Matrix

Thêm vào `train.py` hoặc tạo script riêng:

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ... sau khi evaluate ...

cm = confusion_matrix(test_labels, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
```

## 🔍 Troubleshooting

### Vấn đề 1: Controller không kết nối được

```bash
# Kiểm tra Ryu đang chạy
ps aux | grep ryu

# Kiểm tra port 6653
sudo netstat -tulpn | grep 6653

# Kill process cũ
sudo pkill ryu-manager

# Restart controller
sudo ryu-manager controller.py
```

### Vấn đề 2: Mininet không tạo được network

```bash
# Clean up Mininet
sudo mn -c

# Remove old OVS bridges
sudo ovs-vsctl list-br | xargs -r -L1 sudo ovs-vsctl del-br

# Restart
sudo python3 topo.py
```

### Vấn đề 3: Không có data trong CSV

```bash
# Kiểm tra controller có đang ghi file không
ls -lh data.csv

# Check permissions
chmod 666 data.csv

# Monitor controller output
# Trong terminal chạy controller, xem có log "Send stats request" không
```

### Vấn đề 4: hping3 không hoạt động

```bash
# Install hping3
sudo apt-get install -y hping3

# Nếu vẫn lỗi, dùng ping flood thay thế
# Trong traffic.py, thay:
# cmd = f'hping3 -S -p 80 --flood {target}'
# bằng:
# cmd = f'ping -f -s 1024 {target}'
```

### Vấn đề 5: Training bị overfitting

- Tăng epsilon để explore nhiều hơn
- Giảm learning rate
- Balance dataset tốt hơn
- Thu thập thêm data

## 📈 Tips để cải thiện hiệu suất

### 1. Thu thập data tốt hơn

```python
# Tăng thời gian collect normal traffic
generator.generate_normal_traffic(duration=300, interval=0.5)

# Vary attack intensities
for duration in [30, 60, 90]:
    generator.generate_icmp_flood(duration=duration)
    time.sleep(30)  # Cool down
```

### 2. Feature engineering

Thêm features mới vào `controller.py`:

```python
# Protocol distribution
# Average packet size
# Flow duration variance
# Inter-arrival time
```

### 3. Hyperparameter tuning

```python
# Trong train.py, thử các giá trị khác nhau:
agent = QLearningAgent(
    state_bins=[20, 20, 15],  # Tăng bins
    learning_rate=0.05,         # Giảm learning rate
    epsilon=0.5,                # Tăng exploration
    epsilon_decay=0.99
)
```

### 4. Ensemble methods

Train nhiều models với random seeds khác nhau và vote:

```python
models = []
for seed in range(5):
    np.random.seed(seed)
    agent = train_agent(features, labels)
    models.append(agent)

# Voting
def predict_ensemble(features):
    votes = [model.predict(features)[0] for model in models]
    return max(set(votes), key=votes.count)
```

## 📝 Kết quả mong đợi

Với dataset tốt (ít nhất 1000+ samples, balanced), model nên đạt:
- **Accuracy**: 85-95%
- **Precision**: 80-90%
- **Recall**: 85-95%
- **F1-Score**: 83-92%

## 🎓 Tài liệu tham khảo

### Papers:
- "DDoS Attack Detection in SDN Using Machine Learning Techniques"
- "Q-Learning Based Network Intrusion Detection"
- "Reinforcement Learning for Network Security"

### Websites:
- Mininet Documentation: http://mininet.org/
- Ryu SDN Framework: https://ryu-sdn.org/
- OpenFlow Specification: https://www.opennetworking.org/

## 📧 Liên hệ & Hỗ trợ

Nếu gặp vấn đề, check:
1. Log của Ryu controller
2. Data.csv có được tạo không
3. Traffic có được generate không (dùng `tcpdump`)

---

**Chúc bạn thành công với đề tài nghiên cứu! 🎉**
