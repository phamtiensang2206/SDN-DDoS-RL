#!/bin/bash
###############################################################################
# run_experiment.sh - Automated SDN DDoS Detection Experiment
# Script tự động chạy toàn bộ quá trình từ setup đến training
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONTROLLER_LOG="controller.log"
MININET_LOG="mininet.log"
DATA_FILE="data.csv"
MODEL_FILE="rl_model.pkl"

# Functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    print_header "Checking Requirements"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 not found!"
        exit 1
    fi
    print_info "Python3: $(python3 --version)"
    
    # Check Mininet
    if ! command -v mn &> /dev/null; then
        print_error "Mininet not found!"
        print_info "Install with: sudo apt-get install mininet"
        exit 1
    fi
    print_info "Mininet: $(mn --version 2>&1 | head -1)"
    
    # Check Ryu
    if ! command -v ryu-manager &> /dev/null; then
        print_error "Ryu controller not found!"
        print_info "Install with: sudo pip3 install ryu"
        exit 1
    fi
    print_info "Ryu: $(ryu-manager --version 2>&1)"
    
    # Check Python packages
    print_info "Checking Python packages..."
    python3 -c "import numpy, pandas, matplotlib, sklearn" 2>/dev/null || {
        print_error "Missing Python packages!"
        print_info "Install with: pip3 install -r requirements.txt"
        exit 1
    }
    print_info "All Python packages installed ✓"
    
    print_info "All requirements satisfied!"
}

cleanup() {
    print_header "Cleaning Up"
    
    # Kill existing processes
    print_info "Killing existing Ryu processes..."
    sudo pkill -9 ryu-manager 2>/dev/null || true
    
    print_info "Cleaning Mininet..."
    sudo mn -c 2>/dev/null || true
    
    # Remove old logs
    rm -f $CONTROLLER_LOG $MININET_LOG
    
    print_info "Cleanup completed!"
}

setup_environment() {
    print_header "Setting Up Environment"
    
    # Make scripts executable
    chmod +x topo.py controller.py traffic.py train.py label_data.py detect_realtime.py
    
    # Backup old data if exists
    if [ -f "$DATA_FILE" ]; then
        backup_file="${DATA_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
        print_warning "Backing up existing data to $backup_file"
        cp $DATA_FILE $backup_file
    fi
    
    # Remove old data for fresh start
    rm -f $DATA_FILE
    
    print_info "Environment ready!"
}

start_controller() {
    print_header "Starting Ryu Controller"
    
    print_info "Starting controller in background..."
    sudo ryu-manager controller.py --verbose > $CONTROLLER_LOG 2>&1 &
    CONTROLLER_PID=$!
    
    # Wait for controller to start
    print_info "Waiting for controller to initialize..."
    sleep 5
    
    # Check if controller is running
    if ps -p $CONTROLLER_PID > /dev/null; then
        print_info "Controller started successfully (PID: $CONTROLLER_PID)"
        echo $CONTROLLER_PID > controller.pid
    else
        print_error "Failed to start controller!"
        cat $CONTROLLER_LOG
        exit 1
    fi
}

generate_traffic() {
    print_header "Generating Traffic"
    
    local scenario=$1
    local duration=$2
    
    print_info "Scenario: $scenario"
    print_info "Duration: $duration seconds"
    
    # Create Python script for traffic generation
    cat > run_traffic.py << 'EOF'
#!/usr/bin/env python3
import sys
from traffic import run_traffic_scenario

if __name__ == '__main__':
    scenario = sys.argv[1] if len(sys.argv) > 1 else 'normal'
    
    # Import in Mininet context
    from mininet.net import Mininet
    from mininet.node import RemoteController, OVSKernelSwitch
    from mininet.link import TCLink
    from topo import DDosTopology
    from mininet.log import setLogLevel
    
    setLogLevel('info')
    
    print(f"Creating network for scenario: {scenario}")
    
    # Create network
    topo = DDosTopology()
    net = Mininet(
        topo=topo,
        switch=OVSKernelSwitch,
        controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6653),
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True
    )
    
    net.start()
    print("Network started, generating traffic...")
    
    # Run traffic scenario
    run_traffic_scenario(net, scenario)
    
    print("Traffic generation completed!")
    net.stop()
EOF
    
    chmod +x run_traffic.py
    
    # Run traffic generation
    print_info "Generating traffic (this may take a while)..."
    sudo python3 run_traffic.py $scenario > $MININET_LOG 2>&1
    
    print_info "Traffic generation completed!"
    
    # Check data collection
    if [ -f "$DATA_FILE" ]; then
        local record_count=$(wc -l < $DATA_FILE)
        print_info "Collected $record_count records in $DATA_FILE"
    else
        print_warning "No data file created!"
    fi
}

label_data() {
    print_header "Labeling Data"
    
    if [ ! -f "$DATA_FILE" ]; then
        print_error "Data file not found!"
        return 1
    fi
    
    print_info "Auto-labeling traffic data..."
    python3 label_data.py --input $DATA_FILE
    
    print_info "Data labeled successfully!"
}

train_model() {
    print_header "Training Q-Learning Model"
    
    if [ ! -f "$DATA_FILE" ]; then
        print_error "Data file not found!"
        return 1
    fi
    
    print_info "Starting training..."
    python3 train.py \
        --data $DATA_FILE \
        --episodes 200 \
        --batch-size 64 \
        --balance undersample \
        --model-output $MODEL_FILE
    
    if [ -f "$MODEL_FILE" ]; then
        print_info "Model trained and saved to $MODEL_FILE"
    else
        print_error "Failed to train model!"
        return 1
    fi
}

run_full_experiment() {
    print_header "Running Full Experiment"
    
    print_info "This will run a complete experiment:"
    print_info "1. Start controller"
    print_info "2. Generate normal traffic (2 min)"
    print_info "3. Generate attack traffic (3 min)"
    print_info "4. Label data"
    print_info "5. Train model"
    print_info ""
    print_info "Total estimated time: ~15-20 minutes"
    print_info ""
    
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Experiment cancelled"
        exit 0
    fi
    
    # Start controller
    start_controller
    
    # Wait a bit
    sleep 3
    
    # Generate normal traffic
    print_info "\n>>> Phase 1: Generating normal traffic..."
    generate_traffic "normal" 120
    
    sleep 5
    
    # Generate attack traffic
    print_info "\n>>> Phase 2: Generating attack traffic..."
    generate_traffic "mixed" 180
    
    # Stop controller
    if [ -f controller.pid ]; then
        controller_pid=$(cat controller.pid)
        print_info "Stopping controller (PID: $controller_pid)..."
        sudo kill $controller_pid 2>/dev/null || true
        rm controller.pid
    fi
    
    # Label data
    label_data
    
    # Train model
    train_model
    
    print_header "Experiment Completed!"
    print_info "Results:"
    print_info "  - Data: $DATA_FILE"
    print_info "  - Model: $MODEL_FILE"
    print_info "  - Controller log: $CONTROLLER_LOG"
    print_info "  - Mininet log: $MININET_LOG"
}

# Main menu
show_menu() {
    echo ""
    echo "========================================="
    echo "  SDN DDoS Detection - Experiment Menu"
    echo "========================================="
    echo "1. Check requirements"
    echo "2. Cleanup environment"
    echo "3. Run full experiment (automated)"
    echo "4. Start controller only"
    echo "5. Generate traffic (manual)"
    echo "6. Label data"
    echo "7. Train model"
    echo "8. Run real-time detection"
    echo "0. Exit"
    echo "========================================="
    echo -n "Select option: "
}

# Main script
main() {
    clear
    print_header "SDN DDoS Detection with Q-Learning"
    
    while true; do
        show_menu
        read choice
        
        case $choice in
            1)
                check_requirements
                ;;
            2)
                cleanup
                ;;
            3)
                check_requirements
                cleanup
                setup_environment
                run_full_experiment
                ;;
            4)
                cleanup
                start_controller
                print_info "Controller is running. Press Ctrl+C to stop."
                tail -f $CONTROLLER_LOG
                ;;
            5)
                echo ""
                echo "Available scenarios:"
                echo "  - normal: Normal traffic only"
                echo "  - icmp: ICMP flood attack"
                echo "  - syn: SYN flood attack"
                echo "  - udp: UDP flood attack"
                echo "  - mixed: Mixed attacks"
                echo "  - full: Complete test (recommended)"
                echo -n "Enter scenario: "
                read scenario
                generate_traffic $scenario
                ;;
            6)
                label_data
                ;;
            7)
                train_model
                ;;
            8)
                if [ ! -f "$MODEL_FILE" ]; then
                    print_error "Model not found! Train model first (option 7)"
                else
                    print_info "Starting real-time detection..."
                    python3 detect_realtime.py --verbose
                fi
                ;;
            0)
                print_info "Exiting..."
                cleanup
                exit 0
                ;;
            *)
                print_error "Invalid option!"
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Run main
main
