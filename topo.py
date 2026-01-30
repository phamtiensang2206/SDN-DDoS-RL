#!/usr/bin/env python3
"""
topo.py - Định nghĩa topology mạng SDN
Tạo mạng SDN với multiple switches và hosts để mô phỏng môi trường thực tế
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink

class DDosTopology(Topo):
    """
    Topology mạng SDN để test DDoS detection
    Bao gồm: 1 controller, 3 switches, 9 hosts
    """
    
    def build(self):
        """
        Xây dựng topology:
        - 3 switches kết nối với nhau tạo thành backbone
        - Mỗi switch kết nối với 3 hosts
        - Bandwidth: 10Mbps, delay: 5ms
        """
        
        # Tạo switches
        info('*** Adding switches\n')
        s1 = self.addSwitch('s1', dpid='0000000000000001')
        s2 = self.addSwitch('s2', dpid='0000000000000002')
        s3 = self.addSwitch('s3', dpid='0000000000000003')
        
        # Tạo hosts
        info('*** Adding hosts\n')
        # Hosts cho switch 1 (normal users)
        h1 = self.addHost('h1', ip='10.0.0.1/24', mac='00:00:00:00:00:01')
        h2 = self.addHost('h2', ip='10.0.0.2/24', mac='00:00:00:00:00:02')
        h3 = self.addHost('h3', ip='10.0.0.3/24', mac='00:00:00:00:00:03')
        
        # Hosts cho switch 2 (normal users + potential attackers)
        h4 = self.addHost('h4', ip='10.0.0.4/24', mac='00:00:00:00:00:04')
        h5 = self.addHost('h5', ip='10.0.0.5/24', mac='00:00:00:00:00:05')
        h6 = self.addHost('h6', ip='10.0.0.6/24', mac='00:00:00:00:00:06')
        
        # Hosts cho switch 3 (servers + attackers)
        h7 = self.addHost('h7', ip='10.0.0.7/24', mac='00:00:00:00:00:07')
        h8 = self.addHost('h8', ip='10.0.0.8/24', mac='00:00:00:00:00:08')
        h9 = self.addHost('h9', ip='10.0.0.9/24', mac='00:00:00:00:00:09')  # Server target
        
        # Kết nối switches với nhau (backbone)
        info('*** Creating switch-to-switch links\n')
        self.addLink(s1, s2, cls=TCLink, bw=10, delay='5ms', loss=0)
        self.addLink(s2, s3, cls=TCLink, bw=10, delay='5ms', loss=0)
        self.addLink(s3, s1, cls=TCLink, bw=10, delay='5ms', loss=0)
        
        # Kết nối hosts với switches
        info('*** Creating host-to-switch links\n')
        # Switch 1
        self.addLink(h1, s1, cls=TCLink, bw=10, delay='2ms')
        self.addLink(h2, s1, cls=TCLink, bw=10, delay='2ms')
        self.addLink(h3, s1, cls=TCLink, bw=10, delay='2ms')
        
        # Switch 2
        self.addLink(h4, s2, cls=TCLink, bw=10, delay='2ms')
        self.addLink(h5, s2, cls=TCLink, bw=10, delay='2ms')
        self.addLink(h6, s2, cls=TCLink, bw=10, delay='2ms')
        
        # Switch 3
        self.addLink(h7, s3, cls=TCLink, bw=10, delay='2ms')
        self.addLink(h8, s3, cls=TCLink, bw=10, delay='2ms')
        self.addLink(h9, s3, cls=TCLink, bw=10, delay='2ms')


def create_network():
    """
    Tạo và khởi động mạng SDN
    """
    info('*** Creating network\n')
    topo = DDosTopology()
    
    # Tạo network với Remote Controller (Ryu)
    net = Mininet(
        topo=topo,
        switch=OVSKernelSwitch,
        controller=lambda name: RemoteController(
            name, 
            ip='127.0.0.1', 
            port=6653
        ),
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True
    )
    
    return net


def start_network():
    """
    Khởi động mạng và CLI
    """
    setLogLevel('info')
    
    net = create_network()
    
    info('*** Starting network\n')
    net.start()
    
    info('*** Testing connectivity\n')
    net.pingAll()
    
    info('*** Network is ready\n')
    info('*** h9 (10.0.0.9) is the target server\n')
    info('*** h1-h6 are normal users, h7-h8 can be attackers\n')
    
    # Mở CLI để tương tác
    CLI(net)
    
    info('*** Stopping network\n')
    net.stop()


if __name__ == '__main__':
    start_network()
