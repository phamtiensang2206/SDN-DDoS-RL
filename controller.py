#!/usr/bin/env python3
"""
controller.py - Ryu SDN Controller với chức năng thu thập flow statistics
Thu thập các metrics: packet count, byte count, duration để phát hiện DDoS
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, ipv4, tcp, udp, icmp
from ryu.lib import hub
import time
import csv
import os
from datetime import datetime


class DDoSController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    
    def __init__(self, *args, **kwargs):
        super(DDoSController, self).__init__(*args, **kwargs)
        
        # MAC learning table
        self.mac_to_port = {}
        
        # Flow statistics storage
        self.datapaths = {}
        self.flow_stats = {}
        
        # Data collection
        self.data_file = 'data.csv'
        self.init_csv()
        
        # Monitor thread
        self.monitor_thread = hub.spawn(self._monitor)
        
        # Traffic counters
        self.packet_count = {}
        self.byte_count = {}
        self.flow_count = {}
        
        self.logger.info("DDoS Detection Controller Started")
    
    def init_csv(self):
        """Khởi tạo file CSV để lưu flow statistics"""
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'switch_id',
                    'src_ip',
                    'dst_ip',
                    'src_port',
                    'dst_port',
                    'protocol',
                    'packet_count',
                    'byte_count',
                    'duration',
                    'packet_rate',
                    'byte_rate',
                    'flow_count',
                    'label'  # 0: normal, 1: attack
                ])
    
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        """Theo dõi khi switch connect/disconnect"""
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.info('Register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
                self.packet_count[datapath.id] = 0
                self.byte_count[datapath.id] = 0
                self.flow_count[datapath.id] = 0
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.info('Unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]
    
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Xử lý khi switch kết nối - cài đặt table-miss flow entry"""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        # Install table-miss flow entry
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                         ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        
        self.logger.info("Switch %s connected", datapath.id)
    
    def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle_timeout=0, hard_timeout=0):
        """Thêm flow entry vào switch"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                   priority=priority, match=match,
                                   instructions=inst,
                                   idle_timeout=idle_timeout,
                                   hard_timeout=hard_timeout)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                   match=match, instructions=inst,
                                   idle_timeout=idle_timeout,
                                   hard_timeout=hard_timeout)
        datapath.send_msg(mod)
    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """Xử lý packet-in message từ switch"""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        
        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        
        self.mac_to_port.setdefault(dpid, {})
        
        # Learn MAC address
        self.mac_to_port[dpid][src] = in_port
        
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD
        
        actions = [parser.OFPActionOutput(out_port)]
        
        # Install flow để tránh packet_in tiếp
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id, idle_timeout=10)
                return
            else:
                self.add_flow(datapath, 1, match, actions, idle_timeout=10)
        
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
        
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                 in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
    
    def _monitor(self):
        """Monitor thread - request flow stats mỗi 5 giây"""
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(5)
    
    def _request_stats(self, datapath):
        """Request flow statistics từ switch"""
        self.logger.debug('Send stats request: %016x', datapath.id)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)
    
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        """Xử lý flow statistics reply"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        body = ev.msg.body
        datapath = ev.msg.datapath
        dpid = datapath.id
        
        total_packets = 0
        total_bytes = 0
        flow_count = 0
        
        for stat in sorted([flow for flow in body if flow.priority == 1],
                          key=lambda flow: (flow.match.get('eth_src', ''),
                                          flow.match.get('eth_dst', ''))):
            
            src_ip = stat.match.get('ipv4_src', 'unknown')
            dst_ip = stat.match.get('ipv4_dst', 'unknown')
            src_port = stat.match.get('tcp_src', stat.match.get('udp_src', 0))
            dst_port = stat.match.get('tcp_dst', stat.match.get('udp_dst', 0))
            
            # Xác định protocol
            protocol = 'unknown'
            if 'tcp_src' in stat.match or 'tcp_dst' in stat.match:
                protocol = 'tcp'
            elif 'udp_src' in stat.match or 'udp_dst' in stat.match:
                protocol = 'udp'
            elif 'icmpv4_type' in stat.match:
                protocol = 'icmp'
            
            packet_count = stat.packet_count
            byte_count = stat.byte_count
            duration = stat.duration_sec + stat.duration_nsec / 1e9
            
            total_packets += packet_count
            total_bytes += byte_count
            flow_count += 1
            
            if duration > 0:
                packet_rate = packet_count / duration
                byte_rate = byte_count / duration
            else:
                packet_rate = 0
                byte_rate = 0
            
            # Lưu vào CSV (label mặc định là 0 - normal, cần update thủ công khi có attack)
            self._save_to_csv(timestamp, dpid, src_ip, dst_ip, src_port, dst_port,
                            protocol, packet_count, byte_count, duration,
                            packet_rate, byte_rate, flow_count, label=0)
        
        # Update counters
        self.packet_count[dpid] = total_packets
        self.byte_count[dpid] = total_bytes
        self.flow_count[dpid] = flow_count
        
        # Log statistics
        self.logger.info('Switch %d: Flows=%d, Packets=%d, Bytes=%d',
                        dpid, flow_count, total_packets, total_bytes)
    
    def _save_to_csv(self, timestamp, switch_id, src_ip, dst_ip, src_port, dst_port,
                     protocol, packet_count, byte_count, duration, packet_rate,
                     byte_rate, flow_count, label):
        """Lưu flow statistics vào CSV file"""
        try:
            with open(self.data_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, switch_id, src_ip, dst_ip, src_port, dst_port,
                    protocol, packet_count, byte_count, duration, packet_rate,
                    byte_rate, flow_count, label
                ])
        except Exception as e:
            self.logger.error(f"Error saving to CSV: {e}")


def main():
    """Main function"""
    from ryu.cmd import manager
    import sys
    
    sys.argv.append('controller.py')
    sys.argv.append('--verbose')
    sys.argv.append('--ofp-tcp-listen-port')
    sys.argv.append('6653')
    
    manager.main()


if __name__ == '__main__':
    main()
