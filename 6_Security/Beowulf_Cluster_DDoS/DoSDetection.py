#!/usr/bin/env python3

import socket
import struct
from datetime import datetime
from time import strptime, mktime

# Packet sniffing on Linux
s = socket.socket(socket.PF_PACKET, socket.SOCK_RAW, 8)

dict = {}

file_txt = open("attack_DDoS.txt", 'a')
t1 = str(datetime.now())

file_txt.write(t1)
file_txt.write("\n")

No_Of_IPs = 15

while True:
    pkt = s.recvfrom(1024)
    ipheader = pkt[0][14:34]
    ip_hdr = struct.unpack("!8sB3s4s4s", ipheader)
    IP = socket.inet_ntoa(ip_hdr[3])
    print("The source of IP: ", IP)

    if (IP not in dict):
        next_visit, visit_count = 0
        start_time = mktime(
            strptime(str(datetime.now()).replace('-', ''), '%Y%m%d %H:%M:%S'))
        dict.update({IP: [visit_count, start_time, next_visit]})
    else:
        next_time = mktime(
            strptime(str(datetime.now()).replace('-', ''), '%Y%m%d %H:%M:%S'))
        dict[IP][2] = next_time
        elapsed_time = dict[IP][2] - next_time
        if(elapsed_time < 3.01):
            dict[IP][0] += 1
            dict[IP][1] = next_time
            print(dict[IP])
        else:
            dict[IP][1] = next_time
            print(dict[IP])

    if(dict[IP][0] > No_Of_IPs):
        line = print(f'DDoS attack is Detected: {IP}')
        file_txt.write(line)
        file_txt.write(IP)
        file_txt.write("\n")
    else:
        dict[IP] = 1
