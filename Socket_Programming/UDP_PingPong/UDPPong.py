#!/usr/bin/env python3

from random import *
from time import *
from socket import *
from sys import *

HOST = '127.0.0.1'
PORT = 80
BUFFER = 1024
RES = True

try:
    server = socket(AF_INET, SOCK_DGRAM)
    print('UDP Server Socket Creation Success.')
except error as e:
    print(f'UDP Server Socket Creation Fail.\nError Code: {str(e[0])} Error Message: {str(e[1])}')
    exit(1)


try:
    server.bind((HOST, PORT))
    print(f'UDP Server Socket Bind to {HOST}:{PORT} Success.')
except error as e:
    print(f'UDP Server Socket Bind to {HOST}:{PORT} Fail.\nError Code: {str(e[0])} Error Message: {str(e[1])}')
    exit(1)

while RES:
    start = time()
    ping, addr = server.recvfrom(BUFFER)
    pong = ping.upper()
    server.sendto(pong, addr)
    end = time()
    RTT = end - start
    if RTT > 1:
        start = time()
    print(f'UDP Server Pong Response: {pong}\nUDP Client Ping Request: {ping}\nRTT: {round(RTT, 5)}s')
    