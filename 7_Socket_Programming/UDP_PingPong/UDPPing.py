#!/usr/bin/env python3
from random import *
from time import *
from socket import *
from sys import *

HOST = '127.0.0.1'
PORT = 80
BUFFER = 1024
ADDR = (HOST, PORT)
REQ = True

try:
  client = socket(AF_INET, SOCK_DGRAM)
  print('UDP Client Socket Creation Success.')
except error as e:
  print(f'UDP Client Socket Creation Fail.\nError Code: {str(e[0])} Error Message: {str(e[1])}')
  exit(1)

while REQ:
  ping = b'ping world!'
  start = time()
  client.sendto(ping, ADDR)
  try:
    pong, svr = client.recvfrom(BUFFER)
    end = time()
    RTT = end - start
    if RTT < 1:
      print(f'UDP Client Ping Request: {ping}\nUDP Server Pong Response: {pong}\nRTT: {round(RTT, 5)}s')
      REQ = False
    elif RTT > 1:
      print(f'Error: RTT {round(RTT, 5)} > 1s. UDP Server Response Lost.')
  except timeout:
    print(f'REQUEST TIMED OUT.')