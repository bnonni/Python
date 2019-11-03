#!/usr/bin/env python3
from random import *
from sys import *

def a_inverse(a, modulus):
    # print(type(modulus))
    for i in range(modulus):
        if (((i * a) % modulus) == 1):
            return i
    return 0
        
def GCD(inp1, inp2):
    while(inp1 != 0 and inp2 != 0):
        if(inp1 >= inp2):
            inp1 = inp1 - inp2
        else:
            inp2 = inp2 - inp1
    if(inp1 == 1 and inp2 == 0):
        return True
    elif(inp2 == 1 and inp1 == 0):
        return True
    else: 
        return False
    
def KeyGenerator(L4PN):
    a = 0
    b = 0
    a_inv = 0
    A_list = []
    for i in range(L4PN):
        if(GCD(i, L4PN)):
            A_list.append(i)
    rand_index = int(random() * len(A_list))
    a = A_list[rand_index]
    a_inv = a_inverse(A_list[rand_index], L4PN)
    b = rand_index
    k = str(a) + ' ' + str(a_inv) + ' ' + str(b)
    return k

def encryption(key, plain_text):
    # print(plain_text)
    cipher_text = ''
    key = key.split(' ')
    a = int(key[0])
    b = int(key[2])
    plain_asci = list(plain_text)
    # print(plain_asci)
    for i in range(len(plain_asci)):
        tmp = str(ord(plain_asci[i]) * a + b)
        cipher_text += tmp + ' '
    return cipher_text

def decryption(key, pid, cipher_text):
    cipher_text = cipher_text.split(' ')
    plain_text = ''
    key = key.split(' ')
    a = int(key[0])
    a_inverse = int(key[1])
    b = int(key[2])
    for i in range(len(cipher_text)):
        if cipher_text[i] == '':
            pass
        else:
            cipher_text[i] = int(cipher_text[i])
    del cipher_text[-1]
    for i in range(len(cipher_text)):
        temp = ((cipher_text[i] - b) * a_inverse) % pid
        # print('temp', temp, '\n')
        if temp < 0:
            temp = temp + pid
            # print('temp', temp)
        plain_text += (plain_text + str(temp) + ' ')
        # print('plain_text', plain_text)
    return plain_text

def writeKeys(key_file, pid):
    wr = open('keys.txt', 'w')
    for line in range(10):
        wr.write(KeyGenerator(pid))
        wr.write('\n')
    
def readKeys(key_file):
    rd = open(key_file, 'r')
    keys = [] # list to save lines
    while True:
        key = rd.readline()
        if not key:
            break
        keys.append(key.strip())
    return keys
    
def initCipher(message):
    panther_id = 6449
    writeKeys('keys.txt', panther_id)
    keys = readKeys('keys.txt')
    for key in keys:
        # print('key:', key)
        cipher_text = encryption(key, message)
        print('cipher_text:', cipher_text)
    for key in keys:
        plain_text = decryption(key, panther_id, cipher_text)
        print('plain_text:', plain_text)
    
if __name__ == '__main__':
    initCipher('Apple')