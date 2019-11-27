#!/usr/bin/env python3
from AffineCipher import *
from random import *
from sys import *
import os
   
def initCipher(panther_id, message, file_name):
    writeKeys(file_name, panther_id)
    keys = readKeys(file_name)
    for i,k in enumerate(keys):
        print(f'Message: {message}')
        print(f'key #{i}: {k}')
        cipher_code = encryption(k, message)
        print('Encryption Code:', cipher_code)
        plain_code = decryption(k, panther_id, cipher_code)
        print('Decryption Code:', plain_code)
        plain_text = decryptMessage(plain_code)
        print('Decrypted Message:', plain_text)
        print()
    
if __name__ == '__main__':
        message = argv[1]
        initCipher(6449, message, 'keys.txt')