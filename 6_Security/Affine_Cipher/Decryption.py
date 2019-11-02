#!/usr/bin/env python

def decryption(a, a_inverse, b, L4PN, ciphertext):
    plaintext = ""
    int_ciph = ciphertext.split("")
    i = 0
    while i < len(int_ciph):
        temp = ((int(int_ciph[i]) - b) * a_inverse) % L4PN
        if temp < 0:
            temp = temp + L4PN
        plaintext = plaintext + temp + ""
        i += 1
    return plaintext

