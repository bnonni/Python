#!/usr/bin/env python

def Encryption(key, a, b, a_inverse, LP4N, content):
    ciphertext = ""
    ASCcode = strToInt(content)
    Int_ASC = ASCcode.split("")
    i = 0
    while i < Int_ASC.length:
        temp = int(Int_ASC[i]) * a + b
        ciphertext = ciphertext + temp + ""
        i += 1
    return ciphertext