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
        return True;
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
    k = str(a) + " " + str(a_inv) + " " + str(b)
    return k

def strToInt(strng):
    c = list(strng); 
    result = ""
    for i in range(len(c)):
        temp = c[i]
        result += temp + " "
        return result

<<<<<<< Updated upstream

def encryption(key, L4PN, content):
    a = key[0]
    b = key[2]
    ciphertext = ""
    content_str = strToInt(content)
    content_lst = list(content.split(" "))
    for i in range(len(content_lst)):
        temp = int(content_lst[i]) * a + b
        ciphertext = ciphertext + temp + ""
    return ciphertext
    
panther_id = 6449
key = KeyGenerator(panther_id)
encryption(key, panther_id, "Apple")
=======
def Encryption(key, a, b, a_inverse, LP4N, content):
    ciphertext = ""
    ASCcode = strToInt(content)
    Int_ASC = ACScode.split("")
    i = 0
    while i < Int_ACS.length:
        ciphertext = ciphertext + temp + ""
        i += 1
    return ciphertext
>>>>>>> Stashed changes
