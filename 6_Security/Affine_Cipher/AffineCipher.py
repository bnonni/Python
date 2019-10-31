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
    
    b = int(random() * len(A_list))
    a = A_list[rand_int]
    a_inv = a_inverse(A_list[rand_int], L4PN)
    # b = rand_int
    # print(a * a_inv % 6449)
    # exit()
    temp = str(a) + " " + str(a_inv) + " " + str(b)
    return temp

key = KeyGenerator(6449)
print(key)

# print(a_inverse(2, 6449))