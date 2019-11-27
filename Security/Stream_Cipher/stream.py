#!/usr/bin/env python3


def strToBinary(strg):
 c = list(strg)
 result = ""
 for i in range(len(c)):
  temp = c[i]
  result += temp + " "
 return result 

def binToStr(bin):
 strg = bin.split(" ")
 for i in range(len(strg)):
  c = list(len(strg))
  z = 0
  for j in range(len(c[i])):
   c[j] = strg[j]
   z += ((c[j] - 48)) << (len(c)-1-j)
  result[i] = (chr) z

print(strToBinary("0110011"))