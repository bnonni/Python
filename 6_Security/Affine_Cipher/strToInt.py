#!/usr/bin/env python3
def strToInt(str_):
    result = ""
    c = list(str_)
    for i in range(len(c)):
        temp = c[i]
        result += temp + " "
    return result

print(strToInt('Apple'))
