#!/usr/bin/env python
def strToInt(str_):
    result = ""
    c = str_.toCharArray()
    i = 0
    while i < c.length:
        temp = c[i]
        result += temp + " "
        i += 1
    return result

