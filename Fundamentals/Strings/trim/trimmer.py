#!/usr/bin/env python3
def trimmer(string):
    trimmed = ''
    arr = string.split()
    for j in range(len(arr)):
        trimmed += arr[j] + ' '
    return trimmed, arr


if __name__ == '__main__':
    to_be_trimmed = '   Hello     World   '
    trimmed, arr = trimmer(to_be_trimmed)
    print(trimmed)
    print(arr)
