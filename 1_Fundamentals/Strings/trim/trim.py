#!/usr/bin/env python3
def trimmest(string):
    stack = []
    trimmed = ''
    for i, c in enumerate(string):
        if(c != ' '):
            first = i
            break

    for i, c in enumerate(string[first:]):
        stack.append(c)

    j = len(stack) - 1
    for i in range(j):
        if(stack[j] == ' '):
            stack.pop(j)
            j -= 1

    for i in stack:
        trimmed += i

    return trimmed, stack


if __name__ == '__main__':
    trimmed, stack = trimmest('   Hello     World   ')
    print(trimmed)
    print(stack)
