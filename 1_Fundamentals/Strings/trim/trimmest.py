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

    stk_len = len(stack)
    a = stk_len - 1

    for i in range(stk_len):
        if(stack[a] == ' '):
            stack.pop(a)
            a -= 1

    for i in stack:
        trimmed += i

    return trimmed


if __name__ == '__main__':
    trimmed = trimmest('   Hello     World   ')
    print(trimmed)
