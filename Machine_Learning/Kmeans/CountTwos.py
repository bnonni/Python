#!/usr/bin/env python3
import matplotlib.pyplot as plt

def num(n):
    s = 0
    while(n > 0):
        if n % 10 == 2:
            s += 1
        n /= 10
        n = int(n)
    return int(s)

def count_twos(yes, s, x, n):
    print('Start!')
    l = []
    while(yes):
        s += num(x)
        l.append((x, s-x))
        x += 1
        if x == n:
            yes = False
        if x % 1000000 == 0:
            print(f'iter: {x}')
    return l

data = count_twos(True, 0, 1, 1000000000)

print('Done counting! Splitting x & y!')
x = []
y = []

for i,p in enumerate(data):
    x.append(data[i][0])
    y.append(data[i][1])
    if i % 10000000 == 0:
        print(f'iter: {i}')

print('Plotting data! Please standby!')
plt.plot(x, y)
plt.show()


