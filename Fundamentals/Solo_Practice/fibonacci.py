#!/usr/bin/env python3
from sys import *
from random import *


def fibonacci(n):
    if n <= 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)


n = randint(1, 50) if len(argv) <= 1 else int(argv[1])
nth_fib = fibonacci(n)
print(nth_fib)
