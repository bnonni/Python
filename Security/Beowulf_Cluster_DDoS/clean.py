#!/usr/bin/env python3
from sys import *
f = argv
for i, a in enumerate(f):
    if a == './clean.py':
        pass
    else:
        print(f'File {i}: {a}')
        with open(a, 'rb+') as f:
            content = f.read()
            f.seek(0)
            f.write(content.replace(b'\r', b''))
            f.truncate()
