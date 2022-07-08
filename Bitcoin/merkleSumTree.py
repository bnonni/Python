from hashlib import sha256

one, two, three, four = '1', '2', '3', '4'

def h(s): return sha256(s.encode()).hexdigest()

leaf0 = h(one)
leaf1 = h(two)
leaf2 = h(three)
leaf3 = h(four)

leaf4 = h(leaf0 + leaf1)
leaf5 = h(leaf2 + leaf3)

root = h(leaf4 + leaf5)

print(f'root hash {root}')