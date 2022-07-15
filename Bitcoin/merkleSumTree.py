from hashlib import sha256

def hash(s):
 return sha256(s.encode())

def digest(h):
 return h.hexdigest()

zero, one, two, three = '0', '1', '2', '3'

leaf0 = hash(zero)
leaf1 = hash(one)
leaf2 = hash(two)
leaf3 = hash(three)

leaf01 = hash( leaf0.hexdigest() + leaf1.hexdigest())
leaf23 = hash( leaf2.hexdigest() + leaf3.hexdigest())

root = hash(leaf01.hexdigest() + leaf23.hexdigest())

print(f'root hash: {root.digest()}')

'''
    (root hash)
       h0123
     ____|____
    |         |
   h01       h23
   _|_       _|_
 h0   h1   h2   h3
 |    |    |    |
 0    1    2    3
'''

class Leaf:
 def __init__(self, sum, hash):
  self.sum = sum
  self.hash = hash

class MerkleSumTree(Leaf):
 def __init__(self, Leaf):
  self.Leaf = Leaf
