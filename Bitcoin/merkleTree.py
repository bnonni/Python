from hashlib import sha1

def h(s): return sha1(s.encode()).hexdigest() # hashing helper function

block1 = "Block 1"
block2 = "Block 2"

digest1 = h(block1)
digest2 = h(block2)

root = h(digest1 + digest2)

assert root == 'd1c6d4f28135f428927a1248d71984a937ee543e'