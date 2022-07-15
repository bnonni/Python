from hashlib import sha256

def hash(s):
 return sha256(s.encode())

def digest(h):
 return h.hexdigest()

block1 = "Block 1"
block2 = "Block 2"

hash1 = hash(block1)
hash2 = hash(block2)

root = hash(hash1 + hash2)

assert root == 'd1c6d4f28135f428927a1248d71984a937ee543e'