from hashlib import sha256

def hash(s):
 return sha256(s.encode())

def digest_bin(h):
 return h.digest()

def hex_digest(h):
 return h.hexdigest()

value = "₿ryan Nonni"
hash_value = hash(value)
binary_digest = digest_bin(hash_value)
digest_hex = hex_digest(hash_value)

print(f'Binary digest of hash({value}): {binary_digest}')

print(f'Hex digest of hash({value}): {digest_hex}')

print(f'First {len(value)} chars of hex digest of hash({value}): {digest_hex[:11]}')