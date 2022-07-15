from hashlib import sha256

def hash(s):
 return sha256(s.encode())

def digest(h):
 return h.hexdigest()

hello_world_hash = hash('Hello World')
print(f'Binary hash of "Hello World" {hello_world_hash}')

print(f'Hex hash of "Hello World" {digest(hello_world_hash)}')