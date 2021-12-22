from hashlib import sha256

def computeSHA256hash(input):
    sha = sha256()
    return sha.update(input.encode('utf-8')).hexdigest()

print(computeSHA256hash('Hello World'))