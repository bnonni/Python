import secrets
from mnemonic import Mnemonic

passphrase = ''
counter = 5
m = Mnemonic('english')
wordlist = m.wordlist
n = len(m.wordlist)

while counter > 0:
 r = secrets.randbelow(n)
 print(r)
 passphrase += f'{wordlist.pop(r)}-'
 counter -= 1

print(passphrase.rstrip('-'))
