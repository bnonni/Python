from hashlib import sha256

UTXO = dict(
    txid="0529558696101c85fd4de3deaf654c5aa7ffbacf795db85a137ec9dec2502d17",
    vout=0,
    address="bcrt1qaja9yk4vxmvjferq6hwln6n4fum6hy99juq8rr",
    label="",
    scriptPubKey="0014ecba525aac36d924e460d5ddf9ea754f37ab90a5",
    amount=50,
    confirmations=110,
    spendable=True,
    solvable=True,
    desc="wpkh([1abf2856/0'/0'/1']03bbb27d64230fd4e15d2ba3e917ef08a9a77e5e0906e2c12bf177a6c71c745a89)#6h39zl8m",
    safe=True,
)

def int_to_little_endian(n, length):
    '''endian_to_little_endian takes an integer and returns the little-endian
    byte sequence of length'''
    return n.to_bytes(length, 'little')

def encode_varint(i):
    '''encodes an integer as a varint'''
    if i < 0xfd:
        return bytes([i])
    elif i < 0x10000:
        return b'\xfd' + int_to_little_endian(i, 2)
    elif i < 0x100000000:
        return b'\xfe' + int_to_little_endian(i, 4)
    elif i < 0x10000000000000000:
        return b'\xff' + int_to_little_endian(i, 8)
    else:
        raise ValueError('integer too large: {}'.format(i))

def serialize(self):
        '''Returns the byte serialization of the transaction'''
        result = int_to_little_endian(self.version, 4)
        result += encode_varint(len(self.tx_ins))
        for tx_in in self.tx_ins:
            result += tx_in.serialize()
        result += encode_varint(len(self.tx_outs))
        for tx_out in self.tx_outs:
            result += tx_out.serialize()
        result += int_to_little_endian(self.locktime, 4)
        return result


def hash(s, d):
    if type(s) is not str:
        s = str(s)
    hash = sha256(s.encode())
    if d:
        return hash.digest()
    return hash


def digest(h):
    return h.digest()