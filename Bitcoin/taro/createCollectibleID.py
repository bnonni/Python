

from hashlib import sha256
from utxo import UTXO
from bry_bitmap import BitMap

def hash(s, d):
    if type(s) is not str:
        s = str(s)
    hash = sha256(s.encode())
    if d:
        return hash.digest()
    return hash

def digest(h):
    return h.digest()

def int_to_little_endian(n, length):
    '''endian_to_little_endian takes an integer and returns the little-endian
    byte sequence of length'''
    return n.to_bytes(length, 'little')

def raw_serialize(self):
        # initialize what we'll send back
        result = b''
        # go through each cmd
        for cmd in self.cmds:
            # if the cmd is an integer, it's an opcode
            if type(cmd) == int:
                # turn the cmd into a single byte integer using int_to_little_endian
                result += int_to_little_endian(cmd, 1)
            else:
                # otherwise, this is an element
                # get the length in bytes
                length = len(cmd)
                # for large lengths, we have to use a pushdata opcode
                if length < 75:
                    # turn the length into a single byte integer
                    result += int_to_little_endian(length, 1)
                elif length > 75 and length < 0x100:
                    # 76 is pushdata1
                    result += int_to_little_endian(76, 1)
                    result += int_to_little_endian(length, 1)
                elif length >= 0x100 and length <= 520:
                    # 77 is pushdata2
                    result += int_to_little_endian(77, 1)
                    result += int_to_little_endian(length, 2)
                else:
                    raise ValueError('too long an cmd')
                result += cmd
        return result

# regtest utxo
genesis_outpoint = hash((UTXO["txid"], UTXO["vout"]), True)  # UTXO being used in txn
asset_tag = hash("Terminus Launch Tweet", True)  # any arb data, typically asset name
asset_meta = hash(
    "https://twitter.com/terminusbtc/status/1532131420071727104", True
)  # any arb data, typically info about asset
output_index = 0  # index of output containing Taro asset
asset_type = 1  # collectible asset
asset_id_preimage = "%s%s%s%s%s" % (
    genesis_outpoint,
    asset_tag,
    asset_meta,
    output_index,
    asset_type,
)
asset_id_hash = hash(asset_id_preimage, False)
asset_id_hexdigest = asset_id_hash.hexdigest()
asset_id = digest(asset_id_hash)
bitmap = BitMap()
asset_id_bitmap = bitmap.fromhexstring(asset_id_hexdigest)
print(
    f"Taro Identifier - Collectible Asset - Terminus Launch Tweet\n \
  Genesis Outpoint Hash: {genesis_outpoint}\n \
  Asset Tag Hash: {asset_tag}\n \
  Asset Meta Hash: {asset_meta}\n \
  Output Index: {output_index}\n \
  Asset Type: {asset_type}\n \
  Asset ID Preimage: {asset_id_preimage}\n\n \
  Asset ID Hexdigest: {asset_id_hexdigest}\n \
  Asset ID Digest: {asset_id_hash.digest()}\n\n \
  Asset ID Size: {asset_id_hash.digest_size}\n \
  Asset ID Bitmap: \n{asset_id_bitmap}"
)