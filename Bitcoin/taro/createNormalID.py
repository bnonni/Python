from hashlib import sha256
from utxo import UTXO

def hash(s):
 return sha256(s.encode())

def digest(h):
 return h.hexdigest()

# regtest utxo

genesis_outpoint = (UTXO['txid'], UTXO['vout'])
asset_tag = "Terminus Tokens"
asset_meta = "https://terminus.money/"
asset_id_hash = hash(str(genesis_outpoint) + digest(hash(asset_tag)) + asset_meta)
asset_id = digest(asset_id_hash)
print(
 f"Taro Identifier\n \
  Asset Tag: {asset_tag}\n \
  Asset Meta: {asset_meta}\n \
  Asset ID (hexdigest): {asset_id}\n \
  Asset ID (bytes): {asset_id_hash.digest()}\n \
  Asset ID Size: {asset_id_hash.digest_size}"
 )
