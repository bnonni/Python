

from hashlib import sha256

def hash(s):
 return sha256(s.encode())

def digest(h):
 return h.hexdigest()

# regtest utxo
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
    safe=True
)
genesis_outpoint = (UTXO['txid'], UTXO['vout'])
asset_tag = "Terminus Launch Tweet"
asset_meta = "https://twitter.com/terminusbtc/status/1532131420071727104?s=20&t=C4gOAAJphCfNsb5IOw247w"
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
