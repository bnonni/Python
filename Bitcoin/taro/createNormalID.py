from bry_bitmap import BitMap
from helpers import hash, digest, serialize, UTXO

# UTXO to bind the taro asset to
genesis_outpoint = (UTXO['txid'], UTXO['vout'])
# genesis_outpoint_hash = hash(genesis_outpoint, True)

# any arb data, typically asset name
asset_tag = "Terminus Tokens"
# asset_tag_hash = hash(asset_tag, True)

# any arb data, typically info about asset
asset_meta = "https://terminus.money/"
# asset_meta_hash = hash(asset_meta, True)
 
# index of output containing Taro asset
output_index = 0

# normal asset
asset_type = 0

# asset id preimage, all fields concatenated
# genesis_point || asset_tag || asset_meta || output_index || asset_type
asset_id_preimage = "%s%s%s%s%s" % (
    genesis_outpoint,
    asset_tag,
    asset_meta,
    output_index,
    asset_type,
)

asset_id_hash_obj = hash(asset_id_preimage, False)
asset_id = asset_id_hash_obj.digest()
asset_id_hexdigest = asset_id_hash_obj.hexdigest()
bitmap = BitMap()
asset_id_bitmap = bitmap.fromhexstring(asset_id_hexdigest)

print(
    f"Taro Identifier\n \
        Genesis Outpoint: {genesis_outpoint}\n \
        Asset Tag (name): {asset_tag} \n \
        Asset Meta (link): {asset_meta} \n \
        Asset ID Preimage: {asset_id_preimage}\n \
        Asset ID (bytes): {asset_id}\n \
        Asset ID (hex): {asset_id_hexdigest}\n \
        Asset ID Size: {asset_id_hash_obj.digest_size}\n \
        Asset ID Bitmap: \n{asset_id_bitmap}"
)
