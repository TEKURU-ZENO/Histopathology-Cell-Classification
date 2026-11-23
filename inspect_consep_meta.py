import json
import os

meta_path = r'c:\Users\Dev Mehta\Desktop\HISPATOLOGY EXPERIMENT\data\raw\consep\metadata.json'
with open(meta_path) as f:
    meta = json.load(f)
print('Metadata keys:', list(meta.keys()))
for k, v in meta.items():
    print(f'{k}: {v}')
