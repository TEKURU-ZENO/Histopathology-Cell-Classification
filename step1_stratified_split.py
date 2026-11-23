import pandas as pd
import numpy as np
import json
import os

df_kept = pd.read_csv('data/splits/lizard_kept_regions.csv')
print(f"Total kept regions: {len(df_kept)}")

seed = 42
np.random.seed(seed)

splits = {'train': [], 'val': [], 'test': []}

for src, group in df_kept.groupby('Source_Category'):
    # Shuffle group deterministically
    indices = group.index.tolist()
    np.random.shuffle(indices)
    
    n = len(indices)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    n_test = n - n_train - n_val
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    print(f"Source {src:12s} (total {n:2d}): train={len(train_idx):2d}, val={len(val_idx):2d}, test={len(test_idx):2d}")
    
    for idx in train_idx:
        row = group.loc[idx]
        splits['train'].append(row.to_dict())
    for idx in val_idx:
        row = group.loc[idx]
        splits['val'].append(row.to_dict())
    for idx in test_idx:
        row = group.loc[idx]
        splits['test'].append(row.to_dict())

# Verify counts and no overlap
train_files = set(r['Filename'] for r in splits['train'])
val_files = set(r['Filename'] for r in splits['val'])
test_files = set(r['Filename'] for r in splits['test'])

assert len(train_files & val_files) == 0, "Train and Val overlap!"
assert len(train_files & test_files) == 0, "Train and Test overlap!"
assert len(val_files & test_files) == 0, "Val and Test overlap!"
assert len(train_files | val_files | test_files) == len(df_kept), "Total files count mismatch!"

print(f"\nSplit Summary:")
print(f"  Train: {len(splits['train'])} regions, {sum(r['Total_Nuclei'] for r in splits['train']):,} nuclei")
print(f"  Val:   {len(splits['val'])} regions, {sum(r['Total_Nuclei'] for r in splits['val']):,} nuclei")
print(f"  Test:  {len(splits['test'])} regions, {sum(r['Total_Nuclei'] for r in splits['test']):,} nuclei")

# Class breakdown per split
breakdown = []
for sname in ['train', 'val', 'test']:
    rs = splits[sname]
    breakdown.append({
        'Split': sname,
        'Regions': len(rs),
        'Total_Nuclei': sum(r['Total_Nuclei'] for r in rs),
        'Epithelial': sum(r['Epithelial'] for r in rs),
        'Inflammatory': sum(r['Inflammatory'] for r in rs),
        'Stromal': sum(r['Stromal'] for r in rs)
    })

df_breakdown = pd.DataFrame(breakdown)
print("\n=== SPLIT PHENOTYPE BREAKDOWN ===")
print(df_breakdown.to_string(index=False))

# Save immutable split files
json_path = 'data/splits/splits.json'
csv_path = 'data/splits/splits.csv'

# Save clean simplified JSON
split_json_data = {
    'metadata': {
        'seed': seed,
        'train_ratio': 0.70,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'stratification': 'Source_Category (DigestPath, CRAG, GlaS)'
    },
    'breakdown': breakdown,
    'splits': {
        sname: [r['Filename'] for r in splits[sname]] for sname in ['train', 'val', 'test']
    },
    'details': splits
}

with open(json_path, 'w') as f:
    json.dump(split_json_data, f, indent=2)

# Save flat CSV
flat_rows = []
for sname in ['train', 'val', 'test']:
    for r in splits[sname]:
        r_copy = dict(r)
        r_copy['Split'] = sname
        flat_rows.append(r_copy)

pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
print(f"\nSaved immutable splits to:\n  - {json_path}\n  - {csv_path}")
