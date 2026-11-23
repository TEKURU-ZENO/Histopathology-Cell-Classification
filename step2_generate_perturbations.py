import os
import json
import cv2
import numpy as np
import pandas as pd
from perturbations.macenko import MacenkoPerturbator
from perturbations.randstainna import RandStainNA

# Load splits
with open('data/splits/splits.json') as f:
    split_info = json.load(f)

df_splits = pd.read_csv('data/splits/splits.csv')
train_df = df_splits[df_splits['Split'] == 'train']
test_df = df_splits[df_splits['Split'] == 'test']

out_test_dir = 'data/processed/perturbed_test'
out_train_views_dir = 'data/processed/cached_training_views'
os.makedirs(out_test_dir, exist_ok=True)
os.makedirs(out_train_views_dir, exist_ok=True)

# 1. Fit RandStainNA strictly on training split
train_image_paths = train_df['ImagePath'].tolist()
rs = RandStainNA()
rs.fit_from_images(train_image_paths)
rs.save('data/processed/randstainna_stats.json')

macenko = MacenkoPerturbator()

# 2. Generate Perturbed Test Sets (Lizard Test: 29 regions; CoNSeP Test: 14 tiles)
print("\n--- Generating Perturbed Test Sets ---")
test_records = []

# (A) Lizard Test
for idx, row in test_df.iterrows():
    fn = row['Filename']
    img_rgb = cv2.cvtColor(cv2.imread(row['ImagePath']), cv2.COLOR_BGR2RGB)
    
    # Save clean
    clean_path = os.path.join(out_test_dir, f"lizard_{fn}_clean.png")
    cv2.imwrite(clean_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    test_records.append({'dataset': 'lizard', 'tile': fn, 'condition': 'clean', 'path': clean_path, 'params': {}})
    
    # Macenko Mild, Moderate, Strong
    for sev in ['mild', 'moderate', 'strong']:
        seed = 1000 + idx * 10 + ['mild', 'moderate', 'strong'].index(sev)
        img_m, p_m = macenko.perturb(img_rgb, severity=sev, seed=seed)
        m_path = os.path.join(out_test_dir, f"lizard_{fn}_macenko_{sev}.png")
        cv2.imwrite(m_path, cv2.cvtColor(img_m, cv2.COLOR_RGB2BGR))
        test_records.append({'dataset': 'lizard', 'tile': fn, 'condition': f'macenko_{sev}', 'path': m_path, 'params': p_m})
        
    # RandStainNA
    seed_rs = 2000 + idx
    img_rs, p_rs = rs.perturb(img_rgb, seed=seed_rs)
    rs_path = os.path.join(out_test_dir, f"lizard_{fn}_randstainna.png")
    cv2.imwrite(rs_path, cv2.cvtColor(img_rs, cv2.COLOR_RGB2BGR))
    test_records.append({'dataset': 'lizard', 'tile': fn, 'condition': 'randstainna', 'path': rs_path, 'params': p_rs})

# (B) CoNSeP Test (14 tiles, 20x resampled)
consep_test_dir = 'data/processed/consep/test'
for i in range(1, 15):
    tile_name = f"test_{i}"
    npz_path = os.path.join(consep_test_dir, f"{tile_name}_20x.npz")
    npz = np.load(npz_path)
    img_rgb = cv2.cvtColor(cv2.imread(str(npz['image_path'])), cv2.COLOR_BGR2RGB)
    
    # Save clean
    clean_path = os.path.join(out_test_dir, f"consep_{tile_name}_clean.png")
    cv2.imwrite(clean_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    test_records.append({'dataset': 'consep', 'tile': tile_name, 'condition': 'clean', 'path': clean_path, 'params': {}})
    
    for sev in ['mild', 'moderate', 'strong']:
        seed = 3000 + i * 10 + ['mild', 'moderate', 'strong'].index(sev)
        img_m, p_m = macenko.perturb(img_rgb, severity=sev, seed=seed)
        m_path = os.path.join(out_test_dir, f"consep_{tile_name}_macenko_{sev}.png")
        cv2.imwrite(m_path, cv2.cvtColor(img_m, cv2.COLOR_RGB2BGR))
        test_records.append({'dataset': 'consep', 'tile': tile_name, 'condition': f'macenko_{sev}', 'path': m_path, 'params': p_m})
        
    seed_rs = 4000 + i
    img_rs, p_rs = rs.perturb(img_rgb, seed=seed_rs)
    rs_path = os.path.join(out_test_dir, f"consep_{tile_name}_randstainna.png")
    cv2.imwrite(rs_path, cv2.cvtColor(img_rs, cv2.COLOR_RGB2BGR))
    test_records.append({'dataset': 'consep', 'tile': tile_name, 'condition': 'randstainna', 'path': rs_path, 'params': p_rs})

pd.DataFrame(test_records).to_csv(os.path.join(out_test_dir, 'perturbed_test_index.csv'), index=False)
print(f"Generated {len(test_records)} perturbed test views across Lizard and CoNSeP.")

# 3. Generate K = 4 Cached Training Views per Training Tile
print("\n--- Generating K=4 Cached Training Views for 136 Training Tiles ---")
train_views_manifest = []

for idx, row in train_df.iterrows():
    fn = row['Filename']
    img_rgb = cv2.cvtColor(cv2.imread(row['ImagePath']), cv2.COLOR_BGR2RGB)
    
    # View 0: Clean
    v0_path = os.path.join(out_train_views_dir, f"{fn}_v0_clean.png")
    cv2.imwrite(v0_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    
    # View 1: RandStainNA draw 1
    img_v1, p1 = rs.perturb(img_rgb, seed=5000 + idx*4 + 1)
    v1_path = os.path.join(out_train_views_dir, f"{fn}_v1_randstainna1.png")
    cv2.imwrite(v1_path, cv2.cvtColor(img_v1, cv2.COLOR_RGB2BGR))
    
    # View 2: RandStainNA draw 2
    img_v2, p2 = rs.perturb(img_rgb, seed=5000 + idx*4 + 2)
    v2_path = os.path.join(out_train_views_dir, f"{fn}_v2_randstainna2.png")
    cv2.imwrite(v2_path, cv2.cvtColor(img_v2, cv2.COLOR_RGB2BGR))
    
    # View 3: Macenko Mild
    img_v3, p3 = macenko.perturb(img_rgb, severity='mild', seed=5000 + idx*4 + 3)
    v3_path = os.path.join(out_train_views_dir, f"{fn}_v3_macenko_mild.png")
    cv2.imwrite(v3_path, cv2.cvtColor(img_v3, cv2.COLOR_RGB2BGR))
    
    # View 4: Macenko Moderate
    img_v4, p4 = macenko.perturb(img_rgb, severity='moderate', seed=5000 + idx*4 + 4)
    v4_path = os.path.join(out_train_views_dir, f"{fn}_v4_macenko_mod.png")
    cv2.imwrite(v4_path, cv2.cvtColor(img_v4, cv2.COLOR_RGB2BGR))
    
    train_views_manifest.append({
        'tile': fn,
        'v0_clean': v0_path,
        'v1_randstainna1': v1_path,
        'v2_randstainna2': v2_path,
        'v3_macenko_mild': v3_path,
        'v4_macenko_mod': v4_path
    })
    if (idx + 1) % 25 == 0 or (idx + 1) == len(train_df):
        print(f"  Processed {idx + 1}/{len(train_df)} training tiles...")

pd.DataFrame(train_views_manifest).to_csv(os.path.join(out_train_views_dir, 'training_views_manifest.csv'), index=False)
print("Step 2 Complete! Stain perturbations and cached views generated.")
