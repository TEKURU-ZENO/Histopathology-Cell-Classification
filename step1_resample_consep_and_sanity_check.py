import os
import glob
import cv2
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt

# Directories
consep_raw = 'data/raw/consep'
consep_proc = 'data/processed/consep'
sanity_dir = 'data/processed/sanity_checks'

os.makedirs(os.path.join(consep_proc, 'train'), exist_ok=True)
os.makedirs(os.path.join(consep_proc, 'test'), exist_ok=True)
os.makedirs(sanity_dir, exist_ok=True)

# CoNSeP mapping:
# 1 = other -> DROP (-1)
# 2 = inflammatory -> 1
# 3 = healthy epithelial -> 0
# 4 = dysplastic/malignant epithelial -> 0
# 5 = fibroblast -> 2
# 6 = muscle -> 2
# 7 = endothelial -> 2
CONSEP_MAP = {
    1: -1,
    2: 1,
    3: 0,
    4: 0,
    5: 2,
    6: 2,
    7: 2
}

def process_consep_split(split_name):
    img_dir = os.path.join(consep_raw, split_name.capitalize(), 'Images')
    lbl_dir = os.path.join(consep_raw, split_name.capitalize(), 'Labels')
    out_dir = os.path.join(consep_proc, split_name.lower())
    
    mat_files = sorted(glob.glob(os.path.join(lbl_dir, '*.mat')))
    print(f"Processing CoNSeP {split_name}: {len(mat_files)} tiles...")
    
    records = []
    for mat_path in mat_files:
        base = os.path.splitext(os.path.basename(mat_path))[0]
        img_path = os.path.join(img_dir, f"{base}.png")
        
        # Load 40x image (1000x1000)
        img_40x = cv2.imread(img_path)
        img_40x = cv2.cvtColor(img_40x, cv2.COLOR_BGR2RGB)
        
        # Resample to 20x (500x500)
        img_20x = cv2.resize(img_40x, (500, 500), interpolation=cv2.INTER_AREA)
        
        # Save 20x image
        save_img_path = os.path.join(out_dir, f"{base}_20x.png")
        cv2.imwrite(save_img_path, cv2.cvtColor(img_20x, cv2.COLOR_RGB2BGR))
        
        # Load labels
        mat = sio.loadmat(mat_path)
        raw_types = mat['inst_type'].flatten()
        raw_centroids = mat['inst_centroid']  # Nx2, (x, y)
        
        # Map labels and filter
        mapped_labels = np.array([CONSEP_MAP.get(int(t), -1) for t in raw_types])
        valid_mask = (mapped_labels >= 0)
        
        kept_labels = mapped_labels[valid_mask]
        # Rescale centroids to 20x: (x * 0.5, y * 0.5)
        kept_centroids_20x = raw_centroids[valid_mask] * 0.5
        
        # Save processed .npz
        save_npz_path = os.path.join(out_dir, f"{base}_20x.npz")
        np.savez(
            save_npz_path,
            image_path=save_img_path,
            centroids=kept_centroids_20x,
            labels=kept_labels,
            raw_types=raw_types[valid_mask]
        )
        
        records.append({
            'tile': base,
            'split': split_name,
            'total_raw': len(raw_types),
            'kept': len(kept_labels),
            'epithelial': int(np.sum(kept_labels == 0)),
            'inflammatory': int(np.sum(kept_labels == 1)),
            'stromal': int(np.sum(kept_labels == 2)),
            'dropped_other': int(np.sum(~valid_mask)),
            'img_20x_path': save_img_path,
            'npz_path': save_npz_path
        })
        
    return pd.DataFrame(records)

df_proc_train = process_consep_split('train')
df_proc_test = process_consep_split('test')
df_proc_all = pd.concat([df_proc_train, df_proc_test], ignore_index=True)
df_proc_all.to_csv(os.path.join(consep_proc, 'consep_resampled_summary.csv'), index=False)
print("CoNSeP 40x -> 20x resampling complete.")

# Sanity Check Overlays:
# Plot a couple of tiles with color-coded centroids
# Class 0 (Epithelial): Red
# Class 1 (Inflammatory): Green
# Class 2 (Stromal): Blue
COLORS = {0: (220, 20, 60), 1: (50, 205, 50), 2: (30, 144, 255)}  # RGB
COLOR_NAMES = {0: 'Epithelial (Red)', 1: 'Inflammatory (Green)', 2: 'Stromal (Blue)'}

def create_overlay_tile(img_rgb, centroids, labels, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(img_rgb)
    
    # Plot points by class
    c_map = {0: 'crimson', 1: 'limegreen', 2: 'dodgerblue'}
    for c_id, c_name in [(0, 'Epithelial'), (1, 'Inflammatory'), (2, 'Stromal')]:
        mask = (labels == c_id)
        if np.any(mask):
            pts = centroids[mask]
            ax.scatter(pts[:, 0], pts[:, 1], c=c_map[c_id], s=18, label=f"{c_name} (n={len(pts)})", alpha=0.85, edgecolors='black', linewidths=0.3)
            
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# 1. Overlay for 2 CoNSeP Test tiles
for tile_name in ['test_1', 'test_2']:
    npz = np.load(os.path.join(consep_proc, 'test', f"{tile_name}_20x.npz"))
    img = cv2.cvtColor(cv2.imread(str(npz['image_path'])), cv2.COLOR_BGR2RGB)
    out_path = os.path.join(sanity_dir, f"consep_{tile_name}_overlay.png")
    create_overlay_tile(img, npz['centroids'], npz['labels'], f"CoNSeP {tile_name} (20x Resampled)", out_path)
    print(f"Saved sanity check overlay: {out_path}")

# 2. Overlay for 2 Lizard tiles (1 dpath, 1 crag, 1 glas)
df_kept = pd.read_csv('data/splits/lizard_kept_regions.csv')
LIZARD_MAP = {2: 0, 1: 1, 3: 1, 4: 1, 5: 1, 6: 2}

for prefix in ['dpath_1', 'crag_1', 'glas_1']:
    row = df_kept[df_kept['Filename'] == prefix].iloc[0]
    img = cv2.cvtColor(cv2.imread(row['ImagePath']), cv2.COLOR_BGR2RGB)
    mat = sio.loadmat(row['LabelPath'])
    raw_classes = mat['class'].flatten()
    centroids = mat['centroid']
    labels = np.array([LIZARD_MAP.get(int(c), -1) for c in raw_classes])
    
    # Crop a 1000x1000 or full view for clean visualization
    h, w, _ = img.shape
    crop_size = min(800, h, w)
    img_sub = img[:crop_size, :crop_size]
    mask = (centroids[:, 0] < crop_size) & (centroids[:, 1] < crop_size)
    
    out_path = os.path.join(sanity_dir, f"lizard_{prefix}_overlay.png")
    create_overlay_tile(img_sub, centroids[mask], labels[mask], f"Lizard {prefix} ({row['Source_Category']})", out_path)
    print(f"Saved sanity check overlay: {out_path}")

print("\nStep 1 Complete! Provenance audit, stratified split, CoNSeP resampling, and sanity checks done.")
