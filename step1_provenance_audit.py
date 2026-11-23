import os
import glob
import pandas as pd
import scipy.io as sio
import numpy as np

# Exact paths
info_csv_path = r'data/raw/lizard_labels/Lizard_Labels/info.csv'
labels_dir = r'data/raw/lizard_labels/Lizard_Labels/Labels'
images1_dir = r'data/raw/lizard_images1/Lizard_Images1'
images2_dir = r'data/raw/lizard_images2/Lizard_Images2'

df_info = pd.read_csv(info_csv_path)
print(f"Total entries in info.csv: {len(df_info)}")

# Find all image paths
all_images = {}
for p in glob.glob(os.path.join(images1_dir, '*.png')) + glob.glob(os.path.join(images2_dir, '*.png')):
    base = os.path.splitext(os.path.basename(p))[0]
    all_images[base] = p

print(f"Total images found: {len(all_images)}")

# Map filename prefix to source category
def get_source_category(filename):
    prefix = filename.split('_')[0].lower()
    if prefix == 'dpath':
        return 'DigestPath'
    elif prefix == 'crag':
        return 'CRAG'
    elif prefix == 'glas':
        return 'GlaS'
    elif prefix == 'consep':
        return 'CoNSeP'
    elif prefix == 'pannuke':
        return 'PanNuke'
    elif prefix == 'tcga':
        return 'TCGA'
    else:
        return 'Unknown'

df_info['Source_Category'] = df_info['Filename'].apply(get_source_category)

# Audit nuclei counts per region
stats = []
for idx, row in df_info.iterrows():
    fn = row['Filename']
    src = row['Source_Category']
    mat_path = os.path.join(labels_dir, f"{fn}.mat")
    
    if not os.path.exists(mat_path):
        print(f"Warning: {mat_path} not found!")
        continue
        
    mat = sio.loadmat(mat_path)
    classes = mat['class'].flatten()
    centroids = mat['centroid']
    
    num_nuclei = len(classes)
    # Count per Lizard class:
    # 1: Neutrophil, 2: Epithelial, 3: Lymphocyte, 4: Plasma, 5: Eosinophil, 6: Connective
    c_counts = {c: int(np.sum(classes == c)) for c in range(1, 7)}
    
    # Harmonized classes into 3-class schema:
    # Epithelial: class 2
    # Inflammatory: classes 1, 3, 4, 5
    # Stromal: class 6
    n_epi = c_counts[2]
    n_inf = c_counts[1] + c_counts[3] + c_counts[4] + c_counts[5]
    n_str = c_counts[6]
    
    stats.append({
        'Filename': fn,
        'Source_Category': src,
        'WSI_Source': row['Source'],
        'Total_Nuclei': num_nuclei,
        'Epithelial': n_epi,
        'Inflammatory': n_inf,
        'Stromal': n_str,
        'Neutrophil': c_counts[1],
        'Lizard_Epithelial': c_counts[2],
        'Lymphocyte': c_counts[3],
        'Plasma': c_counts[4],
        'Eosinophil': c_counts[5],
        'Connective': c_counts[6],
        'ImagePath': all_images.get(fn, ''),
        'LabelPath': mat_path
    })

df_stats = pd.DataFrame(stats)

# Summary table per source
summary = df_stats.groupby('Source_Category').agg(
    Regions=('Filename', 'count'),
    Total_Nuclei=('Total_Nuclei', 'sum'),
    Epithelial=('Epithelial', 'sum'),
    Inflammatory=('Inflammatory', 'sum'),
    Stromal=('Stromal', 'sum')
).reset_index()

print("\n=== PROVENANCE AUDIT TABLE (LIZARD) ===")
print(summary.to_string(index=False))

# Kept vs Dropped
kept = df_stats[df_stats['Source_Category'].isin(['DigestPath', 'CRAG', 'GlaS'])]
dropped = df_stats[~df_stats['Source_Category'].isin(['DigestPath', 'CRAG', 'GlaS'])]

print(f"\nKept (DigestPath, CRAG, GlaS): {len(kept)} regions, {kept['Total_Nuclei'].sum():,} nuclei")
print(f"Dropped (CoNSeP, PanNuke, TCGA): {len(dropped)} regions, {dropped['Total_Nuclei'].sum():,} nuclei")

# Save provenance audit csv
os.makedirs('data/splits', exist_ok=True)
summary.to_csv('data/splits/provenance_audit_summary.csv', index=False)
df_stats.to_csv('data/splits/lizard_all_regions_audit.csv', index=False)
kept.to_csv('data/splits/lizard_kept_regions.csv', index=False)
print("Saved data/splits/provenance_audit_summary.csv and data/splits/lizard_kept_regions.csv")
