import os
import glob
import scipy.io as sio
import numpy as np
import pandas as pd

consep_dir = 'data/raw/consep'
train_img_dir = os.path.join(consep_dir, 'Train', 'Images')
train_lbl_dir = os.path.join(consep_dir, 'Train', 'Labels')
test_img_dir = os.path.join(consep_dir, 'Test', 'Images')
test_lbl_dir = os.path.join(consep_dir, 'Test', 'Labels')

# Classes in CoNSeP:
# 1 = other (DROP)
# 2 = inflammatory -> Class 1 (Inflammatory)
# 3 = healthy epithelial -> Class 0 (Epithelial)
# 4 = dysplastic/malignant epithelial -> Class 0 (Epithelial)
# 5 = fibroblast -> Class 2 (Stromal)
# 6 = muscle -> Class 2 (Stromal)
# 7 = endothelial -> Class 2 (Stromal)

def audit_split(img_dir, lbl_dir, split_name):
    records = []
    for lbl_path in sorted(glob.glob(os.path.join(lbl_dir, '*.mat'))):
        base = os.path.splitext(os.path.basename(lbl_path))[0]
        img_path = os.path.join(img_dir, f"{base}.png")
        mat = sio.loadmat(lbl_path)
        
        # 'inst_type' is Nx1, 'inst_centroid' is Nx2
        inst_type = mat['inst_type'].flatten()
        inst_centroid = mat['inst_centroid']
        
        c_counts = {c: int(np.sum(inst_type == c)) for c in range(1, 8)}
        
        n_other = c_counts[1]
        n_inf = c_counts[2]
        n_epi = c_counts[3] + c_counts[4]
        n_str = c_counts[5] + c_counts[6] + c_counts[7]
        n_kept = n_epi + n_inf + n_str
        
        records.append({
            'Tile': base,
            'Split': split_name,
            'Total_Raw_Nuclei': len(inst_type),
            'Kept_Nuclei': n_kept,
            'Epithelial': n_epi,
            'Inflammatory': n_inf,
            'Stromal': n_str,
            'Dropped_Other': n_other,
            'ImagePath': img_path,
            'LabelPath': lbl_path
        })
    return pd.DataFrame(records)

df_train = audit_split(train_img_dir, train_lbl_dir, 'train')
df_test = audit_split(test_img_dir, test_lbl_dir, 'test')
df_consep = pd.concat([df_train, df_test], ignore_index=True)

summary_consep = df_consep.groupby('Split').agg(
    Tiles=('Tile', 'count'),
    Total_Raw_Nuclei=('Total_Raw_Nuclei', 'sum'),
    Kept_Nuclei=('Kept_Nuclei', 'sum'),
    Epithelial=('Epithelial', 'sum'),
    Inflammatory=('Inflammatory', 'sum'),
    Stromal=('Stromal', 'sum'),
    Dropped_Other=('Dropped_Other', 'sum')
).reset_index()

print("=== CONSEP AUDIT TABLE ===")
print(summary_consep.to_string(index=False))

df_consep.to_csv('data/splits/consep_audit.csv', index=False)
print("Saved data/splits/consep_audit.csv")
