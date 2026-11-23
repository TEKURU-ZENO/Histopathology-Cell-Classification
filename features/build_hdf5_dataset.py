import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import glob
import json
import cv2
import h5py
import numpy as np
import scipy.io as sio
import torch
import pandas as pd
from features.extract_features import extract_crops_and_embed
from features.graph_builder import build_spatial_knn_graph

# Setup paths
h5_path = 'data/processed/features_cache.h5'
splits_json_path = 'data/splits/splits.json'
views_manifest_path = 'data/processed/cached_training_views/training_views_manifest.csv'
consep_proc_dir = 'data/processed/consep'
perturbed_test_dir = 'data/processed/perturbed_test'

with open(splits_json_path) as f:
    splits_data = json.load(f)

LIZARD_MAP = {2: 0, 1: 1, 3: 1, 4: 1, 5: 1, 6: 2}

def load_lizard_labels(mat_path):
    mat = sio.loadmat(mat_path)
    classes = mat['class'].flatten()
    centroids = mat['centroid']
    mapped = np.array([LIZARD_MAP.get(int(c), -1) for c in classes], dtype=np.int64)
    return centroids.astype(np.float32), mapped

def build_hdf5():
    print(f"Building HDF5 feature cache at {h5_path}...")
    
    # Open HDF5 file
    with h5py.File(h5_path, 'w') as h5:
        # 1. PROCESS TRAINING SET
        grp_train = h5.create_group('train')
        train_manifest = pd.read_csv(views_manifest_path)
        print(f"\nProcessing {len(train_manifest)} training regions...")
        
        for idx, row in train_manifest.iterrows():
            fn = row['tile']
            # Find label path from splits
            item = next(x for x in splits_data['details']['train'] if x['Filename'] == fn)
            centroids, labels = load_lizard_labels(item['LabelPath'])
            
            # Read clean view (v0)
            img_clean = cv2.cvtColor(cv2.imread(row['v0_clean']), cv2.COLOR_BGR2RGB)
            emb_clean = extract_crops_and_embed(img_clean, centroids)
            
            # Read 4 perturbed views
            img_v1 = cv2.cvtColor(cv2.imread(row['v1_randstainna1']), cv2.COLOR_BGR2RGB)
            emb_v1 = extract_crops_and_embed(img_v1, centroids)
            
            img_v2 = cv2.cvtColor(cv2.imread(row['v2_randstainna2']), cv2.COLOR_BGR2RGB)
            emb_v2 = extract_crops_and_embed(img_v2, centroids)
            
            img_v3 = cv2.cvtColor(cv2.imread(row['v3_macenko_mild']), cv2.COLOR_BGR2RGB)
            emb_v3 = extract_crops_and_embed(img_v3, centroids)
            
            img_v4 = cv2.cvtColor(cv2.imread(row['v4_macenko_mod']), cv2.COLOR_BGR2RGB)
            emb_v4 = extract_crops_and_embed(img_v4, centroids)
            
            # Build graphs for k in {6, 8, 12}
            e_idx_8, e_attr_8 = build_spatial_knn_graph(centroids, k=8)
            e_idx_6, e_attr_6 = build_spatial_knn_graph(centroids, k=6)
            e_idx_12, e_attr_12 = build_spatial_knn_graph(centroids, k=12)
            
            # Save into HDF5
            reg_grp = grp_train.create_group(fn)
            reg_grp.create_dataset('centroids', data=centroids, compression='gzip')
            reg_grp.create_dataset('labels', data=labels, compression='gzip')
            reg_grp.create_dataset('emb_clean', data=emb_clean, compression='gzip')
            reg_grp.create_dataset('emb_v1', data=emb_v1, compression='gzip')
            reg_grp.create_dataset('emb_v2', data=emb_v2, compression='gzip')
            reg_grp.create_dataset('emb_v3', data=emb_v3, compression='gzip')
            reg_grp.create_dataset('emb_v4', data=emb_v4, compression='gzip')
            
            reg_grp.create_dataset('edge_index_k8', data=e_idx_8.numpy(), compression='gzip')
            reg_grp.create_dataset('edge_attr_k8', data=e_attr_8.numpy(), compression='gzip')
            reg_grp.create_dataset('edge_index_k6', data=e_idx_6.numpy(), compression='gzip')
            reg_grp.create_dataset('edge_attr_k6', data=e_attr_6.numpy(), compression='gzip')
            reg_grp.create_dataset('edge_index_k12', data=e_idx_12.numpy(), compression='gzip')
            reg_grp.create_dataset('edge_attr_k12', data=e_attr_12.numpy(), compression='gzip')
            
            reg_grp.attrs['source'] = item['Source_Category']
            reg_grp.attrs['num_nuclei'] = len(labels)
            
            if (idx + 1) % 20 == 0 or (idx + 1) == len(train_manifest):
                print(f"  Train: cached {idx + 1}/{len(train_manifest)} regions")

        # 2. PROCESS VALIDATION SET
        grp_val = h5.create_group('val')
        val_items = splits_data['details']['val']
        print(f"\nProcessing {len(val_items)} validation regions...")
        
        for idx, item in enumerate(val_items):
            fn = item['Filename']
            centroids, labels = load_lizard_labels(item['LabelPath'])
            img_clean = cv2.cvtColor(cv2.imread(item['ImagePath']), cv2.COLOR_BGR2RGB)
            emb_clean = extract_crops_and_embed(img_clean, centroids)
            
            e_idx_8, e_attr_8 = build_spatial_knn_graph(centroids, k=8)
            e_idx_6, e_attr_6 = build_spatial_knn_graph(centroids, k=6)
            e_idx_12, e_attr_12 = build_spatial_knn_graph(centroids, k=12)
            
            reg_grp = grp_val.create_group(fn)
            reg_grp.create_dataset('centroids', data=centroids, compression='gzip')
            reg_grp.create_dataset('labels', data=labels, compression='gzip')
            reg_grp.create_dataset('emb_clean', data=emb_clean, compression='gzip')
            
            reg_grp.create_dataset('edge_index_k8', data=e_idx_8.numpy(), compression='gzip')
            reg_grp.create_dataset('edge_attr_k8', data=e_attr_8.numpy(), compression='gzip')
            reg_grp.create_dataset('edge_index_k6', data=e_idx_6.numpy(), compression='gzip')
            reg_grp.create_dataset('edge_attr_k6', data=e_attr_6.numpy(), compression='gzip')
            reg_grp.create_dataset('edge_index_k12', data=e_idx_12.numpy(), compression='gzip')
            reg_grp.create_dataset('edge_attr_k12', data=e_attr_12.numpy(), compression='gzip')
            
            reg_grp.attrs['source'] = item['Source_Category']
            reg_grp.attrs['num_nuclei'] = len(labels)

        # 3. PROCESS LIZARD IN-DOMAIN TEST SET
        grp_test_liz = h5.create_group('test_lizard')
        test_items = splits_data['details']['test']
        print(f"\nProcessing {len(test_items)} Lizard in-domain test regions...")
        
        for idx, item in enumerate(test_items):
            fn = item['Filename']
            centroids, labels = load_lizard_labels(item['LabelPath'])
            
            reg_grp = grp_test_liz.create_group(fn)
            reg_grp.create_dataset('centroids', data=centroids, compression='gzip')
            reg_grp.create_dataset('labels', data=labels, compression='gzip')
            
            e_idx_8, e_attr_8 = build_spatial_knn_graph(centroids, k=8)
            e_idx_6, e_attr_6 = build_spatial_knn_graph(centroids, k=6)
            e_idx_12, e_attr_12 = build_spatial_knn_graph(centroids, k=12)
            
            reg_grp.create_dataset('edge_index_k8', data=e_idx_8.numpy(), compression='gzip')
            reg_grp.create_dataset('edge_attr_k8', data=e_attr_8.numpy(), compression='gzip')
            reg_grp.create_dataset('edge_index_k6', data=e_idx_6.numpy(), compression='gzip')
            reg_grp.create_dataset('edge_attr_k6', data=e_attr_6.numpy(), compression='gzip')
            reg_grp.create_dataset('edge_index_k12', data=e_idx_12.numpy(), compression='gzip')
            reg_grp.create_dataset('edge_attr_k12', data=e_attr_12.numpy(), compression='gzip')
            
            # Embeddings for clean + 4 test perturbation conditions
            for cond in ['clean', 'macenko_mild', 'macenko_moderate', 'macenko_strong', 'randstainna']:
                img_path = os.path.join(perturbed_test_dir, f"lizard_{fn}_{cond}.png")
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                emb = extract_crops_and_embed(img, centroids)
                reg_grp.create_dataset(f'emb_{cond}', data=emb, compression='gzip')
                
            reg_grp.attrs['source'] = item['Source_Category']
            reg_grp.attrs['num_nuclei'] = len(labels)

        # 4. PROCESS CONSEP TEST SET (14 tiles)
        grp_test_con = h5.create_group('test_consep')
        print(f"\nProcessing 14 CoNSeP test tiles...")
        
        for i in range(1, 15):
            tile_name = f"test_{i}"
            npz_path = os.path.join(consep_proc_dir, 'test', f"{tile_name}_20x.npz")
            npz = np.load(npz_path)
            centroids = npz['centroids'].astype(np.float32)
            labels = npz['labels'].astype(np.int64)
            
            reg_grp = grp_test_con.create_group(tile_name)
            reg_grp.create_dataset('centroids', data=centroids, compression='gzip')
            reg_grp.create_dataset('labels', data=labels, compression='gzip')
            
            e_idx_8, e_attr_8 = build_spatial_knn_graph(centroids, k=8)
            e_idx_6, e_attr_6 = build_spatial_knn_graph(centroids, k=6)
            e_idx_12, e_attr_12 = build_spatial_knn_graph(centroids, k=12)
            
            reg_grp.create_dataset('edge_index_k8', data=e_idx_8.numpy(), compression='gzip')
            reg_grp.create_dataset('edge_attr_k8', data=e_attr_8.numpy(), compression='gzip')
            reg_grp.create_dataset('edge_index_k6', data=e_idx_6.numpy(), compression='gzip')
            reg_grp.create_dataset('edge_attr_k6', data=e_attr_6.numpy(), compression='gzip')
            reg_grp.create_dataset('edge_index_k12', data=e_idx_12.numpy(), compression='gzip')
            reg_grp.create_dataset('edge_attr_k12', data=e_attr_12.numpy(), compression='gzip')
            
            for cond in ['clean', 'macenko_mild', 'macenko_moderate', 'macenko_strong', 'randstainna']:
                img_path = os.path.join(perturbed_test_dir, f"consep_{tile_name}_{cond}.png")
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                emb = extract_crops_and_embed(img, centroids)
                reg_grp.create_dataset(f'emb_{cond}', data=emb, compression='gzip')
                
            reg_grp.attrs['source'] = 'CoNSeP'
            reg_grp.attrs['num_nuclei'] = len(labels)

    print(f"\nFeature extraction and graph building complete! Saved to {h5_path}")

if __name__ == '__main__':
    build_hdf5()
