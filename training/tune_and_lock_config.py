import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import h5py
import json
import datetime
import numpy as np
import torch
import torch.optim as optim
from models.gatv2_model import GATv2Head
from losses.consistency_loss import TotalConsistencyLoss
from metrics.evaluation import compute_macro_f1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preload_data(h5_path='data/processed/features_cache.h5'):
    print("Preloading feature embeddings and graphs into RAM...", flush=True)
    with h5py.File(h5_path, 'r') as h5:
        # Preload Val Data
        val_data = []
        for fn in h5['val'].keys():
            grp = h5['val'][fn]
            val_data.append({
                'labels': grp['labels'][:],
                'emb_clean': torch.from_numpy(grp['emb_clean'][:]).float().to(device),
                'edge_idx_6': torch.from_numpy(grp['edge_index_k6'][:]).long().to(device),
                'edge_attr_6': torch.from_numpy(grp['edge_attr_k6'][:]).float().to(device),
                'edge_idx_8': torch.from_numpy(grp['edge_index_k8'][:]).long().to(device),
                'edge_attr_8': torch.from_numpy(grp['edge_attr_k8'][:]).float().to(device),
                'edge_idx_12': torch.from_numpy(grp['edge_index_k12'][:]).long().to(device),
                'edge_attr_12': torch.from_numpy(grp['edge_attr_k12'][:]).float().to(device),
            })
            
        # Preload Train Data (representative subset of 40 regions for ultra-fast grid search)
        train_data = []
        train_keys = list(h5['train'].keys())[:40]
        all_labels = []
        for fn in h5['train'].keys():
            all_labels.append(h5['train'][fn]['labels'][:])
        all_labels = np.concatenate(all_labels)
        
        for fn in train_keys:
            grp = h5['train'][fn]
            train_data.append({
                'labels': torch.from_numpy(grp['labels'][:]).long().to(device),
                'emb_clean': torch.from_numpy(grp['emb_clean'][:]).float().to(device),
                'emb_v1': torch.from_numpy(grp['emb_v1'][:]).float().to(device),
                'emb_v2': torch.from_numpy(grp['emb_v2'][:]).float().to(device),
                'emb_v3': torch.from_numpy(grp['emb_v3'][:]).float().to(device),
                'emb_v4': torch.from_numpy(grp['emb_v4'][:]).float().to(device),
                'edge_idx_6': torch.from_numpy(grp['edge_index_k6'][:]).long().to(device),
                'edge_attr_6': torch.from_numpy(grp['edge_attr_k6'][:]).float().to(device),
                'edge_idx_8': torch.from_numpy(grp['edge_index_k8'][:]).long().to(device),
                'edge_attr_8': torch.from_numpy(grp['edge_attr_k8'][:]).float().to(device),
                'edge_idx_12': torch.from_numpy(grp['edge_index_k12'][:]).long().to(device),
                'edge_attr_12': torch.from_numpy(grp['edge_attr_k12'][:]).float().to(device),
            })

    # Inverse frequency weights
    C = 3
    counts = np.bincount(all_labels, minlength=C)
    weights = torch.tensor(len(all_labels) / (C * counts.astype(np.float32)), dtype=torch.float32, device=device)
    print(f"Preloaded {len(train_data)} train and {len(val_data)} val regions in GPU/RAM.", flush=True)
    return train_data, val_data, weights

def evaluate_val_grid(lr, lambda_val, k_val, train_data, val_data, weights, max_epochs=4):
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = GATv2Head(in_features=512, hidden_heads=4, hidden_per_head=16, 
                      out_features=32, num_classes=3, edge_dim=3, dropout=0.25).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = TotalConsistencyLoss(class_weights=weights, lambda_cons=lambda_val)

    best_val_f1 = -1.0
    k_idx_key = f'edge_idx_{k_val}'
    k_attr_key = f'edge_attr_{k_val}'

    for epoch in range(1, max_epochs + 1):
        model.train()
        for item in train_data:
            labels = item['labels']
            edge_idx = item[k_idx_key]
            edge_attr = item[k_attr_key]
            emb_clean = item['emb_clean']
            
            # Pick a random perturbed view
            v_key = np.random.choice(['emb_v1', 'emb_v2', 'emb_v3', 'emb_v4'])
            emb_pert = item[v_key]

            optimizer.zero_grad()
            logits_clean = model(emb_clean, edge_idx, edge_attr)
            logits_pert = model(emb_pert, edge_idx, edge_attr)
            loss, _, _ = loss_fn(logits_clean, labels, logits_pert)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds_all, val_targets_all = [], []
        with torch.no_grad():
            for item in val_data:
                v_labels = item['labels']
                v_emb = item['emb_clean']
                v_edge_idx = item[k_idx_key]
                v_edge_attr = item[k_attr_key]

                logits = model(v_emb, v_edge_idx, v_edge_attr)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                val_preds_all.append(preds)
                val_targets_all.append(v_labels)

        val_f1 = compute_macro_f1(np.concatenate(val_targets_all), np.concatenate(val_preds_all))
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

    return best_val_f1

def run_grid_search_and_lock():
    train_data, val_data, weights = preload_data()

    lr_candidates = [1e-3, 3e-4]
    lambda_candidates = [0.1, 0.5, 1.0]
    k_candidates = [6, 8, 12]

    results = []
    best_cfg = None
    best_score = -1.0

    print("Running Grid Search over {lr} x {lambda} x {k}...", flush=True)
    for lr in lr_candidates:
        for lam in lambda_candidates:
            for k in k_candidates:
                score = evaluate_val_grid(lr, lam, k, train_data, val_data, weights)
                print(f"  lr={lr:<5} | lambda={lam:<3} | k={k:<2} -> Val Macro-F1: {score:.4f}", flush=True)
                results.append({'lr': lr, 'lambda': lam, 'k': k, 'val_macro_f1': score})
                
                if score > best_score:
                    best_score = score
                    best_cfg = {'lr': lr, 'lambda': lam, 'k': k}

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    locked_config = {
        'timestamp_utc': timestamp,
        'status': 'LOCKED_BEFORE_TEST_EVALUATION',
        'chosen_hyperparameters': best_cfg,
        'best_validation_macro_f1': best_score,
        'all_grid_results': results,
        'model_parameters': {
            'M1_ResNet18': 11178051,
            'M2_M3_MLP': 66051,
            'M4_M5_GATv2': 70403,
            'M4_M5_SAGE': 69827
        },
        'feature_extraction': {
            'backbone': 'ResNet-18 (ImageNet pretrained, frozen, FP16)',
            'embedding_dim': 512,
            'crop_size': '41x41 pixels at 20x',
            'resize_choice': 'Bilinear interpolation to 224x224 with ImageNet normalization'
        },
        'graph_construction': {
            'k': best_cfg['k'],
            'symmetrization': 'undirected (E_sym = E union E_T)',
            'edge_features': '3-dim [distance, dx, dy]',
            'self_loops': 'fill_value=mean'
        },
        'seeds': [42, 43, 44]
    }

    os.makedirs('configs', exist_ok=True)
    with open('configs/locked_config.json', 'w') as f:
        json.dump(locked_config, f, indent=2)

    print("\n==========================================", flush=True)
    print("CONFIG LOCKED BEFORE TOUCHING TEST SETS!", flush=True)
    print(f"Timestamp: {timestamp}", flush=True)
    print(f"Chosen Settings: lr={best_cfg['lr']}, lambda={best_cfg['lambda']}, k={best_cfg['k']}", flush=True)
    print(f"Best Val Macro-F1: {best_score:.4f}", flush=True)
    print("Saved to configs/locked_config.json", flush=True)
    print("==========================================", flush=True)

if __name__ == '__main__':
    run_grid_search_and_lock()
