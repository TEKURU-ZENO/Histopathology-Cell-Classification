import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import h5py
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from models.mlp_model import MLPHead
from models.gatv2_model import GATv2Head, SAGEHead
from losses.consistency_loss import TotalConsistencyLoss
from metrics.evaluation import compute_macro_f1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_locked_hyperparameters(config_path='configs/locked_config.json'):
    with open(config_path) as f:
        cfg = json.load(f)
    print(f"Loaded locked config locked at {cfg.get('timestamp_utc')}", flush=True)
    hp = cfg['chosen_hyperparameters']
    return hp['lr'], hp['lambda'], hp['k']

def preload_feature_dataset(h5_path='data/processed/features_cache.h5', k=6):
    print("Preloading feature embeddings and graphs from HDF5...", flush=True)
    train_data = []
    val_data = []
    
    with h5py.File(h5_path, 'r') as h5:
        # Calculate class weights from training set
        all_train_labels = []
        for fn in h5['train'].keys():
            all_train_labels.append(h5['train'][fn]['labels'][:])
        all_train_labels = np.concatenate(all_train_labels)
        
        C = 3
        counts = np.bincount(all_train_labels, minlength=C)
        class_weights = len(all_train_labels) / (C * counts.astype(np.float32))
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

        # Train data
        for fn in h5['train'].keys():
            grp = h5['train'][fn]
            train_data.append({
                'name': fn,
                'labels': grp['labels'][:],
                'emb_clean': grp['emb_clean'][:],
                'emb_v1': grp['emb_v1'][:],
                'emb_v2': grp['emb_v2'][:],
                'emb_v3': grp['emb_v3'][:],
                'emb_v4': grp['emb_v4'][:],
                'edge_index': grp[f'edge_index_k{k}'][:],
                'edge_attr': grp[f'edge_attr_k{k}'][:]
            })

        # Val data
        for fn in h5['val'].keys():
            grp = h5['val'][fn]
            val_data.append({
                'name': fn,
                'labels': grp['labels'][:],
                'emb_clean': grp['emb_clean'][:],
                'edge_index': grp[f'edge_index_k{k}'][:],
                'edge_attr': grp[f'edge_attr_k{k}'][:]
            })

    print(f"Preloaded {len(train_data)} train and {len(val_data)} val regions into RAM.", flush=True)
    return train_data, val_data, class_weights_t

def train_and_evaluate_model(
    model_type, gnn_type, seed, 
    train_data, val_data, class_weights,
    lr=1e-3, lambda_cons=0.1, k=6, max_epochs=25, patience=5,
    h5_path='data/processed/features_cache.h5',
    out_pred_dir='outputs/predictions'
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(out_pred_dir, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    loss_fn = TotalConsistencyLoss(class_weights=class_weights, lambda_cons=lambda_cons)

    # Initialize model
    if model_type in ['m2', 'm3']:
        model = MLPHead(in_features=512, hidden_features=128, num_classes=3, dropout=0.2).to(device)
    elif model_type in ['m4', 'm5']:
        if gnn_type == 'gatv2':
            model = GATv2Head(in_features=512, hidden_heads=4, hidden_per_head=16, 
                              out_features=32, num_classes=3, edge_dim=3, dropout=0.25).to(device)
        elif gnn_type == 'sage':
            model = SAGEHead(in_features=512, hidden_features=64, 
                             out_features=32, num_classes=3, dropout=0.25).to(device)
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_ckpt_path = f"checkpoints/{model_type}_{gnn_type}_s{seed}_best.pt"

    # Train loop
    indices = np.arange(len(train_data))
    for epoch in range(1, max_epochs + 1):
        model.train()
        np.random.shuffle(indices)
        total_loss = 0.0

        for idx in indices:
            item = train_data[idx]
            labels = torch.from_numpy(item['labels']).long().to(device)
            emb_clean = torch.from_numpy(item['emb_clean']).float().to(device)
            
            # Random perturbed view among v1..v4
            v_choice = f'emb_v{np.random.randint(1, 5)}'
            emb_pert = torch.from_numpy(item[v_choice]).float().to(device)

            edge_idx = torch.from_numpy(item['edge_index']).long().to(device)
            edge_attr = torch.from_numpy(item['edge_attr']).float().to(device)

            optimizer.zero_grad()

            if model_type == 'm2':
                # M2: Standard Cross-Entropy on random view
                emb_in = emb_clean if np.random.rand() > 0.5 else emb_pert
                logits = model(emb_in)
                loss, _, _ = loss_fn(logits, labels)
            elif model_type == 'm3':
                # M3: MLP with paired Consistency
                logits_c = model(emb_clean)
                logits_p = model(emb_pert)
                loss, _, _ = loss_fn(logits_c, labels, logits_p)
            elif model_type == 'm4':
                # M4: GNN with Standard Cross-Entropy on random view
                emb_in = emb_clean if np.random.rand() > 0.5 else emb_pert
                if gnn_type == 'gatv2':
                    logits = model(emb_in, edge_idx, edge_attr)
                else:
                    logits = model(emb_in, edge_idx)
                loss, _, _ = loss_fn(logits, labels)
            elif model_type == 'm5':
                # M5: GNN with paired Consistency on identical graph
                if gnn_type == 'gatv2':
                    logits_c = model(emb_clean, edge_idx, edge_attr)
                    logits_p = model(emb_pert, edge_idx, edge_attr)
                else:
                    logits_c = model(emb_clean, edge_idx)
                    logits_p = model(emb_pert, edge_idx)
                loss, _, _ = loss_fn(logits_c, labels, logits_p)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation evaluation
        model.eval()
        val_preds_all, val_targets_all = [], []
        with torch.no_grad():
            for item in val_data:
                v_labels = item['labels']
                v_emb = torch.from_numpy(item['emb_clean']).float().to(device)
                v_edge_idx = torch.from_numpy(item['edge_index']).long().to(device)
                v_edge_attr = torch.from_numpy(item['edge_attr']).float().to(device)

                if model_type in ['m2', 'm3']:
                    v_logits = model(v_emb)
                else:
                    if gnn_type == 'gatv2':
                        v_logits = model(v_emb, v_edge_idx, v_edge_attr)
                    else:
                        v_logits = model(v_emb, v_edge_idx)

                v_preds = torch.argmax(v_logits, dim=-1).cpu().numpy()
                val_preds_all.append(v_preds)
                val_targets_all.append(v_labels)

        val_preds_cat = np.concatenate(val_preds_all)
        val_targets_cat = np.concatenate(val_targets_all)
        val_f1 = compute_macro_f1(val_targets_cat, val_preds_cat)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    print(f"[{model_type.upper()}-{gnn_type} Seed {seed}] Best Val Macro-F1: {best_val_f1:.4f} at epoch {best_epoch}", flush=True)

    # Load best checkpoint for test evaluation
    model.load_state_dict(torch.load(best_ckpt_path))
    model.eval()

    # Run inference on test sets
    conditions = ['clean', 'macenko_mild', 'macenko_moderate', 'macenko_strong', 'randstainna']
    test_dsets = ['test_lizard', 'test_consep']

    with h5py.File(h5_path, 'r') as h5:
        for dset_key in test_dsets:
            dset_name = 'lizard' if 'lizard' in dset_key else 'consep'
            dset_grp = h5[dset_key]
            tile_keys = list(dset_grp.keys())

            for cond in conditions:
                all_probs, all_preds, all_targets, all_centroids, all_tile_ids = [], [], [], [], []

                with torch.no_grad():
                    for t_name in tile_keys:
                        grp = dset_grp[t_name]
                        t_labels = grp['labels'][:]
                        t_centroids = grp['centroids'][:]
                        t_emb = torch.from_numpy(grp[f'emb_{cond}'][:]).float().to(device)
                        t_edge_idx = torch.from_numpy(grp[f'edge_index_k{k}'][:]).long().to(device)
                        t_edge_attr = torch.from_numpy(grp[f'edge_attr_k{k}'][:]).float().to(device)

                        if model_type in ['m2', 'm3']:
                            t_logits = model(t_emb)
                        else:
                            if gnn_type == 'gatv2':
                                t_logits = model(t_emb, t_edge_idx, t_edge_attr)
                            else:
                                t_logits = model(t_emb, t_edge_idx)

                        t_probs = torch.softmax(t_logits, dim=-1).cpu().numpy()
                        t_preds = np.argmax(t_probs, axis=-1)

                        all_probs.append(t_probs)
                        all_preds.append(t_preds)
                        all_targets.append(t_labels)
                        all_centroids.append(t_centroids)
                        all_tile_ids.extend([t_name] * len(t_labels))

                save_pred_path = os.path.join(
                    out_pred_dir,
                    f"{model_type}_{gnn_type}_s{seed}_{dset_name}_{cond}.npz"
                )
                np.savez_compressed(
                    save_pred_path,
                    probs=np.vstack(all_probs),
                    preds=np.concatenate(all_preds),
                    targets=np.concatenate(all_targets),
                    centroids=np.vstack(all_centroids),
                    tile_ids=np.array(all_tile_ids),
                    model_type=model_type,
                    gnn_type=gnn_type,
                    seed=seed,
                    dataset=dset_name,
                    condition=cond
                )

    print(f"[{model_type.upper()}-{gnn_type} Seed {seed}] Saved predictions for {len(conditions)} conditions across Lizard & CoNSeP.", flush=True)
    return best_val_f1

if __name__ == '__main__':
    lr, lambda_cons, k = load_locked_hyperparameters('configs/locked_config.json')
    print(f"--- Training M2 to M5 with Locked Config: lr={lr}, lambda={lambda_cons}, k={k} ---", flush=True)

    train_data, val_data, class_weights = preload_feature_dataset('data/processed/features_cache.h5', k=k)

    seeds = [42, 43, 44]
    
    # 1. Main Models: M2, M3, M4 (GATv2), M5 (GATv2)
    models_to_run = [
        ('m2', 'mlp'),
        ('m3', 'mlp'),
        ('m4', 'gatv2'),
        ('m5', 'gatv2'),
        ('m4', 'sage'),
        ('m5', 'sage')
    ]

    for model_type, gnn_type in models_to_run:
        for seed in seeds:
            print(f"\n>>> Running {model_type.upper()}-{gnn_type} [Seed {seed}] <<<", flush=True)
            train_and_evaluate_model(
                model_type=model_type,
                gnn_type=gnn_type,
                seed=seed,
                train_data=train_data,
                val_data=val_data,
                class_weights=class_weights,
                lr=lr,
                lambda_cons=lambda_cons,
                k=k,
                max_epochs=25,
                patience=5
            )

    print("\n=======================================================", flush=True)
    print("ALL M2-M5 RUNS COMPLETE! Predictions saved to outputs/predictions/", flush=True)
    print("=======================================================", flush=True)
