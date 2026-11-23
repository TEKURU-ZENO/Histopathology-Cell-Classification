import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import glob
import json
import numpy as np
import pandas as pd
import h5py
from metrics.evaluation import (
    compute_macro_f1, compute_ece, compute_brier_score,
    compute_consistency_metrics, compute_c_err_with_null
)

def compute_metrics_for_all_runs(pred_dir='outputs/predictions', h5_path='data/processed/features_cache.h5'):
    print(f"Computing metrics from predictions in {pred_dir}...")
    
    # We will compute per-tile metrics to support slide/unit-level statistical analysis
    # and overall aggregate metrics
    all_pred_files = glob.glob(os.path.join(pred_dir, "*.npz"))
    print(f"Found {len(all_pred_files)} prediction files.")
    
    results = []
    
    # Pre-cache graph structures for Cerr calculation
    k_val = 6
    if os.path.exists('configs/locked_config.json'):
        with open('configs/locked_config.json') as f:
            k_val = json.load(f).get('chosen_hyperparameters', {}).get('k', 6)
            
    graph_cache = {}
    with h5py.File(h5_path, 'r') as h5:
        # Lizard test
        for fn in h5['test_lizard'].keys():
            k_key = f'edge_index_k{k_val}' if f'edge_index_k{k_val}' in h5['test_lizard'][fn] else 'edge_index_k8'
            graph_cache[('lizard', fn)] = h5['test_lizard'][fn][k_key][:]
        # CoNSeP test
        for fn in h5['test_consep'].keys():
            k_key = f'edge_index_k{k_val}' if f'edge_index_k{k_val}' in h5['test_consep'][fn] else 'edge_index_k8'
            graph_cache[('consep', fn)] = h5['test_consep'][fn][k_key][:]

    # Group files by model_type, gnn_type, seed, dataset
    file_map = {}
    for pf in all_pred_files:
        base = os.path.splitext(os.path.basename(pf))[0]
        # format: {model_type}_{gnn_type}_s{seed}_{dataset}_{condition}
        parts = base.split('_')
        m_type = parts[0]
        g_type = parts[1]
        seed_str = parts[2]
        dset = parts[3]
        cond = "_".join(parts[4:])
        
        key = (m_type, g_type, seed_str, dset)
        if key not in file_map:
            file_map[key] = {}
        file_map[key][cond] = pf

    print(f"Grouped into {len(file_map)} evaluation runs.")

    for (m_type, g_type, seed_str, dset), cond_files in file_map.items():
        if 'clean' not in cond_files:
            continue
            
        clean_data = np.load(cond_files['clean'])
        clean_probs = clean_data['probs']
        clean_preds = clean_data['preds']
        targets = clean_data['targets']
        tile_ids = clean_data['tile_ids']
        unique_tiles = np.unique(tile_ids)

        clean_f1 = compute_macro_f1(targets, clean_preds)
        clean_ece = compute_ece(clean_probs, targets)
        clean_brier = compute_brier_score(clean_probs, targets)

        # Slide-level clean metrics
        slide_clean_f1s = {}
        for t in unique_tiles:
            mask = (tile_ids == t)
            slide_clean_f1s[t] = compute_macro_f1(targets[mask], clean_preds[mask])

        # Evaluate across conditions
        for cond, pf in cond_files.items():
            data = np.load(pf)
            probs = data['probs']
            preds = data['preds']

            f1 = compute_macro_f1(targets, preds)
            delta_f1 = clean_f1 - f1
            ece = compute_ece(probs, targets)
            brier = compute_brier_score(probs, targets)

            a_cons, cer = compute_consistency_metrics(clean_preds, preds, targets)

            # Spatial error clustering Cerr on graph (for clean and perturbed)
            # Compute over all tiles with 1,000 shuffles and empirical p-value
            c_err_obs_list, c_err_null_list, c_err_pval_list = [], [], []
            slide_cerr_dict = {}
            for t in unique_tiles:
                mask = (tile_ids == t)
                edge_idx = graph_cache.get((dset, t))
                if edge_idx is not None:
                    tile_errors = (preds[mask] != targets[mask])
                    null_res = compute_c_err_with_null(edge_idx, tile_errors, n_shuffles=1000, seed=42)
                    c_err_obs_list.append(null_res['c_err_obs'])
                    c_err_null_list.append(null_res['c_err_null_mean'])
                    c_err_pval_list.append(null_res['p_value'])
                    slide_cerr_dict[t] = null_res

            avg_cerr_obs = float(np.mean(c_err_obs_list)) if c_err_obs_list else 0.0
            avg_cerr_null = float(np.mean(c_err_null_list)) if c_err_null_list else 0.0
            avg_cerr_pval = float(np.mean(c_err_pval_list)) if c_err_pval_list else 1.0

            # Slide-level scores for statistical tests
            slide_f1_dict = {}
            slide_delta_f1_dict = {}
            slide_ece_dict = {}
            slide_brier_dict = {}

            for t in unique_tiles:
                mask = (tile_ids == t)
                t_f1 = compute_macro_f1(targets[mask], preds[mask])
                slide_f1_dict[t] = t_f1
                slide_delta_f1_dict[t] = slide_clean_f1s[t] - t_f1
                slide_ece_dict[t] = compute_ece(probs[mask], targets[mask])
                slide_brier_dict[t] = compute_brier_score(probs[mask], targets[mask])

            results.append({
                'model_type': m_type,
                'gnn_type': g_type,
                'seed': int(seed_str.replace('s', '')),
                'dataset': dset,
                'condition': cond,
                'macro_f1': f1,
                'clean_f1': clean_f1,
                'delta_f1': delta_f1,
                'ece': ece,
                'brier': brier,
                'a_cons': a_cons,
                'cer': cer,
                'c_err_obs': avg_cerr_obs,
                'c_err_null': avg_cerr_null,
                'c_err_pval': avg_cerr_pval,
                'slide_f1s': slide_f1_dict,
                'slide_delta_f1s': slide_delta_f1_dict,
                'slide_eces': slide_ece_dict,
                'slide_briers': slide_brier_dict,
                'slide_cerrs': slide_cerr_dict
            })

    df_res = pd.DataFrame(results)
    os.makedirs('outputs/metrics', exist_ok=True)
    df_res.to_json('outputs/metrics/detailed_metrics.json', orient='records', indent=2)
    # Save flat summary without dicts
    flat_cols = [c for c in df_res.columns if not c.startswith('slide_')]
    df_res[flat_cols].to_csv('outputs/metrics/summary_metrics.csv', index=False)
    print("Metrics computation complete. Saved detailed_metrics.json and summary_metrics.csv")
    return df_res

if __name__ == '__main__':
    compute_metrics_for_all_runs()
