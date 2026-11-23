import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import json
import time
from training.tune_and_lock_config import run_grid_search_and_lock
from training.train_eval_framework import train_one_model
from training.train_m1 import train_m1_model
from outputs.compute_all_metrics import compute_metrics_for_all_runs
from outputs.generate_paper_tables import generate_all_tables_and_verdicts
from outputs.plot_figure2 import plot_figure_2

def main():
    print("=================================================================")
    print("HISTOPATHOLOGY CELL PHENOTYPING: TIER 1 EXPERIMENTAL EXECUTION")
    print("=================================================================")

    # 1. TUNING & CONFIG LOCKING
    print("\n--- STAGE 1: Hyperparameter Tuning on Validation Split ---")
    run_grid_search_and_lock()

    with open('configs/locked_config.json') as f:
        locked_cfg = json.load(f)

    chosen_lr = locked_cfg['chosen_hyperparameters']['lr']
    chosen_lambda = locked_cfg['chosen_hyperparameters']['lambda']
    chosen_k = locked_cfg['chosen_hyperparameters']['k']
    seeds = locked_cfg['seeds'] # [42, 43, 44]

    print(f"\nUsing locked config: lr={chosen_lr}, lambda={chosen_lambda}, k={chosen_k}")

    # 2. TRAIN M1 (RESNET-18 BASELINE)
    print("\n--- STAGE 2: Training M1 (ResNet-18 Full Fine-Tuning) ---")
    for s in seeds:
        print(f"  Training M1 Seed {s}...")
        train_m1_model(seed=s, max_epochs=8, lr=1e-4)

    # 3. TRAIN M2 TO M5 (FAST HEADS ON CACHED FEATURES)
    print("\n--- STAGE 3: Training M2 to M5 on Cached Features ---")
    models_to_run = [
        ('m2', 'mlp'),
        ('m3', 'mlp'),
        ('m4', 'gatv2'),
        ('m5', 'gatv2'),
        ('m4', 'sage'),
        ('m5', 'sage')
    ]

    for m_type, g_type in models_to_run:
        for s in seeds:
            print(f"  Training {m_type.upper()}-{g_type.upper()} Seed {s}...")
            train_one_model(
                model_type=m_type,
                seed=s,
                lr=chosen_lr,
                lambda_cons=chosen_lambda,
                k=chosen_k,
                max_epochs=25,
                patience=5,
                gnn_type=g_type
            )

    # 4. COMPUTE ALL METRICS
    print("\n--- STAGE 4: Computing Evaluation Metrics ---")
    compute_metrics_for_all_runs()

    # 5. STATISTICAL TESTS & PUBLICATION TABLES
    print("\n--- STAGE 5: Generating Statistical Tests & Tables ---")
    generate_all_tables_and_verdicts()

    # 6. PLOT FIGURE 2
    print("\n--- STAGE 6: Plotting Figure 2 ---")
    plot_figure_2()

    print("\n=================================================================")
    print("ALL EXPERIMENTS, METRICS, TABLES, AND FIGURES SUCCESSFULLY COMPLETED!")
    print("=================================================================")

if __name__ == '__main__':
    main()
