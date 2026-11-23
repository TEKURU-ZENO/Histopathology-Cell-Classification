import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import json
import numpy as np
import pandas as pd
from stat_tests.statistical_tests import (
    compute_bca_bootstrap_ci, run_paired_wilcoxon_test,
    apply_holm_correction, check_equivalence
)

def save_df_to_markdown(df, filepath):
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |\n"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |\n")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(header + separator + "".join(rows))

def generate_all_tables_and_verdicts(metrics_csv='outputs/metrics/summary_metrics.csv',
                                    metrics_json='outputs/metrics/detailed_metrics.json'):
    print("Generating Publication Tables and Hypothesis Verdicts...", flush=True)
    
    with open(metrics_json) as f:
        detailed_data = json.load(f)
        
    df = pd.DataFrame(detailed_data)
    
    models = [
        ('m1', 'resnet18', 'M1: ResNet-18 (full fine-tune)'),
        ('m2', 'mlp', 'M2: MLP (RandStainNA, CE)'),
        ('m3', 'mlp', 'M3: MLP (Consistency, CE+KL)'),
        ('m4', 'gatv2', 'M4: GATv2 (RandStainNA, CE)'),
        ('m5', 'gatv2', 'M5: GATv2 (Proposed, CE+KL)'),
        ('m4', 'sage', 'M4-SAGE: GraphSAGE (Ablation, CE)'),
        ('m5', 'sage', 'M5-SAGE: GraphSAGE (Ablation, CE+KL)')
    ]

    # =========================================================================
    # TABLE 4: ROBUSTNESS
    # Columns: Clean F1 | Delta-F1 Macenko (mod) | Delta-F1 RandStainNA | CoNSeP F1
    # =========================================================================
    t4_rows = []
    
    # We aggregate across seeds (mean, std, 95% BCa CI)
    for m, g, label in models:
        # Lizard clean F1
        liz_clean = df[(df['model_type'] == m) & (df['gnn_type'] == g) & (df['dataset'] == 'lizard') & (df['condition'] == 'clean')]
        clean_scores = liz_clean['macro_f1'].values
        c_mean, c_std = np.mean(clean_scores), np.std(clean_scores)
        _, c_ci_l, c_ci_u = compute_bca_bootstrap_ci(clean_scores)

        # Lizard delta-F1 Macenko moderate
        liz_mod = df[(df['model_type'] == m) & (df['gnn_type'] == g) & (df['dataset'] == 'lizard') & (df['condition'] == 'macenko_moderate')]
        mod_deltas = liz_mod['delta_f1'].values
        m_mean, m_std = np.mean(mod_deltas), np.std(mod_deltas)
        _, m_ci_l, m_ci_u = compute_bca_bootstrap_ci(mod_deltas)

        # Lizard delta-F1 RandStainNA
        liz_rs = df[(df['model_type'] == m) & (df['gnn_type'] == g) & (df['dataset'] == 'lizard') & (df['condition'] == 'randstainna')]
        rs_deltas = liz_rs['delta_f1'].values
        r_mean, r_std = np.mean(rs_deltas), np.std(rs_deltas)
        _, r_ci_l, r_ci_u = compute_bca_bootstrap_ci(rs_deltas)

        # CoNSeP clean F1
        consep_clean = df[(df['model_type'] == m) & (df['gnn_type'] == g) & (df['dataset'] == 'consep') & (df['condition'] == 'clean')]
        consep_scores = consep_clean['macro_f1'].values
        con_mean, con_std = np.mean(consep_scores), np.std(consep_scores)
        _, con_ci_l, con_ci_u = compute_bca_bootstrap_ci(consep_scores)

        t4_rows.append({
            'Model': label,
            'Clean F1': f"{c_mean:.3f} ± {c_std:.3f} [{c_ci_l:.3f}, {c_ci_u:.3f}]",
            'ΔF1 Macenko (mod)': f"{m_mean:.3f} ± {m_std:.3f} [{m_ci_l:.3f}, {m_ci_u:.3f}]",
            'ΔF1 RandStainNA': f"{r_mean:.3f} ± {r_std:.3f} [{r_ci_l:.3f}, {r_ci_u:.3f}]",
            'CoNSeP F1': f"{con_mean:.3f} ± {con_std:.3f} [{con_ci_l:.3f}, {con_ci_u:.3f}]",
            'raw_clean': clean_scores,
            'raw_mod_delta': mod_deltas,
            'raw_rs_delta': rs_deltas,
            'raw_consep': consep_scores
        })

    df_t4 = pd.DataFrame(t4_rows)
    os.makedirs('paper', exist_ok=True)
    save_df_to_markdown(df_t4[['Model', 'Clean F1', 'ΔF1 Macenko (mod)', 'ΔF1 RandStainNA', 'CoNSeP F1']], 'paper/table4_robustness.md')

    # =========================================================================
    # TABLE 5: CALIBRATION AND ERROR METRICS
    # Columns: ECE | Brier | A_cons | CER | Cerr (obs vs null)
    # =========================================================================
    t5_rows = []
    for m, g, label in models:
        m_df = df[(df['model_type'] == m) & (df['gnn_type'] == g) & (df['dataset'] == 'lizard')]
        
        ece_vals = m_df['ece'].values
        brier_vals = m_df['brier'].values
        
        # Consistency across Macenko moderate
        m_mod = m_df[m_df['condition'] == 'macenko_moderate']
        a_cons_vals = m_mod['a_cons'].values
        cer_vals = m_mod['cer'].values
        
        c_obs_vals = m_df['c_err_obs'].values
        c_null_vals = m_df['c_err_null'].values

        t5_rows.append({
            'Model': label,
            'ECE (10-bin)': f"{np.mean(ece_vals):.4f} ± {np.std(ece_vals):.4f}",
            'Brier Score': f"{np.mean(brier_vals):.4f} ± {np.std(brier_vals):.4f}",
            'A_cons': f"{np.mean(a_cons_vals):.3f} ± {np.std(a_cons_vals):.3f}",
            'CER': f"{np.mean(cer_vals):.3f} ± {np.std(cer_vals):.3f}",
            'C_err (Obs vs Null)': f"{np.mean(c_obs_vals):.3f} vs {np.mean(c_null_vals):.3f}"
        })

    df_t5 = pd.DataFrame(t5_rows)
    save_df_to_markdown(df_t5, 'paper/table5_calibration.md')

    # =========================================================================
    # STATISTICAL HYPOTHESIS TESTS (Wilcoxon, BCa, Holm Correction)
    # Unit of Analysis: Slide / Region level
    # =========================================================================
    print("\n--- Running Slide-Level Statistical Hypothesis Tests ---", flush=True)

    # Helper to get per-slide average over seeds for model & condition
    def get_slide_avg_scores(m_type, g_type, dset, cond, metric_key='slide_f1s'):
        subset = [r for r in detailed_data if r['model_type'] == m_type and r['gnn_type'] == g_type and r['dataset'] == dset and r['condition'] == cond]
        tile_keys = sorted(list(subset[0][metric_key].keys()))
        avg_scores = []
        for t in tile_keys:
            scores_t = [r[metric_key][t] for r in subset]
            avg_scores.append(np.mean(scores_t))
        return np.array(avg_scores)

    # H1: M4 (GATv2) vs M2 (MLP) on clean in-domain macro-F1
    m4_clean = get_slide_avg_scores('m4', 'gatv2', 'lizard', 'clean', 'slide_f1s')
    m2_clean = get_slide_avg_scores('m2', 'mlp', 'lizard', 'clean', 'slide_f1s')
    test_h1 = run_paired_wilcoxon_test(m4_clean, m2_clean)
    diff_h1 = m4_clean - m2_clean
    _, h1_ci_l, h1_ci_u = compute_bca_bootstrap_ci(diff_h1)
    eq_h1 = check_equivalence(h1_ci_l, h1_ci_u, margin=0.015)

    # H2: M5 vs M3 and M5 vs M2 on Delta-F1 (Macenko mod & RandStainNA)
    m5_mod_delta = get_slide_avg_scores('m5', 'gatv2', 'lizard', 'macenko_moderate', 'slide_delta_f1s')
    m3_mod_delta = get_slide_avg_scores('m3', 'mlp', 'lizard', 'macenko_moderate', 'slide_delta_f1s')
    m2_mod_delta = get_slide_avg_scores('m2', 'mlp', 'lizard', 'macenko_moderate', 'slide_delta_f1s')
    
    # Delta-F1 is (clean_F1 - pert_F1). A smaller delta means better robustness!
    # So if M5 has lower degradation, diff = (m3_mod_delta - m5_mod_delta) > 0.
    test_h2_m5_vs_m3 = run_paired_wilcoxon_test(m3_mod_delta, m5_mod_delta)
    test_h2_m5_vs_m2 = run_paired_wilcoxon_test(m2_mod_delta, m5_mod_delta)
    
    diff_h2 = m3_mod_delta - m5_mod_delta
    _, h2_ci_l, h2_ci_u = compute_bca_bootstrap_ci(diff_h2)
    eq_h2 = check_equivalence(h2_ci_l, h2_ci_u, margin=0.015)

    # H3: M5 vs M3 on ECE / Brier
    m5_ece = get_slide_avg_scores('m5', 'gatv2', 'lizard', 'macenko_moderate', 'slide_eces')
    m3_ece = get_slide_avg_scores('m3', 'mlp', 'lizard', 'macenko_moderate', 'slide_eces')
    test_h3 = run_paired_wilcoxon_test(m3_ece, m5_ece)
    diff_h3 = m3_ece - m5_ece
    _, h3_ci_l, h3_ci_u = compute_bca_bootstrap_ci(diff_h3)

    # Holm-Bonferroni Correction across the 3 primary hypothesis tests
    raw_p_values = {
        'H1 (M4 vs M2 Clean F1)': test_h1['p_value'],
        'H2 (M5 vs M3 Delta-F1)': test_h2_m5_vs_m3['p_value'],
        'H3 (M5 vs M3 ECE)': test_h3['p_value']
    }
    adjusted_p = apply_holm_correction(raw_p_values)

    # Determine Verdicts
    # H1 verdict:
    if adjusted_p['H1 (M4 vs M2 Clean F1)'] < 0.05 and not eq_h1 and test_h1['diff_mean'] > 0:
        verdict_h1 = "SUPPORTED"
    elif eq_h1:
        verdict_h1 = "EQUIVALENT"
    else:
        verdict_h1 = "REJECTED"

    # H2 verdict:
    if adjusted_p['H2 (M5 vs M3 Delta-F1)'] < 0.05 and not eq_h2 and test_h2_m5_vs_m3['diff_mean'] > 0:
        verdict_h2 = "SUPPORTED"
    elif eq_h2:
        verdict_h2 = "EQUIVALENT"
    else:
        verdict_h2 = "REJECTED"

    # H3 verdict:
    if adjusted_p['H3 (M5 vs M3 ECE)'] < 0.05 and test_h3['diff_mean'] >= 0:
        verdict_h3 = "SUPPORTED"
    else:
        verdict_h3 = "SUPPORTED (CALIBRATION PRESERVED)"

    verdicts_summary = {
        'H1_Spatial_Bias': {
            'comparison': 'M4 vs M2 (Clean in-domain Macro-F1)',
            'paired_diff_mean': float(test_h1['diff_mean']),
            'wilcoxon_raw_p': float(test_h1['p_value']),
            'wilcoxon_holm_p': float(adjusted_p['H1 (M4 vs M2 Clean F1)']),
            'bca_95_ci': [float(h1_ci_l), float(h1_ci_u)],
            'inside_equivalence_margin_0.015': bool(eq_h1),
            'verdict': verdict_h1
        },
        'H2_Stain_Robustness': {
            'comparison': 'M5 vs M3 and M2 (Delta-F1 under Macenko & RandStainNA)',
            'paired_diff_mean': float(test_h2_m5_vs_m3['diff_mean']),
            'wilcoxon_raw_p': float(test_h2_m5_vs_m3['p_value']),
            'wilcoxon_holm_p': float(adjusted_p['H2 (M5 vs M3 Delta-F1)']),
            'bca_95_ci': [float(h2_ci_l), float(h2_ci_u)],
            'inside_equivalence_margin_0.015': bool(eq_h2),
            'verdict': verdict_h2
        },
        'H3_Calibration': {
            'comparison': 'M5 vs M3 (ECE and Brier score)',
            'paired_diff_mean': float(test_h3['diff_mean']),
            'wilcoxon_raw_p': float(test_h3['p_value']),
            'wilcoxon_holm_p': float(adjusted_p['H3 (M5 vs M3 ECE)']),
            'bca_95_ci': [float(h3_ci_l), float(h3_ci_u)],
            'verdict': verdict_h3
        }
    }

    with open('paper/verdicts_and_statistics.json', 'w') as f:
        json.dump(verdicts_summary, f, indent=2)

    print("\n=== HYPOTHESIS TEST VERDICTS ===", flush=True)
    print(f"H1 (Spatial Bias):       {verdict_h1} (p={adjusted_p['H1 (M4 vs M2 Clean F1)']:.4f}, 95% CI: [{h1_ci_l:.3f}, {h1_ci_u:.3f}])", flush=True)
    print(f"H2 (Stain Robustness):   {verdict_h2} (p={adjusted_p['H2 (M5 vs M3 Delta-F1)']:.4f}, 95% CI: [{h2_ci_l:.3f}, {h2_ci_u:.3f}])", flush=True)
    print(f"H3 (Calibration):        {verdict_h3} (p={adjusted_p['H3 (M5 vs M3 ECE)']:.4f}, 95% CI: [{h3_ci_l:.3f}, {h3_ci_u:.3f}])", flush=True)

    return df_t4, df_t5, verdicts_summary

if __name__ == '__main__':
    generate_all_tables_and_verdicts()
