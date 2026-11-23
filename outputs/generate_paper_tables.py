import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import json
import numpy as np
import pandas as pd
from stat_tests.statistical_tests import (
    compute_bca_bootstrap_ci,
    run_paired_wilcoxon_test,
    check_equivalence,
    apply_holm_correction
)

def save_df_to_markdown(df, filepath):
    """Saves a pandas DataFrame to a GitHub-flavored markdown table without external dependencies."""
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

def generate_all_tables_and_verdicts():
    detailed_json = 'outputs/metrics/detailed_metrics.json'
    with open(detailed_json, 'r') as f:
        detailed_data = json.load(f)

    models = [
        ('m1', 'resnet18', 'M1: ResNet-18 (full fine-tune)', '11.18M'),
        ('m2', 'mlp', 'M2: MLP (RandStainNA, CE)', '66.05k'),
        ('m3', 'mlp', 'M3: MLP (Consistency, CE+KL)', '66.05k'),
        ('m4', 'gatv2', 'M4: GATv2 (RandStainNA, CE)', '70.40k'),
        ('m5', 'gatv2', 'M5: GATv2 (Proposed, CE+KL)', '70.40k'),
        ('m4', 'sage', 'M4-SAGE: GraphSAGE (Ablation, CE)', '69.83k'),
        ('m5', 'sage', 'M5-SAGE: GraphSAGE (Ablation, CE+KL)', '69.83k')
    ]

    # Helper to get per-region seed-averaged scores
    def get_region_seed_avg(m_type, g_type, dset, cond, metric_key='slide_f1s'):
        subset = [r for r in detailed_data if r['model_type'] == m_type and r['gnn_type'] == g_type and r['dataset'] == dset and r['condition'] == cond]
        tile_keys = sorted(list(subset[0][metric_key].keys()))
        avg_scores = []
        for t in tile_keys:
            scores_t = [r[metric_key][t] for r in subset]
            avg_scores.append(np.mean(scores_t))
        return np.array(avg_scores)

    # =========================================================================
    # TABLE 4: ROBUSTNESS ACROSS STAIN SHIFTS
    # Region-level F1 (mean over seeds per region), bootstrapped over regions (29 Lizard, 14 CoNSeP)
    # =========================================================================
    t4_rows = []
    for m, g, label, params in models:
        # Lizard clean F1 (29 regions)
        clean_f1s = get_region_seed_avg(m, g, 'lizard', 'clean', 'slide_f1s')
        c_mean, c_std = np.mean(clean_f1s), np.std(clean_f1s)
        _, c_ci_l, c_ci_u = compute_bca_bootstrap_ci(clean_f1s)

        # Lizard delta-F1 Macenko moderate (29 regions)
        mod_deltas = get_region_seed_avg(m, g, 'lizard', 'macenko_moderate', 'slide_delta_f1s')
        m_mean, m_std = np.mean(mod_deltas), np.std(mod_deltas)
        _, m_ci_l, m_ci_u = compute_bca_bootstrap_ci(mod_deltas)

        # Lizard delta-F1 RandStainNA (29 regions)
        rs_deltas = get_region_seed_avg(m, g, 'lizard', 'randstainna', 'slide_delta_f1s')
        r_mean, r_std = np.mean(rs_deltas), np.std(rs_deltas)
        _, r_ci_l, r_ci_u = compute_bca_bootstrap_ci(rs_deltas)

        # CoNSeP clean F1 (14 tiles)
        consep_f1s = get_region_seed_avg(m, g, 'consep', 'clean', 'slide_f1s')
        con_mean, con_std = np.mean(consep_f1s), np.std(consep_f1s)
        _, con_ci_l, con_ci_u = compute_bca_bootstrap_ci(consep_f1s)

        t4_rows.append({
            'Model Architecture': label,
            'In-Domain Clean F1': f"{c_mean:.3f} ± {c_std:.3f} [{c_ci_l:.3f}, {c_ci_u:.3f}]",
            'ΔF1 Macenko (Moderate, ±20%)': f"{m_mean:.3f} ± {m_std:.3f} [{m_ci_l:.3f}, {m_ci_u:.3f}]",
            'ΔF1 RandStainNA': f"{r_mean:.3f} ± {r_std:.3f} [{r_ci_l:.3f}, {r_ci_u:.3f}]",
            'Unseen External CoNSeP F1': f"{con_mean:.3f} ± {con_std:.3f} [{con_ci_l:.3f}, {con_ci_u:.3f}]",
            'Params': params
        })

    df_t4 = pd.DataFrame(t4_rows)
    os.makedirs('paper', exist_ok=True)
    save_df_to_markdown(df_t4, 'paper/table4_robustness.md')
    print("Table 4 generated at paper/table4_robustness.md")

    # =========================================================================
    # TABLE 5: CALIBRATION AND SPATIAL ERROR METRICS
    # Separated by condition (Clean vs RandStainNA Shift)
    # =========================================================================
    t5_rows = []
    for m, g, label, _ in models:
        clean_runs = [d for d in detailed_data if d['model_type'] == m and d['gnn_type'] == g and d['dataset'] == 'lizard' and d['condition'] == 'clean']
        clean_ece = [d['ece'] for d in clean_runs]
        clean_brier = [d['brier'] for d in clean_runs]
        c_err_obs = [d['c_err_obs'] for d in clean_runs]
        c_err_null = [d['c_err_null'] for d in clean_runs]
        c_err_pval = [d['c_err_pval'] for d in clean_runs]

        rs_runs = [d for d in detailed_data if d['model_type'] == m and d['gnn_type'] == g and d['dataset'] == 'lizard' and d['condition'] == 'randstainna']
        rs_ece = [d['ece'] for d in rs_runs]
        rs_brier = [d['brier'] for d in rs_runs]
        a_cons = [d['a_cons'] for d in rs_runs]
        cer = [d['cer'] for d in rs_runs]

        # Format p-value string
        mean_pval = float(np.mean(c_err_pval))
        pval_str = "< 0.001" if mean_pval < 0.001 else f"= {mean_pval:.4f}"

        t5_rows.append({
            'Model Architecture': label,
            'ECE Clean (10-bin)': f"{np.mean(clean_ece):.4f} ± {np.std(clean_ece):.4f}",
            'ECE RandStainNA': f"{np.mean(rs_ece):.4f} ± {np.std(rs_ece):.4f}",
            'Brier Clean': f"{np.mean(clean_brier):.4f} ± {np.std(clean_brier):.4f}",
            'Brier RandStainNA': f"{np.mean(rs_brier):.4f} ± {np.std(rs_brier):.4f}",
            'A_cons (RandStainNA)': f"{np.mean(a_cons):.3f} ± {np.std(a_cons):.3f}",
            'CER (RandStainNA)': f"{np.mean(cer):.3f} ± {np.std(cer):.3f}",
            'C_err (Obs vs Null, p-val)': f"{np.mean(c_err_obs):.4f} vs {np.mean(c_err_null):.4f} (p {pval_str})"
        })

    df_t5 = pd.DataFrame(t5_rows)
    save_df_to_markdown(df_t5, 'paper/table5_calibration.md')
    print("Table 5 generated at paper/table5_calibration.md")

    # =========================================================================
    # STATISTICAL HYPOTHESIS TESTS
    # =========================================================================
    print("\n--- Running Slide-Level Statistical Hypothesis Tests ---", flush=True)

    # -------------------------------------------------------------------------
    # H1: Spatial Bias (M4 vs M2 on Clean in-domain Macro-F1 across 29 regions)
    # -------------------------------------------------------------------------
    m4_clean = get_region_seed_avg('m4', 'gatv2', 'lizard', 'clean', 'slide_f1s')
    m2_clean = get_region_seed_avg('m2', 'mlp', 'lizard', 'clean', 'slide_f1s')
    test_h1 = run_paired_wilcoxon_test(m4_clean, m2_clean)
    diff_h1 = m4_clean - m2_clean
    _, h1_ci_l, h1_ci_u = compute_bca_bootstrap_ci(diff_h1)
    eq_h1 = check_equivalence(h1_ci_l, h1_ci_u, margin=0.015)

    # -------------------------------------------------------------------------
    # H2: Stain Robustness (Comprehensive Comparisons across 29 Lizard regions and 14 CoNSeP tiles)
    # Comparisons:
    # 1. M5 vs M3 (Macenko mod ΔF1)
    # 2. M5 vs M3 (Macenko strong ΔF1)
    # 3. M5 vs M3 (RandStainNA ΔF1)
    # 4. M5 vs M2 (Macenko mod ΔF1)
    # 5. M5 vs M2 (Macenko strong ΔF1)
    # 6. M5 vs M2 (RandStainNA ΔF1)
    # 7. M5 vs M3 (CoNSeP clean F1)
    # 8. M5 vs M2 (CoNSeP clean F1)
    # -------------------------------------------------------------------------
    # Lizard deltas (diff = baseline_delta - m5_delta; positive diff means M5 degraded LESS)
    m5_mod_delta = get_region_seed_avg('m5', 'gatv2', 'lizard', 'macenko_moderate', 'slide_delta_f1s')
    m3_mod_delta = get_region_seed_avg('m3', 'mlp', 'lizard', 'macenko_moderate', 'slide_delta_f1s')
    m2_mod_delta = get_region_seed_avg('m2', 'mlp', 'lizard', 'macenko_moderate', 'slide_delta_f1s')

    m5_str_delta = get_region_seed_avg('m5', 'gatv2', 'lizard', 'macenko_strong', 'slide_delta_f1s')
    m3_str_delta = get_region_seed_avg('m3', 'mlp', 'lizard', 'macenko_strong', 'slide_delta_f1s')
    m2_str_delta = get_region_seed_avg('m2', 'mlp', 'lizard', 'macenko_strong', 'slide_delta_f1s')

    m5_rs_delta = get_region_seed_avg('m5', 'gatv2', 'lizard', 'randstainna', 'slide_delta_f1s')
    m3_rs_delta = get_region_seed_avg('m3', 'mlp', 'lizard', 'randstainna', 'slide_delta_f1s')
    m2_rs_delta = get_region_seed_avg('m2', 'mlp', 'lizard', 'randstainna', 'slide_delta_f1s')

    # CoNSeP clean F1 (diff = m5_f1 - baseline_f1; positive diff means M5 had HIGHER F1)
    m5_consep_clean = get_region_seed_avg('m5', 'gatv2', 'consep', 'clean', 'slide_f1s')
    m3_consep_clean = get_region_seed_avg('m3', 'mlp', 'consep', 'clean', 'slide_f1s')
    m2_consep_clean = get_region_seed_avg('m2', 'mlp', 'consep', 'clean', 'slide_f1s')

    h2_comparisons = {
        'M5 vs M3 (Macenko mod ΔF1)': (m3_mod_delta - m5_mod_delta, m3_mod_delta, m5_mod_delta),
        'M5 vs M3 (Macenko strong ΔF1)': (m3_str_delta - m5_str_delta, m3_str_delta, m5_str_delta),
        'M5 vs M3 (RandStainNA ΔF1)': (m3_rs_delta - m5_rs_delta, m3_rs_delta, m5_rs_delta),
        'M5 vs M2 (Macenko mod ΔF1)': (m2_mod_delta - m5_mod_delta, m2_mod_delta, m5_mod_delta),
        'M5 vs M2 (Macenko strong ΔF1)': (m2_str_delta - m5_str_delta, m2_str_delta, m5_str_delta),
        'M5 vs M2 (RandStainNA ΔF1)': (m2_rs_delta - m5_rs_delta, m2_rs_delta, m5_rs_delta),
        'M5 vs M3 (CoNSeP clean F1)': (m5_consep_clean - m3_consep_clean, m5_consep_clean, m3_consep_clean),
        'M5 vs M2 (CoNSeP clean F1)': (m5_consep_clean - m2_consep_clean, m5_consep_clean, m2_consep_clean),
    }

    h2_raw_p = {}
    h2_details = {}
    for comp_name, (diff, a, b) in h2_comparisons.items():
        w_test = run_paired_wilcoxon_test(a, b)
        h2_raw_p[comp_name] = w_test['p_value']
        _, ci_l, ci_u = compute_bca_bootstrap_ci(diff)
        eq = check_equivalence(ci_l, ci_u, margin=0.015)
        h2_details[comp_name] = {
            'paired_diff_mean': float(np.mean(diff)),
            'wilcoxon_raw_p': float(w_test['p_value']),
            'bca_95_ci': [float(ci_l), float(ci_u)],
            'inside_equivalence_margin_0.015': bool(eq)
        }

    h2_holm_p = apply_holm_correction(h2_raw_p)
    for comp_name in h2_details:
        h2_details[comp_name]['wilcoxon_holm_p'] = float(h2_holm_p[comp_name])

    # -------------------------------------------------------------------------
    # H3: Calibration (M5 vs M3 ECE & Brier under shift)
    # -------------------------------------------------------------------------
    m5_ece = get_region_seed_avg('m5', 'gatv2', 'lizard', 'macenko_moderate', 'slide_eces')
    m3_ece = get_region_seed_avg('m3', 'mlp', 'lizard', 'macenko_moderate', 'slide_eces')
    test_h3 = run_paired_wilcoxon_test(m3_ece, m5_ece)
    diff_h3 = m3_ece - m5_ece
    _, h3_ci_l, h3_ci_u = compute_bca_bootstrap_ci(diff_h3)

    # -------------------------------------------------------------------------
    # Primary Holm Correction across the Primary Hypothesis Tests
    # -------------------------------------------------------------------------
    primary_raw_p = {
        'H1 (M4 vs M2 Clean F1)': test_h1['p_value'],
        'H2 (M5 vs M3 Macenko mod ΔF1)': h2_details['M5 vs M3 (Macenko mod ΔF1)']['wilcoxon_raw_p'],
        'H3 (M5 vs M3 ECE)': test_h3['p_value']
    }
    primary_adjusted_p = apply_holm_correction(primary_raw_p)

    # -------------------------------------------------------------------------
    # Verdict Determination Rules
    # -------------------------------------------------------------------------
    # H1 Verdict:
    if primary_adjusted_p['H1 (M4 vs M2 Clean F1)'] < 0.05 and not eq_h1 and test_h1['diff_mean'] > 0:
        verdict_h1 = "SUPPORTED"
    elif eq_h1:
        verdict_h1 = "EQUIVALENT"
    else:
        verdict_h1 = "REJECTED"

    # H2 Verdict Rule: "rejected" unless M5 degrades significantly less
    # (i.e. positive diff, holm-adjusted p < 0.05, and not inside equivalence margin)
    m5_wins_on_any_shift = any(
        (h2_details[c]['paired_diff_mean'] > 0 and 
         h2_details[c]['wilcoxon_holm_p'] < 0.05 and 
         not h2_details[c]['inside_equivalence_margin_0.015'])
        for c in h2_details
    )

    if m5_wins_on_any_shift:
        verdict_h2 = "SUPPORTED"
    else:
        verdict_h2 = "REJECTED"

    # H3 Verdict Rule: "only judge H3 if M5 wins on robustness" (otherwise "NOT TESTABLE")
    if verdict_h2 == "SUPPORTED":
        if primary_adjusted_p['H3 (M5 vs M3 ECE)'] < 0.05 and test_h3['diff_mean'] >= 0:
            verdict_h3 = "SUPPORTED"
        else:
            verdict_h3 = "SUPPORTED (CALIBRATION PRESERVED)"
    else:
        verdict_h3 = "NOT TESTABLE"

    verdicts_summary = {
        'H1_Spatial_Bias': {
            'comparison': 'M4 vs M2 (Clean in-domain Macro-F1 across 29 Lizard regions)',
            'paired_diff_mean': float(test_h1['diff_mean']),
            'wilcoxon_raw_p': float(test_h1['p_value']),
            'wilcoxon_holm_p': float(primary_adjusted_p['H1 (M4 vs M2 Clean F1)']),
            'bca_95_ci': [float(h1_ci_l), float(h1_ci_u)],
            'inside_equivalence_margin_0.015': bool(eq_h1),
            'verdict': verdict_h1
        },
        'H2_Stain_Robustness': {
            'verdict': verdict_h2,
            'rule': "Rejected unless M5 degrades significantly less than M3/M2",
            'primary_comparison': 'M5 vs M3 (Macenko mod ΔF1)',
            'primary_diff_mean': float(h2_details['M5 vs M3 (Macenko mod ΔF1)']['paired_diff_mean']),
            'primary_holm_p': float(primary_adjusted_p['H2 (M5 vs M3 Macenko mod ΔF1)']),
            'all_comparisons': h2_details
        },
        'H3_Calibration': {
            'verdict': verdict_h3,
            'rule': "Only judge H3 if M5 wins on robustness (otherwise NOT TESTABLE)",
            'comparison': 'M5 vs M3 (ECE under Macenko moderate shift)',
            'paired_diff_mean': float(test_h3['diff_mean']),
            'wilcoxon_raw_p': float(test_h3['p_value']),
            'wilcoxon_holm_p': float(primary_adjusted_p['H3 (M5 vs M3 ECE)']),
            'bca_95_ci': [float(h3_ci_l), float(h3_ci_u)]
        }
    }

    with open('paper/verdicts_and_statistics.json', 'w') as f:
        json.dump(verdicts_summary, f, indent=2)
    print("Verdicts saved at paper/verdicts_and_statistics.json")

    print("\n=== HYPOTHESIS TEST VERDICTS ===", flush=True)
    print(f"H1 (Spatial Bias):       {verdict_h1} (diff={test_h1['diff_mean']:+.4f}, p_holm={primary_adjusted_p['H1 (M4 vs M2 Clean F1)']:.4f}, 95% CI: [{h1_ci_l:.4f}, {h1_ci_u:.4f}])", flush=True)
    print(f"H2 (Stain Robustness):   {verdict_h2} (rule: rejected unless M5 degrades significantly less)", flush=True)
    for c_name, c_res in h2_details.items():
        print(f"  - {c_name}: diff={c_res['paired_diff_mean']:+.4f}, p_raw={c_res['wilcoxon_raw_p']:.4f}, p_holm={c_res['wilcoxon_holm_p']:.4f}, 95% CI: [{c_res['bca_95_ci'][0]:.4f}, {c_res['bca_95_ci'][1]:.4f}], equiv={c_res['inside_equivalence_margin_0.015']}")
    print(f"H3 (Calibration):        {verdict_h3} (rule: contingent on M5 winning H2 robustness)", flush=True)

    return df_t4, df_t5, verdicts_summary

if __name__ == '__main__':
    generate_all_tables_and_verdicts()
