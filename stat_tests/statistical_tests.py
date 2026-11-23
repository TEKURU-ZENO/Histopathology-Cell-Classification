import numpy as np
from scipy import stats
from scipy.stats import norm

def compute_bca_bootstrap_ci(data, stat_fn=np.mean, B=1000, alpha=0.05, seed=42):
    """
    Computes Bias-Corrected and Accelerated (BCa) bootstrap 95% confidence intervals.
    """
    data = np.asarray(data)
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        val = float(data[0])
        return val, val, val

    theta_hat = stat_fn(data)

    rng = np.random.RandomState(seed)
    boot_theta = np.empty(B)
    for b in range(B):
        sample = rng.choice(data, size=n, replace=True)
        boot_theta[b] = stat_fn(sample)

    # Bias correction z0
    prop_less = np.mean(boot_theta < theta_hat)
    prop_less = np.clip(prop_less, 1e-6, 1 - 1e-6)
    z0 = norm.ppf(prop_less)

    # Acceleration factor a using jackknife
    jack_theta = np.empty(n)
    for i in range(n):
        jack_sample = np.delete(data, i)
        jack_theta[i] = stat_fn(jack_sample)

    mean_jack = np.mean(jack_theta)
    num = np.sum((mean_jack - jack_theta) ** 3)
    denom = 6.0 * (np.sum((mean_jack - jack_theta) ** 2) ** 1.5)
    
    if abs(denom) < 1e-10:
        a = 0.0
    else:
        a = num / denom

    # Standard normal quantiles
    z_alpha = norm.ppf(alpha / 2.0)
    z_1_alpha = norm.ppf(1.0 - alpha / 2.0)

    # Adjusted percentiles
    def get_adj_quantile(z_val):
        denom_q = 1.0 - a * (z0 + z_val)
        if abs(denom_q) < 1e-6:
            denom_q = 1e-6
        return norm.cdf(z0 + (z0 + z_val) / denom_q)

    a1 = get_adj_quantile(z_alpha) * 100.0
    a2 = get_adj_quantile(z_1_alpha) * 100.0

    a1 = np.clip(a1, 0.0, 100.0)
    a2 = np.clip(a2, 0.0, 100.0)

    ci_lower = float(np.percentile(boot_theta, a1))
    ci_upper = float(np.percentile(boot_theta, a2))

    return float(theta_hat), ci_lower, ci_upper

def run_paired_wilcoxon_test(scores_a, scores_b):
    """
    Two-sided Wilcoxon signed-rank test on paired differences (a - b).
    """
    diffs = np.asarray(scores_a) - np.asarray(scores_b)
    non_zero = diffs[diffs != 0]
    if len(non_zero) < 5:
        return {'diff_mean': float(np.mean(diffs)), 'statistic': 0.0, 'p_value': 1.0}
        
    res = stats.wilcoxon(diffs, alternative='two-sided')
    return {
        'diff_mean': float(np.mean(diffs)),
        'statistic': float(res.statistic),
        'p_value': float(res.pvalue)
    }

def apply_holm_correction(p_values_dict):
    """
    Applies Holm-Bonferroni correction across a dictionary of test results.
    """
    keys = list(p_values_dict.keys())
    raw_p = [p_values_dict[k] for k in keys]
    
    sorted_indices = np.argsort(raw_p)
    m = len(raw_p)
    
    adj_p = np.zeros(m)
    running_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        p_val = raw_p[idx]
        adj = (m - rank) * p_val
        running_max = max(running_max, adj)
        adj_p[idx] = min(1.0, running_max)
        
    return {keys[i]: float(adj_p[i]) for i in range(m)}

def check_equivalence(ci_lower, ci_upper, margin=0.015):
    """
    Checks if 95% CI is entirely contained within [-margin, +margin].
    """
    return (ci_lower >= -margin) and (ci_upper <= margin)
