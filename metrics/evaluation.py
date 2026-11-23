import numpy as np
import torch
from sklearn.metrics import f1_score

def compute_macro_f1(y_true, y_pred, num_classes=3):
    return float(f1_score(y_true, y_pred, average='macro', labels=list(range(num_classes))))

def compute_ece(probs, y_true, n_bins=10):
    """
    Expected Calibration Error with M=10 equal-width confidence bins.
    probs: (N, C) predicted probabilities
    y_true: (N,) ground-truth class labels
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / N) * np.abs(bin_acc - bin_conf)

    return float(ece)

def compute_brier_score(probs, y_true, num_classes=3):
    """
    Multiclass Brier Score: (1/N) * sum_i sum_c (p_ic - 1[y_i == c])^2
    """
    N = len(y_true)
    one_hot = np.zeros((N, num_classes), dtype=np.float32)
    one_hot[np.arange(N), y_true] = 1.0
    bs = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return float(bs)

def compute_consistency_metrics(clean_preds, pert_preds, y_true):
    """
    A_cons: agreement between clean and perturbed predictions
    CER: consistent error rate (agree on wrong prediction)
    """
    match = (clean_preds == pert_preds)
    a_cons = float(np.mean(match))
    cer = float(np.mean(match & (clean_preds != y_true)))
    return a_cons, cer

def compute_c_err(edge_index, errors):
    """
    Join-count statistic Cerr: fraction of edges whose both ends are misclassified.
    edge_index: (2, E) or list of (src, dst)
    errors: (N,) boolean array where True indicates misclassification
    """
    if len(errors) == 0:
        return 0.0
    
    if isinstance(edge_index, np.ndarray):
        src, dst = edge_index[0], edge_index[1]
    else:
        src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()

    E = len(src)
    if E == 0:
        return 0.0

    both_error = errors[src] & errors[dst]
    return float(np.sum(both_error) / E)

try:
    import numba
    @numba.njit
    def _numba_null_cerr(src, dst, N, n_errors, n_shuffles, E, seed=42):
        np.random.seed(seed)
        scores = np.empty(n_shuffles, dtype=np.float32)
        arr = np.zeros(N, dtype=numba.boolean)
        indices = np.arange(N)
        for i in range(n_shuffles):
            arr.fill(False)
            for j in range(n_errors):
                pick = np.random.randint(j, N)
                indices[j], indices[pick] = indices[pick], indices[j]
                arr[indices[j]] = True
            cnt = 0
            for e in range(E):
                if arr[src[e]] and arr[dst[e]]:
                    cnt += 1
            scores[i] = cnt / E
        return scores
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

def compute_c_err_with_null(edge_index, errors, n_shuffles=1000, seed=42):
    """
    Computes observed Cerr and compares against a null distribution
    from n_shuffles random error permutations on the same graph.
    """
    c_err_obs = compute_c_err(edge_index, errors)
    n_errors = int(np.sum(errors))
    N = len(errors)

    if isinstance(edge_index, torch.Tensor):
        src, dst = edge_index[0].cpu().numpy().astype(np.int64), edge_index[1].cpu().numpy().astype(np.int64)
    else:
        src, dst = edge_index[0].astype(np.int64), edge_index[1].astype(np.int64)
    E = len(src)

    if N <= 1 or n_errors == 0 or n_errors == N or E == 0:
        return {
            'c_err_obs': c_err_obs,
            'c_err_null_mean': c_err_obs,
            'c_err_null_std': 0.0,
            'c_err_diff': 0.0,
            'p_value': 1.0
        }

    if HAS_NUMBA:
        null_scores = _numba_null_cerr(src, dst, N, n_errors, n_shuffles, E, seed=seed)
    else:
        rng = np.random.RandomState(seed)
        null_scores = np.empty(n_shuffles, dtype=np.float32)
        shuffled = np.zeros(N, dtype=bool)
        for i in range(n_shuffles):
            shuffled.fill(False)
            perm = rng.choice(N, size=n_errors, replace=False)
            shuffled[perm] = True
            both = shuffled[src] & shuffled[dst]
            null_scores[i] = float(np.sum(both) / E)

    null_mean = float(np.mean(null_scores))
    null_std = float(np.std(null_scores))
    p_val = float(np.mean(null_scores >= c_err_obs))

    return {
        'c_err_obs': c_err_obs,
        'c_err_null_mean': null_mean,
        'c_err_null_std': null_std,
        'c_err_diff': c_err_obs - null_mean,
        'p_value': p_val
    }
