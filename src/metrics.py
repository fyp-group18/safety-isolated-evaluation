import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def hit_rate_at_k(retrieved_ids_list: list[list[str]], gt_ids_list: list[set[str]], k: int = 5) -> float:
    hits = 0
    for retrieved, gt in zip(retrieved_ids_list, gt_ids_list):
        top_k = retrieved[:k]
        if any(rid in gt for rid in top_k):
            hits += 1
    return hits / len(gt_ids_list) if gt_ids_list else 0.0


def mrr(retrieved_ids_list: list[list[str]], gt_ids_list: list[set[str]]) -> float:
    rr_sum = 0.0
    for retrieved, gt in zip(retrieved_ids_list, gt_ids_list):
        for rank, rid in enumerate(retrieved, 1):
            if rid in gt:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(gt_ids_list) if gt_ids_list else 0.0


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def auroc(y_true: list[int], y_scores: list[float]) -> float | None:
    if len(set(y_true)) < 2:
        return None
    try:
        return roc_auc_score(y_true, y_scores)
    except ValueError:
        return None


def ece(y_true: list[int], y_probs: list[float], n_bins: int = 10) -> float:
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    total_ece = 0.0
    for i in range(n_bins):
        mask = (y_probs >= bin_edges[i]) & (y_probs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_probs[mask].mean()
        total_ece += mask.sum() * abs(bin_acc - bin_conf)
    return total_ece / len(y_true) if len(y_true) > 0 else 0.0


def cohens_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    n = len(labels_a)
    if n == 0:
        return 0.0
    categories = sorted(set(labels_a) | set(labels_b))
    k = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}

    confusion = np.zeros((k, k), dtype=int)
    for a, b in zip(labels_a, labels_b):
        confusion[cat_idx[a]][cat_idx[b]] += 1

    po = np.trace(confusion) / n
    row_sums = confusion.sum(axis=1)
    col_sums = confusion.sum(axis=0)
    pe = (row_sums * col_sums).sum() / (n * n)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def krippendorffs_alpha(labels_a: list, labels_b: list) -> float:
    n = len(labels_a)
    if n == 0:
        return 0.0

    categories = sorted(set(labels_a) | set(labels_b))
    cat_idx = {c: i for i, c in enumerate(categories)}

    coincidence = np.zeros((len(categories), len(categories)), dtype=float)
    for a, b in zip(labels_a, labels_b):
        i, j = cat_idx[a], cat_idx[b]
        coincidence[i][j] += 1
        coincidence[j][i] += 1

    n_c = coincidence.sum(axis=1)
    total = coincidence.sum()

    if total == 0:
        return 0.0

    do = 1.0 - np.trace(coincidence) / total
    de = 1.0 - (n_c * n_c).sum() / (total * total)

    if de == 0:
        return 1.0
    return 1.0 - do / de


def safety_coverage(
    extracted_protocols: list[dict],
    ground_truth_protocols: list[dict],
) -> float:
    if not ground_truth_protocols:
        return 1.0

    matched = 0
    for gt in ground_truth_protocols:
        gt_text = gt.get("text", "").lower()
        gt_words = set(gt_text.split())
        for ext in extracted_protocols:
            ext_text = ext.get("text", "").lower()
            ext_words = set(ext_text.split())
            overlap = len(gt_words & ext_words)
            if overlap >= max(1, len(gt_words) * 0.3):
                matched += 1
                break

    return matched / len(ground_truth_protocols)


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0

    rng = np.random.RandomState(seed)
    arr = np.array(values)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))

    means = np.array(means)
    alpha = (1 - ci) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)
    return float(np.mean(arr)), float(lower), float(upper)
