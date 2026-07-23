"""Pure-Python classification metrics (no numpy/torch) so they are unit-testable
anywhere. The training loop passes plain lists of predictions/labels/probabilities.
"""
from collections import Counter


def accuracy(preds, labels):
    if not labels:
        return 0.0
    correct = sum(1 for p, y in zip(preds, labels) if p == y)
    return correct / len(labels)


def confusion_matrix(preds, labels, num_classes):
    m = [[0] * num_classes for _ in range(num_classes)]
    for p, y in zip(preds, labels):
        m[y][p] += 1
    return m


def per_class_f1(preds, labels, num_classes):
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for p, y in zip(preds, labels):
        if p == y:
            tp[y] += 1
        else:
            fp[p] += 1
            fn[y] += 1
    f1s = []
    for c in range(num_classes):
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1s.append((2 * tp[c] / denom) if denom else 0.0)
    return f1s


def macro_f1(preds, labels, num_classes):
    f1s = per_class_f1(preds, labels, num_classes)
    return sum(f1s) / len(f1s) if f1s else 0.0


def _auc_binary(scores, binary_labels):
    """AUROC via the Mann-Whitney U statistic, with tie-corrected average ranks."""
    n_pos = sum(binary_labels)
    n_neg = len(binary_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")  # undefined for a single class
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    sum_ranks_pos = sum(r for r, y in zip(ranks, binary_labels) if y == 1)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def macro_auroc(probs, labels, num_classes):
    """One-vs-rest macro AUROC. `probs` is a list of per-class probability lists."""
    aucs = []
    for c in range(num_classes):
        scores = [row[c] for row in probs]
        binary = [1 if y == c else 0 for y in labels]
        auc = _auc_binary(scores, binary)
        if auc == auc:  # skip NaN (class absent in this split)
            aucs.append(auc)
    return sum(aucs) / len(aucs) if aucs else float("nan")


def summarize(preds, labels, probs, num_classes):
    return {
        "accuracy": round(accuracy(preds, labels), 4),
        "macro_f1": round(macro_f1(preds, labels, num_classes), 4),
        "macro_auroc": round(macro_auroc(probs, labels, num_classes), 4),
    }
