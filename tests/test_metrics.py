import math
from src.training.metrics import (
    accuracy, confusion_matrix, per_class_f1, macro_f1, macro_auroc, _auc_binary,
)


def test_accuracy_and_confusion():
    preds = [0, 1, 2, 2]
    labels = [0, 1, 1, 2]
    assert accuracy(preds, labels) == 0.75
    cm = confusion_matrix(preds, labels, 3)
    assert cm[1][1] == 1 and cm[1][2] == 1 and cm[2][2] == 1


def test_f1_perfect_and_imperfect():
    assert macro_f1([0, 1, 2], [0, 1, 2], 3) == 1.0
    f1s = per_class_f1([0, 0], [0, 1], 2)
    assert f1s[0] > 0 and f1s[1] == 0.0


def test_auc_binary_perfect_and_random():
    # perfectly separable -> AUC 1.0
    assert _auc_binary([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    # reversed -> AUC 0.0
    assert _auc_binary([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    # tie handling stays within [0,1]
    a = _auc_binary([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1])
    assert 0.0 <= a <= 1.0


def test_macro_auroc_skips_absent_classes():
    probs = [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]]
    labels = [0, 1, 2]
    auc = macro_auroc(probs, labels, 3)
    assert 0.0 <= auc <= 1.0 and not math.isnan(auc)
