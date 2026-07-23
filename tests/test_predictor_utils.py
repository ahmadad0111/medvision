"""Pure-logic tests for inference helpers (no torch needed)."""
from src.inference.predictor import softmax_topk


def test_softmax_topk_orders_and_labels():
    probs = [0.05, 0.6, 0.05, 0.1, 0.1, 0.05, 0.05]
    top = softmax_topk(probs, k=3)
    assert top[0]["class_id"] == 1
    assert top[0]["label"] == "basal cell carcinoma"
    assert top[0]["prob"] == 0.6
    assert [t["class_id"] for t in top] == [1, 3, 4]  # 0.6, 0.1, 0.1 (stable)
    assert len(top) == 3
