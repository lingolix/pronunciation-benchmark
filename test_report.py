"""The metrics everything else rests on, checked against hand-computed values."""

import math

from report import align, fpr_at_recall, roc_auc


def test_roc_auc_perfect_chance_and_ties():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5  # ties count half
    assert math.isnan(roc_auc([0, 0], [0.1, 0.2]))


def test_false_alarms_at_recall():
    labels = [1, 1, 0, 0, 0, 0]
    scores = [0.9, 0.6, 0.7, 0.5, 0.2, 0.1]
    # catching both errors needs threshold 0.6, which also flags the 0.7 correct word
    assert fpr_at_recall(labels, scores, 1.0) == 0.25
    assert fpr_at_recall(labels, scores, 0.5) == 0.0


def test_align_survives_an_inserted_vendor_word():
    words = [{"text": "the"}, {"text": "uh"}, {"text": "cat"}, {"text": "sat."}]
    out = align(["The", "cat", "sat"], words)
    assert [w["text"] for w in out] == ["the", "cat", "sat."]
