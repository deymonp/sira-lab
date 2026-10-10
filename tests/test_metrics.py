from sira_lab.diffview import word_diff
from sira_lab.lab.metrics import auroc, edit_ratio, tpr_at_fpr


def test_tpr_auroc():
    pos = [5, 6, 7, 8]
    neg = [0, 1, 2, 3, 4]
    assert tpr_at_fpr(pos, neg, 0.01) == 1.0
    assert auroc(pos, neg) == 1.0
    assert auroc(neg, pos) == 0.0
    assert tpr_at_fpr([0, 1], [0, 1, 2, 3], 0.01) == 0.0


def test_edit_ratio_and_diff():
    assert edit_ratio("a b c", "a b c") == 1.0
    d = word_diff("the cat sat", "the dog sat")
    assert ("cat ", "removed") in d and ("dog ", "added") in d
