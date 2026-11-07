import math

import krippendorff
import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from indic_judge.metrics import bootstrap_ci, cohen_kappa, confusion, krippendorff_alpha

NAN = np.nan
# Krippendorff (2011) "Computing Krippendorff's alpha-reliability", 4 coders x 12 units
K_DATA = np.array([
    [1, 2, 3, 3, 2, 1, 4, 1, 2, NAN, NAN, NAN],
    [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, NAN, 3],
    [NAN, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, NAN],
    [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, NAN],
])


def test_kappa_hand_computed():
    # confusion [[20,5],[10,15]]: po=0.7, pe=0.5 -> kappa=0.4
    a = [1] * 25 + [2] * 25
    b = [1] * 20 + [2] * 5 + [1] * 10 + [2] * 15
    assert cohen_kappa(a, b, cats=(1, 2)) == pytest.approx(0.4)


@pytest.mark.parametrize("w", [None, "linear", "quadratic"])
def test_kappa_matches_sklearn(w):
    rng = np.random.default_rng(0)
    a = rng.integers(1, 5, 200)
    b = np.clip(a + rng.integers(-1, 2, 200), 1, 4)
    assert cohen_kappa(a, b, weights=w) == pytest.approx(cohen_kappa_score(a, b, weights=w))


def test_kappa_perfect_and_constant():
    assert cohen_kappa([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert math.isnan(cohen_kappa([3, 3, 3], [3, 3, 3]))


def test_alpha_published_values():
    assert krippendorff_alpha(K_DATA, "nominal") == pytest.approx(0.743, abs=1e-3)
    assert krippendorff_alpha(K_DATA, "interval") == pytest.approx(0.849, abs=1e-3)


def test_alpha_ordinal_matches_reference_impl():
    ref = krippendorff.alpha(reliability_data=K_DATA, level_of_measurement="ordinal")
    assert krippendorff_alpha(K_DATA, "ordinal") == pytest.approx(ref, abs=1e-9)


def test_confusion_shape_and_sum():
    m = confusion([1, 2, 4], [1, 3, 4])
    assert m.shape == (4, 4) and m.sum() == 3 and m[1, 2] == 1


def test_bootstrap_ci_brackets_point_and_is_deterministic():
    rng = np.random.default_rng(1)
    a = rng.integers(1, 5, 100)
    b = np.clip(a + rng.integers(-1, 2, 100), 1, 4)
    p, lo, hi = bootstrap_ci(a, b, cohen_kappa, iters=300, seed=5)
    assert lo <= p <= hi
    assert bootstrap_ci(a, b, cohen_kappa, iters=300, seed=5) == (p, lo, hi)
