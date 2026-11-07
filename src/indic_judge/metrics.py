"""Agreement metrics, implemented directly so they can be unit-tested against known values."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

CATS = (1, 2, 3, 4)


def confusion(a: Sequence[int], b: Sequence[int], cats=CATS) -> np.ndarray:
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)), dtype=float)
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    return m


def cohen_kappa(a, b, cats=CATS, weights: str | None = None) -> float:
    """Cohen's kappa. weights=None (unweighted), 'linear' or 'quadratic'."""
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("need two equal-length non-empty rating lists")
    o = confusion(a, b, cats)
    n = o.sum()
    e = np.outer(o.sum(1), o.sum(0)) / n
    k = len(cats)
    i, j = np.indices((k, k))
    if weights is None:
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j) / (k - 1)
    elif weights == "quadratic":
        w = ((i - j) / (k - 1)) ** 2
    else:
        raise ValueError(weights)
    denom = (w * e).sum()
    if denom == 0:  # both raters constant and identical -> perfect but undefined; report nan
        return float("nan")
    return float(1 - (w * o).sum() / denom)


def krippendorff_alpha(data, level: str = "ordinal") -> float:
    """Krippendorff's alpha. data: raters x units array, np.nan = missing.
    level: 'nominal', 'ordinal' or 'interval'."""
    data = np.asarray(data, dtype=float)
    values = np.unique(data[~np.isnan(data)])
    vidx = {v: i for i, v in enumerate(values)}
    V = len(values)
    o = np.zeros((V, V))
    for u in range(data.shape[1]):
        col = data[:, u]
        col = col[~np.isnan(col)]
        m = len(col)
        if m < 2:
            continue
        for x in range(m):
            for y in range(m):
                if x != y:
                    o[vidx[col[x]], vidx[col[y]]] += 1.0 / (m - 1)
    nc = o.sum(1)
    n = nc.sum()
    if n == 0 or V < 2:
        return float("nan")
    if level == "nominal":
        d = 1.0 - np.eye(V)
    elif level == "interval":
        d = (values[:, None] - values[None, :]) ** 2
    elif level == "ordinal":
        d = np.zeros((V, V))
        for c in range(V):
            for k in range(V):
                lo, hi = min(c, k), max(c, k)
                d[c, k] = (nc[lo:hi + 1].sum() - (nc[c] + nc[k]) / 2) ** 2
    else:
        raise ValueError(level)
    do = (o * d).sum()
    de = (np.outer(nc, nc) * d).sum() / (n - 1)
    return float(1 - do / de) if de > 0 else float("nan")


def bootstrap_ci(a, b, stat: Callable, iters=1000, ci=0.95, seed=13) -> tuple[float, float, float]:
    """Paired percentile bootstrap over units. Returns (point, lo, hi)."""
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    point = stat(a, b)
    vals = []
    for _ in range(iters):
        ix = rng.integers(0, len(a), len(a))
        v = stat(a[ix], b[ix])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(point), float(lo), float(hi)


def agreement_row(a, b, iters=1000, ci=0.95, seed=13) -> dict:
    a, b = list(map(int, a)), list(map(int, b))
    k, k_lo, k_hi = bootstrap_ci(a, b, cohen_kappa, iters, ci, seed)
    q, q_lo, q_hi = bootstrap_ci(a, b, lambda x, y: cohen_kappa(x, y, weights="quadratic"),
                                 iters, ci, seed)
    al, al_lo, al_hi = bootstrap_ci(a, b, lambda x, y: krippendorff_alpha(np.vstack([x, y])),
                                    iters, ci, seed)
    return {"n": len(a), "exact_agree": float(np.mean(np.array(a) == np.array(b))),
            "kappa": k, "kappa_lo": k_lo, "kappa_hi": k_hi,
            "qwk": q, "qwk_lo": q_lo, "qwk_hi": q_hi,
            "alpha_ord": al, "alpha_lo": al_lo, "alpha_hi": al_hi}
