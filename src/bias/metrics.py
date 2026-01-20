from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional


def entropy(dist: Dict[str, float]) -> float:
    h = 0.0
    for v in dist.values():
        if v > 0:
            h -= v * math.log(v)
    return float(h)


def js_divergence_dict(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p.keys()) | set(q.keys())
    m = {k: 0.5 * p.get(k, 0.0) + 0.5 * q.get(k, 0.0) for k in keys}

    def kl(a: Dict[str, float], b: Dict[str, float]) -> float:
        s = 0.0
        for k, av in a.items():
            if av <= 0:
                continue
            bv = b.get(k, 0.0)
            if bv <= 0:
                continue
            s += av * math.log(av / bv)
        return float(s)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def js_divergence_list(p: List[float], q: List[float]) -> Optional[float]:
    if not p or not q or len(p) != len(q):
        return None
    m = [(p[i] + q[i]) * 0.5 for i in range(len(p))]

    def kl(a: List[float], b: List[float]) -> float:
        out = 0.0
        for i, av in enumerate(a):
            if av <= 0:
                continue
            bv = b[i]
            if bv <= 0:
                continue
            out += av * math.log(av / bv)
        return float(out)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def js_similarity(p: List[float], q: List[float]) -> Optional[float]:
    jsd = js_divergence_list(p, q)
    if jsd is None:
        return None
    if jsd <= 0:
        return 1.0
    denom = math.log(2.0)
    if denom <= 0:
        return None
    sim = 1.0 - (jsd / denom)
    if sim < 0:
        return 0.0
    if sim > 1:
        return 1.0
    return float(sim)


def popularity_bins(values: List[float], bins: int) -> List[float]:
    if not values or bins <= 0:
        return []
    counts = [0] * bins
    for v in values:
        try:
            x = float(v)
        except Exception:
            continue
        if x < 0:
            idx = 0
        elif x >= 1:
            idx = bins - 1
        else:
            idx = int(x * bins)
            if idx >= bins:
                idx = bins - 1
        counts[idx] += 1
    total = sum(counts)
    if total == 0:
        return []
    return [c / total for c in counts]


def rank_utility(pcts: List[float]) -> Optional[float]:
    if not pcts:
        return None
    weights = [1.0 / math.log2(i + 2) for i in range(len(pcts))]
    denom = sum(weights)
    if denom <= 0:
        return None
    return float(sum(w * p for w, p in zip(weights, pcts)) / denom)


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def mean_bins(bins_list: List[List[float]], bins: int) -> Optional[List[float]]:
    if not bins_list or bins <= 0:
        return None
    filtered = [b for b in bins_list if b and len(b) == bins]
    if not filtered:
        return None
    sums = [0.0] * bins
    for b in filtered:
        for i in range(bins):
            sums[i] += float(b[i])
    denom = float(len(filtered))
    return [s / denom for s in sums]


def gini(counts: Iterable[int]) -> float:
    vals = sorted(int(v) for v in counts if int(v) >= 0)
    if not vals:
        return 0.0
    n = len(vals)
    s = sum(vals)
    if s == 0:
        return 0.0
    num = 0.0
    for i, v in enumerate(vals, start=1):
        num += i * v
    return float((2.0 * num) / (n * s) - (n + 1) / n)
