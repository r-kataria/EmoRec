from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._common import set_plot_style


def _gini(counts: List[int]) -> float:
    vals = sorted(int(v) for v in counts if v is not None)
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


def _exposure_curve(
    exposure_counts: Counter,
    popularity_percentiles: Dict[Any, float],
) -> Optional[Tuple[List[float], List[float], float]]:
    pairs: List[Tuple[float, int]] = []
    for item, count in exposure_counts.items():
        pct = popularity_percentiles.get(item)
        if pct is None:
            continue
        try:
            pct_f = float(pct)
        except Exception:
            continue
        pairs.append((pct_f, int(count)))

    if not pairs:
        return None

    pairs.sort(key=lambda x: x[0])
    total = sum(c for _, c in pairs)
    if total <= 0:
        return None

    xs = [0.0]
    ys = [0.0]
    cum = 0
    for pct, count in pairs:
        cum += count
        xs.append(min(max(pct, 0.0), 1.0))
        ys.append(cum / total)
    if xs[-1] < 1.0:
        xs.append(1.0)
        ys.append(1.0)

    return xs, ys, _gini([c for _, c in pairs])


def plot(
    dataset_counts: Counter,
    counts_by_emotion: Dict[str, Counter],
    popularity_percentiles: Dict[Any, float],
    episodes_by_emotion: Dict[str, int],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    base = _exposure_curve(dataset_counts, popularity_percentiles)
    if not base:
        return
    xs_all, ys_all, gini_all = base

    ordered = sorted(
        counts_by_emotion.keys(),
        key=lambda k: (episodes_by_emotion.get(k, 0), k),
        reverse=True,
    )

    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(xs_all, ys_all, color="#4c78a8", linewidth=2, label=f"all (Gini={gini_all:.2f})")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#999999", linewidth=1)

    cmap = plt.get_cmap("tab10")
    for i, emotion in enumerate(ordered):
        curve = _exposure_curve(counts_by_emotion[emotion], popularity_percentiles)
        if not curve:
            continue
        xs, ys, gini_val = curve
        n = int(episodes_by_emotion.get(emotion, 0))
        ax.plot(
            xs,
            ys,
            linestyle=":",
            linewidth=1.5,
            color=cmap(i % 10),
            label=f"{emotion} (n={n}, Gini={gini_val:.2f})",
        )

    ax.set_title(title)
    ax.set_xlabel("Popularity percentile (items sorted by popularity)")
    ax.set_ylabel("Cumulative exposure share")
    ax.grid(axis="y")
    ncol = min(6, max(2, int(math.ceil(len(ordered) / 8)))) if ordered else 2
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=min(0.8, 0.22 + 0.015 * len(ordered)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

