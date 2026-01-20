from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ._common import set_plot_style


def plot(stats: Dict[str, Any], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    overall = stats.get("overall_mean") or {}
    emo_mean = stats.get("emotion_mean") or {}
    if not overall or not emo_mean:
        return

    decades = sorted(overall.keys())
    counts = stats.get("emotion_counts") or {}
    emotions = sorted(emo_mean.keys(), key=lambda e: counts.get(e, 0), reverse=True)
    if not decades or not emotions:
        return

    set_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 0.35 * len(emotions) + 2.5))
    bottoms = np.zeros(len(emotions), dtype=float)
    cmap = plt.get_cmap("tab20")

    for i, dec in enumerate(decades):
        vals = np.array([emo_mean.get(e, {}).get(dec, 0.0) for e in emotions], dtype=float)
        ax.barh(np.arange(len(emotions)), vals, left=bottoms, color=cmap(i % 20), label=dec)
        bottoms += vals

    ax.set_yticks(np.arange(len(emotions)))
    ax.set_yticklabels([f"{emo} (n={counts.get(emo, 0)})" for emo in emotions])
    ax.set_xlabel("Proportion")
    ax.set_title(title)
    ax.grid(axis="x")
    ncol = min(6, max(2, int(math.ceil(len(decades) / 8)))) if decades else 2
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=min(0.8, 0.22 + 0.015 * len(decades)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

