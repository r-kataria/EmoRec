from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ._common import bootstrap_mean_ci, set_plot_style


def plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    cols = ["p01", "p05", "p10"]
    if any(c not in df for c in cols) or "emotion_top1" not in df:
        return
    counts = df["emotion_top1"].dropna().value_counts()
    labels = counts.index.tolist()
    if not labels:
        return

    x = np.array([1, 5, 10], dtype=float)
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    cmap = plt.get_cmap("tab20")
    for i, lab in enumerate(labels):
        count = int(counts.get(lab, 0))
        sub = df[df["emotion_top1"] == lab]
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for c in cols:
            vals = pd.to_numeric(sub[c], errors="coerce").dropna().to_numpy()
            if len(vals) == 0:
                means.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)
                continue
            means.append(float(np.mean(vals)))
            lo, hi = bootstrap_mean_ci(vals)
            lows.append(lo)
            highs.append(hi)
        color = cmap(i % 20)
        ax.plot(
            x,
            means,
            marker="o",
            linestyle="-",
            linewidth=1.0,
            color=color,
            alpha=0.7,
            label=f"{lab} (n={count})",
        )
        ax.fill_between(x, lows, highs, color=color, alpha=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(["1%", "5%", "10%"])
    ax.set_xlabel("Popularity threshold")
    ax.set_ylabel("Mean coverage")
    ax.set_title(title)
    ax.grid(axis="y")
    ncol = min(6, max(2, int(math.ceil(len(labels) / 8)))) if labels else 2
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=min(0.8, 0.22 + 0.015 * len(labels)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

