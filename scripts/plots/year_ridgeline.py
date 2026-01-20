from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._common import set_plot_style


def plot(df: pd.DataFrame, out_path: Path, title: str, bins: int = 25) -> None:
    import matplotlib.pyplot as plt

    if "mean_year" not in df or "emotion_top1" not in df:
        return
    sub = df[["emotion_top1", "mean_year"]].copy()
    sub["mean_year"] = pd.to_numeric(sub["mean_year"], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return

    counts = sub["emotion_top1"].value_counts()
    emotions = counts.index.tolist()
    years = sub["mean_year"].to_numpy()
    ymin, ymax = np.nanmin(years), np.nanmax(years)
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return

    set_plot_style()
    fig, ax = plt.subplots(figsize=(8.0, 0.35 * len(emotions) + 2.5))
    for i, emo in enumerate(emotions):
        vals = sub[sub["emotion_top1"] == emo]["mean_year"].to_numpy()
        if len(vals) == 0:
            continue
        hist, edges = np.histogram(vals, bins=bins, range=(ymin, ymax), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        y = hist / (hist.max() if hist.max() > 0 else 1.0)
        ax.fill_between(centers, i, i + y, color="#4c78a8", alpha=0.6)
        ax.plot(centers, i + y, color="#4c78a8", linewidth=1.0)
    ax.set_yticks(np.arange(len(emotions)) + 0.2)
    ax.set_yticklabels([f"{emo} (n={int(counts.get(emo, 0))})" for emo in emotions])
    ax.set_xlabel("Mean release year")
    ax.set_title(title)
    ax.grid(axis="x")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

