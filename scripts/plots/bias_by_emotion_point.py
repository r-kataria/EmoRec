from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._common import pretty_metric_name, set_plot_style


def plot(df: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if metric not in df or "emotion_top1" not in df:
        return
    sub = df[["emotion_top1", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return

    stats_df = sub.groupby("emotion_top1")[metric].agg(["mean", "std", "count"])
    stats_df = stats_df[stats_df["count"] > 0].sort_values("mean", ascending=True)
    if stats_df.empty:
        return

    means = stats_df["mean"].to_numpy()
    counts = stats_df["count"].to_numpy()
    stds = stats_df["std"].fillna(0.0).to_numpy()
    ses = np.where(counts > 1, stds / np.sqrt(counts), 0.0)
    cis = 1.96 * ses
    labels = [f"{idx} (n={int(n)})" for idx, n in zip(stats_df.index.tolist(), counts)]

    height = max(4.0, 0.33 * len(labels) + 1.5)
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, height))
    y = np.arange(len(labels))
    ax.errorbar(means, y, xerr=cis, fmt="o", color="#4c78a8", ecolor="#999999", capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(pretty_metric_name(metric))
    ax.set_title(title)
    ax.grid(axis="x")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

