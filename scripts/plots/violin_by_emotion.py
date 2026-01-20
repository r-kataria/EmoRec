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

    stats_df = sub.groupby("emotion_top1")[metric].agg(["count", "median"])
    stats_df = stats_df[stats_df["count"] > 0].sort_values("median", ascending=True)
    if stats_df.empty:
        return

    labels = [f"{idx} (n={int(row['count'])})" for idx, row in stats_df.iterrows()]
    data = [sub[sub["emotion_top1"] == idx][metric].to_numpy() for idx in stats_df.index.tolist()]

    height = max(4.0, 0.35 * len(labels) + 1.5)
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, height))
    parts = ax.violinplot(data, showmeans=False, showmedians=True, vert=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4c78a8")
        pc.set_alpha(0.7)
    ax.set_yticks(np.arange(1, len(labels) + 1))
    ax.set_yticklabels(labels)
    ax.set_xlabel(pretty_metric_name(metric))
    ax.set_title(title)
    ax.grid(axis="x")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

