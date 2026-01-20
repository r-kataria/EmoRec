from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from ._common import pretty_metric_name, set_plot_style


def plot(df: pd.DataFrame, value_col: str, turn_col: str, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if value_col not in df or turn_col not in df or "emotion_top1" not in df:
        return
    sub = df[["emotion_top1", value_col, turn_col]].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return

    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    counts = sub["emotion_top1"].value_counts()
    emotions = counts.index.tolist()
    cmap = plt.get_cmap("tab20")
    for i, emo in enumerate(emotions):
        grp = sub[sub["emotion_top1"] == emo]
        means = grp.groupby(turn_col)[value_col].mean().sort_index()
        ax.plot(
            means.index.to_numpy(),
            means.values,
            color=cmap(i % 20),
            linewidth=1.2,
            label=f"{emo} (n={int(counts.get(emo, 0))})",
        )
    ax.set_xlabel("Recommendation index")
    ax.set_ylabel(pretty_metric_name(value_col))
    ax.set_title(title)
    ax.grid(axis="y")
    ncol = min(6, max(2, int(math.ceil(len(emotions) / 8)))) if emotions else 2
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=min(0.8, 0.22 + 0.015 * len(emotions)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

