from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ._common import pretty_metric_name, set_plot_style


def plot(
    df: pd.DataFrame,
    metrics: List[str],
    out_path: Path,
    title: str,
    max_samples: int = 1000,
    seed: int = 7,
) -> None:
    import matplotlib.pyplot as plt

    if df.empty or "emotion_top1" not in df:
        return
    use_metrics = [m for m in metrics if m in df]
    if not use_metrics:
        return

    sub = df[["emotion_top1"] + use_metrics].copy()
    for m in use_metrics:
        sub[m] = pd.to_numeric(sub[m], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return
    if len(sub) > max_samples:
        sub = sub.sample(n=max_samples, random_state=seed)

    for m in use_metrics:
        vals = sub[m]
        vmin, vmax = float(vals.min()), float(vals.max())
        sub[m] = (vals - vmin) / (vmax - vmin) if vmax > vmin else 0.0

    emotions = sub["emotion_top1"].unique().tolist()
    cmap = plt.get_cmap("tab20")
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(use_metrics))
    for i, emo in enumerate(emotions):
        rows = sub[sub["emotion_top1"] == emo]
        for _, row in rows.iterrows():
            ax.plot(x, row[use_metrics].to_numpy(), color=cmap(i % 20), alpha=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_metric_name(m) for m in use_metrics], rotation=45, ha="right")
    ax.set_ylabel("Normalized value")
    ax.set_title(title)
    ax.grid(axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

