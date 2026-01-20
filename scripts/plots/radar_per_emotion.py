from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ._common import pretty_metric_name, set_plot_style


def plot(df: pd.DataFrame, metrics: List[str], out_dir: Path, title_prefix: str) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    use_metrics = [m for m in metrics if m in df]
    if not use_metrics:
        return
    sub = df[["emotion_top1"] + use_metrics].copy()
    for m in use_metrics:
        sub[m] = pd.to_numeric(sub[m], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return

    for m in use_metrics:
        vmin, vmax = float(sub[m].min()), float(sub[m].max())
        sub[m] = (sub[m] - vmin) / (vmax - vmin) if vmax > vmin else 0.0

    counts = sub["emotion_top1"].value_counts()
    emotions = sub["emotion_top1"].unique().tolist()
    angles = np.linspace(0, 2 * np.pi, len(use_metrics), endpoint=False).tolist()
    angles += angles[:1]

    for emo in emotions:
        rows = sub[sub["emotion_top1"] == emo]
        vals = [float(rows[m].mean()) for m in use_metrics] + [float(rows[use_metrics[0]].mean())]
        set_plot_style()
        fig, ax = plt.subplots(figsize=(5.2, 4.8), subplot_kw={"polar": True})
        ax.plot(angles, vals, color="#4c78a8", linewidth=2)
        ax.fill(angles, vals, color="#4c78a8", alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([pretty_metric_name(m) for m in use_metrics], fontsize=8)
        ax.set_yticklabels([])
        ax.set_title(f"{title_prefix}: {emo} (n={int(counts.get(emo, 0))})", y=1.08)
        fig.tight_layout()
        fig.savefig(out_dir / f"radar_{emo}.png")
        plt.close(fig)
