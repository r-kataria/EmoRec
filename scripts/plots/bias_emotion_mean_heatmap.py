from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ._common import set_plot_style


def plot(df: pd.DataFrame, metrics: List[str], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if df.empty or "emotion_top1" not in df:
        return
    counts = df["emotion_top1"].dropna().value_counts()
    emotions = counts.index.tolist()
    rows = []
    for emo in emotions:
        row = []
        sub = df[df["emotion_top1"] == emo]
        for metric in metrics:
            if metric not in sub:
                row.append(np.nan)
                continue
            vals = pd.to_numeric(sub[metric], errors="coerce")
            row.append(float(vals.mean()) if not vals.isna().all() else np.nan)
        rows.append(row)
    if not rows:
        return

    mat = np.array(rows, dtype=float)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        if np.all(np.isnan(col)):
            continue
        cmin = np.nanmin(col)
        cmax = np.nanmax(col)
        mat[:, j] = (col - cmin) / (cmax - cmin) if cmax > cmin else np.nan

    set_plot_style()
    fig, ax = plt.subplots(figsize=(0.6 * len(metrics) + 3.5, 0.35 * len(emotions) + 2.5))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(emotions)))
    ax.set_yticklabels([f"{emo} (n={int(counts.get(emo, 0))})" for emo in emotions])
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Normalized mean (per metric)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

