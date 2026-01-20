from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ._common import pretty_metric_name, set_plot_style


def plot(df: pd.DataFrame, metrics: List[str], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if df.empty or "emotion_top1" not in df:
        return
    emotions = df["emotion_top1"].dropna().value_counts().index.tolist()
    if not emotions:
        return

    mat = []
    metric_names = []
    for metric in metrics:
        if metric not in df:
            continue
        vals = pd.to_numeric(df[metric], errors="coerce")
        overall_mean = float(vals.mean())
        overall_std = float(vals.std()) if float(vals.std()) > 0 else 1.0
        metric_names.append(metric)
        col = []
        for emo in emotions:
            mvals = pd.to_numeric(df[df["emotion_top1"] == emo][metric], errors="coerce")
            mean = float(mvals.mean()) if not mvals.isna().all() else np.nan
            col.append((mean - overall_mean) / overall_std)
        mat.append(col)
    if not mat:
        return
    mat = np.array(mat, dtype=float).T

    set_plot_style()
    fig, ax = plt.subplots(figsize=(0.6 * len(metric_names) + 3.5, 0.35 * len(emotions) + 2.5))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_xticklabels([pretty_metric_name(m) for m in metric_names], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(emotions)))
    ax.set_yticklabels(emotions)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Effect size (z)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

