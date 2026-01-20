from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ._common import pretty_metric_name, set_plot_style


def plot(df: pd.DataFrame, xcol: str, ycol: str, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if xcol not in df or ycol not in df:
        return
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    mask = ~(x.isna() | y.isna())
    if mask.sum() == 0:
        return
    x = x[mask].to_numpy()
    y = y[mask].to_numpy()
    labels = df.loc[mask, "emotion_top1"].fillna("unknown").to_numpy()
    uniq = sorted(set(labels))
    counts = {lab: int(np.sum(labels == lab)) for lab in uniq}

    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    cmap = plt.get_cmap("tab20")
    for i, lab in enumerate(uniq):
        sel = labels == lab
        ax.scatter(
            x[sel],
            y[sel],
            s=12,
            alpha=0.35,
            color=cmap(i % 20),
            label=f"{lab} (n={counts.get(lab, 0)})",
        )
    ax.set_xlabel(pretty_metric_name(xcol))
    ax.set_ylabel(pretty_metric_name(ycol))
    ax.set_title(title)
    ax.grid(axis="both")
    ncol = min(6, max(2, int(math.ceil(len(uniq) / 8)))) if uniq else 2
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=min(0.8, 0.22 + 0.015 * len(uniq)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

