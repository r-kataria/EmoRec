from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._common import set_plot_style


def plot(df: pd.DataFrame, xcol: str, ycol: str, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if xcol not in df or ycol not in df or "emotion_top1" not in df:
        return
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    mask = ~(x.isna() | y.isna())
    if mask.sum() == 0:
        return
    x = x[mask].to_numpy()
    y = y[mask].to_numpy()
    emos = df.loc[mask, "emotion_top1"].fillna("unknown").to_numpy()

    x_med = float(np.median(x))
    y_med = float(np.median(y))
    quadrants = ["low-low", "low-high", "high-low", "high-high"]

    rows = []
    emotions = sorted(set(emos))
    counts = {emo: int(np.sum(emos == emo)) for emo in emotions}
    for emo in emotions:
        sel = emos == emo
        xs = x[sel]
        ys = y[sel]
        if len(xs) == 0:
            continue
        rows.append(
            [
                np.mean((xs < x_med) & (ys < y_med)),
                np.mean((xs < x_med) & (ys >= y_med)),
                np.mean((xs >= x_med) & (ys < y_med)),
                np.mean((xs >= x_med) & (ys >= y_med)),
            ]
        )

    if not rows:
        return
    mat = np.array(rows, dtype=float)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(5.0, 0.35 * len(emotions) + 2.5))
    cmap = plt.get_cmap("Blues").copy()
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(quadrants)))
    ax.set_xticklabels(quadrants, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(emotions)))
    ax.set_yticklabels([f"{emo} (n={counts.get(emo, 0)})" for emo in emotions])
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Proportion")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

