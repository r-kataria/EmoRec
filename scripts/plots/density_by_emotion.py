from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._common import ensure_dir, pretty_metric_name, set_plot_style


def plot(df: pd.DataFrame, xcol: str, ycol: str, out_dir: Path, title_prefix: str) -> None:
    import matplotlib.pyplot as plt

    if xcol not in df or ycol not in df:
        return
    ensure_dir(out_dir)
    emotions = df["emotion_top1"].dropna().value_counts().index.tolist()
    for emo in emotions:
        sub = df[df["emotion_top1"] == emo]
        x = pd.to_numeric(sub[xcol], errors="coerce")
        y = pd.to_numeric(sub[ycol], errors="coerce")
        mask = ~(x.isna() | y.isna())
        if mask.sum() == 0:
            continue
        x = x[mask].to_numpy()
        y = y[mask].to_numpy()
        set_plot_style()
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ax.hexbin(x, y, gridsize=20, cmap="Blues", mincnt=1)
        ax.set_xlabel(pretty_metric_name(xcol))
        ax.set_ylabel(pretty_metric_name(ycol))
        ax.set_title(f"{title_prefix}: {emo}")
        ax.grid(axis="both")
        fig.tight_layout()
        fig.savefig(out_dir / f"{xcol}_{ycol}_{emo}.png")
        plt.close(fig)

