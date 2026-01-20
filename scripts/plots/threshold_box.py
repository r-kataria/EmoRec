from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._common import set_plot_style


def plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    cols = ["p01", "p05", "p10"]
    series = [df[c].dropna().to_numpy() if c in df else np.array([]) for c in cols]
    if all(len(s) == 0 for s in series):
        return
    set_plot_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    try:
        ax.boxplot(
            series,
            tick_labels=["P@1%", "P@5%", "P@10%"],
            showfliers=False,
            widths=0.5,
            patch_artist=True,
            boxprops={"facecolor": "#f58518", "alpha": 0.6},
            medianprops={"color": "#222222"},
        )
    except TypeError:
        ax.boxplot(
            series,
            labels=["P@1%", "P@5%", "P@10%"],
            showfliers=False,
            widths=0.5,
            patch_artist=True,
            boxprops={"facecolor": "#f58518", "alpha": 0.6},
            medianprops={"color": "#222222"},
        )
    ax.set_title(title)
    ax.set_ylabel("Coverage distribution")
    ax.grid(axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

