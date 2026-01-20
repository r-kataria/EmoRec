from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._common import set_plot_style


def plot(df: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    series = df[metric].dropna()
    if series.empty:
        return
    set_plot_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.hist(series, bins=20, color="#72b7b2", edgecolor="#2f2f2f", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.set_ylabel("Episodes")
    ax.grid(axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

