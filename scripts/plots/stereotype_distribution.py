from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._common import set_plot_style


def plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if "stereotype_label" not in df:
        return
    counts = df["stereotype_label"].value_counts().sort_index()
    if counts.empty:
        return
    total = float(counts.sum())
    if total <= 0:
        return
    labels = ["Non-biased", "Biased"]
    values = [(counts.get(0, 0) / total) * 100.0, (counts.get(1, 0) / total) * 100.0]
    set_plot_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.bar(labels, values, color=["#54a24b", "#e45756"])
    ax.set_title(title)
    ax.set_ylabel("Percent of episodes")
    ax.grid(axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

