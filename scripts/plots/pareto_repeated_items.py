from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np

from ._common import set_plot_style


def plot(counts_by_emotion: Dict[str, Counter], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if not counts_by_emotion:
        return
    emotions = sorted(counts_by_emotion.keys(), key=lambda e: sum(counts_by_emotion[e].values()), reverse=True)
    if not emotions:
        return

    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cmap = plt.get_cmap("tab20")
    for i, emo in enumerate(emotions):
        freqs = sorted(counts_by_emotion[emo].values(), reverse=True)
        total = sum(freqs)
        if total <= 0:
            continue
        cum = np.cumsum(freqs) / total
        x = np.linspace(0, 1, len(cum), endpoint=True)
        ax.plot(x, cum, color=cmap(i % 20), linewidth=1.2, label=f"{emo} (n={total})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    ax.set_xlabel("Share of items (sorted)")
    ax.set_ylabel("Cumulative share of repeats")
    ax.set_title(title)
    ax.grid(axis="y")
    ncol = min(6, max(2, int(math.ceil(len(emotions) / 8)))) if emotions else 2
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=min(0.8, 0.22 + 0.015 * len(emotions)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

