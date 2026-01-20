from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ._common import set_plot_style


def plot(
    counts_by_emotion: Dict[str, Counter],
    episodes_by_emotion: Optional[Dict[str, int]],
    out_path: Path,
    title: str,
    ks: Tuple[int, int, int] = (10, 50, 100),
) -> None:
    import matplotlib.pyplot as plt

    rows = []
    for emo, counts in counts_by_emotion.items():
        freqs = sorted(counts.values(), reverse=True)
        total = sum(freqs)
        if total <= 0:
            continue
        shares = [float(sum(freqs[:k]) / total) for k in ks]
        n = int(episodes_by_emotion.get(emo, 0)) if episodes_by_emotion else int(sum(counts.values()))
        rows.append((emo, n, shares))
    if not rows:
        return
    rows.sort(key=lambda x: x[1], reverse=True)
    labels = [f"{r[0]} (n={r[1]})" for r in rows]
    data = np.array([r[2] for r in rows], dtype=float)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(8.2, 0.35 * len(labels) + 2.5))
    y = np.arange(len(labels))
    height = 0.25
    for i, k in enumerate(ks):
        ax.barh(y + (i - 1) * height, data[:, i], height=height, label=f"top{k}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Share of exposure")
    ax.set_title(title)
    ax.grid(axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(ks), fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

