from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from ._common import set_plot_style


def plot(stats: Dict[str, Any], out_dir: Path, title_prefix: str, top_k: int = 5) -> None:
    import matplotlib.pyplot as plt

    overall = stats.get("overall_mean") or {}
    emo_mean = stats.get("emotion_mean") or {}
    if not overall or not emo_mean:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = stats.get("emotion_counts") or {}
    for emo, dist in emo_mean.items():
        deltas = {g: float(dist.get(g, 0.0) - overall.get(g, 0.0)) for g in overall}
        top = sorted(deltas.items(), key=lambda x: x[1], reverse=True)[:top_k]
        if not top:
            continue
        labels = [g for g, _ in top][::-1]
        values = [v for _, v in top][::-1]
        set_plot_style()
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        y = np.arange(len(labels))
        ax.hlines(y=y, xmin=0, xmax=values, color="#c7c7c7", linewidth=1.0)
        ax.plot(values, y, "o", color="#4c78a8", markersize=5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Lift vs overall")
        ax.set_title(f"{title_prefix}: {emo} (n={counts.get(emo, 0)})")
        ax.grid(axis="x")
        fig.tight_layout()
        fig.savefig(out_dir / f"top_uplift_{emo}.png")
        plt.close(fig)

