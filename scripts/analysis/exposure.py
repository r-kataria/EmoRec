from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def gini(counts: Iterable[int]) -> float:
    vals = sorted(int(v) for v in counts if v is not None)
    if not vals:
        return 0.0
    n = len(vals)
    s = sum(vals)
    if s == 0:
        return 0.0
    num = 0.0
    for i, v in enumerate(vals, start=1):
        num += i * v
    return float((2.0 * num) / (n * s) - (n + 1) / n)


def exposure_by_emotion_table(
    counts_by_emotion: Dict[str, Counter],
    episodes_by_emotion: Dict[str, int],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for emotion, counts in counts_by_emotion.items():
        freqs = list(counts.values())
        if not freqs:
            continue
        total = int(sum(freqs))
        rows.append(
            {
                "emotion": emotion,
                "episodes": int(episodes_by_emotion.get(emotion, 0)),
                "unique_items": int(len(freqs)),
                "total_mentions": total,
                "gini": gini(freqs),
                "mean_mentions_per_item": float(total / len(freqs)) if freqs else None,
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame([])

