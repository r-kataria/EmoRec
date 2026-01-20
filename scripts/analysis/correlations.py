from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pandas as pd


def pearson_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    if len(x) < 3 or len(set(y)) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2:
            return float("nan")
        if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def _normal_approx_p(r: float, n: int) -> float:
        if not np.isfinite(r) or n < 3 or abs(r) >= 1:
            return float("nan")
        t = abs(r) * math.sqrt((n - 2) / max(1e-12, 1.0 - r * r))
        # Normal approximation (good for large n).
        phi = 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))
        return float(2.0 * (1.0 - phi))

    pr = _pearson_r(x, y)
    pp = _normal_approx_p(pr, len(x))

    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    sr = _pearson_r(rx, ry)
    sp = _normal_approx_p(sr, len(rx))

    try:
        from scipy import stats  # type: ignore

        pr_obj = stats.pearsonr(x, y)
        sr_obj = stats.spearmanr(x, y)
        pr, pp, sr, sp = (
            float(pr_obj.statistic),
            float(pr_obj.pvalue),
            float(sr_obj.correlation),
            float(sr_obj.pvalue),
        )
    except Exception:
        pass

    return pr, pp, sr, sp


def corr_table(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []
    labels = sorted(df["emotion_top1"].dropna().unique().tolist())
    for metric in metrics:
        if metric not in df:
            continue
        series = df[metric].astype(float)
        if series.isna().all():
            continue
        for lab in labels:
            mask = ~series.isna()
            x = series[mask].to_numpy()
            y = (df.loc[mask, "emotion_top1"] == lab).astype(int).to_numpy()
            pr, pp, sr, sp = pearson_spearman(x, y)
            rows.append(
                {
                    "metric": metric,
                    "emotion": lab,
                    "n": int(len(x)),
                    "pearson_r": pr,
                    "pearson_p": pp,
                    "spearman_r": sr,
                    "spearman_p": sp,
                }
            )
    return pd.DataFrame(rows)
